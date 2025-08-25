"""This code is modified based on https://github.com/tsinghua-fib-lab/COLA
Stance Detection with Collaborative Role-Infused LLM-Based Agents (ICWSM 2024)
"""

import os
from math import exp

from openai import OpenAI
from returns.result import safe, Result, ResultE
from returns.maybe import maybe
from structured_logprobs import add_logprobs

from ._base import AnalysisOutput, StanceOutput
from .utils import exp_retry

# assign experts for target
target_role_map = {
    "Atheism": "theologian",
    "Climate Change is a Real Concern": "environmental scientist",
    "Feminist Movement": "sociologist",
    "Hillary Clinton": "political scientist",
    "Legalization of Abortion": "sociologist",
    "Donald Trump": "political scientist",
}


LINGUIST_INSTRUCTION = """
Accurately and concisely explain the linguistic elements in the sentence and how these elements affect meaning, 
including grammatical structure, tense and inflection, virtual speech, rhetorical devices, lexical choices and so on. 
Do nothing else."""

EXPERT_INSTRUCTION = """
Accurately and concisely explain the key elements contained in the quote, 
such as characters, events, parties, religions, etc. 
Also explain their relationship with {target} (if exist). 
Do nothing else."""

TOP_USER_INSTRUCTION = """
Analyze the following sentence, focusing on the content, hashtags, 
Internet slang and colloquialisms, emotional tone, implied meaning, and so on. 
Do nothing else."""

STANCE_ANALYSIS_INSTRUCTION = """
'''{tweet}'''
<<<{ling_response}>>>
[[[{expert_response}]]]
---{user_response}---

You think the attitude behind the sentence surrounded by ''' ''' is {stance} of {target}.
The content enclosed by <<< >>> represents linguistic analysis. The content within [[[ ]]] represents the analysis of a {role}. 
The content enclosed by --- ---  represents the analysis of a heavy social media user. 
Identify the top three pieces of evidence from these that best support your opinion and argue for your opinion.
"""

FINAL_JUDGEMENT_INST = """
Determine whether the sentence is in favor of or against {target}, or is irrelevant to {target}.
Sentence: {tweet}
Judge this in relation to the following arguments:
Arguments that the attitude is in favor: {favor_response}
Arguments that the attitude is against: {against_response}
Choose from:\n A: Against\nB: Favor\nC: Irrelevant\n 
Constraint: Answer with only the option above that is most accurate and nothing else.
"""


class COLA:
    """Zero-shot stance analysis using OpenAI API"""

    def __init__(self, model: str = "gpt-4.1-nano"):

        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None, "OPENAI_API_KEY environment variable is not set"

        self.model = model
        self.client = OpenAI(api_key=api_key)

    @maybe
    @exp_retry
    def get_completion_with_role(
        self, role: str, instruction: str, content: str
    ) -> str | None:
        """Get completion from OpenAI API with specified role.

        Args:
            client (OpenAI): The OpenAI client instance.
            model (str): The model to use for completion.
            role (str): The role of the user (e.g., "linguist", "expert").
            instruction (str): The instruction for the model.
            content (str): The content to analyze.

        Returns:
            str | None: The model's response or None if an error occurred.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are a {role}."},
                {"role": "user", "content": f"{instruction}\n{content}"},
            ],
            temperature=0,
        )

        choice = response.choices[0]

        return choice.message.content

    @maybe
    @exp_retry
    def get_completion(self, prompt: str) -> str | None:
        """Get completion from OpenAI API.

        Args:
            client (OpenAI): The OpenAI client instance.
            model (str): The model to use for completion.
            prompt (str): The prompt to send to the model.

        Returns:
            str | None: The model's response or None if an error occurred.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        choice = response.choices[0]
        return choice.message.content

    def linguist_analysis(self, tweet: str):
        """Let an expert linguist analyze the tweet."""

        return self.get_completion_with_role("linguist", LINGUIST_INSTRUCTION, tweet)

    def expert_analysis(self, tweet: str, target: str):
        """Let a domain expert analyze the tweet."""
        role = target_role_map.get(target, "expert")
        instruction = EXPERT_INSTRUCTION.format(target=target)
        return self.get_completion_with_role(role, instruction, tweet)

    def user_analysis(self, tweet: str):
        """Let a heavy social media user analyze the tweet."""
        return self.get_completion_with_role(
            "heavy social media user", TOP_USER_INSTRUCTION, tweet
        )

    def stance_analysis(
        self,
        tweet: str,
        ling_response: str,
        expert_response: str,
        user_response: str,
        target: str,
        stance: str,
    ):
        """Try to do stance analysis with a given stance"""
        role = target_role_map.get(target, "expert")
        prompt = STANCE_ANALYSIS_INSTRUCTION.format(
            tweet=tweet,
            ling_response=ling_response,
            expert_response=expert_response,
            user_response=user_response,
            target=target,
            stance=stance,
            role=role,
        )
        return self.get_completion(prompt)

    @safe
    def final_judgement(
        self, tweet: str, favor_response: str, against_response: str, target: str
    ):
        """
        By the COLA authors:
        This is an example of a prompt for the final judgement stage.
        Most of the time, this prompt can be used directly.
        For some targets, it needs to include a more detailed explanation of the task
        to achieve the performance reported in our paper.
        We believe that automating the addition of specific explanations and evaluation criteria
        for the task is a direction for future improvement.
        If you have any questions, feel free to discuss them with me!
        """
        prompt = FINAL_JUDGEMENT_INST.format(
            tweet=tweet,
            favor_response=favor_response,
            against_response=against_response,
            target=target,
        )

        # NOTE: We added structured decoding in this step
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            logprobs=True,
            response_format=StanceOutput,
            temperature=0,  # for reproduce
        )
        chat_completion = add_logprobs(completion)
        response = chat_completion.value
        probs = chat_completion.log_probs[0]
        # apply `exp` to all values of probs
        probs = {label: exp(logit) for label, logit in probs.items()}
        output = response.choices[0].message.content

        if output is None:
            raise ValueError("OpenAI API returned no output")

        stance = StanceOutput.model_validate_json(output)

        return stance, probs.get("label")

    def analyze(self, text: str, target: str | None = None) -> ResultE[AnalysisOutput]:
        """Using COLA to analyze the stance of a given text"""

        tweet = text
        if target is None:
            target = "the topic"

        return Result.do(
            final_response
            # Step 1: Linguist analysis
            for ling_response in self.linguist_analysis(tweet)
            # Step 2: Expert analysis
            for expert_response in self.expert_analysis(tweet, target)
            # Step 3: Heavy social media user analysis
            for user_response in self.user_analysis(tweet)
            # Step 4: Debate
            for favor_response in self.stance_analysis(
                tweet,
                ling_response,
                expert_response,
                user_response,
                target,
                "in favor",
            )
            for against_response in self.stance_analysis(
                tweet,
                ling_response,
                expert_response,
                user_response,
                target,
                "against",
            )
            # Step 5: Final judgement
            for final_response in self.final_judgement(
                tweet, favor_response, against_response, target
            )
        )
