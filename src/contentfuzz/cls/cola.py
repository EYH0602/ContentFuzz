"""This code is modified based on https://github.com/tsinghua-fib-lab/COLA
Stance Detection with Collaborative Role-Infused LLM-Based Agents (ICWSM 2024)
"""

from returns.result import safe, Result, ResultE
from google.genai.types import (
    GenerateContentConfig,
    AutomaticFunctionCallingConfig,
    ThinkingConfig,
)

from ._base import AnalysisOutput
from .utils import classify_w_prob, get_vertexai_client
from ..utils import exp_retry

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
    """Zero-shot stance analysis using Google Gemini API"""

    def __init__(self, model: str = "gemini-2.5-flash-lite") -> None:
        # self.model, self.client = _get_model_and_client(model)
        self.model = model
        self.client = get_vertexai_client()

    @safe
    @exp_retry
    def get_completion_with_role(self, role: str, instruction: str, tweet: str) -> str:
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

        # NOTE: original OpenAI version
        # saved in comments for reference
        # response = self.client.chat.completions.create(
        #     model=self.model,
        #     messages=[
        #         {"role": "system", "content": f"You are a {role}."},
        #         {"role": "user", "content": f"{instruction}\n{tweet}"},
        #     ],
        #     temperature=0,
        # )
        #
        # choice = response.choices[0]
        # content: str | None = choice.message.content
        # if content is None:
        #     raise ValueError("OpenAI API returned no output")

        # return content

        response = self.client.models.generate_content(
            model=self.model,
            contents=f"{instruction}\n{tweet}",
            config=GenerateContentConfig(
                temperature=0,
                system_instruction=f"You are a {role}.",
                automatic_function_calling=AutomaticFunctionCallingConfig(disable=True),
                thinking_config=ThinkingConfig(
                    thinking_budget=0,  # disable thinking
                ),
            ),
        )

        text = response.text
        if text is None:
            raise ValueError("Gemini API returned no output")
        return text

    @safe
    @exp_retry
    def get_completion(self, prompt: str) -> str:
        """Get completion from OpenAI API.

        Args:
            client (OpenAI): The OpenAI client instance.
            model (str): The model to use for completion.
            prompt (str): The prompt to send to the model.

        Returns:
            str: The model's response or None if an error occurred.
        """
        # NOTE: origional OpenAI version saved in comments for reference
        # response = self.client.chat.completions.create(
        #     model=self.model,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0,
        # )

        # choice = response.choices[0]
        # content: str | None = choice.message.content
        # if content is None:
        #     raise ValueError("OpenAI API returned no output")

        # return content

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0,
                automatic_function_calling=AutomaticFunctionCallingConfig(disable=True),
                thinking_config=ThinkingConfig(
                    thinking_budget=0,  # disable thinking
                ),
            ),
        )
        text = response.text
        if text is None:
            raise ValueError("Gemini API returned no output")
        return text

    def linguist_analysis(self, tweet: str) -> ResultE[str]:
        """Let an expert linguist analyze the tweet."""

        return self.get_completion_with_role("linguist", LINGUIST_INSTRUCTION, tweet)

    def expert_analysis(self, tweet: str, target: str) -> ResultE[str]:
        """Let a domain expert analyze the tweet."""
        role = target_role_map.get(target, "expert")
        instruction = EXPERT_INSTRUCTION.format(target=target)
        return self.get_completion_with_role(role, instruction, tweet)

    def user_analysis(self, tweet: str) -> ResultE[str]:
        """Let a heavy social media user analyze the tweet."""
        return self.get_completion_with_role(
            "heavy social media user", TOP_USER_INSTRUCTION, tweet
        )

    def stance_analysis(  # pylint: disable=R0913,R0917
        self,
        tweet: str,
        ling_response: str,
        expert_response: str,
        user_response: str,
        target: str,
        stance: str,
    ) -> ResultE[str]:
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

    def final_judgement(
        self, tweet: str, favor_response: str, against_response: str, target: str
    ) -> ResultE[AnalysisOutput]:
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

        return classify_w_prob(self.client, self.model, None, prompt)

    def analyze(self, text: str, target: str) -> ResultE[AnalysisOutput]:
        """Using COLA to analyze the stance of a given text"""

        tweet = text

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
                tweet,
                favor_response,
                against_response,
                target,
            )
        )
