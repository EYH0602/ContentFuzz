"""This code is modified based on https://github.com/tsinghua-fib-lab/COLA
Stance Detection with Collaborative Role-Infused LLM-Based Agents (ICWSM 2024)
"""

from google.genai.types import (
    AutomaticFunctionCallingConfig,
    GenerateContentConfig,
    ThinkingConfig,
)
from returns.result import Result, ResultE, safe

from ..utils import Language, exp_retry
from ._base import AnalysisOutput
from ._cola_prompts import (
    PROMPT_SETS,
    ROLE_TRANSLATIONS,
    STANCE_TRANSLATIONS,
    COLAStance,
    Role,
)
from .utils import classify_w_prob, get_vertexai_client

# assign experts for target
target_role_map: dict[str, Role] = {
    "Atheism": "theologian",
    "Climate Change is a Real Concern": "environmental scientist",
    "Feminist Movement": "sociologist",
    "Hillary Clinton": "political scientist",
    "Legalization of Abortion": "sociologist",
    "Donald Trump": "political scientist",
}


class COLA:
    """Zero-shot stance analysis using Google Gemini API"""

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        language: Language = "en",
    ) -> None:
        # self.model, self.client = _get_model_and_client(model)
        self.model = model
        self.client = get_vertexai_client()
        self.language: Language = language
        try:
            self.prompts = PROMPT_SETS[language]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unsupported language: {language}") from exc

    def _translate_role(self, role: Role) -> str:
        """Translate role name to the appropriate language."""
        return ROLE_TRANSLATIONS[self.language][role]

    def _translate_stance(self, stance: COLAStance) -> str:
        """Translate stance to the appropriate language."""
        return STANCE_TRANSLATIONS[self.language][stance]

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

        # NOTE: original OpenAI version,saved in comments for reference
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
        # NOTE: original OpenAI version, saved in comments for reference
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
        translated_role = self._translate_role("linguist")
        return self.get_completion_with_role(
            translated_role, self.prompts.linguist_instruction, tweet
        )

    def expert_analysis(self, tweet: str, target: str) -> ResultE[str]:
        """Let a domain expert analyze the tweet."""
        role: Role = target_role_map.get(target, "expert")
        translated_role = self._translate_role(role)
        instruction = self.prompts.expert_instruction.format(target=target)
        return self.get_completion_with_role(translated_role, instruction, tweet)

    def user_analysis(self, tweet: str) -> ResultE[str]:
        """Let a heavy social media user analyze the tweet."""
        translated_role = self._translate_role("heavy social media user")
        return self.get_completion_with_role(
            translated_role, self.prompts.user_instruction, tweet
        )

    def stance_analysis(  # pylint: disable=R0913,R0917
        self,
        tweet: str,
        ling_response: str,
        expert_response: str,
        user_response: str,
        target: str,
        stance: COLAStance,
    ) -> ResultE[str]:
        """Try to do stance analysis with a given stance"""
        role: Role = target_role_map.get(target, "expert")
        translated_role = self._translate_role(role)
        translated_stance = self._translate_stance(stance)
        prompt = self.prompts.stance_instruction.format(
            tweet=tweet,
            ling_response=ling_response,
            expert_response=expert_response,
            user_response=user_response,
            target=target,
            stance=translated_stance,
            role=translated_role,
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
        prompt = self.prompts.final_judgement_instruction.format(
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

    def batched_analysis(
        self, tasks: list[tuple[str, str]], batch_size: int = 8
    ) -> list[ResultE[AnalysisOutput]]:
        """Batch mode is not yet supported for COLA."""
        return []
