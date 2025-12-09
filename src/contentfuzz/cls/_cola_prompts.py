"""prompt templates for COLA."""

from dataclasses import dataclass

from ..utils import Language


@dataclass(frozen=True)
class PromptSet:
    """Container for COLA instructions in a given language."""

    linguist_instruction: str
    expert_instruction: str
    user_instruction: str
    stance_instruction: str
    final_judgement_instruction: str


EN_LINGUIST_PROMPT = """
Accurately and concisely explain the linguistic elements in the sentence and how these elements affect meaning,
including grammatical structure, tense and inflection, virtual speech, rhetorical devices, lexical choices and so on.
Do nothing else."""

EN_EXPERT_PROMPT = """
Accurately and concisely explain the key elements contained in the quote,
such as characters, events, parties, religions, etc.
Also explain their relationship with {target} (if exist).
Do nothing else."""

EN_USER_PROMPT = """
Analyze the following sentence, focusing on the content, hashtags,
Internet slang and colloquialisms, emotional tone, implied meaning, and so on.
Do nothing else."""

EN_STANCE_PROMPT = """
'''{tweet}'''
<<<{ling_response}>>>
[[[{expert_response}]]]
---{user_response}---

You think the attitude behind the sentence surrounded by ''' ''' is {stance} of {target}.
The content enclosed by <<< >>> represents linguistic analysis. The content within [[[ ]]] represents the analysis of a {role}.
The content enclosed by --- ---  represents the analysis of a heavy social media user.
Identify the top three pieces of evidence from these that best support your opinion and argue for your opinion.
"""
EN_FINAL_PROMPT = """
Determine whether the sentence is in favor of or against {target}, or is irrelevant to {target}.
Sentence: {tweet}
Judge this in relation to the following arguments:
Arguments that the attitude is in favor: {favor_response}
Arguments that the attitude is against: {against_response}
Choose from:\n A: Against\nB: Favor\nC: Irrelevant\n
Constraint: Answer with only the option above that is most accurate and nothing else.
"""

ZH_LINGUIST_PROMPT = """
准确而简洁地解释句子中的语言要素，以及这些要素如何影响意义，
包括语法结构、时态和词形变化、虚拟语气、修辞手法、词语选择等。
不要做任何其他事情。"""

ZH_EXPERT_PROMPT = """
准确而简洁地解释引文中包含的关键元素，
例如人物、事件、群体、宗教等。
同时解释它们与 {target} 的关系（若存在）。
不要做任何其他事情。"""

ZH_USER_PROMPT = """
分析下面的句子，重点关注内容、社交媒体标签（hashtags）、
网络用语和口语表达、情绪语气、隐含意义等。
不要做任何其他事情。"""

ZH_STANCE_PROMPT = """
'''{tweet}'''
<<<{ling_response}>>>
[[[{expert_response}]]]
---{user_response}---

你认为被 ''' ''' 包围的句子所表达的态度，是对 {target} 的 {stance}。
<<< >>> 中的内容代表语言学分析；[[[ ]]] 中的内容代表 {role} 提供的分析；--- --- 中的内容代表社交媒体重度用户的分析。
请从这些内容中找出最能支撑你观点的三条证据，并据此证明你的立场。
"""

ZH_FINAL_PROMPT = """
判断这句话是支持 {target}、反对 {target}，还是与 {target} 无关。
句子：{tweet}

请根据以下论据进行判断：
支持的论据：{favor_response}
反对的论据：{against_response}

请选择：
A: Against（反对）
B: Favor（支持）
C: Irrelevant（无关）

限制条件：只能输出最准确选项对应的选项，不得添加其他内容。
"""

EN_PROMPTS = PromptSet(
    linguist_instruction=EN_LINGUIST_PROMPT,
    expert_instruction=EN_EXPERT_PROMPT,
    user_instruction=EN_USER_PROMPT,
    stance_instruction=EN_STANCE_PROMPT,
    final_judgement_instruction=EN_FINAL_PROMPT,
)

ZH_PROMPTS = PromptSet(
    linguist_instruction=ZH_LINGUIST_PROMPT,
    expert_instruction=ZH_EXPERT_PROMPT,
    user_instruction=ZH_USER_PROMPT,
    stance_instruction=ZH_STANCE_PROMPT,
    final_judgement_instruction=ZH_FINAL_PROMPT,
)

PROMPT_SETS: dict[Language, PromptSet] = {
    "en": EN_PROMPTS,
    "zh": ZH_PROMPTS,
}
