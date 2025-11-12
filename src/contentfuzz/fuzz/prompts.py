# https://hastewire.com/blog/how-to-rewrite-ai-writing-without-losing-meaning-essential-guide
INSTRUCTION = """
You are a helpful writing assistant and an avid social media user.
Your role is to help the content creator refine their post to make it more engaging and shareable.
Improve the writing and flow while **keeping the post's original meaning intact**.
**Do not change the author's stance** (their position or opinion on the topic) **or the target topic** of the post.
Make sure to **preserve the original tone, style, and sentiment** of the writing, maintaining the author's voice.
Only **make minimal edits**: the goal is to polish the text, not to overhaul it.
Output **only** the revised text, and do not include any explanations.
Always **keep the content in the same language** as the original post (no translation or dialect change).
Do not extensively use emojis or hashtags unless they were present in the original text.
"""

INSTRUCTION_CN = """
你是一个乐于助人的写作助理，同时也是一个活跃的社交媒体用户。
你的角色是帮助内容创作作者润色他们的帖子，让他们的帖子变得更加有吸引力和传播性。
请保持帖子本身原意不变的前提下，提高写作和文章流畅度。
请不要改变作者的原本立场（他们对主题的态度或观点）或帖子的目标主题。
请务必保留原有的语气，风格和看法保留作者个人表达。
只进行最小幅度的修改：目标是润色文本，而不是重写。
输出只提供修改后的文本，不要附加任何解释。
从始至终保持和原文相同的语言（不要翻译或者转换方言）。
除非是原文中已经有的表情符号或话题标签，否则不要过度使用表情符号或话题标签。
"""

REWRITE = """
The current text is {stance} towards the {target}.
Without changing its meaning, please rewrite the following text:
```
{text}
```
"""

REWRITE_CH = """
当前的文本关于{target}是{stance}的。
在不改变当前含义的情况下，请重新写以下文本：
```
{text}
```
"""


STEER = """
The current text is {stance} towards the {target}.
Without changing its meaning, please rewrite it **slightly** more {direction}.
Your output should still look like {stance} towards the {target},
but with a more {direction} tone,
and the {direction} should ONLY be noticed when reading really closely.

The text:
```
{text}
```
"""

TLDR = """
The current text is {stance} towards the {target}.
Please write a "Too Long; Didn't Read" (TLDR) summary of the text.
However, without changing its meaning, please write the TLDR **slightly** more {direction}.
Please add a `TL;DR:` notifier at the beginning of the TLDR, in the same language as the original text.

The text:
```
{text}
```
"""

TAGS = """
The current text is {stance} towards the {target}.
Please write five hash-tags for the text.
However, among the hash-tags, please make two of them **slightly** more {direction}.

The text:
```
{text}
```
"""
