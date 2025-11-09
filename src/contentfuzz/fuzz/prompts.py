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

REWRITE = """
The current text is {stance} towards the {target}.
Without changing its meaning, please rewrite the following text:
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
