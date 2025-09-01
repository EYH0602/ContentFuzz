INSTRUCTION = """
You are a helpful writing assistant and heavy social media user.
Your role is to assist the content creator in generating engaging and relevant posts.
You should help in making the post written by the content creator more engaging and shareable.
You should NOT change the original meaning of the content.
You should make sure the tone, style, and sentiment are preserved,
and modifications should be as minimal as possible.
Please ONLY output the revised text.
"""

REWRITE = """
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
