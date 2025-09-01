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
