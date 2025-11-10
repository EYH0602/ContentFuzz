from contentfuzz.stance_dataset.utils import remove_hash_tags


def test_remove_hash_tags():
    text_with_tags = (
        "This is a sample text with #hashtag1# and some more text #hashtag2#."
    )
    cleaned_text = remove_hash_tags(text_with_tags)
    assert (
        cleaned_text
        == "This is a sample text with hashtag1 and some more text hashtag2."
    )

    text_without_tags = "This is a sample text without hashtags."
    cleaned_text = remove_hash_tags(text_without_tags)
    assert cleaned_text == text_without_tags

    text_with_nested_tags = "Check out #nested#tag# example."
    cleaned_text = remove_hash_tags(text_with_nested_tags)
    assert cleaned_text == "Check out nested tag example."

    text_w_nested_tags = "American conservatism has everything to do with religion with all the good stuff taking out of it. #SemST"
    cleaned_text = remove_hash_tags(text_w_nested_tags)
    assert (
        cleaned_text
        == "American conservatism has everything to do with religion with all the good stuff taking out of it."
    )

    text = "Just know that you're not an accident and that you're here by HIS divine providence.  #tcot #pjnet #ccot #christian #SemST"
    cleaned_text = remove_hash_tags(text)
    assert (
        cleaned_text
        == "Just know that you're not an accident and that you're here by HIS divine providence."
    )

    text = "@Jaikrishnashree #italianchachi420 UDF/LDF made #Kerala the most backward southern state & cluelss #Hindus keep voting #marxism #SemST"
    cleaned_text = remove_hash_tags(text)
    assert (
        cleaned_text
        == "@Jaikrishnashree italianchachi420 UDF/LDF made Kerala the most backward southern state & cluelss Hindus keep voting marxism"
    )

    leading_tag = "#SemST Faith matters."
    assert remove_hash_tags(leading_tag) == "Faith matters."

    trailing_after_punct = "Faith matters! #SemST #Amen"
    assert remove_hash_tags(trailing_after_punct) == "Faith matters!"

    inline_semst = "Topic #SemST discussion stays civil."
    assert remove_hash_tags(inline_semst) == "Topic discussion stays civil."
