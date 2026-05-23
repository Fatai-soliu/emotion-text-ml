## Clean text data


def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r"http\S+", "", text)        # remove URLs
    text = re.sub(r"@\w+", "", text)           # remove mentions
    text = re.sub(r"#(\w+)", r"\1", text)      # keep hashtag word
    text = re.sub(r"[^a-z\s]", "", text)       # remove punctuation/symbols
    return text