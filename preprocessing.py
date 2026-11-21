
import html
import re
import spacy


nlp = spacy.load("en_core_web_sm")

NEGATIONS = {"no", "not", "nor", "never", "n't"}


def clean_text(text: str) -> str:
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_text(text: str) -> str:
    cleaned_text = clean_text(text)
    doc = nlp(cleaned_text)

    tokens = []

    for token in doc:
        if token.is_space or token.is_punct:
            continue

        if token.is_stop and token.lower_ not in NEGATIONS:
            continue

        lemma = token.lemma_.lower().strip()

        if lemma:
            tokens.append(lemma)

    return " ".join(tokens)
