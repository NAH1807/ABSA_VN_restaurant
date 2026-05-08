import re
import pandas as pd


def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_label(label_text):
    """
    FOOD#Positive;SERVICE#Negative
    =>
    [
        ("FOOD", "Positive"),
        ("SERVICE", "Negative")
    ]
    """

    results = []

    if label_text.strip() == "":
        return results

    pairs = label_text.split(";")

    for pair in pairs:
        if "#" in pair:
            aspect, sentiment = pair.split("#")
            results.append((aspect.strip(), sentiment.strip()))

    return results