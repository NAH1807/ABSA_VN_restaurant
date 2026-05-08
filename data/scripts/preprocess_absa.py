import os
import re
import pandas as pd
from utils import clean_text


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "..", "raw")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "processed")
        

def parse_label_line(label_text):

    results = []

    for item in re.findall(r"\{([^}]+)\}", label_text):

        parts = item.split(",", 1)

        if len(parts) != 2:
            continue

        aspect_part = parts[0].strip()
        sentiment = parts[1].strip()

        if "#" in aspect_part:
            aspect = aspect_part.split("#", 1)[0].strip()
            results.append((aspect, sentiment))

    return results


def load_dataset(file_path):

    data = []
    block = []

    with open(file_path, "r", encoding="utf-8") as f:

        for raw_line in f:
            line = raw_line.strip()

            if line == "":
                if block:
                    sentence = clean_text(" ".join(block[:-1]))
                    labels = parse_label_line(block[-1])

                    if sentence and labels:
                        data.append({
                            "sentence": sentence,
                            "labels": labels
                        })

                    block = []
                continue

            block.append(line)

        if block:
            sentence = clean_text(" ".join(block[:-1]))
            labels = parse_label_line(block[-1])

            if sentence and labels:
                data.append({
                    "sentence": sentence,
                    "labels": labels
                })

    return data


def save_dataframe(data, output_path):

    rows = []

    for item in data:

        sentence = item["sentence"]

        for aspect, sentiment in item["labels"]:

            rows.append({
                "sentence": sentence,
                "aspect": aspect,
                "sentiment": sentiment
            })

    df = pd.DataFrame(rows)

    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


def main():

    os.makedirs(
        os.path.join(OUTPUT_DIR, "sentiment_classification"),
        exist_ok=True
    )

    files = {
        "train": "train.txt",
        "dev": "dev.txt",
        "test": "test.txt"
    }

    for split, filename in files.items():

        path = os.path.join(RAW_DIR, filename)

        dataset = load_dataset(path)

        output_path = os.path.join(
            OUTPUT_DIR,
            "sentiment_classification",
            f"{split}.csv"
        )

        save_dataframe(dataset, output_path)


if __name__ == "__main__":
    main()