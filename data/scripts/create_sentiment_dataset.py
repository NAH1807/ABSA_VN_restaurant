import os
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "..", "processed", "sentiment_classification")


def main():

    for split in ["train", "dev", "test"]:

        path = os.path.join(INPUT_DIR, f"{split}.csv")

        df = pd.read_csv(path)

        df = df.dropna()

        df["text"] = (
            "[ASPECT] "
            + df["aspect"]
            + " [TEXT] "
            + df["sentence"]
        )

        output_path = os.path.join(INPUT_DIR, f"{split}_sentiment.csv")

        df.to_csv(output_path, index=False)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()