import pandas as pd
from .utils import translate_stance


def load_c_stance(file_path: str) -> pd.DataFrame:
    """load and preprocess C-STANCE dataset
    Preprocessing includes:
    1. Renaming columns to common standards for all datasets (text, stance, target)
    2. Dropping unnecessary columns
    3. Translating stance labels to English
    """
    assert file_path.endswith(".csv"), "Input file must be a CSV file"
    df = pd.read_csv(file_path)
    df = df.rename(
        columns={
            "Text": "text",
            "Stance 1": "stance",
            "Target 1": "target",
        }
    )
    # drop all other columns
    df = df[["text", "stance", "target"]]
    df["stance"] = df["stance"].apply(translate_stance)
    return df
