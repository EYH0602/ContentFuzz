import pandas as pd

from contentfuzz._types import is_valid_stance
from contentfuzz.stance_dataset import StanceDataEntry, StanceDataset

_REQUIRED_COLUMNS = {"stance", "text", "target", "new_text", "predicted", "error"}


def get_success_tasks(df: pd.DataFrame) -> StanceDataset:
    """
    Extracts successful tasks from the given DataFrame.
    For cross model success rate,
    we use the fuzzed text,
    and check if the model's prediction matches the original stance.
    """

    assert _REQUIRED_COLUMNS.issubset(df.columns)

    if df.empty:
        return []

    error_series = (
        df["error"] if "error" in df.columns else pd.Series(pd.NA, index=df.index)
    )

    success_mask = df["predicted"].notna() & error_series.isna()
    successful_tasks = df[success_mask]
    tasks = []
    for _, row in successful_tasks.iterrows():
        stance = str(row["stance"])
        new_text = str(row["new_text"])
        target = str(row["target"])
        assert is_valid_stance(stance), f"Invalid stance: {stance}"

        cross_model_entry: StanceDataEntry = {
            "text": new_text,
            "target": target,
            "stance": stance,
        }
        tasks.append(cross_model_entry)
    return tasks


def compute_cross_model_esr(df: pd.DataFrame) -> float:
    """
    Compute cross model escape success rate (ESR).
    """

    # check if DataFrame has required columns
    assert {"truth", "predicted", "confidence"}.issubset(df.columns)

    attack_success = (df["truth"] != df["predicted"]).astype(int)
    accuracy = attack_success.mean()
    return round(float(accuracy), 4)
