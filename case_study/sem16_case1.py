import orjson
from returns.result import Failure, Success

from contentfuzz.cls import Encoder
from contentfuzz.fuzz import Fuzzer, Mutator
from contentfuzz.fuzz.seed_scheduler import PriorityScheduler
from contentfuzz.stance_dataset import StanceDataEntry


def main():
    """reproduce a simple fuzzing run for case study"""
    task: StanceDataEntry = {
        "stance": "Against",
        "target": "Atheism",
        "text": "I am human. I look forward to the extinction of humanity with eager anticipation. We deserve nothing less.",
    }
    mutator = Mutator()
    model_path = "saved_models/FacebookAI/roberta-base/sem16"
    analyzer = Encoder(model_path)
    scheduler = PriorityScheduler()
    fuzzer = Fuzzer(analyzer, mutator, scheduler)
    match fuzzer.runs(task, n_iters=10):
        case Success(((mutated_text, stance, confidence), iter_cnt)):
            log_obj = task | {
                "new_text": mutated_text,
                "predicted": stance,
                "confidence": confidence,
                "iteration": iter_cnt,
            }
            print("Fuzzing succeeded:")
            print(orjson.dumps(log_obj, option=orjson.OPT_INDENT_2).decode())
        case Failure(e):
            print(f"Fuzzing failed: {e}")


if __name__ == "__main__":
    main()
