# ContentFuzz

Content Fuzzing for Escaping Information Cocoons in Digital Social Media

## Requirements

### Python

We use Python 3.10.
We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/)
to manage your Python dependencies.

```sh
cd ContentFuzz
uv sync # create a virtual environment, and install dependencies
```

### API keys

Please either put the API keys in your environment variables, 
or create a `.env` file in the
project root with the following content.
ContentFuzz talks to Google Gemini through two different client stacks:

- Set `VERTEXAI_API_KEY` so the stance analyzers (`zeroshot`, `cola`, `encoder`) can request log probabilities from the Vertex AI endpoints.
- Set `GEMINI_API_KEY` so the fuzzing mutator can call the public Gemini API when generating rewritten posts.


## Running ContentFuzz

### Stance Analysis

`src/run_cls.py` performs the baseline stance classification run that feeds every
other experiment. The script automatically loads the requested dataset (one of
`c-stance-a`, `c-stance-b`, `sem16`, or `vast`), resumes from partially written
results, and prints accuracy/F1 once the JSONL file is complete.

```sh
uv run src/run_cls.py DATASET ANALYZER \
  -m gemini-2.5-flash-lite \
  -b 1 \
  -o results/ANALYZER+MODEL+DATASET.jsonl
```

- `ANALYZER` must be one of `zeroshot`, `cola`, or `encoder`.
- `-m/--model` forwards the LLM identifier to the analyzer (Gemini Flash Lite by default).
- `-n/--sample-n` randomly samples the dataset before the run when you only need a subset.
- `-o/--output-result-path` overrides the auto-generated path (`results/{analyzer}+{model}+{dataset}.jsonl`).
- `-b/--batch_size` controls the async batch size; increase it carefully based on GPU memory or rate limits.

Example (full dataset, zeroshot analyzer):

```sh
uv run src/run_cls.py c-stance-a zeroshot -m gemini-2.5-flash-lite -b 4
```

You can re-evaluate an existing run at any time:

```sh
uv run src/eval_cls.py results/zeroshot+gemini-2.5-flash-lite+c-stance-a.jsonl
```

### Content Fuzzing

`src/run_fuzz.py` takes the correctly classified rows from a baseline run and
applies the mutation-based fuzzer. The script streams new rows to a JSONL file,
prints fuzzing metrics, and emits a `*.fuzzer_stat.json` file containing the
temperature-scheduling histogram captured during the run.

```sh
uv run src/run_fuzz.py DATASET ANALYZER PATH/TO/CLS_RESULTS.jsonl \
  --fuzzer-model gemini-2.5-flash-lite \
  --cls-model gemini-2.5-flash-lite \
  --schedule priority \
  --mutate-n 5 \
  --n-iters 300 \
  --attack-output-path fuzz/custom.jsonl
```

- `cls_output_path` must be the JSONL file produced by `run_cls.py` for the matching dataset.
- `--fuzzer-model` selects the Gemini model used to rewrite the text, while `--cls-model` determines how the analyzer re-scores fuzzed posts.
- Passing `--temperature <value>` disables scheduling and uses the fixed float. Omitting the flag enables adaptive scheduling between 0.0 and 2.0.
- `--schedule` accepts `fifo`, `priority`, `random`, or `priority_random`.
- `--sample-n` lets you fuzz only a subset of the correctly classified rows.

To inspect the fuzzing metrics once the run finishes:

```sh
uv run src/eval_fuzz.py fuzz/encoder+saved_models--FacebookAI--roberta-base+vast+vast=gemini-2.5-flash-lite+temp-sched+priority+iters-300.jsonl
```

### Cross-Model Evaluation

Use `src/run_cross_model.py` to replay the successful fuzzed samples with a different analyzer/model combination and measure cross-model escape success rate (ESR).

```sh
uv run src/run_cross_model.py DATASET ANALYZER fuzz/<baseline>.jsonl \
  -m gemini-2.5-flash-lite \
  -b 2
```

The script automatically records new predictions in `cross_model/{dataset}` and reports the ESR so you can compare robustness across models.

### Experiments

We provide all of our experiment scripts in `experiments/`.
You can run them directly, or modify them to run your own experiments.

If you encounter CUDA out-of-memory errors,
please adjust the batch sizes in `src/contentfuzz/evaluate.py` accordingly.

### Comparison to Other Methods

We also provide scripts to run other baseline methods for comparison as submodules.

```sh
git submodule update --init --recursive
```

Please refer to the README files in each submodule for instructions on how to run them.
