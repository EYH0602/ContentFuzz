# ContentFuzz

Content Fuzzing for Escaping Information Cocoons in Digital Social Media

## Requirements

### Datasets

The C-STANCE datasets are pulled directly from Hugging Face:

- yfhe/C-STANCE-A
- yfhe/C-STANCE-B

No Git submodules are needed for datasets.

### Python

We use Python 3.10.
We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/)
to manage your Python dependencies.

```sh
cd ContentFuzz
uv sync # create a virtual environment, and install dependencies
```

## Running ContentFuzz

### Stance Analysis

```sh
uv run src/run_cls.py c-stance-a zero-shot
```

If `-o` is not given, the result will be saved to
`f"results/{analyzer_name}+{model}+{dataset_name}.jsonl"`.

For CLI interface detains, please see the help page.

```sh
uv run src/run_cls.py -h
```

Please use `-b` to set batch size for number of tasks to process in parallel.

- For encoder, please adjust batch size based on your VRAM size.
- For LLMs, please adjust batch size based on your rate limit. We use this to control number of async requests.

To evaluate the analysis results:

```sh
uv run src/eval_cls.py results/zero-shot+gpt-4.1+c-stance-a.jsonl
```

### Content Fuzzing

```sh
uv run src/run_fuzz.py c-stance-a zeroshot
```

We provide all of our experiment scripts in `experiments/`.
You can run them directly, or modify them to run your own experiments.

To evaluate the fuzzing results:

```sh
uv run src/eval_fuzz.py fuzz/encoder+saved_models--FacebookAI--roberta-base+vast+vast=gemini-2.5-flash-lite+temp-sched+priority+iters-300.jsonl
```

If you encounter CUDA out-of-memory errors,
please adjust the batch sizes in `src/contentfuzz/evaluate.py` accordingly.

### Comparison to Other Methods

We also provide scripts to run other baseline methods for comparison as submodules.

```sh
git submodule update --init --recursive
```

Please refer to the README files in each submodule for instructions on how to run them.
