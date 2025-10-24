# ContentFuzz

Content Fuzzing for Escaping Information Cocoons in Digital Social Media

## Requirements

### Datasets

The C-STANCE datasets are pulled directly from Hugging Face:

- yfhe/C-STANCE-A
- yfhe/C-STANCE-B

No Git submodules are needed for datasets.

### Python

We use Python 3.12.
We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage your Python dependencies.

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

To evaluate the analysis results:

```sh
uv run src/eval.py results/zero-shot+gpt-4.1+c-stance-a.jsonl
```
