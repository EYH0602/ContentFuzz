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

```sh
uv run src/main.py
```
