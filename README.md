# ContentFuzz

Content Fuzzing for Escaping Information Cocoons in Digital Social Media

## Requirements

### Datasets

First, you need to get the dataset C-STANCE

```sh
git submodule update --init --recursive
```

### Python

We use Python 3.12.
We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage your Python dependencies.

```sh
cd ContentFuzz
uv sync # create a virtual environment, and install dependencies
```

## Running ContentFuzz

```sh
uv run src/main.py C-STANCE/c_stance_dataset/subtaskA/raw_test_all_onecol.csv
```
