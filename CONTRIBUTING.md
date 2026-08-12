# Contributing

Thank you for contributing to **EEG Feature Atlas**.

The project uses **Python**, **uv** for environment and dependency management, and **marimo** for interactive examples.

## Development Setup

Clone the repository and create the local environment:

```bash
git clone https://github.com/bdmi-research/eeg-feature-atlas.git
cd eeg-feature-atlas
uv sync
```

`uv` will create and manage the local `.venv` environment automatically.

Do not create or commit a separate virtual environment.

## Running Python

Run Python files through `uv`:

```bash
uv run python path/to/script.py
```

Run marimo:

```bash
uv run marimo edit
```

Or open a specific notebook:

```bash
uv run marimo edit library_examples/mne/01_example.py
```

## Dependencies

Use `uv` to manage Python packages.

Add a dependency with:

```bash
uv add package-name
```

Do not use `pip install` directly and do not manually edit `uv.lock`.

After pulling changes that modify dependencies, run:

```bash
uv sync
```

Both `pyproject.toml` and `uv.lock` should be committed to the repository.

## Code Guidelines

* Use Python for all examples and utilities.
* Prefer small, focused examples over large notebooks.
* Keep library-specific API calls visible in examples.
* Move code to `utils/` only when it is genuinely reused by multiple examples.
* Use public, automatically downloadable, or synthetic data whenever possible.
* Do not commit local datasets, virtual environments, credentials, or machine-specific files.
* Keep examples reproducible on a clean checkout of the repository.

## Before Committing

Make sure that:

```bash
uv sync
```

completes successfully and that the example you changed can be executed from the repository root.
