# Library Examples

This directory contains small, focused examples of Python libraries useful for EEG feature analysis and visualization.

The goal of these examples is to explore the capabilities of each library before building larger analysis pipelines.

## Guidelines

Each example should:

1. Demonstrate one clearly defined method or feature.
2. Use a small public, automatically downloadable, or synthetic dataset.
3. Keep the API of the demonstrated library visible in the example.
4. Explain what is being calculated and why it may be useful for EEG analysis.
5. Visualize the result when appropriate.
6. Include links to the relevant library documentation or scientific source.
7. Be reproducible from a clean checkout of the repository.

Examples should remain small and easy to understand. Shared code should only be moved to `utils/` when it is genuinely reused by multiple examples.

## Running Examples

Install the project environment from the repository root:

```bash
uv sync
```

Open a marimo notebook:

```bash
uv run marimo edit library_examples/mne/01_spectrum.py
```

Examples should also be executable without the editor:

```bash
uv run python library_examples/mne/01_spectrum.py
```

## Current Libraries

Initial examples will explore:

* MNE-Python
* YASA
* NeuroKit2
* AntroPy

Additional libraries and methods will be added as the project develops.
