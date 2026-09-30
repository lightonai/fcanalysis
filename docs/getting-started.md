# Getting started

[Documentation](README.md) · Next: [How conversations are loaded](conversations.md)

Use Python 3.14 or newer. From a checkout of this repository, install the locked
environment with [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked
```

Run the following commands from the repository root.

## Run a loader without downloading data

The [weather adapter](examples/weather_loader.py) uses synthetic input with the
package's shared validation and curation:

```sh
uv run python docs/examples/weather_loader.py
```

Expected output:

```text
{'read': 3, 'converted': 3, 'valid': 2, 'retained': 1}
{'invalid_arguments': 1}
{'role': 'assistant', 'content': 'Sunny.'}
```

The three inputs are a valid weather conversation, a duplicate with another ID,
and a row passing a number where the tool requires a city string. Validation
rejects the third; curation keeps the first of the two valid copies. The default
removes native reasoning from the answer. The original row remains in `raw`.
Loading never executes recorded tool calls.

## Work with the output

Start Python with `uv run python`, or put this code in a script at the repository
root. It loads one synthetic row and analyzes its call pattern:

```python
import json

from docs.examples.weather_loader import example_row, load
from fcanalysis.core import analyze_sample

samples, report = load([json.dumps(example_row())])
sample = samples[0]
analysis = analyze_sample(sample.messages)
print([pattern.value for pattern in analysis.turn_patterns])  # ['single_call']
```

`docs.examples.weather_loader` is the tutorial file in this checkout, not an
installed package API. See [the conversation format](conversations.md) for the
output fields, [analysis](analysis.md) for more measurements, and
[Adding a loader](adding-loaders.md) to adapt the example to your own format.

## Load a supported dataset

Each source has its own adapter. For example, Dolci loads its pinned release:

```python
from fcanalysis.loaders.dolci import load

samples, report = load()
print(report.summary())
```

This can download the full source and use substantial disk space. `load()`
holds the returned list in memory; `iter_load()` avoids keeping that whole list.
See [Using loaders](loading.md) for the catalogue, iteration, source selection,
reasoning settings, and curation.

## Develop the package

```sh
uv run pytest
uv run prek run --all-files
```

Default tests exclude the full dataset tests marked `e2e`. The
[pre-commit hooks](../prek.toml) run formatting, lint, and type checks; formatting
hooks can edit files.

The code is [MIT licensed](../LICENSE). Adapter docstrings record source dataset
terms; [third-party notices](third-party-notices.md) cover incorporated material.
