# fcanalysis

Python tools for loading, validating, deduplicating, and analyzing
function-calling datasets.

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/). From the repository root:

```sh
uv sync --locked
uv run python docs/examples/weather_loader.py
```

The example uses synthetic data and runs offline.

- [Documentation](docs/README.md)
- [Use a loader](docs/loading.md)
- [Add a loader](docs/adding-loaders.md)

Run tests with `uv run pytest`.

[MIT License](LICENSE). Datasets retain their own licenses.
[Third-party notices](docs/third-party-notices.md).
