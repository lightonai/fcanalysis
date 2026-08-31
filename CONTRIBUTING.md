# Contributing

Contributions should stay focused on function-calling dataset loading,
normalization, validation, and analysis. Open an issue before proposing a new
dataset or a material public-interface change so that provenance, license,
memory cost, and fixture requirements are clear.

## Development setup

The project requires Python 3.14 and uses `uv`:

```sh
git clone https://github.com/lightonai/fcanalysis.git
cd fcanalysis
uv sync --locked
```

Run the release checks before submitting a change:

```sh
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
uv run prek run --all-files
bash -n scripts/*.sh
git diff --check
uv build --clear
```

## Data and fixture policy

Do not commit source datasets, generated samples, semantic outputs, model
artifacts, caches, credentials, logs, or local settings. The tracked fixture
contract is limited to `config.json`, `report.json`, and `output.hash` for each
registered loader/configuration case.

Full outputs and deterministic sample subsets are generated locally and remain
ignored. If a deliberate loader change updates a fixture hash, explain the
behavioral reason, inspect representative diffs locally, regenerate every
affected contract file, and run the relevant full-dataset E2E case.

## Changes to source loaders

A loader change should preserve an explicit three-stage boundary: source
conversion, source-specific policy, then universal filters. Pin source
revisions, report transformations and drop reasons, fail explicitly on invalid
configuration, and add synthetic tests for important success and failure paths.

Review the source dataset card and component licenses before adding support.
The repository's MIT license does not cover dataset content.
