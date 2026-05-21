"""CLI to (re)generate regression fixtures.

Runs the current loader code for every FixtureSpec in tests.matrix
and writes the five fixture files. Existing fixtures are skipped
unless --force is passed.

Usage:
    python -m tests.tools.generate_fixtures              # all specs
    python -m tests.tools.generate_fixtures --loader dolci
    python -m tests.tools.generate_fixtures --fixture dolci/prod
    python -m tests.tools.generate_fixtures --force      # rebuild all

Each spec is one loader run on the full HuggingFace dataset, which for
large loaders (txt360, toolmind, nemotron_terminal) takes minutes per
run. Output goes to tests/fixtures/loaders/{loader}/{config_id}/.
"""

import argparse
import importlib
import sys
import time

from tests.matrix import ALL_SPECS, FixtureSpec, LOADER_MODULES, specs_for_loader
from tests.tools.fixture_io import fixture_dir, write_fixture


def run_one(spec: FixtureSpec, force: bool = False) -> bool:
    out_dir = fixture_dir(spec.loader, spec.config_id)
    if not force and (out_dir / "output.hash").exists():
        print(f"[skip] {spec.fixture_id} (already present)", flush=True)
        return False

    module = importlib.import_module(LOADER_MODULES[spec.loader])
    load = module.load

    started = time.monotonic()
    samples, report = load(
        dataset_config=spec.dataset_config,
        filter_config=spec.filter_config,
        **spec.extra_kwargs,
    )
    load_secs = time.monotonic() - started

    write_started = time.monotonic()
    summary = write_fixture(
        out_dir=out_dir,
        dataset_config=spec.dataset_config,
        filter_config=spec.filter_config,
        extra_kwargs=spec.extra_kwargs,
        report=report,
        samples=samples,
    )
    write_secs = time.monotonic() - write_started

    print(
        f"[done] {spec.fixture_id}: {summary['sample_count']} samples, "
        f"hash={summary['hash'][:12]}..., load={load_secs:.1f}s, "
        f"write={write_secs:.1f}s",
        flush=True,
    )
    return True


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--loader",
        choices=sorted(LOADER_MODULES),
        help="Only regenerate fixtures for one loader.",
    )
    p.add_argument(
        "--fixture",
        help="Only regenerate one fixture, e.g. 'dolci/prod'.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if output.hash already exists.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List all FixtureSpecs and exit without running.",
    )
    return p.parse_args(argv)


def select_specs(args: argparse.Namespace) -> list[FixtureSpec]:
    if args.fixture:
        for spec in ALL_SPECS:
            if spec.fixture_id == args.fixture:
                return [spec]
        raise SystemExit(f"No fixture named {args.fixture!r}")
    if args.loader:
        return specs_for_loader(args.loader)
    return list(ALL_SPECS)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.list:
        for spec in ALL_SPECS:
            print(spec.fixture_id)
        return 0

    specs = select_specs(args)
    print(f"Generating {len(specs)} fixture(s).", flush=True)
    total_started = time.monotonic()

    generated = 0
    for spec in specs:
        if run_one(spec, force=args.force):
            generated += 1

    elapsed = time.monotonic() - total_started
    print(
        f"\nDone. {generated} generated, {len(specs) - generated} skipped, "
        f"total {elapsed:.1f}s."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
