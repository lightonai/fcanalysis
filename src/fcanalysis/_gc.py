import gc
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def gc_disabled() -> Iterator[None]:
    # Disables cyclic GC for the duration of the block; restores prior state
    # and triggers one collection on exit. Loaders process millions of small
    # short-lived dicts where per-generation GC scans dominate runtime; one
    # post-loop collection restores correctness without the per-iteration
    # overhead.
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()
        gc.collect()
