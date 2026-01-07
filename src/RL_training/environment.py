"""
Environment helpers for the roaster simulator.
"""

import sys

from RL_training.aillio_dummy import AillioDummy as Roaster


class SuppressOutput:
    """Context manager to suppress stdout."""

    def __init__(self):
        self._devnull = None
        self._original_stdout = None

    def __enter__(self):
        import os

        self._original_stdout = sys.stdout
        self._devnull = open(os.devnull, 'w')
        sys.stdout = self._devnull
        return self

    def __exit__(self, *args):
        sys.stdout = self._original_stdout
        if self._devnull:
            self._devnull.close()


def create_roaster(debug: bool = False) -> Roaster:
    """Create a roaster instance with optional debug output."""
    if debug:
        roaster = Roaster()
    else:
        with SuppressOutput():
            roaster = Roaster()
    roaster.AILLIO_DEBUG = debug
    return roaster
