"""Minimal package initializer for the SCGN runtime bundle.

The SCGN training path imports individual restoration modules.  The simulator
exports are intentionally not imported here because simulator sources are not
part of this frozen Stage 2 bundle and eager package imports would block every
otherwise valid SCGN import.
"""

__all__: list[str] = []
