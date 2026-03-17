"""Backward-compatible shim — delegates to the registry.

Existing code that does ``from hycut.datavault.loaders import get_datasets``
will keep working unchanged.
"""

from .registry import (
    get_datasets,
    get_dataloaders,
    get_dataset,
    list_datasets,
    get_spec,
)
from .utils import get_default_transforms

__all__ = [
    "get_datasets",
    "get_dataloaders",
    "get_dataset",
    "list_datasets",
    "get_spec",
    "get_default_transforms",
]
