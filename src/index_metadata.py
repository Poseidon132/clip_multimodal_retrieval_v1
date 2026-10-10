"""Bind an embedding index to the exact encoder/adaptation configuration."""
from __future__ import annotations

import hashlib
from pathlib import Path


MODES = ("both", "image-only")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as file:
        for part in iter(lambda: file.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def get_index_metadata(model_name: str, adapter: str | None, adapter_mode: str) -> dict[str, str]:
    if adapter_mode not in MODES:
        raise ValueError(f"adapter_mode must be one of {MODES}")
    if not adapter and adapter_mode != "both":
        raise ValueError("--adapter-mode image-only requires --adapter")
    return {
        "index_model_name": model_name,
        "index_adapter_mode": adapter_mode if adapter else "none",
        "index_adapter_sha256": sha256_file(adapter) if adapter else "",
    }


def validate_index_metadata(
    stored: dict[str, str] | None,
    model_name: str,
    adapter: str | None,
    adapter_mode: str,
) -> None:
    requested = get_index_metadata(model_name, adapter, adapter_mode)
    if stored is None:
        if adapter:
            raise ValueError(
                "This index predates adapter metadata. Rebuild it with src.features "
                "using the SAME --adapter and --adapter-mode used by search."
            )
        return  # Legacy unadapted toy indexes are still supported.
    for key, expected in requested.items():
        if stored[key] != expected:
            raise ValueError(
                f"Index/query encoder mismatch: {key} is {stored[key]!r} in the index "
                f"but {expected!r} was requested. Rebuild the index or fix the search settings."
            )
