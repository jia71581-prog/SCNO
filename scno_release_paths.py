"""Resolve external experiment assets under an installation-specific root."""

import os
from pathlib import Path


def local_asset(relative):
    default = Path(__file__).resolve().parent / "local_assets"
    return str(Path(os.environ.get("SCNO_ASSET_ROOT", default)) / relative)
