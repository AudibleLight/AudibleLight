#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dataset downloading functions, can be called directly inside Python."""

from scripts.download_data.download_fma import main as download_fma
from scripts.download_data.download_gibson import main as download_gibson
from scripts.download_data.download_gibson_waypoints import (
    main as download_gibson_waypoints,
)

# These scripts have optional dependencies
#  we define stub functions here in cases where they aren't installed
#  so that trying to import them doesn't blow up this script

# FSD50K has an optional dependency on soundata
try:
    from scripts.download_data.download_fsd import main as download_fsd
except ImportError as e:
    _import_error_fsd = e

    def download_fsd(*_, **__):
        """Stub for download_fsd when soundata is not installed."""
        raise ImportError(_import_error_fsd)


# RIRs have an optional dependency on mat73
try:
    from scripts.download_data.download_rirs import main as download_rirs
except ImportError as e:
    _import_error_rirs = e

    def download_rirs(*_, **__):
        """Stub for download_rirs when mat73 is not installed."""
        raise ImportError(_import_error_rirs)


__all__ = [
    "download_fma",
    "download_fsd",
    "download_gibson",
    "download_gibson_waypoints",
    "download_rirs",
]
