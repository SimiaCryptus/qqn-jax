"""Projective regions for QQN.

See ``qqn_jax.regions.strategy`` for the full design note. This package
exposes the ``Region`` protocol along with the concrete region
implementations and the composition helper ``Sequential``.
"""

from qqn_jax.regions.strategy import (
    Region,
    RegionInfo,
    RegionState,
    TrustRegionState,
    resolve_region,
)
from qqn_jax.regions.identity import IdentityRegion
from qqn_jax.regions.box import BoxRegion
from qqn_jax.regions.orthant import OrthantRegion
from qqn_jax.regions.quantization import QuantizationRegion
from qqn_jax.regions.no_decrease import NoDecreaseRegion
from qqn_jax.regions.trustregion import TrustRegion
from qqn_jax.regions.sequence import Sequential

__all__ = [
    "Region",
    "RegionInfo",
    "RegionState",
    "TrustRegionState",
    "resolve_region",
    "IdentityRegion",
    "BoxRegion",
    "OrthantRegion",
    "QuantizationRegion",
    "NoDecreaseRegion",
    "TrustRegion",
    "Sequential",
]
