"""Replace only an invalid coarse height with an already qualified RGB profile."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def inside_table(xy: object) -> bool:
    try:
        x, y = xy  # type: ignore[misc]
        return bool(math.isfinite(x) and math.isfinite(y)
                    and -.55 < x < .55 and -.40 < y < .12)
    except (TypeError, ValueError):
        return False


def height_only_invalid(doll: Mapping[str, Any]) -> bool:
    """Retain the original table bounds and reject nonfinite/absent geometry."""
    height = doll.get('height_m')
    return bool(doll.get('geometric_valid') is False and inside_table(doll.get('xy'))
                and isinstance(height, (float, int)) and math.isfinite(height)
                and not .02 < height < .30)


def profile_geometry(doll: Mapping[str, Any], qualified: Mapping[str, Any] | None,
                     diagnostics: Mapping[str, Any]) -> dict[str, Any] | None:
    """The caller must run all original silhouette, size and association gates.

    No CAD silhouette is used as an observed mask. This consumes the result of
    fitting the current observed RGB contour and restores the same geometry
    bounds using its calibrated profile height and fitted table position.
    """
    if not height_only_invalid(doll) or qualified is None or diagnostics.get('valid') is not True:
        return None
    height = qualified.get('calibrated_height_m')
    if (not isinstance(height, (float, int)) or not math.isfinite(height)
            or not .02 < height < .30 or not inside_table(qualified.get('xy'))):
        return None
    return dict(qualified, height_m=height, geometric_valid=True,
                height_source='current_RGB_qualified_CAD_profile',
                coarse_height_m=doll['height_m'])
