"""Current RGB foreground and public profile fitting around an episode track."""
import cv2
import numpy as np
from .contours import TABLE, refine_mask
from ..cameras import project, pixel_scale
from .proposal_masks import proposal_masks
from .profile_geometry import height_only_invalid, profile_geometry


class ObjectObserver:
    def __init__(self, head_size):
        self.head_size = head_size

    def hull(self, xy, rank):
        points = self.head_size.geometry.points[rank] + [xy[0], xy[1], TABLE]
        uv = project(points)
        return cv2.convexHull(uv.astype(np.float32))

    def qualify(self, doll, rank, expected_xy, gates=None):
        gates = {'iou': .65, 'size': .24, 'displacement': .055} if gates is None else gates
        fitted, fit = self.head_size.locate(doll, rank, expected_xy)
        if fitted is None:
            return None, dict(fit, reason='profile_fit_rejected', gates=gates)
        predicted = np.zeros((480, 640), np.uint8)
        observed = np.zeros_like(predicted)
        cv2.fillConvexPoly(predicted, np.rint(self.hull(fitted, rank)).astype(np.int32), 1)
        contour = cv2.convexHull(np.asarray(doll['contour'], np.int32))
        cv2.fillConvexPoly(observed, contour, 1)
        overlap = np.count_nonzero(predicted & observed) / max(1, np.count_nonzero(predicted | observed))
        displacement = float(np.linalg.norm(fitted - expected_xy))
        valid = bool(overlap >= gates['iou'] and fit['size_error'] <= gates['size'] and displacement <= gates['displacement'])
        diag = dict(fit, valid=valid, silhouette_iou=float(overlap), displacement_m=displacement, gates=gates,
                    reason='observed_upright_profile' if valid else 'silhouette_or_association_rejected')
        if not valid:
            return None, diag
        scale = pixel_scale(doll['bottom_pixel'],TABLE)
        return dict(doll, xy=fitted.tolist(), rank=rank,
                    calibrated_height_m=float(np.max(self.head_size.geometry.points[rank][:, 2])),
                    size_metric=float(np.sqrt(doll['mask_area']) * scale),
                    size_pixel_quantum=float(scale), object_observation=diag), diag

    def observe(self, rgb, xy, rank, gates=None):
        """Search is conditional on legal history; acceptance depends on current pixels."""
        hull = self.hull(xy, rank).reshape(-1, 2)
        lower, upper = hull.min(0), hull.max(0)
        # Placement and park-camera drift is larger than CAD hull; CAD still does not paint pixels.
        box = np.r_[lower - [18, 8], upper + [18, 18]].tolist()
        candidates = proposal_masks(rgb, [{'box': box, 'confidence': 1.}])
        diagnostics = []
        accepted = []
        for mask, meta in candidates:
            doll = refine_mask(mask, dict(meta, semantic_prompt='episode_target_RGB_foreground'))
            if doll is None or (not doll['geometric_valid'] and not height_only_invalid(doll)):
                diagnostics.append({'valid': False, 'reason': 'no_valid_observed_contour'})
                continue
            value, diag = self.qualify(doll, rank, np.asarray(xy), gates)
            if not doll['geometric_valid']:
                value = profile_geometry(doll, value, diag)
                diag = dict(diag, coarse_height_m=doll['height_m'],
                            profile_geometry_recovered=value is not None)
            diagnostics.append(diag)
            if value is not None:
                accepted.append((diag['silhouette_iou'], value, mask))
        if not accepted:
            return None, None, {'valid': False, 'search_box': box, 'candidates': diagnostics}
        _, value, mask = max(accepted, key=lambda item: item[0])
        return value, mask, {'valid': True, 'search_box': box, 'candidates': diagnostics}
