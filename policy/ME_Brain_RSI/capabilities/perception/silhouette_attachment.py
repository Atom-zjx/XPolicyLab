"""Current RGB silhouettes for visible objects without interior KLT corners.

The within-episode head appearance chooses pixels, never a layout identity.
Attachment requires an elevated silhouette fit and rejection of the stationary
table hypothesis. An initial mask alone never confirms a grasp.
"""
import cv2
import numpy as np


class SilhouetteAttachment:
    wrist = True
    silhouette = True

    def __init__(self, rgb, mask, ee, prototype, geometry, rank, xy):
        self.geometry = geometry
        self.rank = rank
        self.xy = np.asarray(xy, float).copy()
        self.prototype = prototype
        self.mask = np.asarray(mask, bool)
        self.anchor = self.mask.copy()
        self.origin = np.asarray(ee[:3], float).copy()
        self.confirmed = False
        self.anchor_locked = False
        self.static_shift_px = 0.
        self.lifting = False
        self.streak = 0
        self.tick = 0
        self.last_fit = None
        self.valid = False
        if prototype is not None and prototype.get('mass', 0) >= .5:
            current, _ = self.extract(rgb)
            if current is not None and self.iou(current, self.mask) >= .65:
                self.mask = current
                self.anchor = current.copy()
                self.valid = True

    @staticmethod
    def iou(a, b):
        return float(np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b)))

    @staticmethod
    def descriptor(mask):
        yy, xx = np.nonzero(mask)
        return np.array([xx.mean(), yy.mean(), np.ptp(xx), np.ptp(yy)], float)

    def extract(self, rgb):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        delta = np.abs(hsv[:, :, 0].astype(float) - self.prototype['hue'])
        delta = np.minimum(delta, 180 - delta)
        foreground = ((delta < 10) & (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 45)).astype(np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
        choices = []
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < 400 or area > foreground.size * .8:
                continue
            mask = labels == index
            overlap = self.iou(mask, self.mask)
            if overlap > .4:
                choices.append((overlap, mask))
        choices.sort(key=lambda item: -item[0])
        diag = {'candidate_ious': [item[0] for item in choices]}
        if not choices or (len(choices) > 1 and choices[0][0] - choices[1][0] < .15):
            return None, diag
        return choices[0][1], diag

    def begin_lift(self, ee):
        self.origin = np.asarray(ee[:3], float).copy()
        self.anchor = self.mask.copy()
        self.lifting = True
        self.streak = 0

    def lock_attachment(self):
        self.anchor = self.mask.copy()
        self.anchor_locked = True
        self.lifting = False

    def update(self, rgb, ee):
        self.tick += 1
        if not self.valid:
            return {'valid': False, 'reason': 'silhouette_initialization_unqualified'}
        current, extraction = self.extract(rgb)
        if current is None:
            self.valid = False
            return {'valid': False, 'reason': 'current_rgb_silhouette_missing', **extraction}
        previous = self.descriptor(self.mask)
        observed = self.descriptor(current)
        anchor = self.descriptor(self.anchor)
        frame_motion = float(np.linalg.norm(observed[:2] - previous[:2]))
        anchor_motion = float(np.linalg.norm(observed[:2] - anchor[:2]))
        anchor_iou = self.iou(current, self.anchor)
        size_error = float(np.max(np.abs(observed[2:] / np.maximum(anchor[2:], 1) - 1)))
        self.mask = current
        rise = float(ee[2] - self.origin[2])
        # A table object can be perfectly visible, so geometry alone must also
        # distinguish elevated from table-static before confirming attachment.
        if self.lifting and rise > .08 and self.tick % 4 == 0:
            fitted, fit = self.geometry.fit_lift(current, ee, self.rank, self.xy, rise)
            table_hull = self.geometry.hull(self.xy, ee, self.rank)
            table_mask = np.zeros(current.shape, np.uint8)
            if table_hull is not None:
                cv2.fillConvexPoly(table_mask, np.rint(table_hull).astype(np.int32), 1)
            table_iou = self.iou(current, table_mask.astype(bool))
            qualified = bool(fitted is not None and anchor_iou >= .65 and
                             anchor_motion < 25 and size_error < .25 and
                             anchor_iou - table_iou > .20)
            self.streak = self.streak + 1 if qualified else 0
            self.last_fit = dict(fit, table_static_iou=table_iou,
                                 current_anchor_iou=anchor_iou, qualified=qualified)
            if self.streak >= 2:
                self.confirmed = True
        consistent = bool(anchor_iou >= .60 and size_error < .30)
        return {'valid': consistent, 'tracker': 'current_rgb_silhouette',
                'support_pixels': int(current.sum()), 'attached_error_px': frame_motion,
                'locked_error_px': anchor_motion if self.anchor_locked and consistent else None,
                'absolute_object_motion_px': anchor_motion, 'rise_m': rise,
                'confirmed': self.confirmed, 'anchor_locked': self.anchor_locked,
                'anchor_iou': anchor_iou, 'size_error': size_error,
                'lift_geometry': self.last_fit, 'qualification_streak': self.streak,
                **extraction}
