"""Accepted RGB/FK head projection; no simulator camera metadata access."""
from contextlib import contextmanager
from contextvars import ContextVar
import cv2
import numpy as np

_CURRENT = ContextVar('lxm6wuec_nesting_head_camera')


class HeadCamera:
    def __init__(self, record):
        if record.get('source') != 'rgb_fingertip_fk':
            raise ValueError('Head requires existing RGB-visible fingertip/FK calibration')
        if not (record.get('accepted') is True or record.get('quality', {}).get('accepted') is True):
            raise ValueError('Head RGB calibration is not accepted')
        self.projection = np.asarray(record.get('projection'), float)
        if self.projection.shape != (3, 4) or not np.isfinite(self.projection).all():
            raise ValueError('Head projection must be finite 3x4')
        self.inverse = np.linalg.inv(self.projection[:, :3])
        self.center = -self.inverse @ self.projection[:, 3]
        intrinsic, _, _, *_ = cv2.decomposeProjectionMatrix(self.projection)
        intrinsic = intrinsic / intrinsic[2, 2]
        self.focal = float(np.sqrt(abs(intrinsic[0, 0] * intrinsic[1, 1])))
        self.depth_row = self.projection[2] / np.linalg.norm(self.projection[2, :3])
        if self.focal < 1 or not np.isfinite(self.center).all():
            raise ValueError('Invalid finite pinhole head calibration')

    def project(self, points):
        p = np.asarray(points, float).reshape(-1, 3)
        q = np.column_stack((p, np.ones(len(p)))) @ self.projection.T
        if np.any(abs(q[:, 2]) < 1e-9):
            raise ValueError('Projection at infinity')
        return q[:, :2] / q[:, 2:3]

    def rays(self, pixels):
        uv = np.asarray(pixels, float).reshape(-1, 2)
        return np.column_stack((uv, np.ones(len(uv)))) @ self.inverse.T

    def on_plane(self, pixels, z):
        rays = self.rays(pixels)
        if np.any(abs(rays[:, 2]) < 1e-10):
            raise ValueError('Ray parallel to known support plane')
        return self.center + rays * ((float(z)-self.center[2])/rays[:, 2])[:, None]

    def pixel_scale(self, pixel, z):
        point = self.on_plane([pixel], z)[0]
        # Same camera-plane metric as upstream depth/FX, with the calibrated
        # focal scale. This does not introduce a different table-area metric.
        return float(abs(np.r_[point, 1.] @ self.depth_row) / self.focal)


@contextmanager
def use_camera(camera):
    token = _CURRENT.set(camera)
    try:
        yield camera
    finally:
        _CURRENT.reset(token)


def head_camera():
    return _CURRENT.get()


def project(points):
    return head_camera().project(points)


def on_plane(pixels, z):
    return head_camera().on_plane(pixels, z)


def ray(u, v):
    return head_camera().rays([[u, v]])[0]


def pixel_scale(pixel, z):
    return head_camera().pixel_scale(pixel, z)
