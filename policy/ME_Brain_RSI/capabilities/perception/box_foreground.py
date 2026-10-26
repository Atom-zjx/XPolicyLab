"""Semantic-box foreground from current local RGB background contrast."""
import cv2
import numpy as np
from .proposal_masks import iou


def seeded_proposal_masks(rgb, proposals):
    merged = []
    for proposal in sorted(proposals, key=lambda p: -p['confidence']):
        if not any(iou(proposal['box'], q['box']) > .45 for q in merged):
            merged.append(proposal)
    output = []
    height, width = rgb.shape[:2]
    for proposal in merged:
        box = np.asarray(proposal['box'])
        x0, y0 = np.maximum(0, np.floor(box[:2] - 10)).astype(int)
        x1, y1 = np.minimum([width, height], np.ceil(box[2:] + 10)).astype(int)
        crop = rgb[y0:y1, x0:x1].copy()
        h, w = crop.shape[:2]
        if min(h, w) < 14:
            continue
        a, b = np.maximum(1, np.floor(box[:2] - [x0, y0])).astype(int)
        c, d = np.minimum([w-1, h-1], np.ceil(box[2:] - [x0, y0])).astype(int)
        interior = np.zeros((h, w), np.uint8)
        interior[b:d, a:c] = 1
        lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
        background = lab[interior == 0]
        cv2.setRNGSeed(0)
        _, labels, centers = cv2.kmeans(background, 3, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, .2),
            1, cv2.KMEANS_PP_CENTERS)
        residual = np.linalg.norm(lab[:, :, None] - centers, axis=3).min(axis=2)
        threshold = max(12., float(np.percentile(residual[interior == 0], 99)))
        core = cv2.erode(interior, np.ones((3, 3), np.uint8)) > 0
        seeds = core & (residual > threshold)
        if seeds.sum() < 10:
            continue
        mask = np.where(interior, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)
        mask[seeds] = cv2.GC_FGD
        cv2.grabCut(crop, mask, None, np.zeros((1, 65)), np.zeros((1, 65)),
                    4, cv2.GC_INIT_WITH_MASK)
        foreground = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 30:
            continue
        rx, ry, rw, rh = cv2.boundingRect(contour)
        # A tiny reflective fragment is not the object represented by the semantic box.
        if rw < .65 * (c-a) or rh < .65 * (d-b):
            continue
        full = np.zeros((height, width), np.uint8)
        cv2.fillPoly(full, [contour + [x0, y0]], 1)
        output.append((full, {
            'box': [rx+x0, ry+y0, rx+rw+x0, ry+rh+y0],
            'confidence': proposal['confidence'], 'mask_area': int(full.sum()),
            'semantic_prompt': 'DINO_box_current_RGB_background_seeds',
            'proposal_box': proposal['box'], 'foreground_seed_pixels': int(seeds.sum()),
            'background_residual_threshold': threshold,
        }))
    return output
