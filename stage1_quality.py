import cv2
import numpy as np


def check_input_quality(image_rgb):
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError('Input image is empty.')

    h, w = image_rgb.shape[:2]
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    paper_mask = ((v > 0.85) & (s < 0.20)).astype(np.uint8) * 255
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    mean_v = float(v.mean())
    paper_ratio = float(np.count_nonzero(paper_mask) / paper_mask.size)
    min_dim = min(h, w)

    status = 'PASS'
    reasons = []

    if min_dim < 800:
        status = 'WARN'
        reasons.append('Low resolution')
    if mean_v < 0.25 or mean_v > 0.92:
        status = 'FAIL'
        reasons.append('Brightness outside acceptable range')
    elif mean_v < 0.35 or mean_v > 0.85:
        status = 'WARN' if status != 'FAIL' else status
        reasons.append('Brightness may be suboptimal')
    if paper_ratio < 0.05:
        status = 'FAIL'
        reasons.append('White cardboard insufficient')
    elif paper_ratio < 0.10:
        status = 'WARN' if status != 'FAIL' else status
        reasons.append('White cardboard area is small')
    if blur_score < 25:
        status = 'WARN' if status != 'FAIL' else status
        reasons.append('Possible blur')

    return {
        'status': status,
        'reasons': reasons or ['OK'],
        'paper_mask': paper_mask,
        'metrics': {
            'Height': h,
            'Width': w,
            'Minimum dimension': min_dim,
            'Mean brightness': round(mean_v, 4),
            'White-cardboard ratio': round(paper_ratio, 4),
            'Blur score': round(blur_score, 2),
        },
    }
