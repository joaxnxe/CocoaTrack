import cv2
import numpy as np


def _resize_long_side(image_rgb, target=1280):
    h, w = image_rgb.shape[:2]
    scale = min(1.0, target / max(h, w))
    if scale == 1.0:
        return image_rgb.copy(), scale
    return cv2.resize(image_rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA), scale


def prepare_image(image_rgb, target_long_side=1280, crop_margin=20):
    resized, scale = _resize_long_side(image_rgb, target_long_side)
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    paper_pixels = (((v > int(0.85 * 255)) & (s < int(0.20 * 255))).astype(np.uint8) * 255)
    cleaned = cv2.morphologyEx(paper_pixels, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('White cardboard could not be detected.')

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 5000:
        raise ValueError('Detected white-cardboard region is too small.')

    hull = cv2.convexHull(largest)
    full_roi = np.zeros(resized.shape[:2], dtype=np.uint8)
    cv2.drawContours(full_roi, [hull], -1, 255, -1)

    x, y, w, h = cv2.boundingRect(hull)
    x1 = max(0, x - crop_margin)
    y1 = max(0, y - crop_margin)
    x2 = min(resized.shape[1], x + w + crop_margin)
    y2 = min(resized.shape[0], y + h + crop_margin)

    cropped = resized[y1:y2, x1:x2].copy()
    roi = full_roi[y1:y2, x1:x2].copy()
    prepared = cv2.bitwise_and(cropped, cropped, mask=roi)

    return {
        'resized_rgb': resized,
        'paper_pixel_mask': paper_pixels,
        'paper_roi_mask': roi,
        'prepared_rgb': prepared,
        'crop_box': (x1, y1, x2, y2),
        'scale': scale,
    }
