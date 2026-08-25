import cv2
import numpy as np
import pandas as pd


def extract_features(prepared_rgb, label_image, pods):
    hsv = cv2.cvtColor(prepared_rgb, cv2.COLOR_RGB2HSV)
    rows = []

    for pod in pods:
        pod_id = pod['PodID']
        mask = (label_image == pod_id).astype(np.uint8) * 255
        contour = pod['Contour']
        x, y, w, h = pod['BBox']
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        rw, rh = cv2.minAreaRect(contour)[1]
        long_side, short_side = max(rw, rh), min(rw, rh)
        ar = long_side / short_side if short_side > 0 else np.nan
        solidity = area / hull_area if hull_area > 0 else np.nan
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else np.nan

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        h_vals = hsv[:, :, 0][mask > 0].astype(float)
        s_vals = hsv[:, :, 1][mask > 0].astype(float)
        v_vals = hsv[:, :, 2][mask > 0].astype(float)

        rows.append({
            'PodID': pod_id,
            'Area': round(area, 2),
            'AR': round(float(ar), 4),
            'Solidity': round(float(solidity), 4),
            'Circularity': round(float(circularity), 4),
            'CentroidX': round(float(xs.mean()), 2),
            'CentroidY': round(float(ys.mean()), 2),
            'BBoxX': x,
            'BBoxY': y,
            'BBoxWidth': w,
            'BBoxHeight': h,
            'H_Mean': round(float(h_vals.mean()), 4),
            'S_Mean': round(float(s_vals.mean()), 4),
            'V_Mean': round(float(v_vals.mean()), 4),
        })

    return pd.DataFrame(rows)
