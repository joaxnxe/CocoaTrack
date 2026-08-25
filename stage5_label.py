import cv2
import numpy as np


def label_pods(prepared_rgb, accepted_regions):
    label_image = np.zeros(prepared_rgb.shape[:2], dtype=np.int32)
    overlay = prepared_rgb.copy()
    pods = []

    for pod_id, region in enumerate(accepted_regions, start=1):
        contour = region['contour']
        x, y, w, h = region['bbox']
        cv2.drawContours(label_image, [contour], -1, int(pod_id), -1)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(overlay, str(pod_id), (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        pods.append({'PodID': pod_id, 'BBox': (x, y, w, h), 'Contour': contour})

    return {'label_image': label_image, 'overlay_ids': overlay, 'pods': pods}
