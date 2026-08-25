import cv2
import numpy as np


def prepare_colour_masks(prepared_rgb, paper_roi_mask):
    """Create mutually exclusive green-yellow and red-brown masks.

    Stage 3 does not choose the final image class. Both masks are combined for
    Stage 4 detection. Stage 7 later classifies accepted candidates by overlap.

    Important correction:
    A pixel cannot belong to both masks. Strong green evidence is protected
    from the red-brown mask so green pods are not counted as red-brown merely
    because the earlier red mask was very broad.
    """
    if prepared_rgb is None or prepared_rgb.size == 0:
        raise ValueError("Prepared image is empty.")
    if paper_roi_mask is None or paper_roi_mask.size == 0:
        raise ValueError("Paper ROI mask is empty.")
    if prepared_rgb.shape[:2] != paper_roi_mask.shape[:2]:
        raise ValueError("Prepared image and paper ROI mask have different sizes.")

    roi = paper_roi_mask.astype(np.uint8)
    if roi.max() == 1:
        roi *= 255

    hsv = cv2.cvtColor(prepared_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(prepared_rgb, cv2.COLOR_RGB2LAB)

    h = hsv[:, :, 0].astype(np.int16)
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)

    r = prepared_rgb[:, :, 0].astype(np.int16)
    g = prepared_rgb[:, :, 1].astype(np.int16)
    b = prepared_rgb[:, :, 2].astype(np.int16)
    lab_a = lab[:, :, 1].astype(np.int16)
    lab_b = lab[:, :, 2].astype(np.int16)

    inside = roi > 0

    # ---------------------------------------------------------
    # GREEN-YELLOW EVIDENCE
    # OpenCV hue uses 0-179.
    # ---------------------------------------------------------
    green_hue = (h >= 18) & (h <= 95)
    green_rgb = (g >= r - 8) & (g >= b - 10)
    yellow_rgb = (r >= b + 8) & (g >= b + 8) & (np.abs(r - g) <= 55)

    green_raw = (
        green_hue
        & (green_rgb | yellow_rgb)
        & (s >= 22)
        & (v >= 15)
        & inside
    )

    # Strong green evidence is used to protect green pixels from the broad
    # red-brown rules below.
    strong_green = (
        green_raw
        & (
            (g >= r + 3)
            | ((h >= 25) & (h <= 90))
            | ((lab_a <= 128) & (lab_b >= 125))
        )
    )

    # ---------------------------------------------------------
    # RED-BROWN EVIDENCE
    # Narrower than the previous version to prevent green pods from being
    # swallowed by a very broad red mask.
    # ---------------------------------------------------------
    red_hue = ((h <= 20) | (h >= 160)) & (s >= 18)
    brown_hue = (h > 20) & (h <= 38) & (s >= 18)

    red_rgb = (r >= g + 6) & (r >= b + 8)
    brown_rgb = (r >= g - 2) & (g >= b + 5) & (r >= b + 12)
    lab_red = lab_a >= 132

    red_raw = (
        (
            (red_hue & (red_rgb | lab_red))
            | (brown_hue & brown_rgb & (lab_a >= 128))
            | (red_rgb & (lab_a >= 134) & (s >= 15))
        )
        & (v >= 8)
        & inside
    )

    # Mutual exclusivity: clear green-yellow evidence wins over red-brown.
    red_raw &= ~strong_green

    # Any remaining overlap is resolved using colour evidence instead of
    # allowing the same pixel to vote twice.
    overlap = green_raw & red_raw
    if np.any(overlap):
        green_strength = (
            np.maximum(g - r, 0)
            + np.maximum(g - b, 0)
            + np.maximum(95 - np.abs(h - 55), 0)
        )
        red_strength = (
            np.maximum(r - g, 0)
            + np.maximum(r - b, 0)
            + np.maximum(lab_a - 128, 0) * 2
        )
        green_wins = overlap & (green_strength >= red_strength)
        red_wins = overlap & ~green_wins
        red_raw[green_wins] = False
        green_raw[red_wins] = False

    green_mask = green_raw.astype(np.uint8) * 255
    red_mask = red_raw.astype(np.uint8) * 255

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, open_kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, close_kernel)
    green_mask = cv2.bitwise_and(green_mask, roi)

    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, open_kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, close_kernel)
    red_mask = cv2.bitwise_and(red_mask, roi)

    # Enforce exclusivity again after morphology.
    overlap_after = cv2.bitwise_and(green_mask, red_mask)
    red_mask[overlap_after > 0] = 0

    combined_mask = cv2.bitwise_or(green_mask, red_mask)
    combined_mask = cv2.bitwise_and(combined_mask, roi)
    combined_rgb = cv2.bitwise_and(prepared_rgb, prepared_rgb, mask=combined_mask)

    return {
        "hsv": hsv,
        "green_yellow_mask": green_mask,
        "red_brown_mask": red_mask,
        "combined_mask": combined_mask,
        "combined_rgb": combined_rgb,
        "counts": {
            "Green-yellow pixels": int(np.count_nonzero(green_mask)),
            "Red-brown pixels": int(np.count_nonzero(red_mask)),
            "Overlapping pixels after resolution": int(
                np.count_nonzero(cv2.bitwise_and(green_mask, red_mask))
            ),
            "Combined detection pixels": int(np.count_nonzero(combined_mask)),
        },
    }


# Compatibility alias for older code.
def classify_majority_colour(prepared_rgb, paper_roi_mask):
    result = prepare_colour_masks(prepared_rgb, paper_roi_mask)
    result["majority_mask"] = result["combined_mask"]
    result["majority_only_rgb"] = result["combined_rgb"]
    result["selected_rgb"] = result["combined_rgb"]
    result["majority_class"] = "DETERMINED AFTER STAGE 4"
    result["majority_percentage"] = 0.0
    return result
