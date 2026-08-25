import cv2
import numpy as np


def _odd(value: int, minimum: int = 1) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def detect_pods_with_watershed(
    prepared_rgb,
    majority_mask,
    minimum_area=1000,
    maximum_area=50000,
    clean_opening_size=25,
    clean_closing_size=13,
    core_opening_size=17,
    core_distance=25,
    minimum_seed_area=1,
    minimum_solidity=0.4,
    minimum_aspect_ratio=0.8,
    maximum_aspect_ratio=6.0,
):
    """Detect candidate cocoa pods with manually adjustable Stage 4 settings."""

    if prepared_rgb is None or prepared_rgb.size == 0:
        raise ValueError("Prepared image is empty.")

    if majority_mask is None or majority_mask.size == 0:
        raise ValueError("Detection mask is empty.")

    if prepared_rgb.shape[:2] != majority_mask.shape[:2]:
        raise ValueError("Prepared image and detection mask have different sizes.")

    if minimum_area > maximum_area:
        raise ValueError("Minimum area cannot be greater than maximum area.")

    if minimum_aspect_ratio > maximum_aspect_ratio:
        raise ValueError("Minimum aspect ratio cannot exceed maximum aspect ratio.")

    mask = majority_mask.astype(np.uint8)
    if mask.max() == 1:
        mask *= 255

    opening_size = _odd(clean_opening_size, 1)
    closing_size = _odd(clean_closing_size, 1)
    core_size = _odd(core_opening_size, 3)

    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_size, opening_size)),
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size)),
    )

    thick = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_size, core_size)),
    )

    distance = cv2.distanceTransform(thick, cv2.DIST_L2, 5)
    seeds = (distance >= float(core_distance)).astype(np.uint8) * 255

    seeds = cv2.morphologyEx(
        seeds,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    number_of_seed_labels, seed_labels, seed_stats, _ = cv2.connectedComponentsWithStats(
        seeds,
        connectivity=8,
    )

    filtered_seeds = np.zeros_like(seeds)

    for label_id in range(1, number_of_seed_labels):
        seed_area = int(seed_stats[label_id, cv2.CC_STAT_AREA])
        if seed_area >= int(minimum_seed_area):
            filtered_seeds[seed_labels == label_id] = 255

    marker_count, markers = cv2.connectedComponents(filtered_seeds)
    markers = markers.astype(np.int32) + 1

    background_mask = cv2.bitwise_not(cleaned)
    markers[background_mask > 0] = 1

    unknown_region = cv2.subtract(cleaned, filtered_seeds)
    markers[unknown_region > 0] = 0

    watershed_input = cv2.cvtColor(prepared_rgb, cv2.COLOR_RGB2BGR)
    watershed_markers = cv2.watershed(watershed_input, markers)

    output_image = prepared_rgb.copy()
    accepted_mask = np.zeros_like(cleaned)
    rejected_mask = np.zeros_like(cleaned)
    regions = []

    for label_value in np.unique(watershed_markers):
        if label_value <= 1:
            continue

        region_mask = (watershed_markers == label_value).astype(np.uint8) * 255
        region_mask = cv2.bitwise_and(region_mask, cleaned)

        contours, _ = cv2.findContours(
            region_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))

        if area <= 0:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        rotated_width, rotated_height = cv2.minAreaRect(contour)[1]
        long_side = max(rotated_width, rotated_height)
        short_side = min(rotated_width, rotated_height)
        aspect_ratio = long_side / short_side if short_side > 0 else 999.0

        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / hull_area if hull_area > 0 else 0.0

        perimeter = float(cv2.arcLength(contour, True))
        circularity = (
            4.0 * np.pi * area / (perimeter ** 2)
            if perimeter > 0
            else 0.0
        )

        accepted = (
            float(minimum_area) <= area <= float(maximum_area)
            and float(minimum_aspect_ratio) <= aspect_ratio <= float(maximum_aspect_ratio)
            and solidity >= float(minimum_solidity)
        )

        region = {
            "contour": contour,
            "mask": region_mask,
            "accepted": accepted,
            "area": area,
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
            "circularity": circularity,
            "bbox": (x, y, width, height),
        }
        regions.append(region)

        if accepted:
            cv2.drawContours(accepted_mask, [contour], -1, 255, -1)
        else:
            cv2.drawContours(rejected_mask, [contour], -1, 255, -1)

    accepted_regions = [region for region in regions if region["accepted"]]

    for pod_index, region in enumerate(accepted_regions, start=1):
        x, y, width, height = region["bbox"]
        cv2.rectangle(
            output_image,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            3,
        )
        cv2.putText(
            output_image,
            f"Pod {pod_index}",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

    boundary_image = prepared_rgb.copy()
    boundary_image[watershed_markers == -1] = [255, 0, 255]

    settings = {
        "Minimum area": int(minimum_area),
        "Maximum area": int(maximum_area),
        "Clean opening size": opening_size,
        "Clean closing size": closing_size,
        "Core opening size": core_size,
        "Core distance": float(core_distance),
        "Minimum seed area": int(minimum_seed_area),
        "Minimum solidity": float(minimum_solidity),
        "Minimum aspect ratio": float(minimum_aspect_ratio),
        "Maximum aspect ratio": float(maximum_aspect_ratio),
    }

    return {
        "cleaned_mask": cleaned,
        "thick_objects": thick,
        "distance_map": distance,
        "seed_mask": filtered_seeds,
        "watershed_markers": watershed_markers,
        "boundary_image": boundary_image,
        "accepted_mask": accepted_mask,
        "rejected_mask": rejected_mask,
        "output_image": output_image,
        "regions": regions,
        "accepted_regions": accepted_regions,
        "seed_count": max(marker_count - 1, 0),
        "settings": settings,
    }
