from pathlib import Path
import time

import cv2
import numpy as np
from edge_impulse_linux.image import ImageImpulseRunner


import platform

_MODEL_DIR = Path(__file__).resolve().parent / "models"

_system = platform.system().lower()
_machine = platform.machine().lower()

if _system == "linux" and _machine in {"x86_64", "amd64"}:
    MODEL_PATH = _MODEL_DIR / "cocoa_detector_linux.eim"
else:
    MODEL_PATH = _MODEL_DIR / "cocoa_detector.eim"

MODEL_SIZE = 320

# Duplicate boxes from overlapping tiles will be removed
NMS_IOU_THRESHOLD = 0.30


def _map_box_to_original(bb, original_shape):
    """
    Convert Edge Impulse 320x320 fit-shortest coordinates
    back to the supplied image/tile coordinates.
    """
    h, w = original_shape[:2]

    scale = MODEL_SIZE / min(w, h)

    resized_w = w * scale
    resized_h = h * scale

    crop_x = max(
        (resized_w - MODEL_SIZE) / 2.0,
        0.0,
    )
    crop_y = max(
        (resized_h - MODEL_SIZE) / 2.0,
        0.0,
    )

    x1 = int(
        round(
            (bb["x"] + crop_x)
            / scale
        )
    )

    y1 = int(
        round(
            (bb["y"] + crop_y)
            / scale
        )
    )

    x2 = int(
        round(
            (
                bb["x"]
                + bb["width"]
                + crop_x
            )
            / scale
        )
    )

    y2 = int(
        round(
            (
                bb["y"]
                + bb["height"]
                + crop_y
            )
            / scale
        )
    )

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    )


def _iou(box_a, box_b):
    ax1 = box_a["x"]
    ay1 = box_a["y"]
    ax2 = ax1 + box_a["width"]
    ay2 = ay1 + box_a["height"]

    bx1 = box_b["x"]
    by1 = box_b["y"]
    bx2 = bx1 + box_b["width"]
    by2 = by1 + box_b["height"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(
        0,
        inter_x2 - inter_x1,
    )
    inter_h = max(
        0,
        inter_y2 - inter_y1,
    )

    intersection = (
        inter_w
        * inter_h
    )

    area_a = (
        box_a["width"]
        * box_a["height"]
    )

    area_b = (
        box_b["width"]
        * box_b["height"]
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def _non_max_suppression(boxes):
    """
    Remove repeated detections caused by overlapping tiles.
    Keep the highest-confidence version of each pod.
    """
    boxes = sorted(
        boxes,
        key=lambda b: b["value"],
        reverse=True,
    )

    kept = []

    while boxes:
        best = boxes.pop(0)

        kept.append(best)

        boxes = [
            box
            for box in boxes
            if _iou(best, box)
            < NMS_IOU_THRESHOLD
        ]

    return kept


def _build_tiles(image_rgb):
    """
    Full image + four overlapping 2x2 tiles.

    The tiles enlarge small cocoa pods relative to the
    model's 320x320 input.
    """
    h, w = image_rgb.shape[:2]

    mid_x = w // 2
    mid_y = h // 2

    overlap_x = int(w * 0.12)
    overlap_y = int(h * 0.12)

    tile_boxes = [
        # top-left
        (
            0,
            0,
            min(
                w,
                mid_x + overlap_x,
            ),
            min(
                h,
                mid_y + overlap_y,
            ),
        ),

        # top-right
        (
            max(
                0,
                mid_x - overlap_x,
            ),
            0,
            w,
            min(
                h,
                mid_y + overlap_y,
            ),
        ),

        # bottom-left
        (
            0,
            max(
                0,
                mid_y - overlap_y,
            ),
            min(
                w,
                mid_x + overlap_x,
            ),
            h,
        ),

        # bottom-right
        (
            max(
                0,
                mid_x - overlap_x,
            ),
            max(
                0,
                mid_y - overlap_y,
            ),
            w,
            h,
        ),
    ]

    tiles = []

    for x1, y1, x2, y2 in tile_boxes:

        tile = image_rgb[
            y1:y2,
            x1:x2
        ]

        tiles.append(
            (
                tile,
                x1,
                y1,
            )
        )

    return tiles


def _infer_one_image(
    runner,
    image_rgb,
    offset_x=0,
    offset_y=0,
    source="tile",
):
    """
    Run Edge Impulse on one full image or tile,
    then return coordinates relative to the full image.
    """

    features, _ = (
        runner.get_features_from_image_auto_studio_settings(
            image_rgb
        )
    )

    result = runner.classify(
        features
    )

    boxes = result.get(
        "result",
        {},
    ).get(
        "bounding_boxes",
        [],
    )

    mapped = []

    for bb in boxes:

        x, y, width, height = (
            _map_box_to_original(
                bb,
                image_rgb.shape,
            )
        )

        mapped.append(
            {
                "x": x + offset_x,
                "y": y + offset_y,
                "width": width,
                "height": height,
                "value": float(
                    bb.get(
                        "value",
                        0.0,
                    )
                ),
                "label": bb.get(
                    "label",
                    "0",
                ),
                "source": source,
            }
        )

    return mapped


def _make_rectangle_contour(
    x,
    y,
    width,
    height,
):
    return np.array(
        [
            [[x, y]],
            [[x + width, y]],
            [[x + width, y + height]],
            [[x, y + height]],
        ],
        dtype=np.int32,
    )


def detect_pods_with_edge_impulse(
    prepared_rgb,
    combined_mask=None,
    **kwargs,
):
    """
    CocoaTrack ML detector using:

        full image
        +
        four overlapping tiles
        +
        duplicate suppression

    No HSV or colour mask is used for detection.
    """

    if prepared_rgb is None:
        raise ValueError(
            "prepared_rgb cannot be None"
        )

    image_rgb = np.asarray(
        prepared_rgb
    ).copy()

    output_image = (
        image_rgb.copy()
    )

    accepted_mask = np.zeros(
        image_rgb.shape[:2],
        dtype=np.uint8,
    )

    rejected_mask = np.zeros_like(
        accepted_mask
    )

    start_time = (
        time.perf_counter()
    )

    all_boxes = []

    with ImageImpulseRunner(
        str(MODEL_PATH)
    ) as runner:

        runner.init()

        # ---------------------------------------------
        # 1. FULL IMAGE
        # ---------------------------------------------
        full_boxes = _infer_one_image(
            runner,
            image_rgb,
            source="full",
        )

        all_boxes.extend(
            full_boxes
        )

        # ---------------------------------------------
        # 2. FOUR OVERLAPPING TILES
        # ---------------------------------------------
        tiles = _build_tiles(
            image_rgb
        )

        for tile_number, (
            tile_rgb,
            offset_x,
            offset_y,
        ) in enumerate(
            tiles,
            start=1,
        ):

            tile_boxes = (
                _infer_one_image(
                    runner,
                    tile_rgb,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    source=(
                        f"tile_{tile_number}"
                    ),
                )
            )

            all_boxes.extend(
                tile_boxes
            )

    inference_seconds = (
        time.perf_counter()
        - start_time
    )

    # ---------------------------------------------
    # 3. REMOVE DUPLICATES
    # ---------------------------------------------
    final_boxes = (
        _non_max_suppression(
            all_boxes
        )
    )

    accepted_regions = []

    for index, bb in enumerate(
        final_boxes,
        start=1,
    ):

        x = int(bb["x"])
        y = int(bb["y"])
        width = int(bb["width"])
        height = int(bb["height"])

        contour = (
            _make_rectangle_contour(
                x,
                y,
                width,
                height,
            )
        )

        region_mask = np.zeros_like(
            accepted_mask
        )

        cv2.drawContours(
            region_mask,
            [contour],
            -1,
            255,
            -1,
        )

        cv2.drawContours(
            accepted_mask,
            [contour],
            -1,
            255,
            -1,
        )

        area = float(
            cv2.contourArea(
                contour
            )
        )

        perimeter = float(
            cv2.arcLength(
                contour,
                True,
            )
        )

        hull = cv2.convexHull(
            contour
        )

        hull_area = float(
            cv2.contourArea(
                hull
            )
        )

        solidity = (
            area / hull_area
            if hull_area > 0
            else 0.0
        )

        aspect_ratio = (
            max(
                width,
                height,
            )
            / max(
                min(
                    width,
                    height,
                ),
                1,
            )
        )

        circularity = (
            4.0
            * np.pi
            * area
            / max(
                perimeter ** 2,
                1.0,
            )
        )

        confidence = float(
            bb["value"]
        )

        accepted_regions.append(
            {
                "contour": contour,
                "mask": region_mask,
                "bbox": (
                    x,
                    y,
                    width,
                    height,
                ),
                "area": area,
                "aspect_ratio": (
                    aspect_ratio
                ),
                "solidity": solidity,
                "circularity": (
                    circularity
                ),
                "accepted": True,
                "confidence": (
                    confidence
                ),
                "detection_source": (
                    bb["source"]
                ),
            }
        )

        cv2.rectangle(
            output_image,
            (x, y),
            (
                x + width,
                y + height,
            ),
            (0, 255, 0),
            3,
        )

        # Large pod number only
        label = str(index)

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.15,
            3,
        )

        label_x = x + 4
        label_y = y + text_height + 10

        # White background so the number stays readable
        cv2.rectangle(
            output_image,
            (x, y),
            (
                x + text_width + 14,
                y + text_height + baseline + 24,
            ),
            (255, 255, 255),
            -1,
        )

        # Border around number label
        cv2.rectangle(
            output_image,
            (x, y),
            (
                x + text_width + 14,
                y + text_height + baseline + 24,
            ),
            (0, 0, 0),
            2,
        )

        cv2.putText(
            output_image,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.15,
            (0, 0, 0),
            3,
        )

    return {
        "cleaned_mask": accepted_mask,
        "thick_objects": accepted_mask,
        "seed_mask": accepted_mask,

        "boundary_image": (
            output_image.copy()
        ),

        "output_image": output_image,

        "accepted_mask": (
            accepted_mask
        ),

        "rejected_mask": (
            rejected_mask
        ),

        "accepted_regions": (
            accepted_regions
        ),

        "rejected_regions": [],

        "seed_count": len(
            accepted_regions
        ),

        "detection_count": len(
            accepted_regions
        ),

        "raw_detection_count": len(
            all_boxes
        ),

        "inference_seconds": (
            inference_seconds
        ),

        "detector": (
            "Edge Impulse YOLO-Pro "
            "+ tiled inference"
        ),
    }
