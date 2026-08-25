import io
import cv2
from PIL import Image

from stage1_quality import check_input_quality
from stage2_prepare import prepare_image
from stage3_hsv import prepare_colour_masks
from stage4_watershed import detect_pods_with_watershed
from stage5_label import label_pods
from stage6_features import extract_features
from stage7_classify import classify_pod_colours


def run_pipeline(image_rgb, params=None):
    params = params or {}

    stage1 = check_input_quality(image_rgb)
    stage2 = prepare_image(image_rgb)
    stage3 = prepare_colour_masks(stage2["prepared_rgb"], stage2["paper_roi_mask"])

    stage4 = detect_pods_with_watershed(
        prepared_rgb=stage3["combined_rgb"],
        majority_mask=stage3["combined_mask"],
        minimum_area=params.get("minimum_area", 3000),
        maximum_area=params.get("maximum_area", 190000),
        core_opening_size=params.get("core_opening_size", 17),
        core_distance=params.get("core_distance", 8),
        minimum_solidity=params.get("minimum_solidity", 0.82),
        minimum_aspect_ratio=params.get("minimum_aspect_ratio", 1.25),
        maximum_aspect_ratio=params.get("maximum_aspect_ratio", 4.5),
    )

    stage5 = label_pods(stage2["prepared_rgb"], stage4["accepted_regions"])
    stage6 = extract_features(stage2["prepared_rgb"], stage5["label_image"], stage5["pods"])
    stage7 = classify_pod_colours(
        stage6,
        stage5["label_image"],
        stage3["green_yellow_mask"],
        stage3["red_brown_mask"],
        red_brown_weight=params.get("red_brown_weight", 1.15),
        minimum_evidence_coverage=params.get("minimum_evidence_coverage", 0.10),
        minimum_score_margin=params.get("minimum_score_margin", 0.08),
    )

    final_overlay = stage5["overlay_ids"].copy()
    for _, row in stage7.iterrows():
        cv2.putText(
            final_overlay,
            str(row["Pod_Color"]),
            (int(row["CentroidX"]), int(row["CentroidY"])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            2,
        )

    return {
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "stage4": stage4,
        "stage5": stage5,
        "stage6": stage6,
        "stage7": stage7,
        "candidate_majority": stage7.attrs.get("image_majority", "UNCERTAIN"),
        "green_candidate_count": stage7.attrs.get("green_candidate_count", 0),
        "red_candidate_count": stage7.attrs.get("red_candidate_count", 0),
        "uncertain_candidate_count": stage7.attrs.get("uncertain_candidate_count", 0),
        "final_overlay": final_overlay,
    }


def image_bytes(image_rgb):
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    return buffer.getvalue()
