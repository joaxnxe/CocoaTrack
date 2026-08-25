import numpy as np
import pandas as pd


def classify_pod_colours(
    features_df: pd.DataFrame,
    label_image,
    green_yellow_mask,
    red_brown_mask,
    minimum_evidence_coverage: float = 0.08,
    minimum_score_margin: float = 0.05,
) -> pd.DataFrame:
    """Classify accepted candidates using mutually exclusive mask overlap.

    There is no red-brown weighting. Every green-yellow pixel and red-brown
    pixel has one equal vote. Each accepted candidate is first classified
    individually. The final image majority is then based on candidate counts,
    never on full-image branch pixels.
    """
    output = features_df.copy()

    if output.empty:
        for column, dtype in (
            ("Green_Pixels", "int64"),
            ("Red_Brown_Pixels", "int64"),
            ("Green_Percentage", "float64"),
            ("Red_Brown_Percentage", "float64"),
            ("Colour_Evidence_Coverage", "float64"),
            ("Score_Margin", "float64"),
            ("Initial_Colour", "object"),
            ("Initial_Confidence", "float64"),
            ("Image_Majority", "object"),
            ("Pod_Color", "object"),
        ):
            output[column] = pd.Series(dtype=dtype)
        output.attrs.update(
            image_majority="UNCERTAIN",
            green_candidate_count=0,
            red_candidate_count=0,
            uncertain_candidate_count=0,
        )
        return output

    labels = np.asarray(label_image)
    green = np.asarray(green_yellow_mask) > 0
    red = np.asarray(red_brown_mask) > 0

    if labels.shape != green.shape or labels.shape != red.shape:
        raise ValueError("Label image and colour masks must have matching sizes.")

    # Defensive exclusivity. Green pixels are removed from red if an older
    # Stage 3 file accidentally produced overlap.
    red = red & ~green

    rows = []

    for _, feature_row in output.iterrows():
        pod_id = int(feature_row["PodID"])
        pod = labels == pod_id
        total_pixels = int(np.count_nonzero(pod))

        green_pixels = int(np.count_nonzero(pod & green))
        red_pixels = int(np.count_nonzero(pod & red))
        evidence_pixels = green_pixels + red_pixels

        if total_pixels == 0:
            green_fraction = red_fraction = coverage = margin = confidence = 0.0
            colour = "UNCERTAIN"
        else:
            green_fraction = green_pixels / total_pixels
            red_fraction = red_pixels / total_pixels
            coverage = evidence_pixels / total_pixels
            margin = abs(green_fraction - red_fraction)

            if coverage < minimum_evidence_coverage:
                colour = "UNCERTAIN"
                confidence = coverage * 100.0
            elif margin < minimum_score_margin:
                # Tie-break with the candidate's measured mean hue only when
                # overlap evidence is genuinely close.
                hue = float(feature_row.get("H_Mean", -1.0))
                if 18 <= hue <= 95:
                    colour = "GREEN-YELLOW"
                elif (0 <= hue <= 38) or (160 <= hue <= 179):
                    colour = "RED-BROWN"
                else:
                    colour = "UNCERTAIN"
                confidence = max(green_fraction, red_fraction) * 100.0
            elif green_fraction > red_fraction:
                colour = "GREEN-YELLOW"
                confidence = green_fraction / max(
                    green_fraction + red_fraction, 1e-9
                ) * 100.0
            else:
                colour = "RED-BROWN"
                confidence = red_fraction / max(
                    green_fraction + red_fraction, 1e-9
                ) * 100.0

        rows.append(
            {
                "Green_Pixels": green_pixels,
                "Red_Brown_Pixels": red_pixels,
                "Green_Percentage": round(green_fraction * 100.0, 2),
                "Red_Brown_Percentage": round(red_fraction * 100.0, 2),
                "Colour_Evidence_Coverage": round(coverage * 100.0, 2),
                "Score_Margin": round(margin * 100.0, 2),
                "Initial_Colour": colour,
                "Initial_Confidence": round(confidence, 2),
            }
        )

    overlap_df = pd.DataFrame(rows, index=output.index)
    for column in overlap_df.columns:
        output[column] = overlap_df[column]

    green_count = int((output["Initial_Colour"] == "GREEN-YELLOW").sum())
    red_count = int((output["Initial_Colour"] == "RED-BROWN").sum())
    uncertain_count = int((output["Initial_Colour"] == "UNCERTAIN").sum())

    if green_count == 0 and red_count == 0:
        image_majority = "UNCERTAIN"
    elif green_count >= red_count:
        image_majority = "GREEN-YELLOW"
    else:
        image_majority = "RED-BROWN"

    output["Image_Majority"] = image_majority

    # Preserve the requested majority-guided final output, while keeping each
    # candidate's true overlap decision visible in Initial_Colour.
    output["Pod_Color"] = image_majority

    output.attrs.update(
        image_majority=image_majority,
        green_candidate_count=green_count,
        red_candidate_count=red_count,
        uncertain_candidate_count=uncertain_count,
        minimum_evidence_coverage=float(minimum_evidence_coverage),
        minimum_score_margin=float(minimum_score_margin),
    )

    return output
