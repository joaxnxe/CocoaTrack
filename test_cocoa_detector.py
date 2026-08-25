import cv2
from edge_impulse_linux.image import ImageImpulseRunner

MODEL = "/Users/joanne/Desktop/CocoaTrack_Adjustable_Stage4/models/cocoa_detector.eim"
IMAGE = "/Users/joanne/Desktop/COCOA PROGRESS/EI TESTING/T11_AngleB.JPEG"

with ImageImpulseRunner(MODEL) as runner:
    info = runner.init()

    img = cv2.imread(IMAGE)
    if img is None:
        raise RuntimeError("Could not load image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    features, cropped = runner.get_features_from_image_auto_studio_settings(img)
    result = runner.classify(features)

    boxes = result["result"].get("bounding_boxes", [])

    print(f"\nDetected cocoa pods: {len(boxes)}")

    for i, bb in enumerate(boxes, start=1):
        print(
            f"Pod {i}: confidence={bb['value']:.2%}, "
            f"x={bb['x']}, y={bb['y']}, "
            f"w={bb['width']}, h={bb['height']}"
        )

        cv2.rectangle(
            cropped,
            (bb["x"], bb["y"]),
            (bb["x"] + bb["width"], bb["y"] + bb["height"]),
            (255, 0, 0),
            2
        )

    cv2.imwrite(
        "cocoa_detection_debug.jpg",
        cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
    )

    print("\nSaved: cocoa_detection_debug.jpg")
