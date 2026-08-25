# CocoaTrack

CocoaTrack is an image-based cocoa monitoring prototype integrating cocoa pod detection, pod-region refinement, colour-based maturity estimation, and harvestable dry bean yield estimation.

## System Flow

Field Image
-> YOLO-Pro Cocoa Pod Detection
-> Tiled Detection + NMS
-> GrabCut + Watershed Segmentation
-> HSV / CIELAB Colour Analysis
-> Maturity Estimation
-> Ripe Pod Counting
-> Estimated Dry Bean Yield

## Main Components

- Edge Impulse YOLO-Pro cocoa pod detector
- Full-image + tiled inference
- GrabCut and watershed pod-region refinement
- HSV and CIELAB colour feature extraction
- Maturity classes: UNRIPE, HALF-RIPE, RIPE, UNCERTAIN
- Harvestable dry bean yield estimation: Ydry = Nripe / 25

## Local Runtime

The current detector is deployed as an Edge Impulse EIM compiled for Apple Silicon macOS.

Model architecture:
- macOS
- ARM64
- Mach-O executable

The complete inference pipeline is therefore currently intended to run locally on Apple Silicon.

The current EIM cannot run directly on Streamlit Community Cloud because Streamlit Community Cloud uses Linux.

## Python Environment

Developed using Python 3.11.

Install dependencies:

    python -m pip install -r requirements.txt

Run CocoaTrack locally:

    python -m streamlit run app.py

## Yield Interpretation

The yield estimate represents dry bean yield associated with ripe pods visible and successfully detected in the analysed image.

It should not automatically be interpreted as whole-tree yield.

## Prototype Status

CocoaTrack is a research prototype intended for demonstration and further development.
