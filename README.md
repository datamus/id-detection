# ID Detection using Webcam

This project uses a YOLOv8 model to detect and track various types of IDs in real-time using a webcam. It is designed to recognize different document types such as passports, driver's licenses, and certificates, providing a robust solution for document identification and tracking.

## Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/datamus/id-detection.git
   cd id-detection

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt

3. Run the script:
   ```sh
   python id-detection.py

## Requirements
1. Python 3.x
2. OpenCV
3. Ultralytics YOLO

## Description
This script utilizes the YOLOv8 model to detect and track various types of IDs (such as passports, driver's licenses, etc.) in real-time using a webcam feed. The process involves the following steps:
1. Model Initialization: The YOLOv8 model is loaded and ready to process video frames.
2. Webcam Feed: The script captures frames from the webcam.
3. Detection and Tracking: Each frame is processed to detect and track documents, with track history maintained for accurate tracking.
4. Visualisation: The detected documents are annotated on the frame with bounding boxes and tracking lines.

Key features include:
-Detection of multiple document types with specified dimensions.
-Calculation of width-to-height ratios for accurate bounding box display.
-Real-time tracking with visualization of tracking lines and bounding boxes.
-Overlay of a boundary box to assess the containment of detected documents.

Upon introduction of the ID in the frame, an overlayed boundary box is created at the center of the frame. This boundary box reflects the ratios of the detected ID type. Proximity calculations activate the coloring of the boundary box when the ID document is within the perimeter at a certain distance from the corners.

## Future Work
This is still a work in progress. The next steps include:
1. Adding direction/prompts for the user to guide the ID into the centered overlayed boundary box.
2. Capturing frames when the boundary box coloring is activated (proximity conditions are satisfied).
3. Performing image quality assessment of captured frames.
4. Extracting text from fields when image quality conditions are met.
