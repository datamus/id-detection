# src/id_detection.py

from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import math

def main():
    """
    Main function to run the ID detection using webcam.
    """
    # Load the YOLOv8 model
    model = YOLO('YOLOv8n_combined_dataset.pt')

    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Store the track history with enhanced structure
    track_history = defaultdict(lambda: {'centroids': [], 'boxes': []})

    # Initialise a dictionary to store document types for each Track ID:
    document_types = {}

    # Dictionary of document dimensions in mm (not actually necessary, just for full understanding)
    document_dimensions = {
        'PASSPORT': (125, 88),
        'REGISTRATION_DOCUMENTS': (210, 297),
        'BIRTH_CERTIFICATE': (210, 297),
        'CITIZENSHIP_CERTIFICATE': (210, 297),
        'DRIVERS_LICENCE': (85.6, 54),
        'DRIVERS_LICENSE': (85.6, 54),
        'PHOTOID': (85.6, 54),
        'MARRIAGE_CERTIFICATE': (210, 297),
        'NON_PHOTOID': (85.6, 54)
    }

    # Calculate and store width to height ratios
    document_ratios = {doc_type: dimensions[0] / dimensions[1] for doc_type, dimensions in document_dimensions.items()}

    # Assume a standard height for the overlayed boxes
    standard_height = 150 # pixels

    # Calculate widths based on the ratio for the standard height
    document_widths = {doc_type: int(ratio * standard_height) for doc_type, ratio in document_ratios.items()}

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            annotated_frame = frame
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, max_det=1, imgsz=992)
            frame_height, frame_width, _ = frame.shape
            frame_center_x = frame_width // 2
            frame_center_y = frame_height // 2
            if results and hasattr(results[0], 'boxes') and results[0].boxes:
                if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    class_ids = results[0].boxes.cls.int().cpu().tolist()  # Accessing class IDs

                    # Visualize the results on the frame (tracking line)
                    annotated_frame = results[0].plot()

                    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                        x, y, w, h = box.numpy()
                        class_name = results[0].names[class_id]  # Get the class name using class_id
                        document_type = class_name.split('.')[0]

                        document_types[track_id] = document_type

                        top_left = (x - w // 2, y - h // 2)
                        top_right = (x + w // 2, y - h // 2)
                        bottom_left = (x - w // 2, y + h // 2)
                        bottom_right = (x + w // 2, y + h // 2)

                        centroid = (float(x + w / 2), float(y + h / 2))
                        track_history[track_id]['centroids'].append(centroid)
                        track_history[track_id]['boxes'].append((float(x), float(y), float(w), float(h)))

                        if len(track_history[track_id]['centroids']) > 30:
                            track_history[track_id]['centroids'].pop(0)
                            track_history[track_id]['boxes'].pop(0)

                        if track_id in document_types:
                            doc_type = document_types[track_id]
                            box_width = document_widths[doc_type]
                            box_height = standard_height
                            top_left_x = frame_center_x - box_width // 2
                            top_left_y = frame_center_y - box_height // 2

                            # Calculate corners of the overlayed boundary box
                            overlay_top_left = (top_left_x, top_left_y)
                            overlay_top_right = (top_left_x + box_width, top_left_y)
                            overlay_bottom_left = (top_left_x, top_left_y + box_height)
                            overlay_bottom_right = (top_left_x + box_width, top_left_y + box_height)

                            # Calculate distances for each corner
                            distance_top_left = math.sqrt(
                                math.pow(overlay_top_left[0] - top_left[0], 2) + math.pow(overlay_top_left[1] - top_left[1], 2))
                            distance_top_right = math.sqrt(
                                math.pow(overlay_top_right[0] - top_right[0], 2) + math.pow(overlay_top_right[1] - top_right[1], 2))
                            distance_bottom_left = math.sqrt(
                                math.pow(overlay_bottom_left[0] - bottom_left[0], 2) + math.pow(overlay_bottom_left[1] - bottom_left[1], 2))
                            distance_bottom_right = math.sqrt(
                                math.pow(overlay_bottom_right[0] - bottom_right[0], 2) + math.pow(overlay_bottom_right[1] - bottom_right[1], 2))

                            threshold = 100

                            # Calculate the containment and distance conditions
                            is_contained = (
                                top_left[0] >= top_left_x and top_right[0] <= top_left_x + box_width and 
                                top_left[1] >= top_left_y and bottom_left[1] <= top_left_y + box_height
                            )

                            are_corners_within_threshold = (distance_top_left < threshold and 
                                                            distance_top_right < threshold and 
                                                            distance_bottom_left < threshold and 
                                                            distance_bottom_right < threshold)
                            # Draw the overlayed boundary box based on conditions
                            if is_contained and are_corners_within_threshold:
                                box_color = (0, 255, 0)  # Green color for boundary box
                            else:
                                box_color = (255, 255, 255)  # White color for boundary box

                            cv2.rectangle(annotated_frame, (top_left_x, top_left_y), (top_left_x + box_width, top_left_y + box_height), box_color, 3)
                            cv2.circle(annotated_frame, (int(top_left_x), int(top_left_y)), 5, (255, 0, 0), -1)
                            cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                            cv2.circle(annotated_frame, (int(top_left[0]), int(top_left[1])), 5, (0, 255, 0), -1)

                        # Draw the tracking lines
                        # points = np.array(track_history[track_id]['centroids'], dtype=np.int32).reshape((-1, 1, 2))
                        # cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=1)

                        # Print class name along with other details
                        print(f"Track ID: {track_id}, Class: {class_name}, Box Coordinates: (x: {x}, y: {y}, width: {w}, height: {h})")
                        print(f"  Corners: Top-Left: {top_left}, Top-Right: {top_right}, Bottom-Left: {bottom_left}, Bottom-Right: {bottom_right}")
                        print(f"  Overlay Corners: Top-Left: {overlay_top_left}, Top-Right: {overlay_top_right}, Bottom-Left: {overlay_bottom_left}, Bottom-Right: {overlay_bottom_right}")
                        print(f"Distance Top-Left: {distance_top_left}, Distance Top-Right: {distance_top_right}, Distance Bottom-Left: {distance_bottom_left}, Distance Bottom-Right: {distance_bottom_right}")


                else:
                    continue  # Skip this frame if no track IDs are present

                # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
