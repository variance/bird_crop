import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse # Import moved to top
import os # Needed for checking file existence

class BirdCropper:
    # Renamed process_one to process_single
    def __init__(self, model_path="yolov8n.pt", process_single=True):
        self.model = YOLO(model_path)
        self.class_id = 14  # COCO class ID for 'bird'
        self.process_single = process_single # Store the option

    def detect_and_crop(self, img_path, output_dir, conf=0.5):
        img_path_obj = Path(img_path) # Work with Path object for consistency
        output_dir_obj = Path(output_dir) # Work with Path object

        img = cv2.imread(str(img_path_obj))
        if img is None:
            print(f"Error: Could not read image {img_path_obj}")
            return []

        results = self.model.predict(source=img, conf=conf, verbose=False)

        crops = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()

            detections = []
            for box, cls, conf_score in zip(boxes, classes, confidences):
                 if int(cls) == self.class_id:
                    detections.append({'box': box, 'conf': conf_score})

            # Optional sorting if you want the *highest confidence* single bird
            # if self.process_single and detections:
            #    detections = sorted(detections, key=lambda x: x['conf'], reverse=True)

            for i, det in enumerate(detections):
                box = det['box']
                x1, y1, x2, y2 = map(int, box)
                # Ensure coordinates are valid
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

                # Ensure crop area is valid
                if x1 >= x2 or y1 >= y2:
                    print(f"Warning: Invalid crop dimensions for {img_path_obj.name}, skipping.")
                    continue

                crop = img[y1:y2, x1:x2]

                # --- Filename Logic ---
                base_stem = img_path_obj.stem
                if self.process_single:
                    # Default filename for single mode
                    crop_filename = f"{base_stem}.jpg"
                    crop_path = output_dir_obj / crop_filename
                    # Check if it exists ONLY if we are in single mode
                    if crop_path.exists():
                        # If the simple name exists, fall back to indexed name
                        print(f"Warning: File {crop_path} already exists. Using indexed name.")
                        crop_filename = f"{base_stem}_crop_{i}.jpg"
                        crop_path = output_dir_obj / crop_filename
                else:
                    # Always use index if processing multiple
                    crop_filename = f"{base_stem}_crop_{i}.jpg"
                    crop_path = output_dir_obj / crop_filename
                # --- End Filename Logic ---

                try:
                    cv2.imwrite(str(crop_path), crop)
                    crops.append(crop_path)
                    print(f"Saved crop to: {crop_path}") # Added print for clarity
                except Exception as e:
                    print(f"Error writing crop file {crop_path}: {e}")

                # If process_single is True, break after processing the first bird
                if self.process_single:
                    break # Exit the inner loop after the first bird crop

            # If process_single is True, we can also break the outer loop
            if self.process_single and crops:
                 break # Exit the outer loop ('for r in results:')

        return crops

# Keep the rest of the file (__main__ block) as it was in the previous version

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and crop birds from an image.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--output", type=str, default=None, help="Path to the output directory for cropped images. Defaults to a 'cropped' subdirectory next to the input image.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to the YOLOv8 model file.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detection.")
    # Renamed the boolean flag. '--no-single' sets args.single to False. If flag is absent, args.single is True.
    # Using '--multiple' as the flag to turn OFF single mode is clearer
    parser.add_argument("--multiple", dest='single', action='store_false', help="Process and save all detected birds (use indexed filenames). Default is to process only one bird (use simpler filename if possible).")
    # The default value for 'single' is implicitly True because of action='store_false'

    args = parser.parse_args()

    input_path_obj = Path(args.input) # Use Path object early

    if args.output is None:
        # Handle potential errors if input is not a file
        if not input_path_obj.is_file():
             print(f"Error: Input path {args.input} is not a valid file.")
             exit(1)
        args.output = str(input_path_obj.parent / "cropped")

    output_path_obj = Path(args.output) # Use Path object

    print(f"Input image: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Model path: {args.model}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Process only a single bird: {args.single}") # Print the value being used (using the new name)

    # Pass the model path and the boolean flag to the constructor using the new name
    cropper = BirdCropper(model_path=args.model, process_single=args.single)

    output_path_obj.mkdir(parents=True, exist_ok=True) # Use parents=True to create intermediate dirs

    print(f"Detecting and cropping birds...")
    # Pass Path objects for consistency, though the method handles strings too
    cropped_files = cropper.detect_and_crop(str(input_path_obj), str(output_path_obj), conf=args.conf)

    if cropped_files:
        print(f"Successfully saved {len(cropped_files)} cropped image(s) to {args.output}:")
        for file in cropped_files:
            print(f"- {file.name}") # file is already a Path object
    else:
        print("No birds detected or cropped.")
