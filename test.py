import os
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import sys

print("=" * 60)
print("DEBUG: Starting Hand Gesture Recognition")
print("=" * 60)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"DEBUG: Current directory: {current_dir}")

model_path = os.path.join(current_dir, "keras_model.h5")
labels_path = os.path.join(current_dir, "labels.txt")

print(f"DEBUG: Looking for model at: {model_path}")
print(f"DEBUG: Model exists: {os.path.exists(model_path)}")
print(f"DEBUG: Looking for labels at: {labels_path}")
print(f"DEBUG: Labels exist: {os.path.exists(labels_path)}")

# Initialize video capture
print("\nDEBUG: Initializing camera...")
cap_ob = cv2.VideoCapture(0)
if not cap_ob.isOpened():
    print("ERROR: Could not open camera!")
    sys.exit(1)
print("DEBUG: Camera opened successfully")

# Initialize hand detector
print("DEBUG: Initializing hand detector...")
try:
    detector = HandDetector(maxHands=1)
    print("DEBUG: Hand detector initialized")
except Exception as e:
    print(f"ERROR: Failed to initialize hand detector: {e}")
    sys.exit(1)

# Load the classifier
print("DEBUG: Loading classifier...")
try:
    classifier = Classifier(model_path, labels_path)
    print("DEBUG: Classifier loaded successfully")
except Exception as e:
    print(f"ERROR: Failed to load classifier: {e}")
    print(
        f"ERROR: Make sure keras_model.h5 and labels.txt are in: {current_dir}")
    sys.exit(1)

# Parameters
offset = 25
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
          "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
confidence_threshold = 0.7

print("\n" + "=" * 60)
print("DEBUG: Everything loaded! Opening camera window...")
print("Make sure your hand is visible in the camera")
print("Press 'q' to quit")
print("=" * 60 + "\n")

frame_count = 0
hand_detected_count = 0

try:
    while True:
        frame_count += 1
        success, img = cap_ob.read()

        if not success:
            print("ERROR: Failed to read frame from camera")
            break

        # Print every 30 frames (about once per second at 30fps)
        if frame_count % 30 == 0:
            print(
                f"DEBUG: Processing frame {frame_count}... Hands detected so far: {hand_detected_count}")

        try:
            hands, img = detector.findHands(img)

            if hands:
                hand_detected_count += 1
                hand = hands[0]
                x, y, w, h = hand['bbox']

                # Ensure boundaries are within image
                y1, y2 = max(0, y - offset), min(y + h + offset, img.shape[0])
                x1, x2 = max(0, x - offset), min(x + w + offset, img.shape[1])

                # Create white background image
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y1:y2, x1:x2]

                if imgCrop.size > 0:
                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize

                    # Get prediction
                    prediction, index = classifier.getPrediction(imgWhite)
                    confidence = prediction[index]
                    predicted_label = labels[index] if index < len(
                        labels) else "Unknown"

                    print(
                        f"PREDICTION: {predicted_label} | Confidence: {confidence:.4f}")

                    # Add text overlay on main image
                    if confidence > confidence_threshold:
                        cv2.putText(img, f"{predicted_label} ({confidence:.2f})",
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0), 2)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # Show cropped images
                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

        except Exception as e:
            print(f"ERROR during processing: {e}")
            import traceback
            traceback.print_exc()

        # Display main video
        cv2.imshow("Hand Gesture Recognition", img)

        # Press 'q' to quit
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("\nDEBUG: User pressed 'q', exiting...")
            break

except KeyboardInterrupt:
    print("\nDEBUG: Interrupted by user (Ctrl+C)")
except Exception as e:
    print(f"ERROR: Unexpected error: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("DEBUG: Cleaning up...")
    cap_ob.release()
    cv2.destroyAllWindows()
    print(
        f"DEBUG: Processed {frame_count} frames, detected hands in {hand_detected_count} frames")
    print("DEBUG: Program ended")
