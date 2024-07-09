import cv2
import sys
import numpy as np

# Define filter modes
PREVIEW = 0
BLUR = 1
FEATURES = 2
CANNY = 3

# Parameters for feature detection
feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=15, blockSize=9)

# Set the default source to the first camera
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]  # We assume it can be a camera index or a file path

# Initialize filter mode and loop control
image_filter = PREVIEW
alive = True

# Create a window for display
win_name = "Camera Filters"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Open the video source
source = cv2.VideoCapture(s)
if not source.isOpened():
    print(f"Error: Could not open video source {s}")
    sys.exit()

# Main loop to read frames and apply filters
while alive:
    has_frame, frame = source.read()
    if not has_frame:
        print("Error: Frame not read correctly")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Apply the selected filter
    if image_filter == PREVIEW:
        result = frame
    elif image_filter == CANNY:
        result = cv2.Canny(frame, 80, 150)
    elif image_filter == BLUR:
        result = cv2.blur(frame, (13, 13))
    elif image_filter == FEATURES:
        result = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
        if corners is not None:
            for x, y in np.float32(corners).reshape(-1, 2):
                cv2.circle(result, (int(x), int(y)), 10, (0, 255, 0), 1)

    # Show the frame with the applied filter
    cv2.imshow(win_name, result)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 27 is the escape key
        alive = False
    elif key == ord('c'):
        image_filter = CANNY
        print("Canny edge detection selected")
    elif key == ord('b'):
        image_filter = BLUR
        print("Blurring selected")
    elif key == ord('f'):
        image_filter = FEATURES
        print("Feature detection selected")
    elif key == ord('p'):
        image_filter = PREVIEW
        print("Preview mode selected")

# Release the video source and close the window
source.release()
cv2.destroyAllWindows()
