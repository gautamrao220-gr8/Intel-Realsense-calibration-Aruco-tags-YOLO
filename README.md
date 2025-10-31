# Intel-Realsense-camera-calibration-and-YOLO
- Collect chessboard images for calibration (press 'c' to capture, 'q' to finish capture phase).
- Run OpenCV calibration, save intrinsics to disk.
- Start detection loop: detect objects with YOLO and compute their 3D position w.r.t the camera
  using the depth frame and calibration intrinsics.
- Use the provided chessboard image for calibration or use custom size and update changes in code.
- Object detection using YOLO.
- Making ROI in the camera frame using Aruco tags to avoid object beyond roi.Aruco tags code provided, print on a non glossy paper.
