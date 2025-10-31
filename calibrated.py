import os
import time
import json
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
yolo_model = YOLO("yolov8n.pt")

# ---------- User config ----------
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE_M = 0.024
CALIB_DIR = "calibration"
CALIB_FILE = os.path.join(CALIB_DIR, "camera_calib.json")
CAPTURED_DIR = os.path.join(CALIB_DIR, "captures")
# ---------------------------------

os.makedirs(CALIB_DIR, exist_ok=True)
os.makedirs(CAPTURED_DIR, exist_ok=True)

def start_realsense():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    align_to = rs.stream.color
    align = rs.align(align_to)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[INFO] Depth scale: {depth_scale} meters/unit")
    return pipeline, align, depth_scale

def capture_chessboard_images(pipeline, align, max_images=25):
    print("[INFO] Entering capture mode. Press 'c' to capture image, 'q' to quit and calibrate.")
    captured = 0
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        if not color_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        display = color.copy()
        cv2.putText(display, f"Captured: {captured}/{max_images} | c: capture, q: quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Capture (Color Aligned)", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            fname = os.path.join(CAPTURED_DIR, f"capture_{int(time.time())}.png")
            cv2.imwrite(fname, color)
            print(f"[INFO] Saved {fname}")
            captured += 1
            if captured >= max_images:
                print("[INFO] Reached max captures.")
                break
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()
    files = sorted([os.path.join(CAPTURED_DIR, f) for f in os.listdir(CAPTURED_DIR) if f.endswith(".png")])
    return files

def calibrate_from_images(images, chessboard_size=CHESSBOARD_SIZE, square_size=SQUARE_SIZE_M):
    objp = np.zeros((chessboard_size[1]*chessboard_size[0], 3), np.float32)
    objp[:, :2] = np.indices((chessboard_size[0], chessboard_size[1])).T.reshape(-1,2)
    objp *= square_size

    objpoints, imgpoints = [], []
    img_shape = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, chessboard_size,
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            objpoints.append(objp)
            vis = img.copy()
            cv2.drawChessboardCorners(vis, chessboard_size, corners2, found)
            cv2.imshow("Corners", vis)
            cv2.waitKey(200)
        else:
            print(f"[WARN] Chessboard not found in {fname}")
    cv2.destroyAllWindows()

    if not objpoints or not imgpoints:
        raise RuntimeError("No valid chessboard detections. Capture more images with the chessboard visible.")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None)

    total_error = 0
    for i in range(len(objpoints)):
        imgpts2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpts2, cv2.NORM_L2)/len(imgpts2)
        total_error += error
    mean_error = total_error / len(objpoints)
    print(f"[INFO] Calibration RMS: {ret}, mean reprojection error: {mean_error}")

    return {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "reprojection_error": float(mean_error),
        "image_size": img_shape
    }

def save_calibration(data, path=CALIB_FILE):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved calibration to {path}")

def load_calibration(path=CALIB_FILE):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    data["camera_matrix"] = np.array(data["camera_matrix"], dtype=np.float64)
    data["dist_coeffs"] = np.array(data["dist_coeffs"], dtype=np.float64)
    return data

# ---------- Helper ----------
def pixel_to_point_calibrated(depth_frame, calib, pixel):
    u, v = int(round(pixel[0])), int(round(pixel[1]))
    depth = depth_frame.get_distance(u, v)
    if depth <= 0:
        return None
    fx, fy = calib["camera_matrix"][0,0], calib["camera_matrix"][1,1]
    cx, cy = calib["camera_matrix"][0,2], calib["camera_matrix"][1,2]
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return (x, y, z)

# ---------- Main ----------
def main():
    pipeline, align, depth_scale = start_realsense()
    try:
        calib = load_calibration()
        if calib is None:
            print("[INFO] No calibration found. Starting capture.")
            captures = capture_chessboard_images(pipeline, align, max_images=20)
            if len(captures) == 0:
                print("[ERROR] No captures saved. Exiting.")
                return
            calib_data = calibrate_from_images(captures)
            save_calibration(calib_data)
            calib = load_calibration()
        else:
            print(f"[INFO] Loaded calibration from {CALIB_FILE}")

        print("[INFO] Starting detection loop. Press 'q' to quit.")
        cv2.namedWindow("Color", cv2.WINDOW_AUTOSIZE)

        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())

            # ✅ Undistort using calibration
            color = cv2.undistort(color, calib["camera_matrix"], calib["dist_coeffs"])

            # YOLO detection
            results = yolo_model(color)[0]
            boxes = results.boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                point_3d = pixel_to_point_calibrated(depth_frame, calib, (cx, cy))

                if point_3d:
                    X, Y, Z = point_3d
                    cv2.rectangle(color, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.circle(color, (cx, cy), 5, (0,0,255), -1)
                    cv2.putText(color, f"X={X*100:.1f}cm Y={Y*100:.1f}cm Z={Z*100:.1f}cm",
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

            cv2.imshow("Color", color)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Stopped camera.")

if __name__ == "__main__":
    main()
