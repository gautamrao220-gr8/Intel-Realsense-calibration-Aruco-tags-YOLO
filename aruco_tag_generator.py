import cv2
import numpy as np
import os

# Choose ArUco dictionary
# Common options: DICT_4X4_50, DICT_5X5_100, DICT_6X6_250, DICT_7X7_1000
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Folder to save generated tags
save_dir = "aruco_tags"
os.makedirs(save_dir, exist_ok=True)

# IDs of markers you want to generate (0–249 for DICT_6X6_250)
marker_ids = [0, 1, 2, 3, 4]   # you can add more

# Marker image size in pixels
marker_size = 700

for marker_id in marker_ids:
    img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    filename = os.path.join(save_dir, f"aruco_{marker_id}.png")
    cv2.imwrite(filename, img)
    print(f"Saved marker ID {marker_id} to {filename}")

print("All markers generated in folder:", os.path.abspath(save_dir))
