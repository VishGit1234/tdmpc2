import cv2
import numpy as np
import glob

# === Settings ===
chessboard_size = (10, 7)  # inner corners per chessboard row and column
square_size = 0.025       # in meters or any unit
marker_length=0.018
calibration_images_path = '/Users/markchiu/Pictures/chessboard/*.jpg'  # path to your chessboard images

# === Prepare object points ===
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# === Arrays to store points ===
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# === Load images ===
images = glob.glob(calibration_images_path)
print(images)
image_shape = (0,0)
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray)
    image_shape = gray.shape

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                          criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners_subpix)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners_subpix, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# === Camera Calibration ===
print(image_shape)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

# === Print Results ===
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# == Save results ==
np.save('camera_matrix.npy',mtx)
np.save('dist_coeffs.npy',dist)
np.save('rvecs.npy',rvecs)
np.save('tvecs.npy',tvecs)
