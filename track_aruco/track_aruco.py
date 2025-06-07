import numpy as np
import cv2 as cv

# DEFINE CAMERA POSE (TRANSLATION + ORIENTATION)
# Assumes camera is facing down the table.

rvec_wc = np.array([np.pi, 0.0, 0.0])
R_wc, _ = cv.Rodrigues(rvec_wc) # rotation matrix world-coordinate-space camera
t_wc = np.array([0.0, 0.0, 1.0]) # translation vector world-coordinate-space camera


aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco. DICT_4X4_50)
parameters = cv.aruco.DetectorParameters()
# Create the ArUco detector
detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

camera_matrix = np.load('camera_matrix.npy')
dist_coeff = np.load('dist_coeffs.npy')


cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Detect the markers
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(corners, markerLength=0.05, cameraMatrix=camera_matrix, distCoeffs=dist_coeff)
        # assume only one
        cv.aruco.drawDetectedMarkers(frame, corners, ids)
        cv.drawFrameAxes(frame, camera_matrix, dist_coeff, rvecs[0], tvecs[0], 0.01)
        # format text
        rvec_text = f"rvec: {np.round(rvecs[0][0], 2)}"
        tvec_text = f"tvec: {np.round(tvecs[0][0], 2)}"

        # Choose a position near the marker
        corner = corners[0][0][0]  # top-left corner of the marker
        x, y = int(corner[0]), int(corner[1])

        # Display rvec and tvec in corner of box
        cv.putText(frame, rvec_text, (x, y - 20), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        cv.putText(frame, tvec_text, (x, y + 20), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

        # compute aruco world coordinate
        R_ca, _ = cv.Rodrigues(rvecs[0])
        R_wa = R_wc @ R_ca
        t_wa = R_wc @ tvecs[0].reshape(3,1) + t_wc.reshape(3,1)
        rvec_wa, _ = cv.Rodrigues(R_wa)
        print(f"World Coord {t_wa} Orientation: {rvec_wa}")
        # cv.imshow('Detected Markers', frame)

 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()