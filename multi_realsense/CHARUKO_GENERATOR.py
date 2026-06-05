import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

board = cv2.aruco.CharucoBoard((3,4), 0.04, 0.02, aruco_dict)

img = board.generateImage((1000, 1400))
cv2.imwrite("charuco.png", img)