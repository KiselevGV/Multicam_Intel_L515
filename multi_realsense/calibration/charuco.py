import cv2
import numpy as np


class CharucoCalibrator:
    def __init__(self, square_size=0.0575, marker_size=0.0287):
        self.SQUARE = square_size
        self.MARKER = marker_size
        self.SIZE = (5, 7)

        self.dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        self.board = cv2.aruco.CharucoBoard(
            self.SIZE,
            self.SQUARE,
            self.MARKER,
            self.dict
        )

        self.detector = cv2.aruco.ArucoDetector(self.dict)

    def detect_pose(self, image, K):
        dist = np.zeros((5, 1))

        corners, ids, _ = self.detector.detectMarkers(image)

        if ids is None or len(ids) < 4:
            return None

        ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, image, self.board
        )

        if ch_ids is None or len(ch_ids) < 6:
            return None

        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            ch_corners, ch_ids, self.board, K, dist, None, None
        )

        if not ok:
            return None

        # 🔥 ФИЛЬТР ПО РАССТОЯНИЮ
        distance = np.linalg.norm(tvec)
        if distance < 0.3 or distance > 2.5:
            return None

        return rvec, tvec

    def to_matrix(self, rvec, tvec):
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T


# 🔥 ПРАВИЛЬНОЕ УСРЕДНЕНИЕ
def average_transform(Ts):
    t = np.mean([T[:3, 3] for T in Ts], axis=0)

    R_sum = np.zeros((3, 3))
    for T in Ts:
        R_sum += T[:3, :3]

    U, _, Vt = np.linalg.svd(R_sum)
    R = U @ Vt

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T