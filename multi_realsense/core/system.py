import numpy as np
import time
import cv2
import pyrealsense2 as rs
import open3d as o3d

from camera.camera import Camera
from calibration.charuco import CharucoCalibrator, average_transform
from fusion.fusion import Fusion


class MultiCameraSystem:
    def __init__(self):
        ctx = rs.context()
        devices = ctx.query_devices()

        self.num_cams = len(devices)
        print(f"[INFO] Cameras found: {self.num_cams}")

        if self.num_cams == 0:
            raise Exception("No cameras found")

        self.cameras = []
        for d in devices:
            serial = d.get_info(rs.camera_info.serial_number)
            self.cameras.append(Camera(serial))

        self.calibrator = CharucoCalibrator()
        self.fusion = Fusion()

        self.transforms = []

    def make_grid(self, images):
        cols = int(np.ceil(np.sqrt(len(images))))
        rows = int(np.ceil(len(images) / cols))

        h, w = images[0].shape[:2]
        canvas = np.zeros((rows*h, cols*w, 3), dtype=np.uint8)

        for i, img in enumerate(images):
            r = i // cols
            c = i % cols
            canvas[r*h:(r+1)*h, c*w:(c+1)*w] = img

        return canvas

    def run(self):
        print("=== SYSTEM READY ===")

        # 🔥 РЕЖИМ 1 КАМЕРЫ
        if self.num_cams == 1:
            print("MODE: SINGLE CAMERA")
            print("SPACE = scan | ESC = exit")

            cam = self.cameras[0]
            cam.start_rgb()

            while True:
                img, _ = cam.get_color()
                cv2.putText(img, "PRESS SPACE TO SCAN", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

                cv2.imshow("System", img)

                key = cv2.waitKey(1)

                if key == 27:
                    break

                if key == 32:
                    cam.stop()

                    cloud = self.capture_single()

                    o3d.visualization.draw_geometries([cloud])
                    o3d.io.write_point_cloud("result_single.ply", cloud)

                    print("[INFO] DONE → result_single.ply")
                    break

            cam.stop()
            cv2.destroyAllWindows()
            return

        # 🔵 МУЛЬТИКАМЕРНЫЙ РЕЖИМ
        print("STEP 1: calibration (SPACE)")
        print("STEP 2: scan (SPACE)")

        for cam in self.cameras:
            cam.start_rgb()

        calibrated = False

        while True:
            images = []
            ready = True

            for i, cam in enumerate(self.cameras):
                img, frame = cam.get_color()
                K = cam.get_intrinsics(frame)

                pose = self.calibrator.detect_pose(img, K)

                if not calibrated:
                    if pose is not None:
                        cv2.putText(img, f"Cam {i}: OK", (10,30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    else:
                        cv2.putText(img, f"Cam {i}: NO", (10,30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        ready = False
                else:
                    cv2.putText(img, "READY TO SCAN", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

                images.append(img)

            cv2.imshow("System", self.make_grid(images))

            key = cv2.waitKey(1)

            if key == 27:
                break

            if key == 32:

                # 🔵 КАЛИБРОВКА
                if not calibrated and ready:
                    print("[INFO] CALIBRATING...")

                    self.transforms = []

                    for i, cam in enumerate(self.cameras):
                        Ts = []

                        for _ in range(40):
                            img, frame = cam.get_color()
                            K = cam.get_intrinsics(frame)

                            pose = self.calibrator.detect_pose(img, K)
                            if pose is None:
                                continue

                            T = self.calibrator.to_matrix(*pose)
                            Ts.append(np.linalg.inv(T))

                        if len(Ts) < 10:
                            raise Exception(f"Cam {i}: not enough data")

                        self.transforms.append(average_transform(Ts))

                    np.save("extrinsics.npy", self.transforms)

                    calibrated = True
                    print("[INFO] CALIBRATION DONE")
                    print("REMOVE BOARD → PLACE OBJECT → SPACE")

                # 🔴 СКАН
                elif calibrated:
                    for cam in self.cameras:
                        cam.stop()

                    cloud = self.capture_multi()

                    o3d.visualization.draw_geometries([cloud])
                    o3d.io.write_point_cloud("result_multi.ply", cloud)

                    print("[INFO] DONE → result_multi.ply")
                    break

        for cam in self.cameras:
            cam.stop()

        cv2.destroyAllWindows()

    # =========================
    # 🔴 SINGLE CAMERA
    # =========================
    def capture_single(self):
        cam = self.cameras[0]
        cam.start_depth()

        frames_list = []

        for _ in range(30):  # 🔥 больше кадров
            frames = cam.pipeline.wait_for_frames()
            depth = cam.filter_depth(frames.get_depth_frame())
            color = frames.get_color_frame()

            frames_list.append(
                cam.depth_to_cloud(depth, color)
            )

        merged = frames_list[0]
        for c in frames_list[1:]:
            merged += c

        merged = merged.voxel_down_sample(0.002)

        cam.stop()

        return merged

    # =========================
    # 🔵 MULTI CAMERA
    # =========================
    def capture_multi(self):
        clouds = []

        for i, cam in enumerate(self.cameras):
            print(f"[INFO] Camera {i}")

            cam.start_depth()

            frames_list = []

            for _ in range(20):
                frames = cam.pipeline.wait_for_frames()
                depth = cam.filter_depth(frames.get_depth_frame())
                color = frames.get_color_frame()

                frames_list.append(
                    cam.depth_to_cloud(depth, color)
                )

            merged = frames_list[0]
            for c in frames_list[1:]:
                merged += c

            merged = merged.voxel_down_sample(0.002)

            if i < len(self.transforms):
                merged.transform(self.transforms[i])

            clouds.append(merged)

            cam.stop()
            time.sleep(0.3)

        return self.fusion.merge(clouds)