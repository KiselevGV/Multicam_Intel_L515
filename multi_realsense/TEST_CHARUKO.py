import pyrealsense2 as rs
import numpy as np
import cv2

# -----------------------------
# Получаем все камеры
# -----------------------------
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) < 2:
    raise Exception("Нужно минимум 2 камеры")

serials = [d.get_info(rs.camera_info.serial_number) for d in devices]

print("Найдено камер:", serials)

# -----------------------------
# Запускаем pipelines
# -----------------------------
pipelines = []

for serial in serials:
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    pipelines.append(pipeline)

# -----------------------------
# ArUco
# -----------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict)

try:
    while True:
        images = []

        for i, pipe in enumerate(pipelines):
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()

            if not color:
                continue

            img = np.asanyarray(color.get_data())

            corners, ids, _ = detector.detectMarkers(img)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(img, corners, ids)

                cv2.putText(
                    img,
                    f"Cam {i} | Markers: {len(ids)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    img,
                    f"Cam {i} | No markers",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            images.append(img)

        # -----------------------------
        # Объединяем в одно окно
        # -----------------------------
        if len(images) == 2:
            combined = np.hstack(images)
        else:
            combined = images[0]

        cv2.imshow("Two Cameras", combined)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    for p in pipelines:
        p.stop()

    cv2.destroyAllWindows()