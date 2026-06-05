import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs


# =========================
# CHARUCO SETTINGS
# =========================
CHARUCO_SIZE = (5, 7)
SQUARE_SIZE = 0.0575   # 57.5 мм
MARKER_SIZE = 0.0287   # 28.7 мм

RGB_W, RGB_H = 640, 480
DEPTH_W, DEPTH_H = 640, 480
FPS = 30
MAX_CAMERAS = 2


class AppState:
    def __init__(self):
        self.WIN_NAME = "Two RealSense Live Fusion"
        self.pitch = math.radians(-10)
        self.yaw = math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.calibrated = False

    def reset(self):
        self.pitch = 0
        self.yaw = 0
        self.distance = 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


state = AppState()


# =========================
# CHARUCO
# =========================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

board = cv2.aruco.CharucoBoard(
    CHARUCO_SIZE,
    SQUARE_SIZE,
    MARKER_SIZE,
    aruco_dict
)

detector = cv2.aruco.ArucoDetector(aruco_dict)


def rvec_tvec_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = tvec.flatten().astype(np.float32)

    return T


def average_transform(transforms):
    t = np.mean([T[:3, 3] for T in transforms], axis=0)

    R_sum = np.zeros((3, 3), dtype=np.float32)
    for T in transforms:
        R_sum += T[:3, :3]

    U, _, Vt = np.linalg.svd(R_sum)
    R = U @ Vt

    T_avg = np.eye(4, dtype=np.float32)
    T_avg[:3, :3] = R
    T_avg[:3, 3] = t

    return T_avg


def get_intrinsics_from_frame(frame):
    intr = frame.profile.as_video_stream_profile().intrinsics

    return np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ], dtype=np.float32)


def detect_charuco_pose(image, K):
    dist = np.zeros((5, 1))

    corners, ids, _ = detector.detectMarkers(image)

    if ids is None or len(ids) < 4:
        return None, image

    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
        corners,
        ids,
        image,
        board
    )

    if ch_ids is None or len(ch_ids) < 6:
        return None, image

    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        ch_corners,
        ch_ids,
        board,
        K,
        dist,
        None,
        None
    )

    if not ok:
        return None, image

    cv2.drawFrameAxes(image, K, dist, rvec, tvec, 0.15)

    return (rvec, tvec), image


# =========================
# CAMERA INIT
# =========================
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) < 2:
    raise RuntimeError("Нужно минимум 2 камеры RealSense")

serials = [
    devices[i].get_info(rs.camera_info.serial_number)
    for i in range(min(MAX_CAMERAS, len(devices)))
]

print("[INFO] Cameras:", serials)

pipelines = []
pointclouds = []
decimations = []

for serial in serials:
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, RGB_W, RGB_H, rs.format.bgr8, FPS)

    profile = pipeline.start(config)

    sensor = profile.get_device().first_depth_sensor()

    try:
        sensor.set_option(rs.option.visual_preset, 4)
        sensor.set_option(rs.option.laser_power, 100)
    except Exception:
        pass

    pipelines.append(pipeline)

    pc = rs.pointcloud()
    pointclouds.append(pc)

    dec = rs.decimation_filter()
    dec.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
    decimations.append(dec)

time.sleep(2)


# =========================
# RENDER
# =========================
out = np.empty((720, 1280, 3), dtype=np.uint8)


def mouse_cb(event, x, y, flags, param):
    global out

    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:
        h, w = out.shape[:2]

        dx = x - state.prev_mouse[0]
        dy = y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx ** 2 + dy ** 2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)


cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)


def view(v):
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def project(v):
    h, w = out.shape[:2]
    view_aspect = float(h) / w

    with np.errstate(divide="ignore", invalid="ignore"):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * (
            w * view_aspect,
            h
        ) + (
            w / 2.0,
            h / 2.0
        )

    znear = 0.03
    proj[v[:, 2] < znear] = np.nan

    return proj


def pointcloud_render(out_img, verts, colors):
    if len(verts) == 0:
        return

    v = view(verts)
    s = v[:, 2].argsort()[::-1]

    proj = project(v[s])

    h, w = out_img.shape[:2]

    j, i = proj.astype(np.int32).T

    m = (
        (i >= 0) &
        (i < h) &
        (j >= 0) &
        (j < w)
    )

    out_img[i[m], j[m]] = colors[s][m]


def transform_points(points, T):
    if len(points) == 0:
        return points

    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homo = np.hstack([points, ones])

    transformed = (T @ homo.T).T

    return transformed[:, :3]


def make_preview_grid(images):
    resized = [cv2.resize(img, (640, 480)) for img in images]
    return np.hstack(resized)


# =========================
# CALIBRATION
# =========================
def calibrate_two_cameras(num_frames=50):
    print("[INFO] Calibration started...")

    transforms = []

    for cam_id, pipeline in enumerate(pipelines):
        Ts = []

        for _ in range(num_frames):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            K = get_intrinsics_from_frame(color_frame)

            pose, _ = detect_charuco_pose(img.copy(), K)

            if pose is None:
                continue

            rvec, tvec = pose

            T_cam_board = rvec_tvec_to_matrix(rvec, tvec)

            # camera -> board/world
            T_board_cam = np.linalg.inv(T_cam_board)

            Ts.append(T_board_cam)

        if len(Ts) < 10:
            raise RuntimeError(f"Camera {cam_id}: мало кадров для калибровки")

        T_avg = average_transform(Ts)
        transforms.append(T_avg)

        print(f"[INFO] Camera {cam_id}: calibration frames = {len(Ts)}")
        print(f"[INFO] Camera {cam_id} position:", T_avg[:3, 3])

    print("[INFO] Calibration done")

    return transforms


camera_to_world = [np.eye(4, dtype=np.float32) for _ in pipelines]


# =========================
# MAIN LOOP
# =========================
try:
    while True:
        now = time.time()

        all_verts = []
        all_colors = []

        preview_images = []
        ready = True

        if not state.paused:
            for cam_id, pipeline in enumerate(pipelines):
                frames = pipeline.wait_for_frames()

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if not depth_frame or not color_frame:
                    ready = False
                    continue

                depth_frame = decimations[cam_id].process(depth_frame)

                color_image = np.asanyarray(color_frame.get_data())

                if not state.calibrated:
                    K = get_intrinsics_from_frame(color_frame)
                    pose, img_draw = detect_charuco_pose(color_image.copy(), K)

                    if pose is not None:
                        cv2.putText(
                            img_draw,
                            f"Cam {cam_id}: OK",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                    else:
                        cv2.putText(
                            img_draw,
                            f"Cam {cam_id}: NO",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2
                        )
                        ready = False

                    preview_images.append(img_draw)

                else:
                    pointclouds[cam_id].map_to(color_frame)
                    points = pointclouds[cam_id].calculate(depth_frame)

                    verts = np.asanyarray(points.get_vertices()).view(
                        np.float32
                    ).reshape(-1, 3)

                    tex = np.asanyarray(points.get_texture_coordinates()).view(
                        np.float32
                    ).reshape(-1, 2)

                    h, w = color_image.shape[:2]

                    px = np.clip(
                        (tex[:, 0] * w).astype(np.int32),
                        0,
                        w - 1
                    )

                    py = np.clip(
                        (tex[:, 1] * h).astype(np.int32),
                        0,
                        h - 1
                    )

                    colors = color_image[py, px]

                    valid = verts[:, 2] > 0

                    verts = verts[valid]
                    colors = colors[valid]

                    verts = transform_points(
                        verts,
                        camera_to_world[cam_id]
                    )

                    all_verts.append(verts)
                    all_colors.append(colors)

        out.fill(0)

        if not state.calibrated:
            if preview_images:
                preview = make_preview_grid(preview_images)

                if ready:
                    cv2.putText(
                        preview,
                        "SPACE = CALIBRATE",
                        (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2
                    )
                else:
                    cv2.putText(
                        preview,
                        "SHOW CHARUCO TO BOTH CAMERAS",
                        (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

                cv2.imshow(state.WIN_NAME, preview)

        else:
            if all_verts:
                verts = np.vstack(all_verts)
                colors = np.vstack(all_colors)

                # фильтр по области после объединения
                valid = (
                    (verts[:, 2] > -2.0) &
                    (verts[:, 2] < 2.0)
                )

                verts = verts[valid]
                colors = colors[valid]

                pointcloud_render(out, verts, colors)

            dt = time.time() - now

            cv2.setWindowTitle(
                state.WIN_NAME,
                f"Two RealSense Fusion {1.0 / max(dt, 1e-6):.1f} FPS"
            )

            cv2.imshow(state.WIN_NAME, out)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            state.reset()

        if key == ord("p"):
            state.paused ^= True

        if key == ord("d"):
            state.decimate = (state.decimate + 1) % 3

            for dec in decimations:
                dec.set_option(
                    rs.option.filter_magnitude,
                    2 ** state.decimate
                )

        if key == ord("s"):
            cv2.imwrite("utils/out_two_cameras.png", out)

        if key == ord(" "):
            if not state.calibrated and ready:
                camera_to_world = calibrate_two_cameras()
                state.calibrated = True
                print("[INFO] Now showing fused point cloud")

        if key in (27, ord("q")):
            break

finally:
    for pipeline in pipelines:
        pipeline.stop()

    cv2.destroyAllWindows()