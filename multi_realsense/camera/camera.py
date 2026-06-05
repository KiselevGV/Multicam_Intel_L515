import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time


class Camera:
    def __init__(self, serial):
        self.serial = serial
        self.pipeline = rs.pipeline()

    def start_rgb(self, width=640, height=480):
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        self.pipeline.start(config)

    def start_depth(self, width=640, height=480):
        config = rs.config()
        config.enable_device(self.serial)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

        self.pipeline.start(config)

    def stop(self):
        self.pipeline.stop()

    def get_color(self):
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data()), color

    def get_intrinsics(self, frame):
        intr = frame.profile.as_video_stream_profile().intrinsics
        return np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ])

    # 🔥 ФИЛЬТР DEPTH
    def filter_depth(self, depth_frame):
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()

        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)

        return depth_frame

    def depth_to_cloud(self, depth_frame, color_frame, min_dist=0.2, max_dist=1.0):
        import open3d as o3d
        import numpy as np
        import pyrealsense2 as rs

        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)

        vtx = np.asanyarray(points.get_vertices())
        tex = np.asanyarray(points.get_texture_coordinates())

        color_image = np.asanyarray(color_frame.get_data())
        h, w, _ = color_image.shape

        xyz, rgb = [], []

        for i in range(len(vtx)):
            x, y, z = vtx[i]

            # 🔥 ГЛАВНЫЙ ФИЛЬТР
            if z == 0 or z < min_dist or z > max_dist:
                continue

            u, v = tex[i]
            px = int(u * w)
            py = int(v * h)

            if 0 <= px < w and 0 <= py < h:
                xyz.append([x, y, z])
                rgb.append(color_image[py, px] / 255.0)

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(xyz)
        cloud.colors = o3d.utility.Vector3dVector(rgb)

        return cloud