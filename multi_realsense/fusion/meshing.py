import open3d as o3d
import numpy as np


class Mesher:
    def create_mesh(self, cloud):
        print("[INFO] CLEANING CLOUD...")

        # =========================================
        # 1. ЖЁСТКАЯ ОЧИСТКА
        # =========================================
        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=50,
            std_ratio=1.0
        )

        cloud = cloud.voxel_down_sample(0.002)

        # =========================================
        # 2. CROP (ОЧЕНЬ ВАЖНО)
        # =========================================
        bbox = cloud.get_axis_aligned_bounding_box()
        bbox = bbox.scale(0.9, bbox.get_center())

        cloud = cloud.crop(bbox)

        print("[INFO] NORMALS...")

        # =========================================
        # 3. НОРМАЛИ (КРИТИЧНО)
        # =========================================
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.015,
                max_nn=50
            )
        )

        cloud.orient_normals_consistent_tangent_plane(100)

        # =========================================
        # 4. POISSON (БОЛЕЕ СТАБИЛЬНЫЙ)
        # =========================================
        print("[INFO] POISSON...")

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud,
            depth=10
        )

        densities = np.asarray(densities)

        # =========================================
        # 5. ЖЁСТКАЯ ФИЛЬТРАЦИЯ
        # =========================================
        threshold = np.quantile(densities, 0.1)

        vertices_to_remove = densities < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        # =========================================
        # 6. ВТОРОЙ CROP
        # =========================================
        bbox = cloud.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)

        # =========================================
        # 7. СГЛАЖИВАНИЕ (ОСТОРОЖНО)
        # =========================================
        mesh = mesh.filter_smooth_laplacian(2)

        mesh.compute_vertex_normals()

        print("[INFO] MESH READY")

        return mesh