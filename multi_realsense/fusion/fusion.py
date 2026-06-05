import open3d as o3d
import numpy as np


class Fusion:
    def merge(self, clouds):
        result = clouds[0]

        result.estimate_normals()

        for c in clouds[1:]:
            c.estimate_normals()

            reg = o3d.pipelines.registration.registration_icp(
                c,
                result,
                0.02,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )

            c.transform(reg.transformation)
            result += c

        # 🔥 удаление шума
        result, _ = result.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )

        return result.voxel_down_sample(0.003)