import numpy as np
from numpy import ndarray
import ot
import pathlib
from pathlib import Path

class DDEvaluater:
    """DDEvaluater class.
    
    This class is used to evaluate a distance between two financial point clouds.
    """
    def __init__(
        self,
    ) -> None:
        """initialization."""

    def calc_ot_distance(
        self,
        point_cloud1: ndarray,
        point_cloud2: ndarray
    ) -> float:
        """Calculate the optimal transport distance between two point clouds.

        Args:
            point_cloud1 (ndarray): The first point cloud.
            point_cloud2 (ndarray): The second point cloud.

        Returns:
            ot_distance (float): The optimal transport distance between the two financial point clouds.
        """
        num_points1: int = point_cloud1.shape[0]
        dim_points1: int = point_cloud1.shape[1]
        num_points2: int = point_cloud2.shape[0]
        dim_points2: int = point_cloud2.shape[1]
        if not num_points1 == num_points2:
            raise ValueError("The number of points in the two point clouds must be equal.")
        if not dim_points1 == dim_points2:
            raise ValueError("The dimension of the points in the two point clouds must be equal.")
        point_cloud1 = point_cloud1.astype(np.float64)
        point_cloud2 = point_cloud2.astype(np.float64)
        cost_matrix: ndarray = ot.dist(point_cloud1, point_cloud2)
        prob_mass1, prob_mass2 = (
            np.ones((num_points1,)) / num_points1, np.ones((num_points2,)) / num_points2
        )
        transport_plan = ot.emd(prob_mass1, prob_mass2, cost_matrix)
        ot_distance: float = np.sum(transport_plan * cost_matrix)
        return ot_distance

        