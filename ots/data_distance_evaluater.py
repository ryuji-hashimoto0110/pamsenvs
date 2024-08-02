from abc import abstractmethod
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from matplotlib.pyplot import Figure
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import ndarray
from numpy.random import Generator
import ot
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from typing import Optional

class DDEvaluater:
    """DDEvaluater class.
    
    This class is used to evaluate a distance between two financial point clouds.
    """
    def __init__(
        self,
        seed: int = 42,
        ticker_path_dic: dict[str | int, Path] = {},
    ) -> None:
        """initialization."""
        self.prng: Generator = np.random.default_rng(seed)
        self.ticker_path_dic: dict[str | int, Path] = ticker_path_dic
        self.ticker_point_clouds_dic: dict[str | int, ndarray] = {}

    def calc_ot_distance(
        self,
        point_cloud1: ndarray,
        point_cloud2: ndarray,
        is_per_bit: bool = True
    ) -> float:
        """Calculate the optimal transport distance between two point clouds.

        Args:
            point_cloud1 (ndarray): The first point cloud.
            point_cloud2 (ndarray): The second point cloud.
            is_per_bit (bool): Whether to divide the distance by the dimension of the points.

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
        if is_per_bit:
            ot_distance /= dim_points1
        return ot_distance
    
    @abstractmethod
    def get_point_cloud_from_path(self, *args, **kwargs) -> ndarray:
        pass

    def get_point_cloud_from_ticker(
        self,
        ticker: str | int,
        num_points: int,
        save2dic: bool = True
    ) -> ndarray:
        """Get a point cloud from a ticker.
        
        Args:
            ticker (str | int): The ticker of the target data.
            num_points (int): The number of points in the point cloud.
            save2dic (bool): whether to save  the point cloud into ticker_point_clouds_dic.
            
        Returns:
            point_cloud (ndarray): The point cloud of the financial instrument.
        """
        if ticker in self.ticker_point_clouds_dic:
            point_cloud: ndarray = self.ticker_point_clouds_dic[ticker]
            if point_cloud.shape[0] < num_points:
                raise ValueError(f"The number of points in the point cloud is less than {num_points}.")
            point_cloud: ndarray = self.prng.choice(
                point_cloud, num_points, replace=False
            )
            return point_cloud
        else:
            if ticker not in self.ticker_path_dic:
                raise ValueError(f"The ticker {ticker} is not in the path dictionary.")
            data_path: Path = self.ticker_path_dic[ticker]
            point_cloud: ndarray = self.get_point_cloud_from_path(num_points, data_path)
            if save2dic:
                self.ticker_point_clouds_dic[ticker] = point_cloud
            return point_cloud

    def create_ot_distance_matrix(
        self,
        num_points: int,
        tickers: Optional[list[str | int]] = None,
        save_path: Optional[Path] = None,
        return_distance_matrix: bool = False,
    ) -> Optional[ndarray]:
        if tickers is None:
            tickers: list[str | int] = list(self.ticker_path_dic.keys())
        num_tickers: int = len(tickers)
        distance_matrix: ndarray = np.zeros(
            (num_tickers, num_tickers), dtype=np.float64
        )
        for i in range(num_tickers):
            for j in range(i+1, num_tickers):
                ticker1: str | int = tickers[i]
                ticker2: str | int = tickers[j]
                point_cloud1: ndarray = self.get_point_cloud_from_ticker(ticker1, num_points)
                point_cloud2: ndarray = self.get_point_cloud_from_ticker(ticker2, num_points)
                ot_distance: float = self.calc_ot_distance(point_cloud1, point_cloud2)
                distance_matrix[i, j] = ot_distance
                distance_matrix[j, i] = ot_distance
        if save_path is not None:
            distance_df: DataFrame = pd.DataFrame(distance_matrix, index=tickers, columns=tickers)
            distance_df.to_csv(save_path)
        if return_distance_matrix:
            return distance_matrix
        
    def draw_distance_matrix(
        self,
        tickers: list[str | int],
        distance_matrix: ndarray,
        save_path: Path,
        title: str = "OT Distance Matrix",
        cmap: str = "viridis"
    ) -> None:
        """Draw heatmap of the distance matrix.
        
        Args:
            tickers (list[str | int]): The list of tickers.
            distance_matrix (ndarray): The distance matrix.
            save_path (Path): The path to save the heatmap.
            title (str): The title of the heatmap.
            cmap (str): The colormap of the heatmap.

        Returns:
            None
        """
        num_tickers: int = len(tickers)
        fig: Figure = plt.figure(figsize=(10, 10), dpi=50)
        ax: Axes = fig.add_subplot(111)
        cax = ax.matshow(distance_matrix, cmap=cmap)
        fig.colorbar(cax)
        ax.set_title(title)
        ax.set_xticks(range(num_tickers))
        ax.set_yticks(range(num_tickers))
        ax.set_xticklabels(tickers, rotation=45)
        ax.set_yticklabels(tickers)
        parent_path: Path = save_path.parent
        if not parent_path.exists():
            parent_path.mkdir(parents=True)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.savefig(save_path)
        plt.close()

    def _hist_points(
        self,
        ax: Axes,
        point_cloud1d: ndarray,
        color: str = "blue",
        label: Optional[str] = None,
    ) -> None:
        """Draw a histogram of the points. point_cloud1d is a 1D array.
        
        Args:
            ax (Axes): The axes of the plot.
            point_cloud1d (ndarray): The 1D array of the points.
            color (str): The color of the histogram.
            label (str): The label of the histogram.

        Returns:
            None
        """
        assert point_cloud1d.ndim == 1
        ax.hist(
            point_cloud1d, bins=100, alpha=0.5,
            label=label, color=color
        )
        
    def _scatter_points2d(
        self,
        ax: Axes,
        point_cloud2d: ndarray,
        color: str = "blue",
        label: Optional[str] = None,
    ) -> None:
        """Draw a scatter plot of the points. point_cloud2d is a 2D array.
        
        Args:
            ax (Axes): The axes of the plot.
            point_cloud2d (ndarray): The 2D array of the points.
            color (str): The color of the scatter plot.
            label (str): The label of the scatter plot.

        Returns:
            None
        """
        assert point_cloud2d.shape[1] == 2
        ax.scatter(
            point_cloud2d[:, 0], point_cloud2d[:, 1],
            label=label, color=color
        )
        
    def _scatter_points3d(
        self,
        ax: Axes3D,
        point_cloud3d: ndarray,
        color: str = "blue",
        label: Optional[str] = None,
    ) -> None:
        """Draw a scatter plot of the points. point_cloud3d is a 3D array.
        
        Args:
            ax (Axes): The axes of the plot.
            point_cloud3d (ndarray): The 3D array of the points.
            color (str): The color of the scatter plot.
            label (str): The label of the scatter plot.
        
        Returns:
            None
        """
        assert point_cloud3d.shape[1] == 3
        ax.scatter(
            point_cloud3d[:, 0], point_cloud3d[:, 1], point_cloud3d[:, 2],
            label=label, color=color
        )

    def _draw_points(
        self,
        ax: Axes | Axes3D,
        point_cloud: ndarray,
        draw_dims: Optional[list[int]] = None,
        color: str = "blue",
        label: Optional[str] = None,
    ) -> None:
        """Draw the points in the point cloud.
        
        Args:
            ax (Axes): The axes of the plot.
            point_cloud (ndarray): The point cloud.
            draw_dims (list[int]): The dimensions to draw.
            color (str): The color of the points.
            label (str): The label of the points.

        Returns:
            None
        """
        point_dim: int = point_cloud.shape[1]
        hist_points: bool = False
        scatter_points2d: bool = False
        scatter_points3d: bool = False
        if draw_dims is not None:
            if len(draw_dims) == 1:
                point_cloud1d: ndarray = point_cloud[:, draw_dims].flatten()
                hist_points = True
            elif len(draw_dims) == 2:
                point_cloud2d: ndarray = point_cloud[:, draw_dims]
                scatter_points2d = True
            elif len(draw_dims) == 3:
                point_cloud3d: ndarray = point_cloud[:, draw_dims]
                scatter_points3d = True
            else:
                raise ValueError("The number of dimensions to draw must be less than or equal to 3.")
        else:
            if point_dim == 1:
                point_cloud1d: ndarray = point_cloud.flatten()
                hist_points = True
            elif point_dim == 2:
                point_cloud2d: ndarray = point_cloud
                scatter_points2d = True
            elif point_dim == 3:
                point_cloud3d: ndarray = point_cloud
                scatter_points3d = True
            else:
                raise ValueError("Specify the dimensions to draw.")
        if hist_points:
            assert isinstance(ax, Axes)
            self._hist_points(ax, point_cloud1d, color, label)
        elif scatter_points2d:
            assert isinstance(ax, Axes)
            self._scatter_points2d(ax, point_cloud2d, color, label)
        elif scatter_points3d:  
            assert isinstance(ax, Axes3D)
            self._scatter_points3d(
                ax, point_cloud3d, color, label
            )
        else:
            raise ValueError("Could not draw anything.")
        
    @abstractmethod
    def draw_points(self, *args, **kwargs) -> None:
        pass



        