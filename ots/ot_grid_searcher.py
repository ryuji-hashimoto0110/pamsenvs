from .data_distance_evaluater import DDEvaluater
import json
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
import pathlib
from pathlib import Path
curr_path: Path = pathlib.Path(__file__).resolve().parents[0]
parent_path: Path = curr_path.parents[0]
import sys
sys.path.append(str(parent_path))
from rich import print
from rich.console import Console
from rich.table import Table
from stylized_facts import SimulationEvaluater
from typing import Any
from typing import Optional
from typing import TypeVar

VarID = TypeVar("VarID")

class OTGridSearcher:
    def __init__(
        self,
        initial_seed: int,
        dd_evaluater: DDEvaluater,
        show_process: bool = True,
        base_config: Optional[dict[str, Any]] = None,
        base_config_path: Optional[str | Path] = None,
        target_variables_config: Optional[dict[str, Any]] = None,
        target_variables_config_path: Optional[str | Path] = None,
        use_simulator_given_runner: bool = False,
        num_simulations: int = 1500,
        temp_txts_path: Optional[str | Path] = None,
        temp_tick_dfs_path: Optional[str | Path] = None,
        temp_ohlcv_dfs_path: Optional[str | Path] = None,
        temp_all_time_ohlcv_dfs_path: Optional[str | Path] = None,
        path_to_calc_point_clouds: Optional[str | Path] = None,
        transactions_path: Optional[str | Path] = None,
        session1_transactions_file_name: Optional[str] = None,
        session2_transactions_file_name: Optional[str] = None,
        num_points: int = 1000
    ) -> None:
        """initialization.

        Args:
            initial_seed (int): initial seed.
            dd_evaluater (DDEvaluater): DDEvaluater which has point clouds of real datas.
            base_config (dict[str, Any], optional): Base config. Defaults to None.
            base_config_path (Path, optional): Base config path. Defaults to None.
            target_variables_config (dict[str, Any], optional): Target variables config. Defaults to None.
            target_variables_config_path (Path, optional): Target variables config path. Defaults to None.
            use_simulator_given_runner (bool): Whether to use SimulatorGivenRunner or not. Defaults to False.
            num_simulations (int, optional): Number of simulations. Defaults to 1500.
            temp_txts_path (str | Path, optional): Temporal txts path. Defaults to None.
            temp_tick_dfs_path (str | Path, optional): Temporal tick dfs path. Defaults to None.
            temp_ohlcv_dfs_path (str | Path, optional): Temporal ohlcv dfs path. Defaults to None.
            temp_all_time_ohlcv_dfs_path (str | Path, optional): Temporal all time ohlcv dfs path. Defaults to None.
            path_to_calc_point_clouds (str | Path, optional): Path to calculate point clouds. Defaults to None.
                path_to_calc_point_clouds must be identical with either temp_ohlcv_dfs_path or temp_all_time_ohlcv_dfs_path.
            transactions_path (Path | str, optional): Transactions path. Defaults to None.
                Ex) 'pamsenvs/datas/real_datas/intraday/flex_transactions/1min/9202'
            session1_transactions_file_name (str, optional): Session1 transactions file name. Defaults to None.
            session2_transactions_file_name (str, optional): Session2 transactions file name. Defaults to None.
            num_points (int, optional): Number of points to calculate OT distances. Defaults to 1000.
        """
        self.initial_seed: int = initial_seed
        self.real_tickers: list[int] = list(dd_evaluater.ticker_path_dic.keys())
        self.dd_evaluater: DDEvaluater = dd_evaluater
        self.show_process: bool = show_process
        self.base_config: dict[str, Any] = self._get_config(
            config=base_config,
            config_path=self._convert_str2path(base_config_path, mkdir=False)
        )
        self.target_variables_config: dict[str, Any] = self._get_config(
            config=target_variables_config,
            config_path=self._convert_str2path(target_variables_config_path, mkdir=False)
        )
        self.id2var_dic: dict[VarID, list[str]] = self._get_target_variables(
            target_variables_config=self.target_variables_config, show_variables=self.show_process
        )
        self.var_ids: list[VarID] = list(self.id2var_dic.keys())
        self.result_df: DataFrame = self._create_result_df()
        self.use_simulator_given_runner: bool = use_simulator_given_runner
        self.num_simulations: int = num_simulations
        self.temp_txts_path: Optional[Path] = self._convert_str2path(
            temp_txts_path, mkdir=True
        )
        self.temp_tick_dfs_path: Optional[Path] = self._convert_str2path(
            temp_tick_dfs_path, mkdir=True
        )
        self.temp_ohlcv_dfs_path: Optional[Path] = self._convert_str2path(
            temp_ohlcv_dfs_path, mkdir=True
        )
        self.temp_all_time_ohlcv_dfs_path: Optional[Path] = self._convert_str2path(
            temp_all_time_ohlcv_dfs_path, mkdir=True
        )
        self.path_to_calc_point_clouds: Optional[Path] = self._convert_str2path(
            path_to_calc_point_clouds, mkdir=False
        )
        self.transactions_path: Optional[Path | str] = transactions_path
        self.session1_transactions_file_name: Optional[str] = session1_transactions_file_name
        self.session2_transactions_file_name: Optional[str] = session2_transactions_file_name
        self.num_points: int = num_points

    def _get_config(
        self,
        config: Optional[dict[str, Any]],
        config_path: Optional[Path]
    ) -> dict[str, Any]:
        """get config from config or config path.

        In OTGridSearcher, both direct specification of config and
        specification of config path are allowed.
        """
        if config is not None:
            return config
        elif config_path is not None:
            config: dict[str, Any] = json.load(
                fp=open(str(config_path), mode="r")
            )
            return config
        else:
            raise ValueError("Specify either config or config path.")
        
    def _convert_str2path(
        self,
        path_name: Optional[str | Path],
        mkdir: bool
    ) -> Optional[Path]:
        """convert str to path.
        
        Args:
            path_name (str | Path, optional): Name of path. If path_name is str, it will be
                converted to Path. path_name is recommended to be an absolute path.
            mkdir (bool): Whether to make directory if path does not exist.
        """
        if path_name is None:
            return None
        elif isinstance(path_name, Path):
            path: Path = path_name.resolve()
        elif isinstance(path_name, str):
            path: Path = pathlib.Path(path_name).resolve()
        else:
            raise ValueError(
                f"path_name must be either str or pathlib.Path. path_name={path_name}"
            )
        if not path.exists():
            if mkdir:
                path.mkdir(parents=True)
        return path
        
    def _get_target_variables(
        self,
        target_variables_config: dict[str, Any],
        show_variables: bool = True
    ) -> dict[VarID, list[str]]:
        """create index and variables dictionary.

        target_variables_config consists of Variabe IDs as keys and the list of
        two components as values:
        keys to the target variables and target candidates to search.

        {
            0: [["Agent", "chartWeight"], [{"expon": [0.0]}, ..., {"expon": [3.0]}]],
            1: [["Agent", "noiseWeight"], [{"expon": [0.0]}, ..., {"expon": [3.0]}]],
            2: [["Agent", "riskAversionTerm"], [0.02, ..., 0.20]],
            3: [["Agent", "timeWindowSize"], [10, ..., 1000]]
        }

        This method extract Variable Ids and their names as a dictionary:

        {0: "chartWeight", 1: "noiseWeight", 2: "riskAversionTerm", 3: "timeWindowSize"}
        
        and shows the target variables names and the corresponding candidate values to search.

        +------------------+------------------+------------------+
        |               ID |             Name | Values to search |
        +------------------+------------------+------------------+
        |                0 |      chartWeight | [0.0, ..., 3.0]  |
        |                1 |      noiseWeight | [0.0, ..., 3.0]  |
        |                2 | riskAversionTerm | [0.02, ..., 0.20]|
        |                3 |   timeWindowSize | [10, ..., 1000]  |
        +------------------+------------------+------------------+

        Args:
            target_variables_config (dict[str, Any]): Target variables config.
            show_variables (bool, optional): Whether to show the target variables or not. Defaults to True.

        Returns:
            id2var_dic (dict[VarID, list[str]]): Index and variables dictionary.
        """
        table: Table = Table(title="Target Variables")
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Name", justify="right", style="cyan")
        table.add_column("Values to search", justify="left", style="magenta")
        id2var_dic: dict[VarID, list[str]] = {}
        for var_id, (var_names, var_values) in target_variables_config.items():
            var_name: str = var_names[-1]
            id2var_dic[var_id] = var_name
            if isinstance(var_values[0], dict):
                var_values_str: str = \
                    f"{list(var_values[0].values())}, ..., {list(var_values[-1].values())}"
            else:
                var_values_str: str = f"{var_values[0]}, ..., {var_values[-1]}"
            table.add_row(str(var_id), var_name, var_values_str)
        if show_variables:
            console: Console = Console()
            console.print(table)
        return id2var_dic
    
    def _create_result_df(self) -> DataFrame:
        """create result DataFrame.

        This method creates a DataFrame to store the results of the simulations.
        """
        idvars: list[str] = [f"{var_id}:{self.id2var_dic[var_id]}" for var_id in self.var_ids]
        ottickers: list[str] = [f"ot({ticker})" for ticker in self.real_tickers]
        columns: list[str] = ["simulation id"] + idvars + ottickers
        result_df: DataFrame = pd.DataFrame(columns=columns)
        result_df: DataFrame = result_df.set_index("simulation id")
        return result_df
        
    def simulate_all_combs(
        self,
        result_save_path: Path
    ) -> None:
        """Simulate all combinations of variables.
        
        This method conducts simulations by the following procedure.
            1. Initialize combination dictionary. Seealso: _init_comb_dic.
               comb_dic represents the current combination of variables.
            2. Create number of combinations dictionary. Seealso: _create_num_comb_dic.
            3. For all combinations of variables:
                1. Modify base_config by given comb_dic. Seealso: _set_specific_config.
                2. Make all temporal folders empty. Seealso: _make_temp_folders_empty.
                3. Set simulation_evaluater.
                4. Conduct simulations for num_simulations times.
                5. Save simulation results to txts_path, tick_dfs_path, ohlcv_dfs_path,
                   and all_time_ohlcv_dfs_path.
                6. Calculate OT distances between the result of the simulation and the real datas stored in dd_evaluater.
                7. Update result_df by the variable values and the OT distances.
                8. Update combination dictionary. Seealso: _update_comb_dic.

        Args:
            result_save_path (Path): Path to save the result.
        """
        comb_dic: dict[VarID, int] = self._init_comb_dic()
        num_comb_dic: dict[VarID, int] = self._create_num_comb_dic()
        sim_id: int = 0
        while True:
            specific_config, variable_values = self._set_specific_config(
                self.base_config, comb_dic
            )
            self._make_temp_folders_empty()
            sim_evaluater: SimulationEvaluater = SimulationEvaluater(
                initial_seed=self.initial_seed,
                show_process=self.show_process,
                config=specific_config,
                specific_name="temp",
                txts_path=self.temp_txts_path,
                tick_dfs_path=self.temp_tick_dfs_path,
                ohlcv_dfs_path=self.temp_ohlcv_dfs_path,
                all_time_ohlcv_dfs_path=self.temp_all_time_ohlcv_dfs_path,
                transactions_path=self.transactions_path,
                session1_transactions_file_name=self.session1_transactions_file_name,
                session2_transactions_file_name=self.session2_transactions_file_name,
            )
            assert self.temp_txts_path is not None
            start_date, end_date = sim_evaluater.simulate_multiple_times(
                self.num_simulations, self.use_simulator_given_runner
            )
            assert self.temp_tick_dfs_path is not None
            sim_evaluater.process_flex()
            assert self.temp_ohlcv_dfs_path is not None
            assert self.transactions_path is not None
            assert self.session1_transactions_file_name is not None
            assert self.session2_transactions_file_name is not None
            sim_evaluater.check_stylized_facts(check_asymmetry=False)
            assert self.temp_all_time_ohlcv_dfs_path is not None
            sim_evaluater.concat_ohlcv(start_date=start_date, end_date=end_date)
            if not (
                self.path_to_calc_point_clouds == self.temp_ohlcv_dfs_path or
                self.path_to_calc_point_clouds == self.temp_all_time_ohlcv_dfs_path
            ):
                raise ValueError(
                    "path_to_calc_point_clouds must be identical with either temp_ohlcv_dfs_path or temp_all_time_ohlcv_dfs_path."
                )
            self.dd_evaluater.add_ticker_path(
                ticker_path=self.path_to_calc_point_clouds, ticker_name="temp"
            )
            art_point_cloud: ndarray = self.dd_evaluater.get_point_cloud_from_ticker(
                ticker_name="temp", num_points=self.num_points, save2dic=False
            )
            ot_distances: list[float] = []
            for ticker in self.real_tickers:
                real_point_cloud: ndarray = self.dd_evaluater.get_point_cloud_from_ticker(
                    ticker_name=ticker, num_points=self.num_points, save2dic=True
                )
                ot_distance: float = self.dd_evaluater.calc_ot_distance(
                    art_point_cloud, real_point_cloud, is_per_bit=True
                )
                ot_distances.append(ot_distance)
            new_results: list[int | float] = variable_values + ot_distances
            assert len(self.result_df.columns) == len(new_results)
            self.result_df.loc[sim_id] = new_results
            comb_dic, is_break = self._update_comb_dic(comb_dic, num_comb_dic)
            if is_break:
                break
        self.result_df.to_csv(result_save_path)

    def _init_comb_dic(self) -> dict[VarID, int]:
        """Initialize combination dictionary.
        
        .simulate_all_configs method conducts grid search for the target variables.
        This method initializes the combination dictionary to search the target variables.
        """
        init_comb_dic: dict[VarID, int] = {
            var_id: 0 for var_id in self.var_ids
        }
        return init_comb_dic

    def _create_num_comb_dic(self) -> dict[VarID, int]:
        """Create number of combinations dictionary.
        
        num_comb_dic[var_id] denotes the number of candidate values for the target variable var_id.
        """
        num_comb_dic: dict[VarID, list[int]] = {
            var_id: len(self.target_variables_config[var_id][-1]) \
                for var_id in self.var_ids
        }
        return num_comb_dic
    
    def _update_comb_dic(
        self,
        comb_dic: dict[VarID, int],
        num_comb_dic: dict[VarID, int]
    ) -> tuple[dict[VarID, int], bool]:
        """Update combination dictionary.
        
        This method updates the combination dictionary to search the target variables.
        If the search is completed, return True.

        Args:
            comb_dic (dict[VarID, list[int]]): Combination dictionary.
            num_comb_dic (dict[VarID, list[int]]): Number of combinations dictionary created by _create_num_comb_dic.
        
        Returns:
            comb_dic (dict[VarID, list[int]]): Updated combination dictionary.
            is_break (bool): Whether the search is completed or not.
        """
        is_break: bool = False
        last_var_id: VarID = self.var_ids[-1]
        comb_dic[last_var_id] += 1
        if all(
            num_comb_dic[var_id] - 1 <= comb_dic[var_id] for var_id in self.var_ids
        ):
            is_break = True
        elif num_comb_dic[last_var_id] <= comb_dic[last_var_id]:
            comb_dic[last_var_id] = 0
            for var_id in reversed(self.var_ids[:-1]):
                comb_dic[var_id] += 1
                if comb_dic[var_id] < num_comb_dic[var_id]:
                    break
                else:
                    comb_dic[var_id] = 0
        return comb_dic, is_break

    def _set_specific_config(
        self,
        comb_dic: dict[VarID, list[int]]
    ) -> tuple[dict[str, Any], list[float | int]]:
        """Modify base_config by given comb_dic.
        
        Args:
            comb_dic (dict[VarID, list[int]]): Combination dictionary.
        
        Returns:
            specific_config (dict[str, Any]): Specific config.
            variable_values (list[float | int]): Variable values for the specific config.
        """
        specific_config: dict[str, Any] = self.base_config.copy()
        variable_values: list[float | int] = []
        for var_id in self.var_ids:
            var_names, var_values = self.target_variables_config[var_id]
            new_value: float | int = var_values[comb_dic[var_id]]
            specific_config = self._change_value_in_nested_dic(
                specific_config, var_names, new_value
            )
            variable_values.append(new_value)
        return specific_config, variable_values

    def _change_value_in_nested_dic(
        self,
        dic: dict[str, Any],
        nested_keys: list[str],
        new_value: float | int
    ) -> dict[str, Any]:
        """Change values in a specific config.

        dic is a dictionary which has nested keys like followings.

        {
            "key1": {
                "key2": 0,
                "key3": {
                    "key4": 1, "key5": 2
                },
            },
        }

        This method changes the value in dic by given nested keys and new value.
        If nested_keys = ["key1", "key2"], new_value = 3, then the result will be:

        {
            "key1": {
                "key2": 3,
                "key3": {
                    "key4": 1, "key5": 2
                },
            },
        }
        
        Args:
            dic (dict[str, Any]): nested dictionary.
            nested_keys (list[str]): Nested keys to change the value.
            new_value (float | int): New value to set.
        
        Returns:
            dic (dict[str, Any]): dic with new value.
        """
        new_dic: dict[str, Any] = dic.copy()
        curr_dic: dict[str, Any] = new_dic
        for key in nested_keys[:-1]:
            if key in curr_dic:
                curr_dic = curr_dic[key]
                if not isinstance(curr_dic, dict):
                    raise ValueError("curr_dic is not a dictionary any more.")
            else:
                raise ValueError(f"{key} is not found in {curr_dic}.")
        curr_dic[nested_keys[-1]] = new_value
        return new_dic
    
    def _make_temp_folders_empty(self) -> None:
        """Make all temporal folders empty.
        """
        self._unlink_all_children_files(self.temp_txts_path)
        self._unlink_all_children_files(self.temp_tick_dfs_path)
        self._unlink_all_children_files(self.temp_ohlcv_dfs_path)

    def _unlink_all_children_files(self, path: Optional[Path]) -> None:
        """Unlink all children files in the path.
        """
        if path is not None:
            for file in path.iterdir():
                file.unlink()


        