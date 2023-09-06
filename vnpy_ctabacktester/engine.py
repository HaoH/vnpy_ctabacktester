import logging
from ast import List
import importlib
import traceback
from datetime import datetime
from threading import Thread
from pathlib import Path
from inspect import getfile
from glob import glob
from types import ModuleType
from pandas import DataFrame
from typing import Optional

from src.config import logConfig
from src.engine.backtesting_engine import ExBacktestingEngine
from src.signals.divergence_detector import DivergenceDetector
from src.signals.supertrend_detector import SupertrendDetector
from vnpy.event import Event, EventEngine
from vnpy.trader.engine import BaseEngine, MainEngine
from vnpy.trader.constant import Interval
from vnpy.trader.utility import extract_vt_symbol, get_file_path, save_json
from vnpy.trader.object import HistoryRequest, TickData, ContractData, BarData, TradeData, OrderData
from vnpy.trader.datafeed import BaseDatafeed, get_datafeed
from vnpy.trader.database import BaseDatabase, get_database

import vnpy_ctastrategy
from vnpy_ctastrategy import CtaTemplate, TargetPosTemplate, StopOrder
from vnpy_ctastrategy.backtesting import (
    BacktestingEngine,
    OptimizationSetting,
    BacktestingMode, load_bar_data
)


APP_NAME = "CtaBacktester"

EVENT_BACKTESTER_LOG = "eBacktesterLog"
EVENT_BACKTESTER_BACKTESTING_FINISHED = "eBacktesterBacktestingFinished"
EVENT_BACKTESTER_OPTIMIZATION_FINISHED = "eBacktesterOptimizationFinished"


class BacktesterEngine(BaseEngine):
    """
    For running CTA strategy backtesting.
    """

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """"""
        super().__init__(main_engine, event_engine, APP_NAME)

        self.classes: dict = {}
        self.backtesting_engine: ExBacktestingEngine = None
        self.thread: Thread = None

        self.datafeed: BaseDatafeed = get_datafeed()
        self.database: BaseDatabase = get_database()

        # Backtesting reuslt
        self.result_df: DataFrame = None
        self.result_statistics: dict = None

        # Optimization result
        self.result_values: list = None
        self.history_data_weekly: list[BarData] = None

        # logger
        self.logger = None

        # settings
        self.engine_settings = {}
        self.optimization_settings = {}

    def init_engine(self) -> None:
        """"""
        self.write_log("初始化CTA回测引擎")

        # self.backtesting_engine = BacktestingEngine()
        self.backtesting_engine = ExBacktestingEngine()

        # Redirect log from backtesting engine outside.
        self.backtesting_engine.output = self.write_log

        self.load_strategy_class()
        self.write_log("策略文件加载完成")

        self.init_datafeed()

    def init_datafeed(self) -> None:
        """
        Init datafeed client.
        """
        result: bool = self.datafeed.init(self.write_log)
        if result:
            self.write_log("数据服务初始化成功")

    def write_log(self, msg: str) -> None:
        """"""
        event: Event = Event(EVENT_BACKTESTER_LOG)
        event.data = msg
        self.event_engine.put(event)

    def load_backtesting_settings(self, settings: dict):
        engine = self.backtesting_engine
        engine.clear_data()
        if "start" in settings.keys() and "start_dt" not in settings.keys():
            settings["start_dt"] = settings['start']
        if "end" in settings.keys() and "end_dt" not in settings.keys():
            settings["end_dt"] = settings['end']
        self.engine_settings = settings
        engine.set_parameters(**settings)
        if "optimizations" in settings:
            self.optimization_settings = settings["optimizations"]

    def load_backtesting_data(self, backtesting_data: dict):

        self.load_backtesting_settings(backtesting_data['parameters'])
        engine: BacktestingEngine = self.backtesting_engine

        for key, trade in backtesting_data["trades"].items():
            trade.pop('vt_symbol')
            trade.pop('vt_orderid')
            trade.pop('vt_tradeid')
            engine.trades[key] = TradeData(**trade)
        engine.trade_count = backtesting_data["trade_count"]

        for key, stop_order in backtesting_data["stop_orders"].items():
            engine.stop_orders[key] = StopOrder(**stop_order)
        engine.stop_order_count = backtesting_data["stop_order_count"]

        for key, order in backtesting_data["limit_orders"].items():
            order.pop('vt_symbol')
            order.pop('vt_orderid')
            engine.limit_orders[key] = OrderData(**order)
        engine.limit_order_count = backtesting_data["limit_order_count"]

        # 加载数据
        engine.load_data()
        engine.initialize_daily_results()

        self.result_df = engine.calculate_result()
        engine.calculate_statistics()
        engine.result_statistics = backtesting_data["statistics"]
        self.result_statistics = backtesting_data["statistics"]

    def load_strategy_class(self) -> None:
        """
        Load strategy class from source code.
        """
        # get the path of current package
        home_path: Path = Path.home()
        path1 = home_path.joinpath('Workspace', 'quant_life', 'src', 'strategy')
        # app_path: Path = Path(src.__file__).parent
        # path1: Path = app_path.joinpath("strategy")
        self.load_strategy_class_from_folder(path1, "src.strategy")
        #
        # path2: Path = Path.cwd().joinpath("strategies")
        # self.load_strategy_class_from_folder(path2, "strategies")


    def load_strategy_class_from_folder(self, path: Path, module_name: str = "") -> None:
        """
        Load strategy class from certain folder.
        """
        for suffix in ["py", "pyd", "so"]:
            pathname: str = str(path.joinpath(f"*.{suffix}"))
            for filepath in glob(pathname):
                filename: str = Path(filepath).stem
                name: str = f"{module_name}.{filename}"
                self.load_strategy_class_from_module(name)

    def load_strategy_class_from_module(self, module_name: str) -> None:
        """
        Load strategy class from module file.
        """
        try:
            module: ModuleType = importlib.import_module(module_name)

            # 重载模块，确保如果策略文件中有任何修改，能够立即生效。
            importlib.reload(module)

            for name in dir(module):
                value = getattr(module, name)
                if (
                    isinstance(value, type)
                    and issubclass(value, CtaTemplate)
                    and value not in {CtaTemplate, TargetPosTemplate}
                ):
                    self.classes[value.__name__] = value
        except:  # noqa
            msg: str = f"策略文件{module_name}加载失败，触发异常：\n{traceback.format_exc()}"
            self.write_log(msg)

    def reload_strategy_class(self) -> None:
        """"""
        self.classes.clear()
        self.load_strategy_class()
        self.write_log("策略文件重载刷新完成")

    def get_strategy_class_names(self) -> list:
        """"""
        return list(self.classes.keys())

    def run_backtesting(
        self,
        class_name: str,
        vt_symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        rate: float,
        slippage: float,
        size: int,
        pricetick: float,
        capital: int,
        ta: dict,
        strategy_setting: dict,
        detector_setting: dict,
        log_filename: str = None
    ) -> None:
        """"""
        self.result_df = None
        self.result_statistics = None

        engine: BacktestingEngine = self.backtesting_engine
        engine.clear_data()

        if interval == Interval.TICK.value:
            mode: BacktestingMode = BacktestingMode.TICK
        else:
            mode: BacktestingMode = BacktestingMode.BAR

        engine.set_parameters(
            vt_symbol=vt_symbol,
            interval=Interval(interval),
            start_dt=start,
            end_dt=end,
            rate=rate,
            slippage=slippage,
            size=size,
            price_tick=pricetick,
            capital=capital,
            mode=mode,
            ta=ta,
            strategy_settings=strategy_setting,
            detector_settings=detector_setting,
            strategy_name=class_name,
            classes=self.classes
        )

        engine.load_data()
        if not engine.history_data:
            self.write_log("策略回测失败，历史数据为空")
            return

        try:
            engine.run_backtesting()
        except Exception:
            msg: str = f"策略回测失败，触发异常：\n{traceback.format_exc()}"
            self.write_log(msg)

            self.thread = None
            return

        self.result_df = engine.calculate_result()
        self.result_statistics = engine.calculate_statistics()

        # Clear thread object handler.
        self.thread = None

        if log_filename:
            log_filepath = get_file_path(f"log/{log_filename}.back")
            engine.save_backtesting_data(log_filepath)

        # Put backtesting done event
        event: Event = Event(EVENT_BACKTESTER_BACKTESTING_FINISHED)
        self.event_engine.put(event)

    def start_backtesting(
        self,
        class_name: str,
        vt_symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        rate: float,
        slippage: float,
        size: int,
        pricetick: float,
        capital: int,
        ta: list,
        strategy_setting: dict,
        detector_setting: dict,
        log_filename: str = None
    ) -> bool:
        if self.thread:
            self.write_log("已有任务在运行中，请等待完成")
            return False

        self.write_log("-" * 40)
        self.thread = Thread(
            target=self.run_backtesting,
            args=(
                class_name,
                vt_symbol,
                interval,
                start,
                end,
                rate,
                slippage,
                size,
                pricetick,
                capital,
                ta,
                strategy_setting,
                detector_setting,
                log_filename
            )
        )
        self.thread.start()

        return True

    def get_result_df(self) -> DataFrame:
        """"""
        return self.result_df

    def get_result_statistics(self) -> dict:
        """"""
        return self.result_statistics

    def get_result_values(self) -> list:
        """"""
        return self.result_values

    def get_default_setting(self, class_name: str) -> dict:
        """"""
        strategy_class: type = self.classes[class_name]
        return strategy_class.get_class_parameters()

    def save_optimization_results(self, log_filename: str):
        result = {
            "parameter": self.engine_settings,
            "result_value": self.result_values
        }
        save_json(log_filename, result)


    def run_optimization(
        self,
        class_name: str,
        vt_symbol: str,
        interval: Interval,
        start: datetime,
        end: datetime,
        rate: float,
        slippage: float,
        size: int,
        pricetick: float,
        capital: int,
        ta: dict,
        strategy_setting: dict,
        detector_setting: dict,
        optimization_setting: OptimizationSetting,
        use_ga: bool,
        max_workers: int,
        log_filename: str = None
    ) -> None:
        """"""
        self.result_values = None

        engine: BacktestingEngine = self.backtesting_engine
        engine.clear_data()

        if interval == Interval.TICK.value:
            mode: BacktestingMode = BacktestingMode.TICK
        else:
            mode: BacktestingMode = BacktestingMode.BAR

        engine.set_parameters(
            vt_symbol=vt_symbol,
            interval=interval,
            start_dt=start,
            end_dt=end,
            rate=rate,
            slippage=slippage,
            size=size,
            price_tick=pricetick,
            capital=capital,
            mode=mode,
            ta=ta,
            strategy_settings=strategy_setting,
            detector_settings=detector_setting,
            strategy_name=class_name,
            classes=self.classes
        )

        # 0则代表不限制
        if max_workers == 0:
            max_workers = None

        if use_ga:
            self.result_values = engine.run_ga_optimization(
                optimization_setting,
                output=False,
                max_workers=max_workers
            )
        else:
            self.result_values = engine.run_bf_optimization(
                optimization_setting,
                output=False,
                max_workers=max_workers
            )

        # Clear thread object handler.
        self.thread = None
        self.write_log("多进程参数优化完成")

        if log_filename:
            log_filepath = get_file_path(f"log/{log_filename}.opt")
            self.save_optimization_results(log_filepath)

        # Put optimization done event
        event: Event = Event(EVENT_BACKTESTER_OPTIMIZATION_FINISHED)
        self.event_engine.put(event)

    def start_optimization(
        self,
        class_name: str,
        vt_symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        rate: float,
        slippage: float,
        size: int,
        pricetick: float,
        capital: int,
        ta: list,
        strategy_setting: dict,
        detector_setting: dict,
        optimization_setting: OptimizationSetting,
        use_ga: bool,
        max_workers: int,
        log_filename: str = None
    ) -> bool:
        if self.thread:
            self.write_log("已有任务在运行中，请等待完成")
            return False

        self.write_log("-" * 40)
        self.thread = Thread(
            target=self.run_optimization,
            args=(
                class_name,
                vt_symbol,
                interval,
                start,
                end,
                rate,
                slippage,
                size,
                pricetick,
                capital,
                ta,
                strategy_setting,
                detector_setting,
                optimization_setting,
                use_ga,
                max_workers,
                log_filename
            )
        )
        self.thread.start()

        return True

    def run_downloading(
        self,
        vt_symbol: str,
        interval: str,
        start: datetime,
        end: datetime
    ) -> None:
        """
        执行下载任务
        """
        self.write_log(f"{vt_symbol}-{interval}开始下载历史数据")

        try:
            symbol, exchange = extract_vt_symbol(vt_symbol)
        except ValueError:
            self.write_log(f"{vt_symbol}解析失败，请检查交易所后缀")
            self.thread = None
            return

        req: HistoryRequest = HistoryRequest(
            symbol=symbol,
            exchange=exchange,
            interval=Interval(interval),
            start=start,
            end=end
        )

        try:
            if interval == "tick":
                data: List[TickData] = self.datafeed.query_tick_history(req, self.write_log)
            else:
                contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)

                # If history data provided in gateway, then query
                if contract and contract.history_data:
                    data: List[BarData] = self.main_engine.query_history(
                        req, contract.gateway_name
                    )
                # Otherwise use RQData to query data
                else:
                    data: List[BarData] = self.datafeed.query_bar_history(req, self.write_log)

            if data:
                if interval == "tick":
                    self.database.save_tick_data(data)
                else:
                    self.database.save_bar_data(data)

                self.write_log(f"{vt_symbol}-{interval}历史数据下载完成, 共{len(data)}条。")
            else:
                self.write_log(f"数据下载失败，无法获取{vt_symbol}的历史数据")
        except Exception:
            msg: str = f"数据下载失败，触发异常：\n{traceback.format_exc()}"
            self.write_log(msg)

        # Clear thread object handler.
        self.thread = None

    def start_downloading(
        self,
        vt_symbol: str,
        interval: str,
        start: datetime,
        end: datetime
    ) -> bool:
        # if self.thread:
        #     self.write_log("已有任务在运行中，请等待完成")
        #     return False

        self.write_log("-" * 40)
        thread = Thread(
            target=self.run_downloading,
            args=(
                vt_symbol,
                interval,
                start,
                end
            )
        )
        thread.start()

        return True

    def get_all_trades(self) -> list:
        """"""
        return self.backtesting_engine.get_all_trades()

    def get_all_orders(self) -> list:
        """"""
        return self.backtesting_engine.get_all_orders()

    def get_all_daily_results(self) -> list:
        """"""
        return self.backtesting_engine.get_all_daily_results()

    def get_history_data(self, interval: Interval = Interval.DAILY) -> list:
        """"""
        if interval == Interval.DAILY:
            return self.backtesting_engine.history_data
        else:
            engine = self.backtesting_engine
            if self.history_data_weekly is None:
                self.history_data_weekly = load_bar_data(
                    engine.symbol,
                    engine.exchange,
                    interval,
                    engine.start,
                    engine.end
                )
            return self.history_data_weekly

    def get_strategy_class_file(self, class_name: str) -> str:
        """"""
        strategy_class: type = self.classes[class_name]
        file_path: str = getfile(strategy_class)
        return file_path