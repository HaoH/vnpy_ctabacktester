import csv
import logging
import logging.config
import subprocess
from datetime import datetime, timedelta
from copy import copy
from typing import List, Tuple

import numpy as np
import pyqtgraph as pg
from pandas import DataFrame
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree.parameterTypes import WidgetParameterItem

from ex_vnpy.logging.config import logConfig
from vnpy.trader.constant import Interval, Direction, Exchange
from vnpy.trader.engine import MainEngine, BaseEngine
from vnpy.trader.ui import QtCore, QtWidgets, QtGui
from vnpy.trader.ui.widget import BaseMonitor, BaseCell, DirectionCell, EnumCell
from vnpy.event import Event, EventEngine
from vnpy.chart import ChartWidget, CandleItem, VolumeItem
from vnpy.trader.utility import load_json, save_json, get_file_path, update_nested_dict
from vnpy.trader.object import BarData, TradeData, OrderData
from vnpy.trader.database import DB_TZ
from vnpy_ctastrategy.backtesting import DailyResult

from ..engine import (
    APP_NAME,
    EVENT_BACKTESTER_LOG,
    EVENT_BACKTESTER_BACKTESTING_FINISHED,
    EVENT_BACKTESTER_OPTIMIZATION_FINISHED,
    OptimizationSetting
)


class BacktesterManager(QtWidgets.QWidget):
    """"""

    setting_filename: str = "cta_backtester_setting.json"

    signal_log: QtCore.Signal = QtCore.Signal(Event)
    signal_backtesting_finished: QtCore.Signal = QtCore.Signal(Event)
    signal_optimization_finished: QtCore.Signal = QtCore.Signal(Event)

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """"""
        super().__init__()

        self.splitter: QtWidgets.QSplitter = None
        self.fileModel: QtWidgets.QFileSystemMode = None
        self.filetreeView: QtWidgets.QTreeView = None
        self.parameter_tree: Parameter = None
        self.log_filename: str = None
        self.logger = None
        self.main_engine: MainEngine = main_engine
        self.event_engine: EventEngine = event_engine

        self.backtester_engine: BaseEngine = main_engine.get_engine(APP_NAME)
        self.class_names: list = []
        self.settings: dict = {}

        self.target_display: str = ""

        self.init_ui()
        self.register_event()
        self.backtester_engine.init_engine()
        self.init_strategy_settings()
        self.load_backtesting_setting()

    def init_strategy_settings(self) -> None:
        """"""
        self.class_names = self.backtester_engine.get_strategy_class_names()
        self.class_names.sort()

        for class_name in self.class_names:
            setting: dict = self.backtester_engine.get_default_setting(class_name)
            self.settings[class_name] = setting

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle("CTA回测")

        # init file browser
        search_layout = QtWidgets.QHBoxLayout()

        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.textChanged.connect(self.filter_by_filenames)  # 连接文本变化信号
        search_layout.addWidget(self.search_edit)

        search_button = QtWidgets.QPushButton('搜索')
        search_button.clicked.connect(self.filter_by_filenames)
        search_layout.addWidget(search_button)

        self.filetreeView = QtWidgets.QTreeView(self)
        self.fileModel = QtWidgets.QFileSystemModel()
        self.filetreeView.setModel(self.fileModel)

        initial_path = '/Users/wukong/.vntrader'
        rootIndex = self.fileModel.setRootPath(initial_path)
        self.filetreeView.setRootIndex(rootIndex)

        self.fileModel.setFilter(QtCore.QDir.NoDotAndDotDot | QtCore.QDir.AllDirs | QtCore.QDir.Files)

        self.fileModel.setNameFilters(['*.back'])       # 设置只显示指定后缀的文件
        self.fileModel.setNameFilterDisables(False)     # 启用后缀过滤

        # self.filemodel.setHeaderData(0, QtCore.Qt.Horizontal, 'File Name') # 设置列名
        # self.filetreeView.setColumnWidth(0, 200)  # 第一列宽度为200像素
        self.filetreeView.setColumnHidden(0, False)     # 显示第一列（文件名）
        self.filetreeView.setColumnHidden(1, True)
        self.filetreeView.setColumnHidden(2, True)
        self.filetreeView.setColumnHidden(3, True)

        self.filetreeView.doubleClicked.connect(self.backtestingfile_double_clicked)
        self.filetreeView.hide()  # 初始状态隐藏文件浏览器

        # init parameter tree
        pg.parametertree.registerParameterItemType('datetime', DatetimeParameterItem)

        start_dt = QtCore.QDateTime.fromString("2016-01-01 00:00:00", "yyyy-MM-dd hh:mm:ss")
        end_dt = QtCore.QDateTime(QtCore.QDate.currentDate(), QtCore.QTime(0, 0))
        params = [
            {'title': '策略名称', 'name': 'strategy_name', 'type': 'list', 'values': ['UniStrategy'], 'value': 'UniStrategy'},
            {'title': 'Symbol', 'name': 'vt_symbol', 'type': 'str', 'value': '600111.SSE'},
            {'title': 'K线周期', 'name': 'interval', 'type': 'list',
             'values': [interval.value for interval in Interval], 'value': Interval.DAILY.value},
            {'title': '开始时间', 'name': 'start_dt', 'type': 'datetime', 'value': start_dt},
            {'title': '结束时间', 'name': 'end_dt', 'type': 'datetime', 'value': end_dt},
            {'title': '信号阈值', 'name': 'threshold', 'type': 'float', 'value': 3.0},
            {'title': '信号总强度', 'name': 'full_strength', 'type': 'float', 'value': 10.0},
            {'title': 'Impulse止损', 'name': 'stoploss_ind_enabled', 'type': 'bool', 'value': True},
            {
                'title': 'Indicators',
                'name': 'ta',
                'type': 'group',
                'children': [
                    {'title': 'MACD', 'name': 'MACD11.params', 'type': 'str', 'value': "11,22,8;"},
                    {'title': 'ATR', 'name': 'ATR13.params', 'type': 'str', 'value': "13;"},
                    {'title': 'Impulse', 'name': 'Impulse11.params', 'type': 'str', 'value': "11,22,8,13;"}
                ]
            },
            {
                'title': 'SupertrendDetector',
                'name': 'SupertrendDetector',
                'type': 'group',
                'children': [
                    {'title': '是否开启', 'name': 'enabled', 'type': 'bool', 'value': True},
                    {'title': '信号强度', 'name': 'weight', 'type': 'float', 'value': 5.0},
                    {'title': '止损比例', 'name': 'stop_loss_rate', 'type': 'float', 'value': 0.08},
                    {'title': '趋势类型', 'name': 'trend_type', 'type': 'list', 'values': ['EVERY', 'PIVOT'],
                     'value': 'EVERY'},
                    {'title': 'Trend Source', 'name': 'trend_source', 'type': 'list',
                     'values': ['open', 'high', 'low', 'close'], 'value': 'close'},
                    {'title': 'ATR Factor', 'name': 'atr_factor', 'type': 'float', 'value': 3},
                    {'title': 'PivotValidBars', 'name': 'valid_bars', 'type': 'int', 'value': 3},
                ]
            },
            {
                'title': 'DivergenceDetector',
                'name': 'DivergenceDetector',
                'type': 'group',
                'children': [
                    {'title': '是否开启', 'name': 'enabled', 'type': 'bool', 'value': False},
                    {'title': '信号强度', 'name': 'weight', 'type': 'float', 'value': 5.0},
                    {'title': '止损比例', 'name': 'stop_loss_rate', 'type': 'float', 'value': 0.08},
                    {'title': '价格source', 'name': 'source', 'type': 'list',
                     'values': ['open', 'high', 'low', 'close'], 'value': 'low'},
                    {'title': 'Pivot Source', 'name': 'pivot_source', 'type': 'list', 'values': ['macd_h', 'close'],
                     'value': 'macd_h'},
                ]
            },

        ]
        param_tree = Parameter.create(name='params', type='group', children=params)
        param_tree.sigTreeStateChanged.connect(self.parameter_changed)
        self.parameter_tree = param_tree

        # 创建一个参数树
        param_tree_widget = ParameterTree()
        param_tree_widget.setHeaderLabels(["参数名称", "参数值"])
        param_tree_widget.setParameters(param_tree, showTop=False)

        # init function button
        backtesting_button: QtWidgets.QPushButton = QtWidgets.QPushButton("开始回测")
        backtesting_button.clicked.connect(self.start_backtesting)

        optimization_button: QtWidgets.QPushButton = QtWidgets.QPushButton("参数优化")
        optimization_button.clicked.connect(self.start_optimization)

        self.result_button: QtWidgets.QPushButton = QtWidgets.QPushButton("优化结果")
        self.result_button.clicked.connect(self.show_optimization_result)
        self.result_button.setEnabled(False)

        downloading_button: QtWidgets.QPushButton = QtWidgets.QPushButton("更新数据")
        downloading_button.clicked.connect(self.start_downloading)

        load_backtesting_button: QtWidgets.QPushButton = QtWidgets.QPushButton("加载回测")
        # load_backtesting_button.clicked.connect(self.load_backtesting_data)
        load_backtesting_button.clicked.connect(self.toggle_file_browser)

        self.order_button: QtWidgets.QPushButton = QtWidgets.QPushButton("委托记录")
        self.order_button.clicked.connect(self.show_backtesting_orders)
        self.order_button.setEnabled(False)

        self.trade_button: QtWidgets.QPushButton = QtWidgets.QPushButton("成交记录")
        self.trade_button.clicked.connect(self.show_backtesting_trades)
        self.trade_button.setEnabled(False)

        self.daily_button: QtWidgets.QPushButton = QtWidgets.QPushButton("每日盈亏")
        self.daily_button.clicked.connect(self.show_daily_results)
        self.daily_button.setEnabled(False)

        self.candle_button: QtWidgets.QPushButton = QtWidgets.QPushButton("K线图表")
        self.candle_button.clicked.connect(self.show_candle_chart)
        # self.candle_button.setEnabled(False)
        self.candle_button.setEnabled(True)

        # edit_button: QtWidgets.QPushButton = QtWidgets.QPushButton("代码编辑")
        # edit_button.clicked.connect(self.edit_strategy_code)

        save_setting_button: QtWidgets.QPushButton = QtWidgets.QPushButton("保存配置")
        save_setting_button.clicked.connect(self.save_settings)

        reload_button: QtWidgets.QPushButton = QtWidgets.QPushButton("策略重载")
        reload_button.clicked.connect(self.reload_strategy_class)

        for button in [
            backtesting_button,
            optimization_button,
            downloading_button,
            load_backtesting_button,
            self.result_button,
            self.order_button,
            self.trade_button,
            self.daily_button,
            self.candle_button,
            # edit_button,
            save_setting_button,
            reload_button
        ]:
            button.setFixedHeight(button.sizeHint().height() * 2)

        result_grid: QtWidgets.QGridLayout = QtWidgets.QGridLayout()

        result_grid.addWidget(load_backtesting_button, 0, 0)
        result_grid.addWidget(self.candle_button, 0, 1)
        result_grid.addWidget(save_setting_button, 0, 2)

        result_grid.addWidget(self.trade_button, 1, 0)
        result_grid.addWidget(self.order_button, 1, 1)
        result_grid.addWidget(self.daily_button, 1, 2)

        result_grid.addWidget(optimization_button, 3, 0)
        result_grid.addWidget(self.result_button, 3, 1)
        result_grid.addWidget(downloading_button, 3, 2)
        # result_grid.addWidget(edit_button, 3, 2)

        # result_grid.addWidget(reload_button, 4, 2)

        left_vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        # left_vbox.addLayout(form)
        left_vbox.addWidget(param_tree_widget)
        left_vbox.addStretch()
        left_vbox.addWidget(backtesting_button)
        left_vbox.addStretch()
        left_vbox.addLayout(result_grid)
        # left_vbox.addStretch()
        # left_vbox.addWidget(optimization_button)
        # left_vbox.addWidget(self.result_button)
        # left_vbox.addStretch()
        # left_vbox.addWidget(edit_button)
        # left_vbox.addWidget(reload_button)

        # init statistic
        self.statistics_monitor: StatisticsMonitor = StatisticsMonitor()

        self.log_monitor: QtWidgets.QTextEdit = QtWidgets.QTextEdit()

        # init result chart
        self.chart: BacktesterChart = BacktesterChart()
        chart: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        chart.addWidget(self.chart)

        self.trade_dialog: BacktestingResultDialog = BacktestingResultDialog(
            self.main_engine,
            self.event_engine,
            "回测成交记录",
            BacktestingTradeMonitor
        )
        self.order_dialog: BacktestingResultDialog = BacktestingResultDialog(
            self.main_engine,
            self.event_engine,
            "回测委托记录",
            BacktestingOrderMonitor
        )
        self.daily_dialog: BacktestingResultDialog = BacktestingResultDialog(
            self.main_engine,
            self.event_engine,
            "回测每日盈亏",
            DailyResultMonitor
        )

        self.candle_dialog: CandleChartDialog = CandleChartDialog()

        middle_vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        middle_vbox.addWidget(self.statistics_monitor)
        middle_vbox.addWidget(self.log_monitor)

        # init layout
        left_hbox: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        left_hbox.addLayout(left_vbox)
        left_hbox.addLayout(middle_vbox)

        left_widget: QtWidgets.QWidget = QtWidgets.QWidget()
        left_widget.setLayout(left_hbox)

        right_vbox: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        right_vbox.addWidget(self.chart)

        right_widget: QtWidgets.QWidget = QtWidgets.QWidget()
        right_widget.setLayout(right_vbox)

        hbox: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        hbox.addWidget(left_widget)
        hbox.addWidget(right_widget)

        visible_widget: QtWidgets.QWidget = QtWidgets.QWidget()
        visible_widget.setLayout(hbox)

        filetree_layout = QtWidgets.QVBoxLayout()
        filetree_layout.addLayout(search_layout)
        filetree_layout.addWidget(self.filetreeView)

        filetree_widget: QtWidgets.QWidget = QtWidgets.QWidget()
        filetree_widget.setLayout(filetree_layout)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.addWidget(filetree_widget)
        self.splitter.addWidget(visible_widget)
        self.splitter.setSizes([0, 1])  # 设置左右两边的初始宽度

        all_box: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        all_box.addWidget(self.splitter)
        self.setLayout(all_box)

    def init_logger(self, symbol: str) -> None:
        # Initialize logger
        self.log_filename = f"{symbol}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        config_path = get_file_path(f"log/{self.log_filename}")

        # 修改初始化参数
        logConfig['handlers']['fileHandler']['filename'] = config_path
        # logConfig['handlers']['pyQtGraphHandler']['log_widget'] = self.log_monitor

        # 加载配置文件并应用
        logging.config.dictConfig(logConfig)

        # 初始化日志
        self.logger = logging.getLogger("Backtesting")

    def load_backtesting_setting(self) -> None:
        """"""
        setting: dict = load_json(self.setting_filename)
        if not setting:
            return

        self.update_parameter_tree(setting)
        self.backtester_engine.load_backtesting_settings(setting)


    def register_event(self) -> None:
        """"""
        self.signal_log.connect(self.process_log_event)
        self.signal_backtesting_finished.connect(
            self.process_backtesting_finished_event)
        self.signal_optimization_finished.connect(
            self.process_optimization_finished_event)

        self.event_engine.register(EVENT_BACKTESTER_LOG, self.signal_log.emit)
        self.event_engine.register(EVENT_BACKTESTER_BACKTESTING_FINISHED, self.signal_backtesting_finished.emit)
        self.event_engine.register(EVENT_BACKTESTER_OPTIMIZATION_FINISHED, self.signal_optimization_finished.emit)

    def parameter_changed(self, param, changes):
        for param, change, data in changes:
            if change == 'value':
                print(f"Parameter '{param.name()}' changed to {data}")

    def process_log_event(self, event: Event) -> None:
        """"""
        msg = event.data
        self.write_log(msg)

    def write_log(self, msg: str, log_level: int = logging.INFO) -> None:
        """"""
        if self.logger:
            if log_level == logging.INFO:
                self.logger.info(msg)
            elif log_level == logging.DEBUG:
                self.logger.debug(msg)
            elif log_level == logging.WARNING:
                self.logger.warning(msg)
            elif log_level == logging.CRITICAL:
                self.logger.critical(msg)
            elif log_level == logging.ERROR:
                self.logger.error(msg)

        timestamp: str = datetime.now().strftime("%H:%M:%S")
        msg: str = f"{timestamp}\t{msg}"
        self.log_monitor.append(msg)

    def process_backtesting_finished_event(self, event: Event) -> None:
        """"""
        statistics: dict = self.backtester_engine.get_result_statistics()
        self.statistics_monitor.set_data(statistics)

        df: DataFrame = self.backtester_engine.get_result_df()
        self.chart.set_data(df)

        self.trade_button.setEnabled(True)
        self.order_button.setEnabled(True)
        self.daily_button.setEnabled(True)

        # Tick data can not be displayed using candle chart
        # interval: str = self.interval_combo.currentText()
        interval: str = self.parameter_tree.param("interval").value()
        if interval != Interval.TICK.value:
            self.candle_button.setEnabled(True)

    def process_optimization_finished_event(self, event: Event) -> None:
        """"""
        self.write_log("请点击[优化结果]按钮查看")
        self.result_button.setEnabled(True)

    def start_backtesting(self) -> None:
        """"""
        if "strategy_name" not in self.parameter_tree.keys():
            self.write_log("请选择要回测的策略")
            return

        new_settings = self.get_parameter_tree_settings()
        vt_symbol = new_settings['vt_symbol']

        # 初始化日志输出
        self.init_logger(vt_symbol)

        # Check validity of vt_symbol
        if "." not in vt_symbol:
            self.write_log("本地代码缺失交易所后缀，请检查")
            return

        _, exchange_str = vt_symbol.split(".")
        if exchange_str not in Exchange.__members__:
            self.write_log("本地代码的交易所后缀不正确，请检查")
            return

        # # Get strategy setting
        # dialog: BacktestingSettingEditor = BacktestingSettingEditor(class_name, old_setting)
        # i: int = dialog.exec()
        # if i != dialog.Accepted:
        #     return
        #
        # new_strategy_setting: dict = dialog.get_setting()
        # new_strategy_setting['stoploss_ind'] = {"name": "Impulse11", "signals": "impulse", "type": "impulse"}
        # self.settings[class_name] = new_strategy_setting

        # # Save backtesting parameters
        backtesting_settings = self.backtester_engine.engine_settings

        update_nested_dict(backtesting_settings, new_settings)
        save_json(self.setting_filename, backtesting_settings)

        result: bool = self.backtester_engine.start_backtesting(
            backtesting_settings["strategy_name"],
            backtesting_settings["vt_symbol"],
            backtesting_settings["interval"],
            backtesting_settings["start_dt"],
            backtesting_settings["end_dt"],
            backtesting_settings["trade_settings"],
            backtesting_settings["ta"],
            backtesting_settings["strategy_settings"],
            backtesting_settings["detector_settings"],
            self.log_filename
        )

        if result:
            self.statistics_monitor.clear_data()
            self.chart.clear_data()

            self.trade_button.setEnabled(False)
            self.order_button.setEnabled(False)
            self.daily_button.setEnabled(False)
            self.candle_button.setEnabled(False)

            self.trade_dialog.clear_data()
            self.order_dialog.clear_data()
            self.daily_dialog.clear_data()
            self.candle_dialog.clear_data()

    def get_parameter_tree_settings(self) -> dict:
        """"""
        def get_parameters_dict(parameters):
            params_dict = {}
            for param in parameters:
                if param.hasChildren():
                    params_dict[param.name()] = get_parameters_dict(param.children())
                else:
                    params_dict[param.name()] = param.value()
            return params_dict

        parameters_dict = get_parameters_dict(self.parameter_tree.children())

        # update parameters_dict values
        parameters_dict['interval'] = Interval(parameters_dict['interval'])
        parameters_dict['start_dt'] = parameters_dict['start_dt'].toPython()
        parameters_dict['end_dt'] = parameters_dict['end_dt'].toPython()

        parameters_dict['strategy_settings'] = {
            "threshold": parameters_dict['threshold'],
            "full_strength": parameters_dict['full_strength'],
            "stoploss_ind": {
                "enabled": parameters_dict['stoploss_ind_enabled'],
            }
        }

        ta_settings = parameters_dict['ta']
        parameters_dict['ta'] = {}
        for ind_name, ind_value in ta_settings.items():
            real_name = ind_name.split('.')[0]
            parameters_dict['ta'][real_name] = {}
            if len(ind_value) > 0 and ind_value[-1] == ';':
                ind_value = ind_value[:-1]
                ind_value = [int(x) for x in ind_value.split(',')]
            parameters_dict['ta'][real_name]['params'] = ind_value

        parameters_dict['detector_settings'] = {}
        detectors = self.backtester_engine.engine_settings['detector_settings'].keys()
        for detector in detectors:
            parameters_dict['detector_settings'][detector] = parameters_dict[detector]

        for param_name in ['threshold', 'full_strength', 'stoploss_ind_enabled']:
            parameters_dict.pop(param_name)

        for param_name in detectors:
            parameters_dict.pop(param_name)

        return parameters_dict

    def update_parameter_tree(self, setting: dict) -> None:
        strategy_name_param = self.parameter_tree.param('strategy_name')
        strategy_name_param.setLimits(self.class_names)
        strategy_name_param.setValue(setting["strategy_name"])
        self.parameter_tree.param('vt_symbol').setValue(setting['vt_symbol'])
        self.parameter_tree.param('interval').setValue(setting['interval'].value)

        start_dt: datetime = setting.get("start_dt", "")
        if start_dt == "":
            start_dt = setting.get("start", "")
        if start_dt != "":
            start_dt = QtCore.QDateTime.fromString(start_dt.strftime("%Y-%m-%d %H:%M:%S"), "yyyy-MM-dd hh:mm:ss")
            self.parameter_tree.param('start_dt').setValue(start_dt)

        end_dt: datetime = setting.get("end_dt", "")
        if end_dt == "":
            end_dt = setting.get("end", "")
        if end_dt != "":
            end_dt = QtCore.QDateTime.fromString(end_dt.strftime("%Y-%m-%d %H:%M:%S"), "yyyy-MM-dd hh:mm:ss")
            self.parameter_tree.param('end_dt').setValue(end_dt)

        strategy_settings = setting['strategy_settings']
        self.parameter_tree.param('threshold').setValue(strategy_settings['threshold'])
        self.parameter_tree.param('full_strength').setValue(strategy_settings['full_strength'])
        if 'enabled' in strategy_settings['stoploss_ind']:
            self.parameter_tree.param('stoploss_ind_enabled').setValue(strategy_settings['stoploss_ind']['enabled'])

        ta_settings = setting['ta']
        ta_param = self.parameter_tree.param('ta')
        for ind_name, ind_value in ta_settings.items():
            param_name = f'{ind_name}.params'
            if param_name in ta_param.names.keys():
                values = [str(x) for x in ind_value['params']]
                ta_param.param(param_name).setValue(f"{','.join(values)};")

        detector_settings = setting['detector_settings']
        for detector_name, detector_setting in detector_settings.items():
            detector_params = self.parameter_tree.param(detector_name)
            detector_params_names = detector_params.names.keys()
            for param_name, param_value in detector_setting.items():
                if param_name in detector_params_names:
                    detector_params.param(param_name).setValue(param_value)

    def toggle_file_browser(self):
        status = self.filetreeView.isHidden()
        if status:
            self.filetreeView.show()
            self.splitter.setSizes([1, 3])
        else:
            self.filetreeView.hide()
            self.splitter.setSizes([0, 1])

    def filter_by_filenames(self):
        search_text = self.search_edit.text()

        if not search_text:
            self.fileModel.setNameFilters(['*.back'])
            self.filetreeView.collapseAll()
        else:
            self.fileModel.setNameFilters([f'*{search_text}*.back'])

    def backtestingfile_double_clicked(self, index: QtCore.QModelIndex):
        file_info = self.fileModel.fileInfo(index)
        file_path = file_info.filePath()
        # 在这里执行双击文件后的动作，例如打开文件或显示文件信息
        if file_path.endswith('.back'):
            self.write_log(f'加载回测文件: {file_path}')
            self.load_backtesting_data(file_path)

    def load_backtesting_data(self, file_name: str = "") -> None:
        # 使用pyqtgraph弹出系统文件选择弹框，然后选择文件
        # file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
        #     self,
        #     "选择文件",
        #     "/Users/wukong/.vntrader/log",
        #     "策略回溯文件(*.back)"
        # )
        #
        if not file_name:
            return

        # 读取file_name文件，json格式
        backtesting_data = load_json(file_name)
        self.backtester_engine.load_backtesting_data(backtesting_data)

        # 根据backtester_engine的参数更新前端组件
        parameters = backtesting_data['parameters']
        self.update_parameter_tree(parameters)

        # 从backtesting_data中读取结果，更新self.statistics_monitor
        statistics: dict = self.backtester_engine.get_result_statistics()
        self.statistics_monitor.set_data(statistics)

        # 从backtesting_data中读取结果，更新self.chart
        df: DataFrame = self.backtester_engine.get_result_df()
        self.chart.set_data(df)

        # 更新回测K线图
        self.candle_dialog.updated = False

        # 更新self.trade_button, self.order_button, self.daily_button, self.candle_button
        self.trade_button.setEnabled(True)
        self.order_button.setEnabled(True)
        self.daily_button.setEnabled(True)
        self.candle_button.setEnabled(True)

    def start_optimization(self) -> None:
        """"""
        new_settings = self.get_parameter_tree_settings()
        strategy_name = new_settings['strategy_name']
        vt_symbol = new_settings['vt_symbol']

        # parameters: dict = self.settings[strategy_name]
        # parameters = backtesting_settings["optimizations"]
        parameters = self.backtester_engine.optimization_settings
        dialog: OptimizationSettingEditor = OptimizationSettingEditor(strategy_name, parameters)
        i: int = dialog.exec()
        if i != dialog.Accepted:
            return

        new_settings["optimizations"] = dialog.parameters

        backtesting_settings = self.backtester_engine.engine_settings
        update_nested_dict(backtesting_settings, new_settings)
        save_json(self.setting_filename, backtesting_settings)

        if not dialog.run_optimization:
            return

        optimization_setting, use_ga, max_workers = dialog.get_setting()
        self.target_display: str = dialog.target_display

        # 初始化日志输出
        self.init_logger(vt_symbol)

        self.backtester_engine.start_optimization(
            strategy_name,
            vt_symbol,
            backtesting_settings["interval"],
            backtesting_settings["start_dt"],
            backtesting_settings["end_dt"],
            backtesting_settings["trade_settings"],
            backtesting_settings["ta"],
            backtesting_settings["strategy_settings"],
            backtesting_settings["detector_settings"],
            optimization_setting,
            use_ga,
            max_workers,
            self.log_filename
        )

        self.result_button.setEnabled(False)

    def start_downloading(self) -> None:
        """"""
        vt_symbol = self.parameter_tree.param('vt_symbol').value()
        start_dt = self.parameter_tree.param('start_dt').value().toPython()
        end_dt = self.parameter_tree.param('end_dt').value().toPython()

        start_dt = start_dt.replace(tzinfo=DB_TZ)
        end_dt = end_dt.replace(tzinfo=DB_TZ)

        self.backtester_engine.start_downloading(
            vt_symbol,
            'd',
            start_dt,
            end_dt
        )

        self.backtester_engine.start_downloading(
            vt_symbol,
            'w',
            start_dt,
            end_dt
        )


    def show_optimization_result(self) -> None:
        """"""
        result_values: list = self.backtester_engine.get_result_values()

        dialog: OptimizationResultMonitor = OptimizationResultMonitor(
            result_values,
            self.target_display
        )
        dialog.exec_()

    def show_backtesting_trades(self) -> None:
        """"""
        if not self.trade_dialog.is_updated():
            trades: List[TradeData] = self.backtester_engine.get_all_trades()
            self.trade_dialog.update_data(trades)

        self.trade_dialog.exec_()

    def show_backtesting_orders(self) -> None:
        """"""
        if not self.order_dialog.is_updated():
            orders: List[OrderData] = self.backtester_engine.get_all_orders()
            self.order_dialog.update_data(orders)

        self.order_dialog.exec_()

    def show_daily_results(self) -> None:
        """"""
        if not self.daily_dialog.is_updated():
            results: List[DailyResult] = self.backtester_engine.get_all_daily_results()
            self.daily_dialog.update_data(results)

        self.daily_dialog.exec_()

    def show_candle_chart(self) -> None:
        """"""
        if not self.candle_dialog.is_updated():
            # clear data first
            self.candle_dialog.clear_data()

            history: list = self.backtester_engine.get_history_data(Interval.DAILY)
            history_w: list = self.backtester_engine.get_history_data(Interval.WEEKLY)

            self.candle_dialog.save_history_data(history, history_w)
            self.candle_dialog.update_history(history)

            trades: List[TradeData] = self.backtester_engine.get_all_trades()
            self.candle_dialog.update_trades(trades)
            self.candle_dialog.save_trades(trades)

            # self.interval_d_btn.clicked.connect(lambda: self.change_period(Interval.DAILY))
            self.candle_dialog.chart.interval_w_btn.clicked.connect(lambda: self.candle_dialog.change_period(Interval.WEEKLY))
            self.candle_dialog.chart.interval_d_btn.clicked.connect(lambda: self.candle_dialog.change_period(Interval.DAILY))

            self.candle_dialog.setWindowTitle(f"回测K线图表-{self.parameter_tree.param('vt_symbol').value()}")

        self.candle_dialog.exec_()

    def save_settings(self) -> None:
        new_settings = self.get_parameter_tree_settings()
        backtesting_settings = self.backtester_engine.engine_settings
        update_nested_dict(backtesting_settings, new_settings)
        save_json(self.setting_filename, backtesting_settings)


    def edit_strategy_code(self) -> None:
        """"""
        class_name: str = self.class_combo.currentText()
        if not class_name:
            return

        file_path: str = self.backtester_engine.get_strategy_class_file(class_name)
        # cmd: list = ["code", file_path]
        cmd: list = ["pycharm", file_path]

        p: subprocess.CompletedProcess = subprocess.run(cmd, shell=True)
        if p.returncode:
            QtWidgets.QMessageBox.warning(
                self,
                "启动代码编辑器失败",
                "请检查是否安装了Visual Studio Code，并将其路径添加到了系统全局变量中！"
            )

    def reload_strategy_class(self) -> None:
        """"""
        self.backtester_engine.reload_strategy_class()

        current_strategy_name: str = self.class_combo.currentText()

        self.class_combo.clear()
        self.init_strategy_settings()

        ix: int = self.class_combo.findText(current_strategy_name)
        self.class_combo.setCurrentIndex(ix)


    def show(self) -> None:
        """"""
        self.showMaximized()


class StatisticsMonitor(QtWidgets.QTableWidget):
    """"""
    KEY_NAME_MAP: dict = {
        "start_date": "首个交易日",
        "end_date": "最后交易日",

        "total_days": "总交易日",
        "profit_days": "盈利交易日",
        "loss_days": "亏损交易日",

        "capital": "起始资金",
        "end_balance": "结束资金",

        "total_return": "总收益率",
        "annual_return": "年化收益",
        "max_drawdown": "最大回撤",
        "max_ddpercent": "百分比最大回撤",

        "total_net_pnl": "总盈亏",
        "total_commission": "总手续费",
        "total_slippage": "总滑点",
        "total_turnover": "总成交额",
        "total_trade_count": "总成交笔数",

        "win_rate_normal": "总胜率",
        "win_rate_weighted": "总加权胜率",
        "loss_rate_weighted": "总加权败率",
        "total_entry_count": "入场总次数",
        "entry_win_count": "入场成功次数",
        "entry_loss_count": "入场失败次数",

        "win_count_8": "入场盈利<8%",
        "win_count_16": "入场盈利<16%",
        "win_count_16a": "入场盈利>=16%",
        "loss_count_2": "入场亏损<=2%",
        "loss_count_5": "入场亏损<=5%",
        "loss_count_8": "入场亏损<=8%",
        "loss_count_8a": "入场亏损>8%",

        "daily_net_pnl": "日均盈亏",
        "daily_commission": "日均手续费",
        "daily_slippage": "日均滑点",
        "daily_turnover": "日均成交额",
        "daily_trade_count": "日均成交笔数",

        "daily_return": "日均收益率",
        "return_std": "收益标准差",
        "sharpe_ratio": "夏普比率",
        "return_drawdown_ratio": "收益回撤比"
    }

    def __init__(self) -> None:
        """"""
        super().__init__()

        self.cells: dict = {}

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        self.setRowCount(len(self.KEY_NAME_MAP))
        self.setVerticalHeaderLabels(list(self.KEY_NAME_MAP.values()))

        self.setColumnCount(1)
        self.horizontalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.setEditTriggers(self.NoEditTriggers)

        for row, key in enumerate(self.KEY_NAME_MAP.keys()):
            cell: QtWidgets.QTableWidgetItem = QtWidgets.QTableWidgetItem()
            self.setItem(row, 0, cell)
            self.cells[key] = cell

    def clear_data(self) -> None:
        """"""
        for cell in self.cells.values():
            cell.setText("")

    def set_data(self, data: dict) -> None:
        """"""
        data["capital"] = f"{data['capital']:,.2f}"
        data["end_balance"] = f"{data['end_balance']:,.2f}"
        data["total_return"] = f"{data['total_return']:,.2%}"
        data["annual_return"] = f"{data['annual_return']:,.2%}"
        data["max_drawdown"] = f"{data['max_drawdown']:,.2f}"
        data["max_ddpercent"] = f"{data['max_ddpercent']:,.2%}"
        data["win_rate_normal"] = f"{data['win_rate_normal']:.2%}"
        data["win_rate_weighted"] = f"{data['win_rate_weighted']:.2%}"
        if 'loss_rate_weighted' in data:
            data["loss_rate_weighted"] = f"{data['loss_rate_weighted']:.2%}"
        # data["total_entry_count"] = f"{data['total_entry_count']}"
        # data["entry_win_count"] = f"{data['entry_win_count']}"
        # data["entry_loss_count"] = f"{data['entry_loss_count']}"
        # data["total_entry_count"] = f"{data['total_entry_count']}"
        # data["entry_win_count"] = f"{data['entry_win_count']}"
        # data["entry_loss_count"] = f"{data['entry_loss_count']}"
        data["total_net_pnl"] = f"{data['total_net_pnl']:,.2f}"
        data["total_commission"] = f"{data['total_commission']:,.2f}"
        data["total_slippage"] = f"{data['total_slippage']:,.2f}"
        data["total_turnover"] = f"{data['total_turnover']:,.2f}"
        data["daily_net_pnl"] = f"{data['daily_net_pnl']:,.2f}"
        data["daily_commission"] = f"{data['daily_commission']:,.2f}"
        data["daily_slippage"] = f"{data['daily_slippage']:,.2f}"
        data["daily_turnover"] = f"{data['daily_turnover']:,.2f}"
        data["daily_trade_count"] = f"{data['daily_trade_count']:,.2f}"
        data["daily_return"] = f"{data['daily_return']:,.2%}"
        data["return_std"] = f"{data['return_std']:,.2%}"
        data["sharpe_ratio"] = f"{data['sharpe_ratio']:,.2f}"
        data["return_drawdown_ratio"] = f"{data['return_drawdown_ratio']:,.2f}"

        for key, cell in self.cells.items():
            value = data.get(key, "")
            cell.setText(str(value))


class BacktestingSettingEditor(QtWidgets.QDialog):
    """
    For creating new strategy and editing strategy parameters.
    """

    def __init__(
        self, class_name: str, parameters: dict
    ) -> None:
        """"""
        super(BacktestingSettingEditor, self).__init__()

        self.class_name: str = class_name
        self.parameters: dict = parameters
        self.edits: dict = {}

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        form: QtWidgets.QFormLayout = QtWidgets.QFormLayout()

        # Add vt_symbol and name edit if add new strategy
        self.setWindowTitle(f"策略参数配置：{self.class_name}")
        button_text: str = "确定"
        parameters: dict = self.parameters

        for name, value in parameters.items():
            type_ = type(value)

            edit: QtWidgets.QLineEdit = QtWidgets.QLineEdit(str(value))
            if type_ is int:
                validator: QtGui.QIntValidator = QtGui.QIntValidator()
                edit.setValidator(validator)
            elif type_ is float:
                validator: QtGui.QDoubleValidator = QtGui.QDoubleValidator()
                edit.setValidator(validator)
            else:
                continue

            form.addRow(f"{name} {type_}", edit)

            self.edits[name] = (edit, type_)

        button: QtWidgets.QPushButton = QtWidgets.QPushButton(button_text)
        button.clicked.connect(self.accept)
        form.addRow(button)

        widget: QtWidgets.QWidget = QtWidgets.QWidget()
        widget.setLayout(form)

        scroll: QtWidgets.QScrollArea = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)

        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addWidget(scroll)
        self.setLayout(vbox)

    def get_setting(self) -> dict:
        """"""
        setting: dict = {}

        for name, tp in self.edits.items():
            edit, type_ = tp
            value_text = edit.text()

            if type_ == bool:
                if value_text == "True":
                    value = True
                else:
                    value = False
            else:
                value = type_(value_text)

            setting[name] = value

        return setting


class BacktesterChart(pg.GraphicsLayoutWidget):
    """"""

    def __init__(self) -> None:
        """"""
        super().__init__(title="Backtester Chart")

        self.dates: dict = {}

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        pg.setConfigOptions(antialias=True)

        # Create plot widgets
        self.balance_plot = self.addPlot(
            title="账户净值",
            axisItems={"bottom": DateAxis(self.dates, orientation="bottom")}
        )
        self.nextRow()

        self.drawdown_plot = self.addPlot(
            title="净值回撤",
            axisItems={"bottom": DateAxis(self.dates, orientation="bottom")}
        )
        self.nextRow()

        self.pnl_plot = self.addPlot(
            title="每日盈亏",
            axisItems={"bottom": DateAxis(self.dates, orientation="bottom")}
        )
        self.nextRow()

        self.distribution_plot = self.addPlot(title="盈亏分布")

        # Add curves and bars on plot widgets
        self.balance_curve = self.balance_plot.plot(
            pen=pg.mkPen("#ffc107", width=3)
        )

        dd_color: str = "#303f9f"
        self.drawdown_curve = self.drawdown_plot.plot(
            fillLevel=-0.3, brush=dd_color, pen=dd_color
        )

        profit_color: str = 'r'
        loss_color: str = 'g'
        self.profit_pnl_bar = pg.BarGraphItem(
            x=[], height=[], width=0.3, brush=profit_color, pen=profit_color
        )
        self.loss_pnl_bar = pg.BarGraphItem(
            x=[], height=[], width=0.3, brush=loss_color, pen=loss_color
        )
        self.pnl_plot.addItem(self.profit_pnl_bar)
        self.pnl_plot.addItem(self.loss_pnl_bar)

        distribution_color: str = "#6d4c41"
        self.distribution_curve = self.distribution_plot.plot(
            fillLevel=-0.3, brush=distribution_color, pen=distribution_color
        )

    def clear_data(self) -> None:
        """"""
        self.balance_curve.setData([], [])
        self.drawdown_curve.setData([], [])
        self.profit_pnl_bar.setOpts(x=[], height=[])
        self.loss_pnl_bar.setOpts(x=[], height=[])
        self.distribution_curve.setData([], [])

    def set_data(self, df) -> None:
        """"""
        if df is None:
            return

        count: int = len(df)

        self.dates.clear()
        for n, date in enumerate(df.index):
            self.dates[n] = date

        # Set data for curve of balance and drawdown
        self.balance_curve.setData(df["balance"])
        self.drawdown_curve.setData(df["drawdown"])

        # Set data for daily pnl bar
        profit_pnl_x: list = []
        profit_pnl_height: list = []
        loss_pnl_x: list = []
        loss_pnl_height: list = []

        for count, pnl in enumerate(df["net_pnl"]):
            if pnl >= 0:
                profit_pnl_height.append(pnl)
                profit_pnl_x.append(count)
            else:
                loss_pnl_height.append(pnl)
                loss_pnl_x.append(count)

        self.profit_pnl_bar.setOpts(x=profit_pnl_x, height=profit_pnl_height)
        self.loss_pnl_bar.setOpts(x=loss_pnl_x, height=loss_pnl_height)

        # Set data for pnl distribution
        hist, x = np.histogram(df["net_pnl"], bins="auto")
        x = x[:-1]
        self.distribution_curve.setData(x, hist)


class DateAxis(pg.AxisItem):
    """Axis for showing date data"""

    def __init__(self, dates: dict, *args, **kwargs) -> None:
        """"""
        super().__init__(*args, **kwargs)
        self.dates: dict = dates

    def tickStrings(self, values, scale, spacing) -> list:
        """"""
        strings: list = []
        for v in values:
            dt = self.dates.get(v, "")
            strings.append(str(dt))
        return strings


class OptimizationSettingEditor(QtWidgets.QDialog):
    """
    For setting up parameters for optimization.
    """
    DISPLAY_NAME_MAP: dict = {
        "总收益率": "total_return",
        "夏普比率": "sharpe_ratio",
        "收益回撤比": "return_drawdown_ratio",
        "日均盈亏": "daily_net_pnl"
    }

    def __init__(
        self, class_name: str, parameters: dict
    ) -> None:
        """"""
        super().__init__()

        self.run_optimization = True
        self.class_name: str = class_name
        self.parameters: dict = parameters
        self.edits: dict = {}

        self.optimization_setting: OptimizationSetting = None
        self.use_ga: bool = False

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        QLabel: QtWidgets.QLabel = QtWidgets.QLabel

        self.target_combo: QtWidgets.QComboBox = QtWidgets.QComboBox()
        self.target_combo.addItems(list(self.DISPLAY_NAME_MAP.keys()))

        self.worker_spin: QtWidgets.QSpinBox = QtWidgets.QSpinBox()
        self.worker_spin.setRange(0, 1000)
        self.worker_spin.setValue(10)
        self.worker_spin.setToolTip("设为0则自动根据CPU核心数启动对应数量的进程")

        self.widgets = []  # List to store widgets

        self.grid_layout: QtWidgets.QGridLayout = QtWidgets.QGridLayout()
        self.grid_layout.addWidget(QLabel("优化目标"), 0, 0)
        self.grid_layout.addWidget(self.target_combo, 0, 1, 1, 2)
        self.grid_layout.addWidget(QLabel("进程上限"), 0, 3)
        self.grid_layout.addWidget(self.worker_spin, 0, 4, 1, 2)

        add_button = QtWidgets.QPushButton("+")
        add_button.clicked.connect(self.add_row)
        remove_button = QtWidgets.QPushButton("-")
        remove_button.clicked.connect(self.remove_row)
        self.grid_layout.addWidget(add_button, 0, 6)
        self.grid_layout.addWidget(remove_button, 0, 7)

        self.grid_layout.addWidget(QLabel("参数名"), 1, 0, 1, 3)
        self.grid_layout.addWidget(QLabel("值类型"), 1, 3)
        self.grid_layout.addWidget(QLabel("值列表"), 1, 4, 1, 3)
        self.grid_layout.addWidget(QLabel("是否启用"), 1, 7)
        # self.grid_layout.addWidget(QLabel("结束"), 1, 3)

        # Add vt_symbol and name edit if add new strategy
        self.setWindowTitle(f"优化参数配置：{self.class_name}")

        # validator: QtGui.QDoubleValidator = QtGui.QDoubleValidator()
        # row: int = 4

        for name, setting in self.parameters.items():
            param_name_edit, type_combo, value_edit, enable_checkbox = self.add_row()
            param_name_edit.setText(name)
            type_combo.setCurrentText(setting['type'])
            value_edit.setText(setting['value'])
            enable_checkbox.setChecked(setting['enabled'])

        # for name, value in self.parameters.items():
        #     type_ = type(value)
        #     if type_ not in [int, float]:
        #         continue
        #
        #     start_edit: QtWidgets.QLineEdit = QtWidgets.QLineEdit(str(value))
        #     step_edit: QtWidgets.QLineEdit = QtWidgets.QLineEdit(str(1))
        #     end_edit: QtWidgets.QLineEdit = QtWidgets.QLineEdit(str(value))
        #
        #     for edit in [start_edit, step_edit, end_edit]:
        #         edit.setValidator(validator)
        #
        #     self.grid_layout.addWidget(QLabel(name), row, 0)
        #     self.grid_layout.addWidget(start_edit, row, 1)
        #     self.grid_layout.addWidget(step_edit, row, 2)
        #     self.grid_layout.addWidget(end_edit, row, 3)
        #
        #     self.edits[name] = {
        #         "type": type_,
        #         "start": start_edit,
        #         "step": step_edit,
        #         "end": end_edit
        #     }
        #
        #     row += 1

        save_button: QtWidgets.QPushButton = QtWidgets.QPushButton("保存")
        save_button.clicked.connect(self.save_optimization_settings)

        parallel_button: QtWidgets.QPushButton = QtWidgets.QPushButton("多进程优化")
        parallel_button.clicked.connect(self.generate_parallel_setting)
        # self.grid_layout.addWidget(parallel_button, row, 0, 1, 4)

        # row += 1
        ga_button: QtWidgets.QPushButton = QtWidgets.QPushButton("遗传算法优化")
        ga_button.clicked.connect(self.generate_ga_setting)
        # self.grid_layout.addWidget(ga_button, row, 0, 1, 4)

        widget: QtWidgets.QWidget = QtWidgets.QWidget()
        widget.setLayout(self.grid_layout)

        scroll: QtWidgets.QScrollArea = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)

        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addWidget(scroll)
        vbox.addWidget(save_button)
        vbox.addWidget(parallel_button)
        vbox.addWidget(ga_button)
        self.setLayout(vbox)


    def add_row(self):
        row = self.grid_layout.rowCount()

        param_name_label = QtWidgets.QLineEdit()
        type_combo = QtWidgets.QComboBox()
        type_combo.addItems(['single', 'tuple', 'string', 'bool'])
        value_edit = QtWidgets.QLineEdit()
        enable_checkbox = QtWidgets.QCheckBox("Enable")

        self.grid_layout.addWidget(param_name_label, row, 0, 1, 3)
        self.grid_layout.addWidget(type_combo, row, 3)
        self.grid_layout.addWidget(value_edit, row, 4, 1, 3)
        self.grid_layout.addWidget(enable_checkbox, row, 7)

        new_row_widgets = (param_name_label, type_combo, value_edit, enable_checkbox)
        self.widgets.append(new_row_widgets)  # Add widgets to list
        return new_row_widgets

    def remove_row(self):
        if len(self.widgets) > 0:
            # row = self.grid_layout.rowCount() - 1
            row = len(self.widgets) - 1

            for widget_tuple in self.widgets[row]:
                widget = widget_tuple
                self.grid_layout.removeWidget(widget)
                widget.deleteLater()

            del self.widgets[row]

    def generate_ga_setting(self) -> None:
        """"""
        self.use_ga: bool = True
        self.run_optimization = False
        self.generate_setting()

    def generate_parallel_setting(self) -> None:
        """"""
        self.use_ga: bool = False
        self.run_optimization = True
        self.generate_setting()

    def save_ui_settings(self):
        self.parameters = {}
        for widget in self.widgets:
            param_name_label, type_combo, value_edit, enable_checkbox = widget
            param_name = param_name_label.text()
            value_type = type_combo.currentText()
            value = value_edit.text()
            enabled = enable_checkbox.isChecked()

            self.parameters[param_name] = {
                'type': value_type,
                'value': value,
                'enabled': enabled
            }

    def save_optimization_settings(self):
        self.save_ui_settings()
        self.run_optimization = False
        self.accept()

    def generate_setting(self) -> None:
        """"""
        self.optimization_setting = OptimizationSetting()

        self.target_display: str = self.target_combo.currentText()
        target_name: str = self.DISPLAY_NAME_MAP[self.target_display]
        self.optimization_setting.set_target(target_name)

        self.save_ui_settings()
        for param_name, param_value in self.parameters.items():
            if param_value["enabled"]:
                converted_value = self.convert_str_to_type(param_value["type"], param_value["value"])
                self.optimization_setting.add_parameter_values(param_name, converted_value)

        self.accept()

    def convert_str_to_type(self, value_type: str, array_as_string: str) -> List:
        """
        type为tuple类型，会将
        "1;2,3;4,5,6;7,8,9,10" 转换成 [(1,), (2, 3), (4, 5, 6), (7, 8, 9, 10)]

        type为 singlel类型，会将
        "0.05;0.06;0.07;0.08" 转换成 [0.05, 0.06, 0.07, 0.08]

        :param type:
        :param array_as_string:
        :return:
        """
        string_array = array_as_string.split(';')
        if value_type == 'tuple':
            converted_array = [tuple(map(int, item.split(','))) for item in string_array]
        elif value_type == 'single':
            converted_array = [float(item) for item in string_array]
        elif value_type == 'bool':
            converted_array = [True if item.lower() == 'true' else False for item in string_array]
        elif value_type == 'string':
            converted_array = string_array

        return converted_array

        # for name, d in self.edits.items():
        #     type_ = d["type"]
        #     start_value = type_(d["start"].text())
        #     step_value = type_(d["step"].text())
        #     end_value = type_(d["end"].text())
        #
        #     if start_value == end_value:
        #         self.optimization_setting.add_parameter(name, start_value)
        #     else:
        #         self.optimization_setting.add_parameter(
        #             name,
        #             start_value,
        #             end_value,
        #             step_value
        #         )

    def get_setting(self) -> Tuple[OptimizationSetting, bool, int]:
        """"""
        return self.optimization_setting, self.use_ga, self.worker_spin.value()


class OptimizationResultMonitor(QtWidgets.QDialog):
    """
    For viewing optimization result.
    """

    def __init__(
        self, result_values: list, target_display: str
    ) -> None:
        """"""
        super().__init__()

        self.result_values: list = result_values
        self.target_display: str = target_display

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle("参数优化结果")
        self.resize(1100, 500)

        # Creat table to show result
        table: QtWidgets.QTableWidget = QtWidgets.QTableWidget()

        table.setColumnCount(2)
        table.setRowCount(len(self.result_values))
        table.setHorizontalHeaderLabels(["参数", self.target_display])
        table.setEditTriggers(table.NoEditTriggers)
        table.verticalHeader().setVisible(False)

        table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents
        )
        table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch
        )

        for n, tp in enumerate(self.result_values):
            setting, target_value, _ = tp
            setting_cell: QtWidgets.QTableWidgetItem = QtWidgets.QTableWidgetItem(str(setting))
            target_cell: QtWidgets.QTableWidgetItem = QtWidgets.QTableWidgetItem(f"{target_value:.2f}")

            setting_cell.setTextAlignment(QtCore.Qt.AlignCenter)
            target_cell.setTextAlignment(QtCore.Qt.AlignCenter)

            table.setItem(n, 0, setting_cell)
            table.setItem(n, 1, target_cell)

        # Create layout
        button: QtWidgets.QPushButton = QtWidgets.QPushButton("保存")
        button.clicked.connect(self.save_csv)

        hbox: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        hbox.addStretch()
        hbox.addWidget(button)

        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addWidget(table)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

    def save_csv(self) -> None:
        """
        Save table data into a csv file
        """
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存数据", "", "CSV(*.csv)")

        if not path:
            return

        with open(path, "w") as f:
            writer = csv.writer(f, lineterminator="\n")

            writer.writerow(["参数", self.target_display])

            for tp in self.result_values:
                setting, target_value, _ = tp
                row_data: list = [str(setting), str(target_value)]
                writer.writerow(row_data)


class BacktestingTradeMonitor(BaseMonitor):
    """
    Monitor for backtesting trade data.
    """

    headers: dict = {
        "tradeid": {"display": "成交号 ", "cell": BaseCell, "update": False},
        "orderid": {"display": "委托号", "cell": BaseCell, "update": False},
        "symbol": {"display": "代码", "cell": BaseCell, "update": False},
        "exchange": {"display": "交易所", "cell": EnumCell, "update": False},
        "direction": {"display": "方向", "cell": DirectionCell, "update": False},
        "offset": {"display": "开平", "cell": EnumCell, "update": False},
        "price": {"display": "价格", "cell": BaseCell, "update": False},
        "volume": {"display": "数量", "cell": BaseCell, "update": False},
        "datetime": {"display": "时间", "cell": BaseCell, "update": False},
        "gateway_name": {"display": "接口", "cell": BaseCell, "update": False},
    }


class BacktestingOrderMonitor(BaseMonitor):
    """
    Monitor for backtesting order data.
    """

    headers: dict = {
        "orderid": {"display": "委托号", "cell": BaseCell, "update": False},
        "symbol": {"display": "代码", "cell": BaseCell, "update": False},
        "exchange": {"display": "交易所", "cell": EnumCell, "update": False},
        "type": {"display": "类型", "cell": EnumCell, "update": False},
        "direction": {"display": "方向", "cell": DirectionCell, "update": False},
        "offset": {"display": "开平", "cell": EnumCell, "update": False},
        "price": {"display": "价格", "cell": BaseCell, "update": False},
        "volume": {"display": "总数量", "cell": BaseCell, "update": False},
        "traded": {"display": "已成交", "cell": BaseCell, "update": False},
        "status": {"display": "状态", "cell": EnumCell, "update": False},
        "datetime": {"display": "时间", "cell": BaseCell, "update": False},
        "gateway_name": {"display": "接口", "cell": BaseCell, "update": False},
    }


class FloatCell(BaseCell):
    """
    Cell used for showing pnl data.
    """

    def __init__(self, content, data) -> None:
        """"""
        content: str = f"{content:.2f}"
        super().__init__(content, data)


class DailyResultMonitor(BaseMonitor):
    """
    Monitor for backtesting daily result.
    """

    headers: dict = {
        "date": {"display": "日期", "cell": BaseCell, "update": False},
        "trade_count": {"display": "成交笔数", "cell": BaseCell, "update": False},
        "start_pos": {"display": "开盘持仓", "cell": BaseCell, "update": False},
        "end_pos": {"display": "收盘持仓", "cell": BaseCell, "update": False},
        "turnover": {"display": "成交额", "cell": FloatCell, "update": False},
        "commission": {"display": "手续费", "cell": FloatCell, "update": False},
        "slippage": {"display": "滑点", "cell": FloatCell, "update": False},
        "trading_pnl": {"display": "交易盈亏", "cell": FloatCell, "update": False},
        "holding_pnl": {"display": "持仓盈亏", "cell": FloatCell, "update": False},
        "total_pnl": {"display": "总盈亏", "cell": FloatCell, "update": False},
        "net_pnl": {"display": "净盈亏", "cell": FloatCell, "update": False},
    }


class BacktestingResultDialog(QtWidgets.QDialog):
    """"""

    def __init__(
        self,
        main_engine: MainEngine,
        event_engine: EventEngine,
        title: str,
        table_class: QtWidgets.QTableWidget
    ) -> None:
        """"""
        super().__init__()

        self.main_engine: MainEngine = main_engine
        self.event_engine: EventEngine = event_engine
        self.title: str = title
        self.table_class: QtWidgets.QTableWidget = table_class

        self.updated: bool = False

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle(self.title)
        self.resize(1100, 600)

        self.table: QtWidgets.QTableWidget = self.table_class(self.main_engine, self.event_engine)

        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.table)

        self.setLayout(vbox)

    def clear_data(self) -> None:
        """"""
        self.updated = False
        self.table.setRowCount(0)

    def update_data(self, data: list) -> None:
        """"""
        self.updated = True

        data.reverse()
        for obj in data:
            self.table.insert_new_row(obj)

    def is_updated(self) -> bool:
        """"""
        return self.updated


class CandleChartDialog(QtWidgets.QDialog):
    """"""

    def __init__(self) -> None:
        """"""
        super().__init__()

        self.updated: bool = False

        self.dt_ix_map: dict = {}
        self.ix_bar_map: dict = {}

        self.high_price = 0
        self.low_price = 0
        self.price_range = 0

        self.items: list = []

        self.interval: Interval = Interval.DAILY
        self.history_data: dict = {}
        self.trades: list = None

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle("回测K线图表")
        self.resize(1400, 800)

        # Create chart widget
        self.chart: ChartWidget = ChartWidget()
        self.chart.add_plot("candle", hide_x_axis=True)
        self.chart.add_plot("volume", maximum_height=200)
        self.chart.add_item(CandleItem, "candle", "candle")
        self.chart.add_item(VolumeItem, "volume", "volume")
        self.chart.add_cursor()
        self.chart.add_period_changer("candle")

        # Create help widget
        # text1: str = "红色虚线 —— 盈利交易"
        # label1: QtWidgets.QLabel = QtWidgets.QLabel(text1)
        # label1.setStyleSheet("color:red")
        #
        # text2: str = "绿色虚线 —— 亏损交易"
        # label2: QtWidgets.QLabel = QtWidgets.QLabel(text2)
        # label2.setStyleSheet("color:#00FF00")
        #
        # text3: str = "黄色向上箭头 —— 买入开仓 Buy"
        # label3: QtWidgets.QLabel = QtWidgets.QLabel(text3)
        # label3.setStyleSheet("color:yellow")
        #
        # text4: str = "黄色向下箭头 —— 卖出平仓 Sell"
        # label4: QtWidgets.QLabel = QtWidgets.QLabel(text4)
        # label4.setStyleSheet("color:yellow")
        #
        # text5: str = "紫红向下箭头 —— 卖出开仓 Short"
        # label5: QtWidgets.QLabel = QtWidgets.QLabel(text5)
        # label5.setStyleSheet("color:magenta")
        #
        # text6: str = "紫红向上箭头 —— 买入平仓 Cover"
        # label6: QtWidgets.QLabel = QtWidgets.QLabel(text6)
        # label6.setStyleSheet("color:magenta")
        #
        # hbox1: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        # hbox1.addStretch()
        # hbox1.addWidget(label1)
        # hbox1.addStretch()
        # hbox1.addWidget(label2)
        # hbox1.addStretch()
        #
        # hbox2: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        # hbox2.addStretch()
        # hbox2.addWidget(label3)
        # hbox2.addStretch()
        # hbox2.addWidget(label4)
        # hbox2.addStretch()
        #
        # hbox3: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        # hbox3.addStretch()
        # hbox3.addWidget(label5)
        # hbox3.addStretch()
        # hbox3.addWidget(label6)
        # hbox3.addStretch()

        # Set layout
        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.chart)
        # vbox.addLayout(hbox1)
        # vbox.addLayout(hbox2)
        # vbox.addLayout(hbox3)
        self.setLayout(vbox)

    def save_history_data(self, history: list, history_w: list):
        self.history_data[Interval.DAILY] = history
        self.history_data[Interval.WEEKLY] = history_w

    def save_trades(self, trades: list):
        self.trades = trades

    def change_period(self, interval: Interval):
        self.clear_data()

        self.interval = interval
        self.update_history(self.history_data[interval])
        self.update_trades(self.trades)

    def update_history(self, history: list) -> None:
        """"""
        self.updated = True
        self.chart.update_history(history)

        for ix, bar in enumerate(history):
            dt_ix = bar.datetime
            if self.interval == Interval.WEEKLY:
                dt_ix = dt_ix + timedelta(days=4 - dt_ix.weekday())
            self.ix_bar_map[ix] = bar
            # self.dt_ix_map[bar.datetime] = ix
            self.dt_ix_map[dt_ix] = ix

            if not self.high_price:
                self.high_price = bar.high_price
                self.low_price = bar.low_price
            else:
                self.high_price = max(self.high_price, bar.high_price)
                self.low_price = min(self.low_price, bar.low_price)

        self.price_range = self.high_price - self.low_price

    def update_trades(self, trades: list) -> None:
        """"""
        trade_pairs: list = generate_trade_pairs(trades, self.interval)

        candle_plot: pg.PlotItem = self.chart.get_plot("candle")

        scatter_data: list = []

        y_adjustment: float = self.price_range * 0.001

        for d in trade_pairs:
            open_ix = self.dt_ix_map[d["open_dt"]]
            close_ix = self.dt_ix_map[d["close_dt"]]
            open_price = d["open_price"]
            close_price = d["close_price"]

            # Trade Line
            x: list = [open_ix, close_ix]
            y: list = [open_price, close_price]

            if d["direction"] == Direction.LONG and close_price >= open_price:
                color: str = "r"
            elif d["direction"] == Direction.SHORT and close_price <= open_price:
                color: str = "r"
            else:
                color: str = "g"

            pen: QtGui.QPen = pg.mkPen(color, width=1.5, style=QtCore.Qt.DashLine)
            item: pg.PlotCurveItem = pg.PlotCurveItem(x, y, pen=pen)

            self.items.append(item)
            candle_plot.addItem(item)

            # Trade Scatter
            open_bar: BarData = self.ix_bar_map[open_ix]
            close_bar: BarData = self.ix_bar_map[close_ix]

            if d["direction"] == Direction.LONG:
                scatter_color: str = "yellow"
                open_symbol: str = "t1"
                close_symbol: str = "t"
                open_side: int = 1
                close_side: int = -1
                open_y: float = open_bar.low_price
                close_y: float = close_bar.high_price
            else:
                scatter_color: str = "magenta"
                open_symbol: str = "t"
                close_symbol: str = "t1"
                open_side: int = -1
                close_side: int = 1
                open_y: float = open_bar.high_price
                close_y: float = close_bar.low_price
            #
            # pen = pg.mkPen(QtGui.QColor(scatter_color))
            # brush: QtGui.QBrush = pg.mkBrush(QtGui.QColor(scatter_color))
            # size: int = 10
            #
            # open_scatter: dict = {
            #     "pos": (open_ix, open_y - open_side * y_adjustment),
            #     "size": size,
            #     "pen": pen,
            #     "brush": brush,
            #     "symbol": open_symbol
            # }
            #
            # close_scatter: dict = {
            #     "pos": (close_ix, close_y - close_side * y_adjustment),
            #     "size": size,
            #     "pen": pen,
            #     "brush": brush,
            #     "symbol": close_symbol
            # }
            #
            # scatter_data.append(open_scatter)
            # scatter_data.append(close_scatter)

            # Trade text
            volume = d["volume"]
            text_color: QtGui.QColor = QtGui.QColor(scatter_color)
            open_text: pg.TextItem = pg.TextItem(f"[{volume}]", color=text_color, anchor=(0.5, 0.5))
            close_text: pg.TextItem = pg.TextItem(f"[{volume}]", color=text_color, anchor=(0.5, 0.5))

            open_text.setPos(open_ix, open_y - open_side * y_adjustment * 3)
            close_text.setPos(close_ix, close_y - close_side * y_adjustment * 3)

            self.items.append(open_text)
            self.items.append(close_text)

            candle_plot.addItem(open_text)
            candle_plot.addItem(close_text)

        # trade_scatter: pg.ScatterPlotItem = pg.ScatterPlotItem(scatter_data)
        # self.items.append(trade_scatter)
        # candle_plot.addItem(trade_scatter)

    def clear_data(self) -> None:
        """"""
        self.updated = False

        candle_plot: pg.PlotItem = self.chart.get_plot("candle")
        for item in self.items:
            candle_plot.removeItem(item)
        self.items.clear()

        self.chart.clear_all()

        self.dt_ix_map.clear()
        self.ix_bar_map.clear()

    def is_updated(self) -> bool:
        """"""
        return self.updated


def generate_trade_pairs(trades: list, interval: Interval) -> list:
    """"""
    long_trades: list = []
    short_trades: list = []
    trade_pairs: list = []

    for trade in trades:
        trade: TradeData = copy(trade)

        if trade.direction == Direction.LONG:
            same_direction: list = long_trades
            opposite_direction: list = short_trades
        else:
            same_direction: list = short_trades
            opposite_direction: list = long_trades

        while trade.volume and opposite_direction:
            open_trade: TradeData = opposite_direction[0]
            open_dt = open_trade.datetime
            close_dt = trade.datetime
            if interval == Interval.WEEKLY:
                open_dt = open_dt + timedelta(days=4 - open_dt.weekday())
                close_dt = close_dt + timedelta(days=4 - close_dt.weekday())

            close_volume = min(open_trade.volume, trade.volume)
            d: dict = {
                "open_dt": open_dt,
                "open_price": open_trade.price,
                "close_dt": close_dt,
                "close_price": trade.price,
                "direction": open_trade.direction,
                "volume": close_volume,
            }
            trade_pairs.append(d)

            open_trade.volume -= close_volume
            if not open_trade.volume:
                opposite_direction.pop(0)

            trade.volume -= close_volume

        if trade.volume:
            same_direction.append(trade)

    return trade_pairs


class DatetimeParameterItem(WidgetParameterItem):
    def makeWidget(self):
        # self.asSubItem = True
        w = QtWidgets.QDateTimeEdit()
        w.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        # w.setMaximumHeight(200)
        # w.sigChanged = w.dateChanged
        w.sigChanged = w.dateTimeChanged
        # w.value = w.date
        w.value = w.dateTime
        # w.setValue = w.setDate
        w.setValue = w.setDateTime
        # self.hideWidget = False
        # self.param.opts.setdefault('default', QtCore.QDate.currentDate())
        self.param.opts.setdefault('default', QtCore.QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss"))
        return w

    def updateDisplayLabel(self, value=None):
        """Update the display label to reflect the value of the parameter."""
        if value is None:
            value = self.param.value()
        self.displayLabel.setText(value.toString("yyyy-MM-dd hh:mm:ss"))
