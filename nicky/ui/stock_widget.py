
import sys
import threading
import datetime
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')
import queue

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from nicky.data.stocks.sohu import sohu
from nicky.ui.widgets import MainPlot, ReturnPlot, HeatmapPlot, AboutDialog, IndicatorPlot


__all__ = ['StockWidget', ]


class StockWidget(QtGui.QWidget):

    def __init__(self):
        super(StockWidget, self).__init__()
        self.initData()
        self.initUI()
        self.stock_indexes_get()

    def initData(self):
        self.code = ''
        self.ktitle = ''
        self.main_data = None
        self.data_type = 'stock'

        self.history = []
        self.now_position = -1

        self.ktype = 'D'
        self.start_date = '2015-01-01'
        #self.end_date = '2018-01-01'
        self.end_date = datetime.date.today().strftime('%Y-%m-%d')
        self.now_date = QtCore.QDate.fromString(self.end_date, 'yyyy-MM-dd')

        self.stock_indexes_list = self.get_data(type='indexes')
        self.stock_indexes_code_list = self.stock_indexes_list.index
        self.stock_indexes_name_list = self.stock_indexes_list['name']

        self.stock_basics_list = self.get_data(type='stocks')
        self.stock_code_list = self.stock_basics_list.index
        self.stock_name_list = self.stock_basics_list[['name']]
        self.stock_info_list = self.stock_basics_list[["industry", "area"]]

        self.industry_list = list(self.get_data(type='industries').keys())

        self.concept_list = list(self.get_data(type='concepts').keys())

        self.area_list = list(self.get_data(type='areas').keys())


    def initUI(self):
        self.setMouseTracking(True)

        self.initLayout()
        self.initToolBars()
        self.initPlotLayout()
        self.initPlots()

    def initLayout(self):
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)

    def initToolBars(self):
        self.initKlineBar()
        self.initTimeSelectionBar()
        self.initAnalysisBar()
        self.initForecastBar()

    def initPlotLayout(self):
        self.plot_layout = QtGui.QStackedLayout()
        self.layout.addLayout(self.plot_layout, self.layout.count(), 0)

    def initPlots(self):

        self.candlestick_plot = MainPlot()
        self.return_plot = ReturnPlot()
        self.heatmap_plot = HeatmapPlot()

        self.plot_layout.addWidget(self.candlestick_plot)
        self.plot_layout.addWidget(self.return_plot)
        self.plot_layout.addWidget(self.heatmap_plot)
        self.indicator_plots = {}

    def loadData(self, data, title, data_type='ohlcv'):
        if data_type == 'ohlcv':
            self.candlestick_plot.loadData(data, title)
            self.plot_layout.setCurrentIndex(0)
        elif data_type == 'return':
            self.return_plot.loadData(data, title)
            self.plot_layout.setCurrentIndex(1)
        elif data_type == 'cluster':
            self.heatmap_plot.loadData(data, title)
            self.plot_layout.setCurrentIndex(2)
        elif data_type.startswith('indicator'):
            indicator_type = data_type.split(':')[1]
            if indicator_type not in self.indicator_plots:
                current_index = self.plot_layout.count()
                indicator_plot = IndicatorPlot(indicator_type)
                indicator_plot.loadData(data, title)
                self.plot_layout.addWidget(indicator_plot)
                self.plot_layout.setCurrentIndex(current_index)
                self.indicator_plots[indicator_type] = current_index
            else:
                self.plot_layout.setCurrentIndex(self.indicator_plots[indicator_type])
                self.plot_layout.currentWidget().loadData(data, title)

        else:
            pass

    def initKlineBar(self):
        self.klinebar = QtGui.QToolBar("stock select bar.")
        self.layout.addWidget(self.klinebar)

        self.klinebar.setAutoFillBackground(True)

        stock_indexes_label = QtGui.QLabel("指数:")
        self.klinebar.addWidget(stock_indexes_label)

        # update k line according to stock index name
        stock_indexes_name_box = QtGui.QComboBox(self)
        stock_indexes_name_box.setEditable(True)
        stock_indexes_name_box.setMinimumContentsLength(8)
        stock_indexes_name_box.activated.connect(self.stock_indexes_get)
        for i in self.stock_indexes_name_list:
            stock_indexes_name_box.addItem(i)
        self.klinebar.addWidget(stock_indexes_name_box)

        # update k line according to stock index code
        stock_indexes_code_box = QtGui.QComboBox(self)
        stock_indexes_code_box.setEditable(True)
        stock_indexes_code_box.setMinimumContentsLength(8)
        stock_indexes_code_box.activated.connect(self.stock_indexes_get)
        for i in self.stock_indexes_code_list:
            stock_indexes_code_box.addItem(i)
        self.klinebar.addWidget(stock_indexes_code_box)

        self.klinebar.addSeparator()

        stock_basic_label = QtGui.QLabel("个股：")
        self.klinebar.addWidget(stock_basic_label)

        # update k line according to stock name
        stock_basic_name_box = QtGui.QComboBox(self)
        stock_basic_name_box.setEditable(True)
        stock_basic_name_box.setMinimumContentsLength(8)
        stock_basic_name_box.activated.connect(self.stock_basic_get)
        for i in self.stock_name_list['name']:
            stock_basic_name_box.addItem(i)
        self.klinebar.addWidget(stock_basic_name_box)

        # update k line according to stock code
        stock_basic_code_box = QtGui.QComboBox(self)
        stock_basic_code_box.setEditable(True)
        stock_basic_code_box.setMinimumContentsLength(8)
        stock_basic_code_box.activated.connect(self.stock_basic_get)
        for i in self.stock_code_list:
            stock_basic_code_box.addItem(i)
        self.klinebar.addWidget(stock_basic_code_box)

        self.klinebar.addSeparator()
        #'''
        # industry_mean
        industry_label = QtGui.QLabel("行业：")
        self.klinebar.addWidget(industry_label)

        industry_box = QtGui.QComboBox(self)
        industry_box.setEditable(True)
        industry_box.setMinimumContentsLength(8)
        industry_box.activated.connect(self.industry_changed)
        for i in self.industry_list:
            industry_box.addItem(i)
        self.klinebar.addWidget(industry_box)

        self.klinebar.addSeparator()

        # concept_mean
        concept_label = QtGui.QLabel("概念：")
        self.klinebar.addWidget(concept_label)

        concept_box = QtGui.QComboBox(self)
        concept_box.setEditable(True)
        concept_box.setMinimumContentsLength(8)
        concept_box.activated.connect(self.concept_changed)
        for i in self.concept_list:
            concept_box.addItem(i)
        self.klinebar.addWidget(concept_box)

        self.klinebar.addSeparator()

        # area_mean
        area_label = QtGui.QLabel("地区：")
        self.klinebar.addWidget(area_label)

        area_box = QtGui.QComboBox(self)
        area_box.setEditable(True)
        area_box.setMinimumContentsLength(4)
        area_box.activated.connect(self.area_changed)
        for i in self.area_list:
            area_box.addItem(i)
        self.klinebar.addWidget(area_box)
        #'''

        self.klinebar.addSeparator()

        self.refreshCandlestickBar()

        updateDbAction = QtGui.QAction("Update All", self)
        updateDbAction.triggered.connect(self.update_db)
        self.klinebar.addAction(updateDbAction)

        self.klinebar.addSeparator()

    def refreshCandlestickBar(self):

        closest_list = sohu.get_closest_list(self.code)

        if hasattr(self, 'closest_label'):
            self.klinebar.removeAction(self.closest_label)
            del self.closest_label
        if hasattr(self, 'closest_box'):
            self.klinebar.removeAction(self.closest_box)
            del self.closest_box
        if hasattr(self, 'closest_result'):
            self.klinebar.removeAction(self.closest_result)
            del self.closest_result

        self.closest_label = self.klinebar.addWidget(QtGui.QLabel("十支最相似股票："))

        if closest_list is not None:
            self.closest_box = QtGui.QComboBox(self)

            closest_list = closest_list[1:11]
            zs_or_cn = self.code[:2]
            basics = sohu.get_stock_basics() if zs_or_cn == 'cn' else sohu.get_index_basics()
            names = [basics.loc[i[-6:]]['name'] for i in closest_list]

            self.closest_box.activated.connect(lambda offset: self.closest_data_load(closest_list[offset], names[offset]))

            for i in names:
                self.closest_box.addItem(i)
            self.closest_box = self.klinebar.addWidget(self.closest_box)
        else:
            self.closest_result = self.klinebar.addWidget(QtGui.QLabel("无数据"))

    def initTimeSelectionBar(self):
        self.timeselectionbar = QtGui.QToolBar("select time range.")
        self.layout.addWidget(self.timeselectionbar)

        self.timeselectionbar.setAutoFillBackground(True)

        changeDateRange = QtGui.QAction( # QtGui.QIcon("icons/bold.png"),
            "起止日期", self)
        changeDateRange.triggered.connect(self.toggleDatetimeBar)

        self.timeselectionbar.addAction(changeDateRange)

        changeDateStart1M = QtGui.QAction("近一月", self)
        changeDateStart1M.triggered.connect(lambda : self.start_date_changed(self.now_date.addMonths(-1)))

        changeDateStart3M = QtGui.QAction("近三月", self)
        changeDateStart3M.triggered.connect(lambda : self.start_date_changed(self.now_date.addMonths(-3)))

        changeDateStart6M = QtGui.QAction("近半年", self)
        changeDateStart6M.triggered.connect(lambda : self.start_date_changed(self.now_date.addMonths(-6)))

        changeDateStart1Y = QtGui.QAction("近一年", self)
        changeDateStart1Y.triggered.connect(lambda : self.start_date_changed(self.now_date.addYears(-1)))

        changeDateStart3Y = QtGui.QAction("近三年", self)
        changeDateStart3Y.triggered.connect(lambda : self.start_date_changed(self.now_date.addYears(-3)))

        changeDateStart5Y = QtGui.QAction("近五年", self)
        changeDateStart5Y.triggered.connect(lambda : self.start_date_changed(self.now_date.addYears(-5)))

        changeDateStart10Y = QtGui.QAction("近十年", self)
        changeDateStart10Y.triggered.connect(lambda : self.start_date_changed(self.now_date.addYears(-10)))

        changeDateStart15Y = QtGui.QAction("十五年", self)
        changeDateStart15Y.triggered.connect(lambda : self.start_date_changed(self.now_date.addYears(-15)))

        changeDateStart20Y = QtGui.QAction("二十年", self)
        changeDateStart20Y.triggered.connect(lambda : self.start_date_changed(self.now_date.addYears(-20)))

        changeDateStartALL = QtGui.QAction("全部数据", self)
        changeDateStartALL.triggered.connect(lambda : self.start_date_changed(self.now_date.addYears(-50)))

        self.timeselectionbar.addAction(changeDateStart1M)
        self.timeselectionbar.addAction(changeDateStart3M)
        self.timeselectionbar.addAction(changeDateStart6M)
        self.timeselectionbar.addAction(changeDateStart1Y)
        self.timeselectionbar.addAction(changeDateStart3Y)
        self.timeselectionbar.addAction(changeDateStart5Y)
        self.timeselectionbar.addAction(changeDateStart10Y)
        self.timeselectionbar.addAction(changeDateStart15Y)
        self.timeselectionbar.addAction(changeDateStart20Y)
        self.timeselectionbar.addAction(changeDateStartALL)

        self.datetimebar = QtGui.QToolBar("set stock date time")
        self.layout.addWidget(self.datetimebar)
        #self.datetimebar.setOrientation(QtCore.Qt.Vertical)

        self.datetimebar.hide()

        #stock_basic_label = QtGui.QLabel("起止日期：")
        #self.datetimebar.addWidget(stock_basic_label)

        start_date = QtGui.QCalendarWidget(self)
        start_date.setGridVisible(True)
        start_date.setFirstDayOfWeek(QtCore.Qt.Monday)
        start_date.clicked[QtCore.QDate].connect(self.start_date_changed)

        end_date = QtGui.QCalendarWidget(self)
        end_date.setGridVisible(True)
        end_date.setFirstDayOfWeek(QtCore.Qt.Monday)
        end_date.clicked[QtCore.QDate].connect(self.end_date_changed)

        self.datetimebar.addWidget(start_date)
        self.datetimebar.addWidget(end_date)

    def update_db(self):
        self.update_db_progress_bar = QtGui.QProgressBar(self)
        self.layout.addWidget(self.update_db_progress_bar)
        nothing = QtGui.QProgressBar(self)
        self.layout.addWidget(nothing)
        nothing.hide()

        def update_progress_bar(progress_queue):
            value = progress_queue.get()
            while value + 0.1 < 100:
                self.update_db_progress_bar.setValue(value)
                value = progress_queue.get()
            self.layout.removeWidget(self.update_db_progress_bar)

        progress_queue = queue.Queue()
        update_db_thread = threading.Thread(target=sohu.update_db, args=(progress_queue, ))
        update_progress_bar_thread = threading.Thread(target=update_progress_bar, args=(progress_queue, ))
        update_db_thread.daemon = True
        update_db_thread.start()
        update_progress_bar_thread.start()

    def initAnalysisBar(self):
        self.analysisbar = QtGui.QToolBar("indicators of stock.")
        self.layout.addWidget(self.analysisbar)

        self.analysisbar.setAutoFillBackground(True)

        #self.analysisbar.hide()

        ohlcv_return_action = QtGui.QAction("Return", self)
        ohlcv_return_action.triggered.connect(self.plot_return)

        self.analysisbar.addAction(ohlcv_return_action)
        self.analysisbar.addSeparator()

        ohlcv_MACD_action = QtGui.QAction("MACD", self)
        ohlcv_MACD_action.triggered.connect(lambda : self.plot_indicator(indicator_type='MACD'))

        self.analysisbar.addAction(ohlcv_MACD_action)
        self.analysisbar.addSeparator()

        ohlcv_CCI_action = QtGui.QAction("CCI", self)
        ohlcv_CCI_action.triggered.connect(lambda : self.plot_indicator(indicator_type='CCI'))

        self.analysisbar.addAction(ohlcv_CCI_action)
        self.analysisbar.addSeparator()

        ohlcv_ATR_action = QtGui.QAction("ATR", self)
        ohlcv_ATR_action.triggered.connect(lambda : self.plot_indicator(indicator_type='ATR'))

        self.analysisbar.addAction(ohlcv_ATR_action)
        self.analysisbar.addSeparator()

        ohlcv_MA_action = QtGui.QAction("MA", self)
        ohlcv_MA_action.triggered.connect(lambda : self.plot_indicator(indicator_type='MA'))

        self.analysisbar.addAction(ohlcv_MA_action)
        self.analysisbar.addSeparator()

        ohlcv_STD_action = QtGui.QAction("STD", self)
        ohlcv_STD_action.triggered.connect(lambda : self.plot_indicator(indicator_type='STD'))

        self.analysisbar.addAction(ohlcv_STD_action)
        self.analysisbar.addSeparator()

        ohlcv_upper_action = QtGui.QAction("Upper Band", self)
        ohlcv_upper_action.triggered.connect(lambda : self.plot_indicator(indicator_type='UBAND'))

        self.analysisbar.addAction(ohlcv_upper_action)
        self.analysisbar.addSeparator()

        ohlcv_lower_action = QtGui.QAction("Lower Band", self)
        ohlcv_lower_action.triggered.connect(lambda : self.plot_indicator(indicator_type='LBAND'))

        self.analysisbar.addAction(ohlcv_lower_action)
        self.analysisbar.addSeparator()

        ohlcv_EWMA_action = QtGui.QAction("EWMA", self)
        ohlcv_EWMA_action.triggered.connect(lambda : self.plot_indicator(indicator_type='EWMA'))

        self.analysisbar.addAction(ohlcv_EWMA_action)
        self.analysisbar.addSeparator()

        ohlcv_MA5_action = QtGui.QAction("MA5", self)
        ohlcv_MA5_action.triggered.connect(lambda : self.plot_indicator(indicator_type='MA5'))

        self.analysisbar.addAction(ohlcv_MA5_action)
        self.analysisbar.addSeparator()

        ohlcv_MA10_action = QtGui.QAction("MA10", self)
        ohlcv_MA10_action.triggered.connect(lambda : self.plot_indicator(indicator_type='MA10'))

        self.analysisbar.addAction(ohlcv_MA10_action)
        self.analysisbar.addSeparator()

        ohlcv_STOK_action = QtGui.QAction("STOK", self)
        ohlcv_STOK_action.triggered.connect(lambda : self.plot_indicator(indicator_type='STOK'))

        self.analysisbar.addAction(ohlcv_STOK_action)
        self.analysisbar.addSeparator()

        ohlcv_STOD_action = QtGui.QAction("STOD", self)
        ohlcv_STOD_action.triggered.connect(lambda : self.plot_indicator(indicator_type='STOD'))

        self.analysisbar.addAction(ohlcv_STOD_action)
        self.analysisbar.addSeparator()

        ohlcv_WVAD_action = QtGui.QAction("WVAD", self)
        ohlcv_WVAD_action.triggered.connect(lambda : self.plot_indicator(indicator_type='WVAD'))

        self.analysisbar.addAction(ohlcv_WVAD_action)
        self.analysisbar.addSeparator()

        ohlcv_MTM6_action = QtGui.QAction("MTM6", self)
        ohlcv_MTM6_action.triggered.connect(lambda : self.plot_indicator(indicator_type='MTM6'))

        self.analysisbar.addAction(ohlcv_MTM6_action)
        self.analysisbar.addSeparator()

        ohlcv_MTM12_action = QtGui.QAction("MTM12", self)
        ohlcv_MTM12_action.triggered.connect(lambda : self.plot_indicator(indicator_type='MTM12'))

        self.analysisbar.addAction(ohlcv_MTM12_action)
        self.analysisbar.addSeparator()

        show_correlation_heatmaps_action = QtGui.QAction('correlation heatmap', self)
        show_correlation_heatmaps_action.triggered.connect(self.toggleDistanceHeatmapBar)

        self.analysisbar.addAction(show_correlation_heatmaps_action)
        self.analysisbar.addSeparator()

        self.correlation_heatmap_bar = QtGui.QToolBar('draw correlation (correlation) heatmap between stocks')
        self.layout.addWidget(self.correlation_heatmap_bar)
        self.correlation_heatmap_bar.setAutoFillBackground(True)
        self.correlation_heatmap_bar.hide()

        show_price_correlation_action = QtGui.QAction("price correlation", self)
        show_price_correlation_action.triggered.connect(self.plot_price_correlation_heatmap)

        self.correlation_heatmap_bar.addAction(show_price_correlation_action)

        show_volatility_correlation_action = QtGui.QAction("volatility correlation", self)
        show_volatility_correlation_action.triggered.connect(self.plot_volatility_correlation_heatmap)

        self.correlation_heatmap_bar.addAction(show_volatility_correlation_action)

        show_simple_return_correlation_action = QtGui.QAction("simple return correlation", self)
        show_simple_return_correlation_action.triggered.connect(self.plot_simple_return_correlation_heatmap)

        self.correlation_heatmap_bar.addAction(show_simple_return_correlation_action)

        show_log_return_correlation_action = QtGui.QAction("log return correlation", self)
        show_log_return_correlation_action.triggered.connect(self.plot_log_return_correlation_heatmap)

        self.correlation_heatmap_bar.addAction(show_log_return_correlation_action)

    def initForecastBar(self):
        self.forecastbar = QtGui.QToolBar("forecast with models.")
        self.layout.addWidget(self.forecastbar)

        self.forecastbar.setAutoFillBackground(True)

        self.forecastbar.hide()

        arima_action = QtGui.QAction('ARIMA', self)
        arima_action.triggered.connect(self.forecast_arima)

        #arimax_action = QtGui.QAction('ARIMAX', self)
        #arimax_action.triggered.connect(self.forecast_arimax)

        dar_action = QtGui.QAction('DAR', self)
        dar_action.triggered.connect(self.forecast_dar)

        self.forecastbar.addAction(arima_action)
        #self.forecastbar.addAction(arimax_action)
        self.forecastbar.addAction(dar_action)

        self.forecastbar.addSeparator()

    def closest_data_load(self, code, title):
        data = self.get_data(code)
        self.load_data(data, title)

    def forecast_arima(self):
        kw = {'ar':4, 'ma':4, 'integ':0}
        predict, fit_info = sohu.get_data_forecasting(self.main_data, target='close', method='ARIMA', forward_step=5, fit_method='MLE', **kw)
        print(predict.tail(), fit_info.summary())

    def forecast_dar(self):
        kw = {'ar':4, 'integ':0}
        predict, fit_info = sohu.get_data_forecasting(self.main_data, target='close', method='DAR', forward_step=5, fit_method='MLE', **kw)
        print(predict.tail(), fit_info.summary())

    def plot_price_correlation_heatmap(self):
        self.plot_correlation_heatmap('close')

    def plot_volatility_correlation_heatmap(self):
        self.plot_correlation_heatmap('volatility')

    def plot_simple_return_correlation_heatmap(self):
        self.plot_correlation_heatmap('simple return')

    def plot_log_return_correlation_heatmap(self):
        self.plot_correlation_heatmap('log return')

    def plot_return(self):
        self.load_data(self.main_data, self.ktitle, 'return')

    def plot_indicator(self, indicator_type):
        self.load_data(self.main_data, self.ktitle, 'indicator:%s'%indicator_type)

    def plot_correlation_heatmap(self, type='close'):

        if type == 'close':
            zs_or_cn = self.code[:2]
            data = sohu.get_dist_matrix(zs_or_cn)
            self.load_data(data, '', 'cluster')
        else:
            pass

    def toggleDatetimeBar(self):
        state = self.datetimebar.isVisible()
        self.datetimebar.setVisible(not state)

    def toggleDistanceHeatmapBar(self):
        state = self.correlation_heatmap_bar.isVisible()
        self.correlation_heatmap_bar.setVisible(not state)

    def stock_indexes_get(self, offset=None):
        if offset is None:
            code = 'zs_000001'
            kline_title = '大盘指数： %s' % '上证指数'
        else:
            code = 'zs_%s' % self.stock_indexes_code_list[offset]
            kline_title = "大盘指数: %s" % self.stock_indexes_name_list[offset]
        self.code = code
        self.data_type = 'stock'
        data = self.get_data(code)
        self.main_data = data
        self.ktitle = kline_title
        self.load_data(data, kline_title)

    def stock_basic_get(self, offset):
        code = self.stock_code_list[offset]
        name = self.stock_name_list['name'][code]
        info = self.stock_info_list.ix[code]
        code = 'cn_%s' % code
        self.code = code
        self.data_type = 'stock'
        data = self.get_data(code)
        kline_title = '''
                <span style="color: white; font-size: 30px;">股票名称: %s</span><br>
                <span style="color: white; font-size: 30px;">股票代码: %s</span><br>
                <span style="color: white; font-size: 30px;">行  业: %s</span><br>
                <span style="color: white; font-size: 30px;">地  区: %s</span><br>
                ''' % (name, code[-6:], info['industry'], info['area'])
        print(info)
        self.main_data = data
        self.ktitle = kline_title
        self.load_data(data, kline_title)

    def industry_changed(self, offset):
        self.code = self.industry_list[offset]
        self.data_type = 'industry'
        data = self.get_data(self.code, type=self.data_type)
        self.main_data = data
        self.ktitle = '行业名称: %s' % self.code
        self.load_data(data, self.ktitle)

    def area_changed(self, offset):
        self.code = self.area_list[offset]
        self.data_type = 'area'
        data = self.get_data(self.code, type=self.data_type)
        self.main_data = data
        self.ktitle = '地域: %s' % self.code
        self.load_data(data, self.ktitle)

    def concept_changed(self, offset):
        self.code = self.concept_list[offset]
        self.data_type = 'concept'
        data = self.get_data(self.code, type=self.data_type)
        self.main_data = data
        self.ktitle = '概念: %s' % self.code
        self.load_data(data, self.ktitle)

    def start_date_changed(self, datetime):
        self.start_date = datetime.toPyDate().strftime("%Y-%m-%d")
        data = self.get_data(self.code, type=self.data_type)
        self.main_data = data
        self.load_data(data, self.ktitle)

    def end_date_changed(self, datetime):
        self.end_date = datetime.toPyDate().strftime("%Y-%m-%d")
        data = self.get_data(self.code, type=self.data_type)
        self.main_data = data
        self.load_data(data, self.ktitle)

    def get_data(self, code='zs_000001', type='stock'):
        if type == 'stock':
            data = sohu.get_hist_data(code, start=self.start_date, end=self.end_date, ktype=self.ktype, indicators=True)
        elif type == 'industry':
            data = sohu.get_industry_mean(code, start=self.start_date, end=self.end_date)
        elif type == 'area':
            data = sohu.get_area_mean(code, start=self.start_date, end=self.end_date)
        elif type == 'concept':
            data = sohu.get_concept_mean(code, start=self.start_date, end=self.end_date)
        elif type == 'stocks':
            data = sohu.get_stock_basics()
        elif type == 'indexes':
            data = sohu.get_index_basics()
        elif type == 'industries':
            data = sohu.get_industry_basics()
        elif type == 'areas':
            data = sohu.get_area_basics()
        elif type == 'concepts':
            data = sohu.get_concept_basics()
        else:
            data = None
        return data

    def load_data(self, data, kline_title, data_type='ohlcv'):
        #self.main_data = data
        self.loadData(data, kline_title, data_type)
        if data_type == 'ohlcv' and hasattr(self, 'klinebar'):
            self.refreshCandlestickBar()

