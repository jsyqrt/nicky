# -*- coding: utf-8 -*-

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pandas as pd
import numpy as np

__all__ = ['MainPlot', 'ReturnPlot', 'HeatmapPlot', 'AboutDialog', 'IndicatorPlot']


class MyStringAxis(pg.AxisItem):
    """时间序列横坐标支持"""

    # 初始化
    def __init__(self, xdict, *args, **kwargs):
        pg.AxisItem.__init__(self, *args, **kwargs)
        self.minVal = 0
        self.maxVal = 0
        self.xdict = xdict
        self.x_values = np.asarray(list(xdict.keys()))
        self.x_strings = list(xdict.values())
        self.setPen(color=(255, 255, 255, 255), width=0.8)
        self.setStyle(
            tickFont=QtGui.QFont("Roman times", 10, QtGui.QFont.Bold),
            autoExpandTextSpace=True)

    # 更新坐标映射表
    def update_xdict(self, xdict):
        self.xdict.update(xdict)
        self.x_values = np.asarray(list(self.xdict.keys()))
        self.x_strings = list(self.xdict.values())

    # 将原始横坐标转换为时间字符串,第一个坐标包含日期
    def tickStrings(self, values, scale, spacing):
        strings = []
        for v in values:
            vs = v * scale
            if vs in self.x_values:
                vstr = self.x_strings[np.abs(self.x_values - vs).argmin()]
                vstr = vstr.strftime('%Y-%m-%d') # %H:%M:%S')
            else:
                vstr = ""
            strings.append(vstr)
        return strings


class MyCodeAxis(pg.AxisItem):

    def __init__(self, xdict, *args, **kwargs):
        pg.AxisItem.__init__(self, *args, **kwargs)
        self.minVal = 0
        self.maxVal = 0
        self.xdict = xdict
        self.x_values = np.asarray(list(xdict.keys()))
        self.x_strings = list(xdict.values())
        self.setPen(color=(255, 255, 255, 255), width=0.8)
        self.setStyle(
            tickFont=QtGui.QFont("Roman times", 6, QtGui.QFont.Bold),
            autoExpandTextSpace=True)

    def update_xdict(self, xdict):
        self.xdict.update(xdict)
        self.x_values = np.asarray(list(self.xdict.keys()))
        self.x_strings = list(self.xdict.values())

    def tickStrings(self, values, scale, spacing):
        strings = []
        for v in values:
            vs = v * scale
            if vs in self.x_values:
                vstr = self.x_strings[np.abs(self.x_values - vs).argmin()]
                #vstr = vstr.strftime('%Y-%m-%d %H:%M:%S')
            else:
                vstr = ""
            strings.append(vstr)
        return strings


class CandlestickItem(pg.GraphicsObject):
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.generatePicture(self.data)

    def generatePicture(self, data, redrew=True):
        self.data = data
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        #w = (self.data[1]['low'] - self.data[0]['high']) / 3. \
            #if len(self.data) > 0 else (0, 1)
        w = 0.4
        for (t, open, close, min, max) in self.data:
            if open > close:
                p.setPen(pg.mkPen('g'))
                p.setBrush(pg.mkBrush('g'))
            else:
                p.setPen(pg.mkPen('r'))
                p.setBrush(pg.mkBrush('r'))
            p.drawLine(QtCore.QPointF(t, min), QtCore.QPointF(t, max))
            p.drawRect(QtCore.QRectF(t - w, open, w * 2, close - open))
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class CrossHair(QtCore.QObject):

    def __init__(self, plot_items, layout):
        self.plot_items = plot_items
        self.layout = layout

        super(CrossHair, self).__init__()

        self.x = 0
        self.y = 0

        self.data = None

        self.vhs = []
        for i in range(len(self.plot_items)):
            vh = [pg.InfiniteLine(angle=90, movable=False), pg.InfiniteLine(angle=0, movable=False)]
            self.vhs.append(vh)
            self.plot_items[i].addItem(self.vhs[i][0], ignoreBounds=True)
            self.plot_items[i].addItem(self.vhs[i][1], ignoreBounds=True)

        self.title = None
        self.title_label = pg.LabelItem()
        self.layout.addItem(self.title_label, row=0, col=0, rowspan=1)

        self.label = pg.LabelItem()
        self.layout.addItem(self.label, row=1, col=0, rowspan=2)

        self.proxy = [pg.SignalProxy(self.plot_items[i].scene().sigMouseMoved, rateLimit=60, slot=lambda evt, i=i: self.mouseMoved(evt, i=i)) for i in range(len(self.plot_items))]

    def update(self, data, title):
        self.data = data
        self.title = title
        self.title_label.setText('''
            <div style="text-align: center; background-color:#000">
            <span style="color: red; font-size: 16px;">%s</span><br>''' % self.title
        )

    def mouseMoved(self, evt, i=0):
        pos = evt[0]
        if self.plot_items[i].sceneBoundingRect().contains(pos):
            self.vhs[i][0].show()
            self.vhs[i][1].show()
            self.x = self.plot_items[i].vb.mapSceneToView(pos).x()
            self.y = self.plot_items[i].vb.mapSceneToView(pos).y()

            if round(self.x) >= 0 and round(self.x) <= (self.data.shape[0] - 1):
                self.info = self.data.iloc[round(self.x)]

                d = self.info

                self.label.setText(
                '''<span style="color: white; font-size: 16px;">年月日：%s</span><br>
                <span style="color: white; font-size: 16px;">开盘价：%.3f</span><br>
                <span style="color: white; font-size: 16px;">收盘价：%.3f</span><br>
                <span style="color: white; font-size: 16px;">最高价：%.3f</span><br>
                <span style="color: white; font-size: 16px;">最低价：%.3f</span><br>
                <span style="color: white; font-size: 16px;">成交量：%.1f</span><br>
                <span style="color: white; font-size: 16px;">波动量：%.3f</span><br>
                <span style="color: white; font-size: 16px;">简单回报率：</span><br>
                <span style="color: white; font-size: 16px;">%.6f</span><br>
                <span style="color: white; font-size: 16px;">对数回报率：</span><br>
                <span style="color: white; font-size: 16px;">%.6f</span><br>
                </div>'''
                % (list(self.data.index)[round(self.x)].date(),
                    d['open'], d['close'], d['high'], d['low'], d['volume'],
                    d['volatility'], d['simple return'], d['log return']
                    ))

            #self.vhs[i][0].setPos(self.x)
            self.vhs[i][1].setPos(self.y)

            for i in range(len(self.vhs)):
                self.vhs[i][0].setPos(self.x)
        else:
            #self.vhs[i][0].hide()
            self.vhs[i][1].hide()


class MainPlot(pg.PlotWidget):
    def __init__(self):
        super(MainPlot, self).__init__()

        self.initData()
        self.initUI()

    def initData(self):
        self.candlestick_plotitem = None
        self.volume_plotitem = None
        self.simple_return_plotitem = None
        self.log_return_plotitem = None

    def initUI(self):
        self.layout = pg.GraphicsLayout()
        self.setCentralItem(self.layout)

        xdict = {}
        self.axisTimes = MyStringAxis(xdict, orientation='bottom')

        self.initCandlestickPlot()
        self.initVolumePlot()
        self.initReturnPlot()

        self.crosshair = CrossHair([self.candlestick_plot, self.volume_plot, self.return_plot], self.layout)

    def initCandlestickPlot(self):
        self.candlestick_axis = pg.AxisItem('left')
        self.candlestick_plot = pg.PlotItem(axisItems={'bottom': self.axisTimes})
        #self.candlestick_vb = pg.ViewBox(enableMouse=False)
        self.candlestick_vb = pg.ViewBox()
        self.candlestick_vb.setXLink(self.candlestick_plot.vb)
        self.candlestick_axis.linkToView(self.candlestick_vb)
        self.candlestick_axis.setLabel('股票价格')
        self.candlestick_plot.hideAxis('left')

        #self.crosshair = Crosshair(self, self)

        #self.candlestick_plot.setMinimumHeight(500)

        #self.layout.nextRow()
        self.layout.addItem(self.candlestick_axis, col=1)
        self.layout.addItem(self.candlestick_plot, col=2)
        self.layout.scene().addItem(self.candlestick_vb)

    def initVolumePlot(self):
        self.volume_axis = pg.AxisItem('left')
        self.volume_plot = pg.PlotItem(axisItems={'bottom': self.axisTimes})
        self.volume_vb = pg.ViewBox()
        self.volume_vb.setXLink(self.volume_plot.vb)
        self.volume_plot.setXLink(self.candlestick_plot)
        self.volume_axis.linkToView(self.volume_vb)
        self.volume_axis.setLabel('成交量')
        self.volume_plot.hideAxis('left')

        #self.candlestick_plot.setMinimumHeight(200)

        self.layout.nextRow()
        self.layout.addItem(self.volume_axis, col=1)
        self.layout.addItem(self.volume_plot, col=2)
        self.layout.scene().addItem(self.volume_vb)

    def initReturnPlot(self):
        self.simple_return_axis = pg.AxisItem('left')
        self.log_return_axis = pg.AxisItem('right')
        self.return_plot = pg.PlotItem(axisItems={'bottom': self.axisTimes})
        self.return_plot.setXLink(self.volume_plot)
        self.return_vb = self.return_plot.vb
        self.simple_return_vb = pg.ViewBox()
        self.log_return_vb = pg.ViewBox()
        self.simple_return_vb.setXLink(self.return_vb)
        self.log_return_vb.setXLink(self.simple_return_vb)
        self.simple_return_axis.linkToView(self.simple_return_vb)
        self.log_return_axis.linkToView(self.log_return_vb)
        self.simple_return_axis.setLabel('simple return')
        self.log_return_axis.setLabel('log return')
        self.return_plot.hideAxis('left')

        #self.candlestick_plot.setMinimumHeight(200)

        self.layout.nextRow()
        self.layout.addItem(self.simple_return_axis, col=1)
        self.layout.addItem(self.return_plot, col=2)
        self.layout.scene().addItem(self.simple_return_vb)
        self.layout.scene().addItem(self.log_return_vb)
        self.layout.addItem(self.log_return_axis, col=3)

    def loadData(self, data, title):

        data.insert(1, 'time_int', np.array(list(range(len(data.index)))))

        self.candlestick_data = data[['time_int', 'open', 'close', 'low', 'high']].to_records(False)

        vd = pd.DataFrame()
        vd['open'] = data.apply(lambda x: 0 if x['close'] >= x['open'] else x['volume'],axis=1)
        vd['close'] = data.apply(lambda x: 0 if x['close'] < x['open'] else x['volume'],axis=1)
        vd['low'] = 0
        vd['high'] = data['volume']
        vd['time_int'] = np.array(list(range(len(data.index))))
        self.volume_data = vd[['time_int','open','close','low','high']].to_records(False)

        self.simple_return_data = list(data['simple return'])
        self.log_return_data = list(data['log return'])

        self.axisTimes.xdict = {}
        xdict = dict(enumerate(data.index.tolist()))
        self.axisTimes.update_xdict(xdict)

        self.crosshair.update(data, title)

        self.update()

        #self.updateViews()

    def update(self):

        if self.candlestick_plotitem is not None:
            self.candlestick_vb.removeItem(self.candlestick_plotitem)
        self.candlestick_plotitem = CandlestickItem(self.candlestick_data)
        self.candlestick_vb.addItem(self.candlestick_plotitem)

        if self.volume_plotitem is not None:
            self.volume_vb.removeItem(self.volume_plotitem)
        self.volume_plotitem = CandlestickItem(self.volume_data)
        self.volume_vb.addItem(self.volume_plotitem)

        if self.simple_return_plotitem is not None:
            self.simple_return_vb.removeItem(self.simple_return_plotitem)
        if self.log_return_plotitem is not None:
            self.log_return_vb.removeItem(self.log_return_plotitem)

        self.simple_return_plotitem = pg.PlotCurveItem(self.simple_return_data, pen='y')
        self.log_return_plotitem = pg.PlotCurveItem(self.log_return_data, pen='r')

        self.simple_return_vb.addItem(self.simple_return_plotitem)
        self.log_return_vb.addItem(self.log_return_plotitem)

        self.candlestick_plot.vb.sigResized.connect(self.updateViews)
        self.volume_plot.vb.sigResized.connect(self.updateViews)
        self.return_vb.sigResized.connect(self.updateViews)

        self.candlestick_vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.volume_vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.simple_return_vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.log_return_vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

    def updateViews(self):
        self.candlestick_vb.setGeometry(self.candlestick_plot.vb.sceneBoundingRect())
        self.volume_vb.setGeometry(self.volume_plot.vb.sceneBoundingRect())
        self.simple_return_vb.setGeometry(self.return_vb.sceneBoundingRect())
        self.log_return_vb.setGeometry(self.return_vb.sceneBoundingRect())


class ReturnPlot(pg.PlotWidget):

    def __init__(self):
        super(ReturnPlot, self).__init__()

        self.simple_return_axis = pg.AxisItem('left')
        self.log_return_axis = pg.AxisItem('right')

        self.simple_return = pg.ViewBox()
        self.log_return = pg.ViewBox()

        self.mainvb = self.simple_return

        self.setWindowTitle('stock return')

        self.layout = pg.GraphicsLayout()
        self.setCentralWidget(self.layout)

        self.layout.addItem(self.simple_return_axis, row=1, col=1)
        self.layout.addItem(self.log_return_axis, row=1, col=3)

        xdict = {}
        self.axisTime = MyStringAxis(xdict, orientation='bottom')

        self.plot_item = pg.PlotItem(axisItems={'bottom': self.axisTime})
        self.default_vb = self.plot_item.vb

        self.plot_item.hideAxis('left')

        self.layout.addItem(self.plot_item, row=1, col=2)

        self.layout.scene().addItem(self.simple_return)
        self.layout.scene().addItem(self.log_return)

        self.simple_return_axis.linkToView(self.simple_return)
        self.log_return_axis.linkToView(self.log_return)

        self.simple_return.setXLink(self.default_vb)
        self.log_return.setXLink(self.simple_return)

        self.simple_return_axis.setLabel('simple return')
        self.log_return_axis.setLabel('log return')

        self.old = [None, None]

        self.crosshair = CrossHair([self.plot_item], self.layout)

    def loadData(self, data, title):
        data.index = pd.to_datetime(data.index)
        #data.insert(1, 'time_int', np.array(list(range(len(data.index)))))

        self.simple_return_data = list(data['simple return'])
        self.log_return_data = list(data['log return'])

        self.axisTime.xdict = {}
        xdict = dict(enumerate(data.index.tolist()))
        self.axisTime.update_xdict(xdict)

        self.crosshair.update(data, title)

        self.update()
        self.updateViews()

    def update(self):

        if self.old[0] == self.old[1] and self.old[0] is None:
            self.simple_return.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
            self.log_return.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        else:
            self.simple_return.removeItem(self.old[0])
            self.log_return.removeItem(self.old[1])

        self.old[0] = pg.PlotCurveItem(self.simple_return_data, pen='y')
        #self.old[1] = CandlestickItem(self.candlestick_data)
        self.old[1] = pg.PlotCurveItem(self.log_return_data, pen='r')

        self.simple_return.addItem(self.old[0])
        self.log_return.addItem(self.old[1])

        self.simple_return.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.log_return.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

        self.default_vb.sigResized.connect(self.updateViews)

    def updateViews(self):
        self.simple_return.setGeometry(self.default_vb.sceneBoundingRect())
        self.log_return.setGeometry(self.default_vb.sceneBoundingRect())


class IndicatorPlot(pg.PlotWidget):

    def __init__(self, indicator_type='MACD'):
        super(IndicatorPlot, self).__init__()

        self.indicator_type = indicator_type

        self.axis = pg.AxisItem('left')

        self.vb = pg.ViewBox()

        self.mainvb = self.vb

        self.setWindowTitle('stock indicator: %s' % self.indicator_type)

        self.layout = pg.GraphicsLayout()
        self.setCentralWidget(self.layout)

        self.layout.addItem(self.axis, row=1, col=1)

        xdict = {}
        self.axisTime = MyStringAxis(xdict, orientation='bottom')

        self.plot_item = pg.PlotItem(axisItems={'bottom': self.axisTime})
        self.default_vb = self.plot_item.vb

        self.plot_item.hideAxis('left')

        self.layout.addItem(self.plot_item, row=1, col=2)

        self.layout.scene().addItem(self.vb)

        self.axis.linkToView(self.vb)

        self.vb.setXLink(self.default_vb)

        self.axis.setLabel('%s' % self.indicator_type)

        self.old = [None]

        self.crosshair = CrossHair([self.plot_item], self.layout)

    def loadData(self, data, title):
        data.index = pd.to_datetime(data.index)

        self.data = list(data[self.indicator_type] if self.indicator_type in data.columns else data['close'])

        self.axisTime.xdict = {}
        xdict = dict(enumerate(data.index.tolist()))
        self.axisTime.update_xdict(xdict)

        self.crosshair.update(data, title)

        self.update()
        self.updateViews()

    def update(self):

        if self.old[0] is None:
            self.vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        else:
            self.vb.removeItem(self.old[0])

        self.old[0] = pg.PlotCurveItem(self.data, pen='r')

        self.vb.addItem(self.old[0])

        self.vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

        self.default_vb.sigResized.connect(self.updateViews)

    def updateViews(self):
        self.vb.setGeometry(self.default_vb.sceneBoundingRect())


class HeatmapPlot(pg.PlotWidget):

    def __init__(self):
        super(HeatmapPlot, self).__init__()

        self.setWindowTitle('heatmap')

        self.layout = pg.GraphicsLayout()
        self.setCentralWidget(self.layout)

        xdict = {}
        self.baxis = MyCodeAxis(xdict, orientation='bottom')
        self.laxis = MyCodeAxis(xdict, orientation='left')

        self.plot_item = pg.PlotItem()
        self.default_vb = self.plot_item.vb

        self.plot_item.getAxis('left').hide()
        self.plot_item.getAxis('bottom').hide()

        self.heatmap_vb = pg.ViewBox()

        self.layout.addItem(self.laxis, row=2, col=0)
        self.layout.addItem(self.plot_item, row=2, col=1)
        self.layout.scene().addItem(self.heatmap_vb)
        self.layout.addItem(self.baxis, row=3, col=1)

        self.heatmap_vb.setXLink(self.default_vb)
        self.baxis.linkToView(self.heatmap_vb)
        self.laxis.linkToView(self.heatmap_vb)

        self.old = None

    def loadData(self, data, title):

        self.mdata = data

        self.baxis.xdict = {}
        self.laxis.xdict = {}
        xdict = dict(enumerate([x[-6:] for x in self.mdata.index.tolist()]))
        self.baxis.update_xdict(xdict)
        self.laxis.update_xdict(xdict)

        self.update()

    def update(self):
        if self.old is None:
            self.heatmap_vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        else:
            self.heatmap_vb.removeItem(self.old)

        self.old = pg.ScatterPlotItem(pxMode=False, symbol='s')   ## Set pxMode=False to allow spots to transform with the view

        def get_color(i, j, data):
            x = data.iloc[i][data.columns[j]]
            return pg.intColor(int((x+1) / 2 * 1000), 1000)

        spots = []
        w = 1
        n = len(self.mdata.index)
        for i in range(n):
            for j in range(n):
                spots.append({'pos': (w*i, w*j), 'size': w, 'pen': {'color': get_color(i, j, self.mdata), 'width': w}, 'brush':get_color(i, j, self.mdata)})
        self.old.addPoints(spots)

        self.heatmap_vb.addItem(self.old)
        self.default_vb.sigResized.connect(self.updateViews)
        self.heatmap_vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

    def updateViews(self):
        self.heatmap_vb.setGeometry(self.default_vb.sceneBoundingRect())


class AboutDialog(QtGui.QDialog):
    def __init__(self, parent=None, window_title='', text=''):
        QtGui.QDialog.__init__(self, parent)

        self.parent = parent
        self.window_title = window_title
        self.text = text

        self.initUI()

    def initUI(self):
        self.textLabel = QtGui.QLabel(self.text, self)

        self.layout = QtGui.QGridLayout()
        self.layout.addWidget(self.textLabel)

        self.setGeometry(500, 200, 360, 250)
        self.setWindowTitle(self.window_title)
        self.setLayout(self.layout)

    def setText(self, text):
        self.textLabel.setText(text)

