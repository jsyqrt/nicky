# -*- coding: utf-8 -*-

import sys
import threading
import datetime
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')
import queue

import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from nicky.data.stocks.sohu import sohu

from nicky.ui.stock_widget import StockWidget

__all__ = ['MainWindow']

class MainWindow(QtGui.QMainWindow):

    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)

        self.initData()
        self.initUI()

    def initData(self):
        pass

    def initUI(self):
        self.initMenubar()
        self.initLeftBar()
        self.initLayout()
        self.initWidgets()

        #self.showMaximized()

    def initLayout(self):
        self.layout = QtGui.QStackedLayout()
        self.main_widget = QtGui.QWidget()
        self.main_widget.setLayout(self.layout)
        self.setCentralWidget(self.main_widget)

    def initWidgets(self):
        self.stock_widget = StockWidget()

        self.layout.addWidget(self.stock_widget)

    def initMenubar(self):
        pass

    def initLeftBar(self):
        self.leftbar = QtGui.QToolBar(self)
        self.addToolBar(QtCore.Qt.LeftToolBarArea, self.leftbar)

        showStockAction = QtGui.QAction("Stock", self)
        showStockAction.triggered.connect(self.showStockWidget)
        self.leftbar.addAction(showStockAction)

    def showStockWidget(self):
        self.layout.setCurrentIndex(0)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    main = MainWindow()
    main.showMaximized()

    sys.exit(app.exec_())
