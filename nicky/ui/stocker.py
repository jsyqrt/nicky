# coding: utf-8
# stocker.py
import sys

from pyqtgraph.Qt import QtGui

from nicky.ui.ui import MainWindow


app = QtGui.QApplication(sys.argv)
main = MainWindow()

main.showMaximized()

sys.exit(app.exec_())
