from PySide import QtGui, QtCore
from logging import Formatter

class QTextHandler(QtGui.QTextEdit):
    def __init__(self, parent=None):
        super(QTextHandler, self).__init__(parent)

    def setFormatter(self, fmt=None):
        self.formatter = Formatter(fmt)

    def handle(self, record):
        self.append(self.formatter.format(record))
