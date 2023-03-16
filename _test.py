import sys
import time

from PySide2.QtCore import *  # Signal()
from PySide2.QtGui import *
from PySide2.QtWidgets import *  # QMainWindow, QWidget, QGridLayout


#qthread 에러 없이 종료하기
class UserButton(QPushButton):

    def __init__(self):
        super(UserButton, self).__init__()
        self._prop = 'false'


    def getter(self):
        return self._prop

    def setter(self, val):
        if self._prop == val:
            return
        self._prop = val
        self.style().polish(self)

    prop = Property(str, fget=getter, fset=setter)


class intervalThread(QThread):
    def __init__(self, b1:UserButton, b2:UserButton):
        super(intervalThread,self).__init__()
        self.working = True
        self.b1 = b1
        self.b2 = b2

    def run(self):
        while self.working:
            if self.b1.prop == 'true':
                self.b1.prop = 'false'
            else:
                self.b1.prop = 'true'
            print(self.b1.prop)
            self.sleep(1)

    def stop(self):
        self.working = False
        self.quit()
        self.wait(5000) #5000ms = 5s

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        # QWidget.__init__(self)

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.button1 = UserButton()
        self.button2 = UserButton()

        self.layout.addWidget(self.button1, 0, 0, 1, 1)
        self.layout.addWidget(self.button2, 0, 1, 1, 1)

        self.thread = intervalThread(self.button1, self.button2)
        self.thread.start()

    def closeEvent(self, e):
        self.hide()
        self.thread.stop()

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()

        if e.key() == Qt.Key_S:
            self.thread.working=True
            self.thread.start()
            print("S")

        if e.key() == Qt.Key_P:
            self.thread.working=False
            print("P")

app = QApplication(sys.argv)
lf = MainWindow()
lf.show()
sys.exit(app.exec_())