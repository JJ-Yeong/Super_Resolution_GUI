import os
import sys

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic

MAIN_PATH = "_QTD/_page01.ui"
SECOND_PATH = "_QTD/_page02.ui"

def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def get_ui(ui_path):
    form = resource_path(ui_path)
    return uic.loadUiType(form)[0]

form_class = get_ui(MAIN_PATH)
form_secondwindow = get_ui(SECOND_PATH)


class Page1(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.setWindowFlags(
            self.windowFlags() |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowSystemMenuHint
            )

    def btn_1_to_2(self):
        idx = stack.currentIndex() + 1
        stack.setCurrentIndex(idx)
        # self.hide()                     # 메인윈도우 숨김
        # self.second = Page2()    #
        # self.second.exec()              # 두번째 창을 닫을 때 까지 기다림
        # self.show()                     # 두번째 창을 닫으면 다시 첫 번째 창이 보여짐짐


class Page2(QMainWindow, form_secondwindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.setWindowFlags(
            self.windowFlags() |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowSystemMenuHint
            )

    def btn_2_to_1(self):
        idx = stack.currentIndex() - 1
        stack.setCurrentIndex(idx)
        # self.close()                  # 클릭시 종료됨.


if __name__ == '__main__':

    app = QApplication(sys.argv)

    #화면 전환용 Widget 설정
    stack = QStackedWidget()
    # stack.setGeometry(0, 0, 2000, 1600)
    # stack.setFrameShape(QFrame.Box)
    page1 = Page1()
    page2 = Page2()

    #Widget 추가
    stack.addWidget(page1)
    stack.addWidget(page2)

    #프로그램 화면을 보여주는 코드
    # stack.setFixedWidth(900)
    # stack.setFixedHeight(500)
    stack.show()

    sys.exit(app.exec_())