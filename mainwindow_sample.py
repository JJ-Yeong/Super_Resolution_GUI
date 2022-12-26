import os
import sys

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic


MAIN_PATH = "_QTD/mainwindow_sample.ui"
PBAR_PATH = "_QTD/progressbar_sample.ui"

def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def get_ui(ui_path):
    form = resource_path(ui_path)
    return uic.loadUiType(form)[0]

form_class = get_ui(MAIN_PATH)
form_class_pbar = get_ui(PBAR_PATH)


class ProgressDialog(QDialog, form_class_pbar):
    def __init__(self, filename: str):
        super().__init__()
        self.pbar: QProgressBar
        self.label_file_name: QLabel
        self.setupUi(self)
        self.setModal(True)
        self.filename = filename

        self.dot = 1
        self.text = f"{self.filename}\nInference..{'.' * self.dot}"
        self.label_file_name.setText(self.text)
        self.show()

        self.timer = QBasicTimer()
        self.step = 0
        self.start_progress()

    def timerEvent(self, e):
        if self.step >= 100:
            self.timer.stop()
            self.close()

        self.text = f"{self.filename}\nInference..{'.' * self.dot}"
        self.label_file_name.setText(self.text)
        self.step += 1
        self.dot = self.dot + 1 if self.dot < 3 else 1
        self.pbar.setValue(self.step)

    def start_progress(self):
        self.timer.start(100, self)

    def btn_stop_progress(self):
        self.close()


class MainWindow(QMainWindow, QWidget, form_class):
    def __init__(self):
        super().__init__()
        self.stack: QStackedWidget
        self.setupUi(self)

    def fnc_next_page(self):
        idx = self.stack.currentIndex() + 1
        self.stack.setCurrentIndex(idx)

    def fnc_previous_page(self):
        idx = self.stack.currentIndex() - 1
        self.stack.setCurrentIndex(idx)

    def fnc_start_progress(self):
        global filename
        filepath, _ = QFileDialog.getOpenFileName(self, 'Open File') 
        if filepath:
            filename = os.path.basename(filepath)
            # self.hide()
            self.pdialog = ProgressDialog(filename)
            self.pdialog.exec()
        # self.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
