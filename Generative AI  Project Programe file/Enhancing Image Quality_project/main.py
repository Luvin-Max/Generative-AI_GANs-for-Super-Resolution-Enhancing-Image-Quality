import sys

import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, \
                            QVBoxLayout, QHBoxLayout, QWidget,\
                            QFileDialog, QGridLayout, QMessageBox
from PyQt5.QtGui import QIcon, QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import cv2


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        # window swtting's
        self.setWindowIcon(QIcon("images/TL_img.png"))
        self.setWindowTitle("Image Enhancer")
        self.setWindowTitle("Image Enhancer")
        self.setGeometry(600, 200, 600, 500)

        # Alinment Items Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # label 2
        self.label2 = QLabel()
        self.label2.setAlignment(Qt.AlignBottom)
        self.layout.addWidget(self.label2)  # This is for Image About Txt

        # img view
        self.selected_img = QLabel()
        self.selected_img.setAlignment(Qt.AlignTop)
        self.layout.addWidget(self.selected_img)

        # label 3
        self.label3 = QLabel()
        self.label3.setAlignment(Qt.AlignBottom)
        self.layout.addWidget(self.label3)  # This is for Image About Txt
        # loader view
        self.loader_label = QLabel("Loading...")
        self.loader_label.hide()
        self.loader_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.loader_label)

        # label
        self.label1 = QLabel("Select the Img:")
        self.label1.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label1)


        # button's
        self.Button_layout = QHBoxLayout()
        self.layout.addLayout(self.Button_layout)  # button layout
        self.Button_layout.setAlignment(Qt.AlignCenter)

        # buttons1
        self.select_button = QPushButton("Select Image")
        self.select_button.clicked.connect(self.load_image)
        self.Button_layout.addWidget(self.select_button)

        # buttons2
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit_image)
        self.Button_layout.addWidget(self.submit_button)

    def load_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg)")
        if filename:
            self.original_image = cv2.imread(filename)
            pixmap = QPixmap(filename)
            self.selected_img.setPixmap(pixmap.scaled(150, 150))
            self.label1.setText("click the Submit Button")
            self.label2.setText("Selecting img:")

        else:
            QMessageBox.warning("Invalid image file.")

    def submit_image(self):
        if hasattr(self, 'original_image'):
            self.loader_label.show()
            QApplication.processEvents()
            enhanced_image = self.original_image
            enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
            height, width, channel = enhanced_image.shape
            bytes_per_line = 3 * width
            q_img = QPixmap(
                QPixmap.fromImage(QImage(enhanced_image.data, width, height, bytes_per_line, QImage.Format_RGB888)))


            # output feeld
            self.loader_label.setPixmap(q_img.scaled(400, 400))
            self.label3.setText("Enhanced image:")
            self.label1.hide()
        else:
            QMessageBox.warning(self, "Warning", "Please select an image first.")


def CreatApp():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


CreatApp()
