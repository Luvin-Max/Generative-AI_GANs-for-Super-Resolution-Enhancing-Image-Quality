import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
import tensorflow as tf
from model import generator

class ImageEnhancer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Enhancer")
        self.initUI()

    def initUI(self):
        # Load the trained generator model
        self.gen_model = generator()
        self.gen_model.load_weights("generator_weights.h5")  # Replace with the path to your trained weights

        # Create widgets
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.button = QPushButton("Load Image", self)
        self.button.clicked.connect(self.load_image)

        # Set layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def load_image(self):
        # Open file dialog to select image
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")

        # Load and enhance the image
        if file_path:
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            enhanced_image = self.enhance_image(image)

            # Display enhanced image
            pixmap = QPixmap.fromImage(QImage(enhanced_image.data, enhanced_image.shape[1], enhanced_image.shape[0],
                                               enhanced_image.strides[0], QImage.Format_RGB888))
            self.label.setPixmap(pixmap)

    def enhance_image(self, image):
        # Preprocess the input image
        preprocessed_image = image.astype(np.float32) / 255.0
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

        # Enhance image using the generator model
        enhanced_image = self.gen_model.predict(preprocessed_image)[0]

        # Postprocess the enhanced image
        enhanced_image = (enhanced_image * 255.0).clip(0, 255).astype(np.uint8)

        return enhanced_image

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageEnhancer()
    window.show()
    sys.exit(app.exec_())
