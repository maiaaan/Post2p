import os.path
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QRectF
from PyQt5.QtGui import QPixmap, QColor, QPen, QPainter, QBrush
import RedCell_functions
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QMainWindow, \
    QWidget, QHBoxLayout, QGraphicsEllipseItem, QGraphicsPixmapItem
import sys


class CustomGraphicsView_red(QGraphicsView):
    objectClicked = pyqtSignal(int)
    def __init__(self, cell_info, Green_Cell, background_image_path=None):
        super().__init__()
        self.setScene(QGraphicsScene())
        self.Green_Cell = Green_Cell
        self.cell_info = cell_info
        self.background_image_path = background_image_path
        self.setBackgroundImage()
        self.drawObjects()

    def setBackgroundImage(self):
        if self.background_image_path:
            pixmap = QPixmap(self.background_image_path)
            if not pixmap.isNull():
                pixmap_item = QGraphicsPixmapItem(pixmap)
                pixmap_item.setZValue(-100)  # Ensure it stays in the background
                self.scene().addItem(pixmap_item)
            else:
                print("Failed to load background image:", self.background_image_path)
        self.drawObjects()

    def drawObjects(self):
        scene = self.scene()
        for i, cell in enumerate(self.cell_info):
            if self.Green_Cell[i][0] == 0:  # Draw only red objects
                color = QColor(Qt.red)
                color.setAlpha(10)
                for x, y in zip(cell['xpix'], cell['ypix']):
                    ellipse = scene.addEllipse(x, y, 1, 1, QPen(color), QBrush(color))
                    ellipse.setData(0, i)

    def mousePressEvent(self, event):
        items = self.items(event.pos())
        for item in items:
            if isinstance(item, QGraphicsEllipseItem):
                object_index = item.data(0)
                self.objectClicked.emit(object_index)
                return
        super().mousePressEvent(event)


class CustomGraphicsView_green(QGraphicsView):
    objectClicked = pyqtSignal(int)

    def __init__(self, cell_info, Green_Cell, background_image_path=None):
        super().__init__()
        self.setScene(QGraphicsScene())
        self.Green_Cell = Green_Cell
        self.cell_info = cell_info
        self.background_image_path = background_image_path
        self.setBackgroundImage()
        self.drawObjects()

    def setBackgroundImage(self):
        if self.background_image_path:
            pixmap = QPixmap(self.background_image_path)
            if not pixmap.isNull():
                pixmap_item = QGraphicsPixmapItem(pixmap)
                pixmap_item.setZValue(-100)
                self.scene().addItem(pixmap_item)
            else:
                print("Failed to load background image:", self.background_image_path)

    def drawObjects(self):
        scene = self.scene()
        for i, cell in enumerate(self.cell_info):
            if self.Green_Cell[i][0] == 1:  # Draw only green objects
                color = QColor(Qt.green)
                color.setAlpha(10)
                for x, y in zip(cell['xpix'], cell['ypix']):
                    ellipse = scene.addEllipse(x, y, 1, 1, QPen(color), QBrush(color))
                    ellipse.setData(0, i)

    def mousePressEvent(self, event):
        items = self.items(event.pos())
        for item in items:
            if isinstance(item, QGraphicsEllipseItem):
                object_index = item.data(0)
                self.objectClicked.emit(object_index)
                return
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self, cell_info, Green_Cell, background_image_path=None):
        super().__init__()
        self.setWindowTitle("Clickable Object Transfer")
        self.setGeometry(100, 100, 1024, 512)

        # Initialize views
        self.Red_view = CustomGraphicsView_red(cell_info, Green_Cell, background_image_path)
        self.Green_view = CustomGraphicsView_green(cell_info, Green_Cell, background_image_path)

        layout = QHBoxLayout()
        layout.addWidget(self.Red_view)
        layout.addWidget(self.Green_view)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Connecting signals
        self.Red_view.objectClicked.connect(lambda idx: self.toggleItem(idx, 1))
        self.Green_view.objectClicked.connect(lambda idx: self.toggleItem(idx, 0))

        # Initialize tracking lists
        self.currentRedObjects = []
        self.currentGreenObjects = []

        # tracking update
        self.updateObjectTracking()

    def toggleItem(self, object_index, target_view):
        # Update the Green_Cell to reflect the new view state
        self.Red_view.Green_Cell[object_index][0] = target_view
        self.Green_view.Green_Cell[object_index][0] = target_view
        self.Red_view.scene().clear()
        self.Green_view.scene().clear()
        self.Red_view.drawObjects()
        self.Green_view.drawObjects()
        self.Red_view.setBackgroundImage()
        self.Green_view.setBackgroundImage()
        self.updateObjectTracking()

    def updateObjectTracking(self):
        # Clear current lists
        self.currentRedObjects.clear()
        self.currentGreenObjects.clear()

        # Re-populate the lists based on current state in Green_Cell
        for idx, state in enumerate(self.Red_view.Green_Cell):
            if state[0] == 0:
                self.currentRedObjects.append(idx)
            else:
                self.currentGreenObjects.append(idx)
        print("Current Red Objects:", self.currentRedObjects)
        print("Current Green Objects:", self.currentGreenObjects)

        return self.currentRedObjects, self.currentGreenObjects

import functions

if __name__ == "__main__":
    Base_path, ops, Mean_image, cell, stat, single_red = RedCell_functions.loadred()
    image_path =  os.path.join(Base_path, "grey_image.jpg")
    cell_info,_ = functions.detect_cell(cell, stat)

    separete_masks = RedCell_functions.single_mask(ops, cell_info)
    thresh = RedCell_functions.DetectRedCellMask(Base_path, min_area = 35 , max_area= 150)
    only_green_mask, only_green_cell, comen_cell, KeepMask, blank2 = \
        RedCell_functions.select_mask(Base_path, thresh, separete_masks, cell_true=2)

    Green_Cell = np.ones((len(cell_info), 2))
    Green_Cell[comen_cell, 0] = 0

    app = QApplication(sys.argv)
    window = MainWindow(cell_info, Green_Cell, image_path)

    window.show()
    app.exec_()


currentRedObjects = window.currentRedObjects
currentGreenObject = window.currentGreenObjects
print("Final1 ", currentRedObjects)
print("Final2 ", currentGreenObject)

