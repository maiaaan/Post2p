from PyQt5.QtCore import QObject, pyqtSignal, Qt, QRectF
from PyQt5.QtGui import QPixmap, QColor, QPen, QPainter, QBrush
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QMainWindow, \
    QWidget, QHBoxLayout, QGraphicsEllipseItem, QGraphicsPixmapItem, QLineEdit, QVBoxLayout
from PyQt5 import QtCore, QtGui, QtWidgets
import os.path
import numpy as np
import sys
import functions
import RedCell_functions
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
                pixmap_item.setZValue(-100)
                self.scene().addItem(pixmap_item)
            else:
                print("Failed to load background image:", self.background_image_path)
        self.drawObjects()

    def drawObjects(self):
        scene = self.scene()
        for i, cell in enumerate(self.cell_info):
            if self.Green_Cell[i][0] == 0:
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
            if self.Green_Cell[i][0] == 1:
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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cell_info, Green_Cell, background_image_path):
        super().__init__()
        self.setupUi(cell_info, Green_Cell, background_image_path)

    def setupUi(self, cell_info, Green_Cell, background_image_path):
        self.setObjectName("MainWindow")
        self.setWindowTitle("You can transfer masks between channels")
        self.setStyleSheet("""
            background-color: rgb(27, 27, 27);
            gridline-color: rgb(213, 213, 213);
            border-top-color: rgb(197, 197, 197);
        """)
        self.red_cell_num  = 200
        self.Green_cell_num= 500
        self.setGeometry(100, 100, 1100, 641)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setAccessibleName("")
        self.label_3.setStyleSheet("color: rgb(167, 167, 167);")
        self.label_3.setTextFormat(QtCore.Qt.RichText)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.lineEdit_Red = QtWidgets.QLineEdit(self.frame)
        self.lineEdit_Red.setEnabled(True)
        self.lineEdit_Red.setObjectName("lineEdit_Red")
        self.verticalLayout_2.addWidget(self.lineEdit_Red)

        self.Red_view = CustomGraphicsView_red(cell_info, Green_Cell, background_image_path)
        self.Green_view = CustomGraphicsView_green(cell_info, Green_Cell, background_image_path)
        self.verticalLayout_2.addWidget(self.Red_view)
        self.horizontalLayout_2.addWidget(self.frame)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setEnabled(True)
        self.frame_2.setToolTip("")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_4 = QtWidgets.QLabel(self.frame_2)
        self.label_4.setAccessibleName("")
        self.label_4.setStyleSheet("color: rgb(167, 167, 167);")
        self.label_4.setTextFormat(QtCore.Qt.RichText)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.lineEdit_Green = QtWidgets.QLineEdit(self.frame_2)
        self.lineEdit_Green.setObjectName("lineEdit_Green")
        self.verticalLayout.addWidget(self.lineEdit_Green)

        self.verticalLayout.addWidget(self.Green_view)
        self.horizontalLayout_5.addWidget(self.frame_2)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_5)
        self.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 954, 21))
        self.menubar.setObjectName("menubar")
        self.menubar.setStyleSheet("color: white;")
        self.menuOpen = QtWidgets.QMenu(self.menubar)
        self.menuOpen.setObjectName("menuOpen")
        self.menuOpen.setStyleSheet("""
                  QMenu {background-color: rgb(200, 200, 200);
                      color: rgb(20, 20, 20);}
              """)
        self.setMenuBar(self.menubar)
        self.actionload_proccesd_file = QtWidgets.QAction(self)
        self.actionload_proccesd_file.setObjectName("actionload_proccesd_file")
        self.menuOpen.addAction(self.actionload_proccesd_file)
        self.menubar.addAction(self.menuOpen.menuAction())
        ####################################################
        # Connecting signals
        self.Red_view.objectClicked.connect(lambda idx: self.toggleItem(idx, 1))
        self.Green_view.objectClicked.connect(lambda idx: self.toggleItem(idx, 0))

        # Initialize tracking lists
        self.currentRedObjects = []
        self.currentGreenObjects = []

        # tracking update
        self.updateObjectTracking()
        self.retranslateUi()

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
        self.lineEdit_Red.setText(str(self.red_cell_num))
        self.lineEdit_Green.setText(str(self.Green_cell_num))

    def updateObjectTracking(self):
        self.currentRedObjects.clear()
        self.currentGreenObjects.clear()



        # Re-populate the lists based on current state in Green_Cell
        for idx, state in enumerate(self.Red_view.Green_Cell):
            if state[0] == 0:
                self.currentRedObjects.append(idx)

            else:
                self.currentGreenObjects.append(idx)
            self.red_cell_num = len(self.currentRedObjects)
            self.Green_cell_num = len(self.currentGreenObjects)
        return self.currentRedObjects, self.currentGreenObjects




    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "You can transfer masks between channels"))
        self.label_3.setText(_translate("MainWindow", "Red/Green"))
        self.label_4.setText(_translate("MainWindow", "Only Green"))
        self.menuOpen.setTitle(_translate("MainWindow", "Open"))
        self.lineEdit_Green.setStyleSheet("color: white")
        self.lineEdit_Green.setText(_translate("MainWindow", f"{self.Green_cell_num}"))
        self.lineEdit_Red.setStyleSheet("color: white")
        self.lineEdit_Red.setText(_translate("MainWindow", f"{self.red_cell_num}"))
        self.actionload_proccesd_file.setText(_translate("MainWindow", "load proccesd file"))

save_red_results = r"F:\VIP_CB1 td Tom\VIP_2_FOD_male\220223\TSeries-02222023-mouse2-001\suite2p\plane0\save"
Base_path = r"F:\VIP_CB1 td Tom\VIP_2_FOD_male\220223\TSeries-02222023-mouse2-001"

if __name__ == "__main__":
    uite2p_path, ops, Mean_image, cell, stat, single_red = RedCell_functions.loadred(Base_path)
    image_path =  os.path.join(save_red_results, "grey_image.jpg")
    cell_info,_ = functions.detect_cell(cell, stat)

    separete_masks = RedCell_functions.single_mask(ops, cell_info)
    thresh = RedCell_functions.DetectRedCellMask(Base_path, min_area = 35 , max_area= 150)
    only_green_mask, only_green_cell, comen_cell, KeepMask, blank2 = \
        RedCell_functions.select_mask(Base_path, thresh, separete_masks, cell_true=2)

    print("only_green_cell", only_green_cell)
    print("comen_cell", comen_cell)

    Green_Cell = np.ones((len(cell_info), 2))
    Green_Cell[comen_cell, 0] = 0

    app = QApplication(sys.argv)
    window = MainWindow(cell_info, Green_Cell, image_path)

    window.show()
    app.exec_()


# currentRedObjects = window.currentRedObjects
# currentGreenObject = window.currentGreenObjects
# print("Final1 ", currentRedObjects)
# print("Final2 ", currentGreenObject)
#

