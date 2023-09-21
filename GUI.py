from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
import os
import numpy as np
from PyQt5.QtCore import QTimer
data_path = r"C:\Users\faezeh.rabbani\Desktop\myfiles\Data\pyr\TSeries-04242023-765fogscreen-001\Results\data"
figure_path = r"C:\Users\faezeh.rabbani\Desktop\myfiles\Data\pyr\TSeries-04242023-765fogscreen-001\Results\Figures"

class Ui_MainWindow(object):

    first_frame_changed = QtCore.pyqtSignal(str)
    def setupUi(self, MainWindow,figure_path, data_path):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(836, 970)
        font = QtGui.QFont()
        font.setFamily("Blackadder ITC")
        MainWindow.setFont(font)
        MainWindow.setStyleSheet("background-color: rgb(54, 84, 107);\n""")

        self.remove_blink = 0
        self.FaceAnalyze = 1
        self.PupilAnalyze = 1
        self.F0 = "sliding"
        self.neuropil = 0.7
        self.first_frame = int
        self.last_frame = int
        self.motion = os.path.join(data_path,"raw_motion.npy")
        self.Motion = np.load(self.motion)
        self.last = len(self.Motion)
        self.first = 0
        self.pupil = os.path.join(data_path,"raw_pupil.npy")
        self.Pupil = np.load(self.pupil)
        self.mean_dF = os.path.join(data_path, "raw_mean_F.npy")
        self.Mean_dF = np.load(self.mean_dF)
        self.first_dF = 0
        self.last_dF = len(self.Mean_dF)


        self.mean_F_image = os.path.join(figure_path, "raw_mean_F.png")
        self.pupil_image = os.path.join(figure_path,"pupil.png")
        self.motion_image = os.path.join(figure_path, "raw_face_motion.png")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.fig_motion = QtWidgets.QGraphicsView(self.centralwidget)
        self.fig_motion.setGeometry(QtCore.QRect(20, 45, 770, 180))
        self.fig_motion.setObjectName("fig_motion")
        self.fig_pupil = QtWidgets.QGraphicsView(self.centralwidget)
        self.fig_pupil.setGeometry(QtCore.QRect(20, 335, 770, 180))
        self.fig_pupil.setObjectName("fig_pupil")
        self.fig_mean_F = QtWidgets.QGraphicsView(self.centralwidget)
        self.fig_mean_F.setGeometry(QtCore.QRect(20, 550, 770, 180))
        self.fig_mean_F.setObjectName("fig_mean_F")
        # Load image file and create QPixmap object
        pixmap1 = QPixmap(self.pupil_image)        
        # Create QGraphicsScene and add the QPixmap object
        scene1 = QGraphicsScene(self.fig_pupil)
        item1 = QGraphicsPixmapItem(pixmap1)
        scene1.addItem(item1)        
        # Set the QGraphicsScene to the QGraphicsView object
        self.fig_pupil.setScene(scene1)
        
        pixmap2 = QPixmap(self.motion_image)
        scene2 = QGraphicsScene(self.fig_motion)
        item2 = QGraphicsPixmapItem(pixmap2)
        scene2.addItem(item2)
        self.fig_motion.setScene(scene2)

        pixmap3 = QPixmap(self.mean_F_image)
        scene3 = QGraphicsScene(self.fig_mean_F)
        item3 = QGraphicsPixmapItem(pixmap3)
        scene3.addItem(item3)
        self.fig_mean_F.setScene(scene3)

        self.line_neuropil = QtWidgets.QLineEdit(self.centralwidget)
        self.line_neuropil.setGeometry(QtCore.QRect(700, 840, 100, 30))
        self.line_neuropil.setObjectName("line_neuropil")


        self.line_first_frame = QtWidgets.QLineEdit(self.centralwidget)
        self.line_first_frame.setGeometry(QtCore.QRect(20, 260, 110, 30))
        self.line_first_frame.setObjectName("line_first_frame")


        self.line_last_frame = QtWidgets.QLineEdit(self.centralwidget)
        self.line_last_frame.setGeometry(QtCore.QRect(170, 260, 110, 30))
        self.line_last_frame.setObjectName("line_last_frame")

     ################
        self.line_F_first_frame = QtWidgets.QLineEdit(self.centralwidget)
        self.line_F_first_frame.setGeometry(QtCore.QRect(20, 780, 110, 30))
        self.line_F_first_frame.setObjectName("line_F_first_frame")


        self.line_F_last_frame = QtWidgets.QLineEdit(self.centralwidget)
        self.line_F_last_frame.setGeometry(QtCore.QRect(170, 780, 110, 30))
        self.line_F_last_frame.setObjectName("line_F_last_frame")
     ###############################

        self.Button_yes = QtWidgets.QPushButton(self.centralwidget)
        self.Button_yes.setGeometry(QtCore.QRect(370, 840, 110, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.Button_yes.setFont(font)
        self.Button_yes.setStyleSheet("color: rgb(255, 255, 255);")
        self.Button_yes.setObjectName("Button_yes")
        self.Button_yes.clicked.connect(self.yes)


        self.Button_F0_hamming = QtWidgets.QPushButton(self.centralwidget)
        self.Button_F0_hamming.setGeometry(QtCore.QRect(370, 900, 110, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(50)
        self.Button_F0_hamming.setFont(font)
        self.Button_F0_hamming.setStyleSheet("color: rgb(255, 255, 255);")
        self.Button_F0_hamming.setObjectName("Hamming Window")
        self.Button_F0_hamming.clicked.connect(self.Hamming_window)

        self.Button_F0_sliding = QtWidgets.QPushButton(self.centralwidget)
        self.Button_F0_sliding.setGeometry(QtCore.QRect(220, 900, 110, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(50)
        self.Button_F0_sliding.setFont(font)
        self.Button_F0_sliding.setStyleSheet("color: rgb(255, 255, 255);")
        self.Button_F0_sliding.setObjectName("Sliding Window")
        self.Button_F0_sliding.clicked.connect(self.sliding_window)

        self.Button_no = QtWidgets.QPushButton(self.centralwidget)
        self.Button_no.setGeometry(QtCore.QRect(220, 840, 110, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.Button_no.setFont(font)
        self.Button_no.setStyleSheet("color: rgb(255, 255, 255);")
        self.Button_no.setObjectName("Button_no")
        self.Button_no.clicked.connect(self.No)

        self.label_y_min = QtWidgets.QLabel(self.centralwidget)
        self.label_y_min.setGeometry(QtCore.QRect(20, 850, 100, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_y_min.setFont(font)
        self.label_y_min.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_y_min.setAlignment(QtCore.Qt.AlignLeft)
        self.label_y_min.setObjectName("label_y_min")
        ##################################################
        self.label_Analyze_face = QtWidgets.QLabel(self.centralwidget)
        self.label_Analyze_face.setGeometry(QtCore.QRect(320,230, 150, 30))
        font = QtGui.QFont()
        font.setWeight(75)
        self.label_Analyze_face.setFont(font)
        self.label_Analyze_face.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_Analyze_face.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Analyze_face.setObjectName("label_Analyze_face")

        self.buttom_face_yes = QtWidgets.QPushButton(self.centralwidget)
        self.buttom_face_yes.setGeometry(QtCore.QRect(340,259, 110, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(50)
        self.buttom_face_yes.setFont(font)
        self.buttom_face_yes.setStyleSheet("color: rgb(255, 255, 255);")
        self.buttom_face_yes.setObjectName("Hamming Window")
        self.buttom_face_yes.clicked.connect(self.FaceYes)

        self.button_face_no = QtWidgets.QPushButton(self.centralwidget)
        self.button_face_no.setGeometry(QtCore.QRect(340,289, 110, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(50)
        self.button_face_no.setFont(font)
        self.button_face_no.setStyleSheet("color: rgb(255, 255, 255);")
        self.button_face_no.setObjectName("Sliding Window")
        self.button_face_no.clicked.connect(self.FaceNO)

        ##################################################
        self.label_Analyze_pupil = QtWidgets.QLabel(self.centralwidget)
        self.label_Analyze_pupil.setGeometry(QtCore.QRect(490,230, 150, 30))
        font = QtGui.QFont()
        font.setWeight(75)
        self.label_Analyze_pupil.setFont(font)
        self.label_Analyze_pupil.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_Analyze_pupil.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Analyze_pupil.setObjectName("label_Analyze_pupil")

        self.buttom_pupil_yes = QtWidgets.QPushButton(self.centralwidget)
        self.buttom_pupil_yes.setGeometry(QtCore.QRect(510,259, 110, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(50)
        self.buttom_pupil_yes.setFont(font)
        self.buttom_pupil_yes.setStyleSheet("color: rgb(255, 255, 255);")
        self.buttom_pupil_yes.setObjectName("Hamming Window")
        self.buttom_pupil_yes.clicked.connect(self.PupilYes)

        self.button_pupil_no = QtWidgets.QPushButton(self.centralwidget)
        self.button_pupil_no.setGeometry(QtCore.QRect(510,289, 110, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(50)
        self.button_pupil_no.setFont(font)
        self.button_pupil_no.setStyleSheet("color: rgb(255, 255, 255);")
        self.button_pupil_no.setObjectName("Sliding Window")
        self.button_pupil_no.clicked.connect(self.PupilNo)
        ##################################################

        self.label_neuropil = QtWidgets.QLabel(self.centralwidget)
        self.label_neuropil.setGeometry(QtCore.QRect(650, 800, 150, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_neuropil.setFont(font)
        self.label_neuropil.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_neuropil.setAlignment(QtCore.Qt.AlignCenter)
        self.label_neuropil.setObjectName("Neuropil")

        self.label_F0 = QtWidgets.QLabel(self.centralwidget)
        self.label_F0.setGeometry(QtCore.QRect(20, 910, 130, 30))
        font = QtGui.QFont()
        font.setWeight(70)
        self.label_F0.setFont(font)
        self.label_F0.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_F0.setAlignment(QtCore.Qt.AlignLeft)
        self.label_F0.setObjectName("label_F0")

        self.label_First_frame = QtWidgets.QLabel(self.centralwidget)
        self.label_First_frame.setGeometry(QtCore.QRect(20, 235, 110, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_First_frame.setFont(font)
        self.label_First_frame.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_First_frame.setAlignment(QtCore.Qt.AlignCenter)
        self.label_First_frame.setObjectName("label_First_frame")

        self.label_last_frame = QtWidgets.QLabel(self.centralwidget)
        self.label_last_frame.setGeometry(QtCore.QRect(170, 235, 110, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_last_frame.setFont(font)
        self.label_last_frame.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_last_frame.setAlignment(QtCore.Qt.AlignCenter)
        self.label_last_frame.setObjectName("label_F_last_frame")

        ########################################

        self.label_F_First_frame = QtWidgets.QLabel(self.centralwidget)
        self.label_F_First_frame.setGeometry(QtCore.QRect(20, 750, 110, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_F_First_frame.setFont(font)
        self.label_F_First_frame.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_F_First_frame.setAlignment(QtCore.Qt.AlignCenter)
        self.label_F_First_frame.setObjectName("label_F_First_frame")

        self.label_F_last_frame = QtWidgets.QLabel(self.centralwidget)
        self.label_F_last_frame.setGeometry(QtCore.QRect(170, 750, 110, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_F_last_frame.setFont(font)
        self.label_F_last_frame.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_F_last_frame.setAlignment(QtCore.Qt.AlignCenter)
        self.label_F_last_frame.setObjectName("label_last_frame")

        ################################################

        self.Button_OK = QtWidgets.QPushButton(self.centralwidget)
        self.Button_OK.setGeometry(QtCore.QRect(690, 900, 110, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.Button_OK.setFont(font)
        self.Button_OK.setStyleSheet("color: rgb(255, 255, 255);")
        self.Button_OK.setObjectName("Button_OK")
        self.Button_OK.clicked.connect(self.get_inputs)
        
        self.label_Motion_Energy = QtWidgets.QLabel(self.centralwidget)
        self.label_Motion_Energy.setGeometry(QtCore.QRect(20, 10, 90, 20))
        self.label_Motion_Energy.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_Motion_Energy.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Motion_Energy.setObjectName("label_Motion_Energy")
        self.label_Pupil_energy = QtWidgets.QLabel(self.centralwidget)
        self.label_Pupil_energy.setGeometry(QtCore.QRect(20, 305, 90, 20))
        self.label_Pupil_energy.setFont(font)
        self.label_Pupil_energy.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_Pupil_energy.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Pupil_energy.setObjectName("label_Pupil_energy")

        self.label_Mean_F = QtWidgets.QLabel(self.centralwidget)
        self.label_Mean_F.setGeometry(QtCore.QRect(20, 525, 150, 20))
        self.label_Mean_F.setFont(font)
        self.label_Mean_F.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_Mean_F.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Mean_F.setObjectName("label_Mean_F")

        self.label_titel = QtWidgets.QLabel(self.centralwidget)
        self.label_titel.setGeometry(QtCore.QRect(160, 0, 471, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_titel.setFont(font)
        self.label_titel.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_titel.setAlignment(QtCore.Qt.AlignCenter)
        self.label_titel.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 836, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def get_inputs(self):
        self.first_frame = int(self.line_first_frame.text())
        self.last_frame = int(self.line_last_frame.text())
        self.first_dF = int(self.line_F_first_frame.text())
        self.last_dF = int(self.line_F_last_frame.text())
        self.neuropil = float(self.line_neuropil.text())
        print("Neuropil impact factor is ", self.neuropil)
        print(self.remove_blink)
    def sliding_window(self):
        self.F0 = "sliding"
        print("F0 will use Sliding window")
    def Hamming_window(self):
        self.F0 = "hamming"
        print("F0 will use Hamming window")
    def yes(self):
        self.remove_blink = 1
        print("blinking frames will be removed")
    def No(self):
        self.remove_blink = 0
        print("No frames for pupil will be removed")
    def FaceYes(self):
        self.FaceAnalyze = 1
        print("face will be Analyzed")
    def FaceNO(self):
        self.FaceAnalyze = 0
        print("face will not be Analyzed")
    def PupilYes(self):
        self.PupilAnalyze = 1
        print("pupil will be analyzed")
    def PupilNo(self):
        self.PupilAnalyze = 0
        print("Pupil will be analyzed")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Setting"))
        self.label_y_min.setText(_translate("MainWindow", "Remove blinking?"))
        self.label_Analyze_face.setText(_translate("MainWindow", "Do you Analyze Face?"))
        self.label_Analyze_pupil.setText(_translate("MainWindow", "Do you Analyze Pupil?"))
        self.label_First_frame.setText(_translate("MainWindow", "First Frame"))
        self.label_F_First_frame.setText(_translate("MainWindow", "F first frame"))
        self.label_F_last_frame.setText(_translate("MainWindow", "F last frame"))
        self.label_last_frame.setText(_translate("MainWindow", "Last Frame"))
        self.Button_OK.setText(_translate("MainWindow", "OK"))
        self.label_Motion_Energy.setText(_translate("MainWindow", "Motion Energy"))
        self.label_Pupil_energy.setText(_translate("MainWindow", "Pupil Energy"))
        self.label_Mean_F.setText(_translate("MainWindow", "Mean raw F for All ROI"))
        self.label_titel.setWhatsThis(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\"><br/></span></p></body></html>"))
        self.label_titel.setText(_translate("MainWindow", "Remove extra frames"))
        self.line_first_frame.setText(_translate("MainWindow",f"{self.first}"))
        self.line_last_frame.setText(_translate("MainWindow",f"{self.last}"))
        self.line_F_first_frame.setText(_translate("MainWindow",f"{self.first_dF}"))
        self.line_F_last_frame.setText(_translate("MainWindow",f"{self.last_dF}"))
        self.Button_yes.setText(_translate("MainWindow", "Yes"))
        self.Button_F0_hamming.setText(_translate("MainWindow", "Hamming Window"))
        self.Button_F0_sliding.setText(_translate("MainWindow", "Sliding Window"))
        self.Button_no.setText(_translate("MainWindow", "NO"))
        self.label_F0.setText(_translate("MainWindow", "F0 Calculation Method"))
        self.label_neuropil.setText(_translate("MainWindow", "Neuropil impact factor"))
        self.line_neuropil.setText(_translate("MainWindow", f"{self.neuropil}"))
        self.button_face_no.setText(_translate("MainWindow", "NO"))
        self.buttom_face_yes.setText(_translate("MainWindow", "Yes"))
        self.button_pupil_no.setText(_translate("MainWindow", "NO"))
        self.buttom_pupil_yes.setText(_translate("MainWindow", "Yes"))




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow,figure_path, data_path)
    MainWindow.show()
    app.exec_()