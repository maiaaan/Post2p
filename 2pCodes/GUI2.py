from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog
import os
class Ui_MainWindow(object):
    def setupUi(self, MainWindow, figure_path, LenData):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(580, 775)
        MainWindow.setStyleSheet("background-color: rgb(27, 27, 27);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.tabGeneral = QtWidgets.QTabWidget(self.centralwidget)
        self.tabGeneral.setStyleSheet("background-color: rgb(27, 27, 27);\n"
"color: rgb(177, 177, 177);")
        self.tabGeneral.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabGeneral.setTabShape(QtWidgets.QTabWidget.Triangular)
        self.tabGeneral.setUsesScrollButtons(True)
        self.tabGeneral.setDocumentMode(True)
        self.tabGeneral.setTabsClosable(False)
        self.tabGeneral.setTabBarAutoHide(False)
        self.tabGeneral.setObjectName("tabGeneral")
        self.tabsetting = QtWidgets.QWidget()
        self.tabsetting.setObjectName("tabsetting")
        #----------------------------variables-----------------------------
        self.recording_date = None
        self.upload_metadata = False
        self.F0_method = 'sliding'
        self.first_frame = 0
        self.last_frame= LenData
        self.mouseLine = ""
        self.mousecode = str
        self.selected_neuron = None
        self.mouse_Genotype = None
        self.sensor = None
        self.sex = None
        self.selected_screen_state = None
        self.session = str
        self.min_AS_win = 75
        self.min_Run_win = 105
        self.min_Rest_win = 45
        self.min_PM_win = 60
        self.speed_filter = 10
        self.num_permutation = 1000
        self.motion_filter = 15
        self.speed_threshold = 0.5
        self.skew_th = 2.5
        self.syn_itter = 10
        self.alpha = 0.7
        self.motion_th = 2
        self.mean_F_image = os.path.join(figure_path, "raw_mean_F.png")
        self.Pupil_image = os.path.join(figure_path, "pupil.png")
        self.facemotion_image = os.path.join(figure_path, "raw_face_motion.png")
        # ---------------------------------------------------------------
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tabsetting)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_7 = QtWidgets.QLabel(self.tabsetting)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_7.setFont(font)
        self.label_7.setInputMethodHints(QtCore.Qt.ImhMultiLine)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_6.addWidget(self.label_7)
        self.line = QtWidgets.QFrame(self.tabsetting)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_6.addWidget(self.line)
        self.verticalLayout_2.addLayout(self.verticalLayout_6)
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.SpinBox_motion_th = QtWidgets.QDoubleSpinBox(self.tabsetting)
        self.SpinBox_motion_th.setObjectName("SpinBox_motion_th")
        self.SpinBox_motion_th.setValue(self.motion_th)
        self.gridLayout_6.addWidget(self.SpinBox_motion_th, 3, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.tabsetting)
        self.label_5.setObjectName("label_5")
        self.gridLayout_6.addWidget(self.label_5, 6, 1, 1, 1)
        self.label_speed_filter = QtWidgets.QLabel(self.tabsetting)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_speed_filter.setFont(font)
        self.label_speed_filter.setObjectName("label_speed_filter")
        self.gridLayout_6.addWidget(self.label_speed_filter, 2, 0, 1, 1)
        self.label_motion_filter = QtWidgets.QLabel(self.tabsetting)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_motion_filter.setFont(font)
        self.label_motion_filter.setObjectName("label_motion_filter")
        self.gridLayout_6.addWidget(self.label_motion_filter, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.tabsetting)
        self.label_2.setObjectName("label_2")
        self.gridLayout_6.addWidget(self.label_2, 4, 0, 1, 1)
        self.spinBox_motion_filter = QtWidgets.QSpinBox(self.tabsetting)
        self.spinBox_motion_filter.setObjectName("spinBox")
        self.gridLayout_6.addWidget(self.spinBox_motion_filter, 1, 1, 1, 1)
        self.spinBox_motion_filter.setValue(self.motion_filter)
        self.lineEdit_min_AS = QtWidgets.QLineEdit(self.tabsetting)
        self.lineEdit_min_AS.setObjectName("lineEdit_min_AS")
        self.gridLayout_6.addWidget(self.lineEdit_min_AS, 7, 0, 1, 1)
        self.lineEdit_min_Run_win = QtWidgets.QLineEdit(self.tabsetting)
        self.lineEdit_min_Run_win.setObjectName("lineEdit_min_Run_win")
        self.gridLayout_6.addWidget(self.lineEdit_min_Run_win, 5, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.tabsetting)
        self.label_3.setObjectName("label_3")
        self.gridLayout_6.addWidget(self.label_3, 4, 1, 1, 1)
        self.lineEdit_min_Rest_win = QtWidgets.QLineEdit(self.tabsetting)
        self.lineEdit_min_Rest_win.setObjectName("lineEdit_min_Rest_win")
        self.gridLayout_6.addWidget(self.lineEdit_min_Rest_win, 5, 1, 1, 1)
        self.label_motionThr = QtWidgets.QLabel(self.tabsetting)
        self.label_motionThr.setObjectName("label_motionThr")
        self.gridLayout_6.addWidget(self.label_motionThr, 2, 1, 1, 1)
        self.label_filter_kernel = QtWidgets.QLabel(self.tabsetting)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_filter_kernel.setFont(font)
        self.label_filter_kernel.setObjectName("label_filter_kernel")
        self.gridLayout_6.addWidget(self.label_filter_kernel, 0, 0, 1, 1)
        self.SpinBox_speed_th = QtWidgets.QDoubleSpinBox(self.tabsetting)
        self.SpinBox_speed_th.setObjectName("SpinBox_speed_th")
        self.SpinBox_speed_th.setValue(self.speed_threshold)
        self.gridLayout_6.addWidget(self.SpinBox_speed_th, 3, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.tabsetting)
        self.label_4.setObjectName("label_4")
        self.gridLayout_6.addWidget(self.label_4, 6, 0, 1, 1)
        self.lineEdit_speed_filter = QtWidgets.QLineEdit(self.tabsetting)
        self.lineEdit_speed_filter.setObjectName("lineEdit_speed_filter")
        self.gridLayout_6.addWidget(self.lineEdit_speed_filter, 1, 0, 1, 1)
        self.lineEdit_min_PM_win = QtWidgets.QLineEdit(self.tabsetting)
        self.lineEdit_min_PM_win.setObjectName("lineEdit_min_PM_win")
        self.gridLayout_6.addWidget(self.lineEdit_min_PM_win, 7, 1, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout_6)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_6 = QtWidgets.QLabel(self.tabsetting)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_5.addWidget(self.label_6)
        self.SpinBox_skew = QtWidgets.QDoubleSpinBox(self.tabsetting)
        self.SpinBox_skew.setObjectName("SpinBox_skew")
        self.SpinBox_skew.setValue(self.skew_th)
        self.verticalLayout_5.addWidget(self.SpinBox_skew)
        self.label_itt = QtWidgets.QLabel(self.tabsetting)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_itt.setFont(font)
        self.label_itt.setObjectName("label_itt")
        self.verticalLayout_5.addWidget(self.label_itt)
        self.spinBox_itt = QtWidgets.QSpinBox(self.tabsetting)
        self.spinBox_itt.setObjectName("spinBox_itt")
        self.spinBox_itt.setValue(self.syn_itter)
        self.verticalLayout_5.addWidget(self.spinBox_itt)
        self.label_11 = QtWidgets.QLabel(self.tabsetting)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_5.addWidget(self.label_11)
        self.lineEdit_permutation = QtWidgets.QLineEdit(self.tabsetting)
        self.lineEdit_permutation.setObjectName("lineEdit_permutation")
        self.verticalLayout_5.addWidget(self.lineEdit_permutation)
        self.comboBox_F0_method = QtWidgets.QComboBox(self.tabsetting)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.comboBox_F0_method.setFont(font)
        self.comboBox_F0_method.setIconSize(QtCore.QSize(16, 16))
        self.comboBox_F0_method.setObjectName("comboBox_F0_method")
        self.comboBox_F0_method.addItem("")
        self.comboBox_F0_method.addItem("")
        self.comboBox_F0_method.addItem("")
        default_index = 2
        self.comboBox_F0_method.setCurrentIndex(default_index)
        self.verticalLayout_5.addWidget(self.comboBox_F0_method)
        self.label_alpha_factor = QtWidgets.QLabel(self.tabsetting)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_alpha_factor.setFont(font)
        self.label_alpha_factor.setObjectName("label_alpha_factor")
        self.verticalLayout_5.addWidget(self.label_alpha_factor)
        self.SpinBox_alpha_factor = QtWidgets.QDoubleSpinBox(self.tabsetting)
        self.SpinBox_alpha_factor.setObjectName("SpinBox_alpha_factor")
        self.SpinBox_alpha_factor.setValue(self.alpha)
        self.verticalLayout_5.addWidget(self.SpinBox_alpha_factor)
        self.generate_figure_checkBox = QtWidgets.QCheckBox(self.tabsetting)
        self.generate_figure_checkBox.setObjectName("generate_figure_checkBox")
        self.verticalLayout_5.addWidget(self.generate_figure_checkBox)
        self.Convolve_checkBox = QtWidgets.QCheckBox(self.tabsetting)
        self.Convolve_checkBox.setObjectName("Convolve_checkBox")
        self.verticalLayout_5.addWidget(self.Convolve_checkBox)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem)
        #------------------------------------------
        self.Metadata_push = QtWidgets.QPushButton(self.tabsetting)
        self.Metadata_push.setObjectName("pushButton")
        self.Metadata_push.clicked.connect(self.get_Metadata)
        self.verticalLayout_5.addWidget(self.Metadata_push)
        #----------------------------------------
        self.savesetting_pushButton = QtWidgets.QPushButton(self.tabsetting)
        self.savesetting_pushButton.setObjectName("pushButton")
        self.savesetting_pushButton.clicked.connect(self.get_setting_input)
        self.verticalLayout_5.addWidget(self.savesetting_pushButton)
        self.verticalLayout_2.addLayout(self.verticalLayout_5)
        self.tabGeneral.addTab(self.tabsetting, "")
        self.General = QtWidgets.QWidget()
        self.General.setObjectName("General")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.General)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_mean_F = QtWidgets.QLabel(self.General)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_mean_F.setFont(font)
        self.label_mean_F.setObjectName("label_mean_F")
        self.verticalLayout_4.addWidget(self.label_mean_F)
        self.graphicsView_mean_F = QtWidgets.QGraphicsView(self.General)
        self.graphicsView_mean_F.setObjectName("graphicsView_mean_F")
        self.verticalLayout_4.addWidget(self.graphicsView_mean_F)
        self.label_pupil = QtWidgets.QLabel(self.General)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_pupil.setFont(font)
        self.label_pupil.setObjectName("label_pupil")
        self.verticalLayout_4.addWidget(self.label_pupil)
        self.graphicsView_pupil = QtWidgets.QGraphicsView(self.General)
        self.graphicsView_pupil.setObjectName("graphicsView_pupil")
        self.verticalLayout_4.addWidget(self.graphicsView_pupil)
        self.label_facemotion = QtWidgets.QLabel(self.General)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_facemotion.setFont(font)
        self.label_facemotion.setObjectName("label_facemotion")
        self.verticalLayout_4.addWidget(self.label_facemotion)
        self.graphicsView_facemotion = QtWidgets.QGraphicsView(self.General)
        self.graphicsView_facemotion.setObjectName("graphicsView_facemotion")
        self.verticalLayout_4.addWidget(self.graphicsView_facemotion)
        self.verticalLayout_7.addLayout(self.verticalLayout_4)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        #-----------------------------------------------
        pixmap = QPixmap(self.facemotion_image)
        scene = QGraphicsScene(self.graphicsView_facemotion)
        item = QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.graphicsView_facemotion.setScene(scene)
        #-----------------------------------------------
        pixmap2 = QPixmap(self.Pupil_image)
        scene2 = QGraphicsScene(self.graphicsView_pupil)
        item2 = QGraphicsPixmapItem(pixmap2)
        scene2.addItem(item2)
        self.graphicsView_pupil.setScene(scene2)
        #----------------------------------------------
        pixmap3 = QPixmap(self.mean_F_image)
        scene3 = QGraphicsScene(self.graphicsView_mean_F)
        item3 = QGraphicsPixmapItem(pixmap3)
        scene3.addItem(item3)
        self.graphicsView_mean_F.setScene(scene3)
        #_____________________Test________________________
        self.checkBox_lag_analyse = QtWidgets.QCheckBox(self.General)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.checkBox_lag_analyse.setFont(font)
        self.checkBox_lag_analyse.setObjectName("checkBox_lag_analyse")
        self.verticalLayout.addWidget(self.checkBox_lag_analyse)
        self.checkBox_generate_metadata = QtWidgets.QCheckBox(self.General)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.checkBox_generate_metadata.setFont(font)
        self.checkBox_generate_metadata.setObjectName("checkBox_generate_metadata")
        self.verticalLayout.addWidget(self.checkBox_generate_metadata)
        self.checkBox_skew_analyse = QtWidgets.QCheckBox(self.General)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.checkBox_skew_analyse.setFont(font)
        self.checkBox_skew_analyse.setObjectName("checkBox_skew_analyse")
        self.verticalLayout.addWidget(self.checkBox_skew_analyse)
        self.checkBox_pupil_analyse = QtWidgets.QCheckBox(self.General)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.checkBox_pupil_analyse.setFont(font)
        self.checkBox_pupil_analyse.setObjectName("checkBox_pupil_analyse")
        self.verticalLayout.addWidget(self.checkBox_pupil_analyse)
        self.checkBox_face_analyse = QtWidgets.QCheckBox(self.General)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.checkBox_face_analyse.setFont(font)
        self.checkBox_face_analyse.setObjectName("checkBox_face_analyse")
        self.verticalLayout.addWidget(self.checkBox_face_analyse)
        self.checkBox_remove_blinking = QtWidgets.QCheckBox(self.General)
        self.checkBox_remove_blinking.setObjectName("checkBox_remove_blinking")
        self.verticalLayout.addWidget(self.checkBox_remove_blinking)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setHorizontalSpacing(8)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.comboBox_sensor = QtWidgets.QComboBox(self.General)
        self.comboBox_sensor.setObjectName("comboBox_sensor")
        self.comboBox_sensor.addItem("")
        self.comboBox_sensor.addItem("")
        self.comboBox_sensor.addItem("")
        self.comboBox_sensor.addItem("")
        self.gridLayout_3.addWidget(self.comboBox_sensor, 1, 0, 1, 1)
        self.comboBox_sex = QtWidgets.QComboBox(self.General)
        self.comboBox_sex.setObjectName("comboBox_sex")
        self.comboBox_sex.addItem("")
        self.comboBox_sex.addItem("")
        self.comboBox_sex.addItem("")
        self.gridLayout_3.addWidget(self.comboBox_sex, 2, 0, 1, 1)
        #------------------------------------
        self.dateEdit = QtWidgets.QDateEdit(self.General)
        self.dateEdit.setObjectName("dateEdit")
        self.gridLayout_3.addWidget(self.dateEdit, 2, 1, 1, 1)
        #------------------------------------
        self.comboBox_screen = QtWidgets.QComboBox(self.General)
        self.comboBox_screen.setObjectName("comboBox_screen")
        self.comboBox_screen.addItem("")
        self.comboBox_screen.addItem("")
        self.comboBox_screen.addItem("")
        self.comboBox_screen.addItem("")
        self.gridLayout_3.addWidget(self.comboBox_screen, 1, 1, 1, 1)
        self.comboBox_Genotype = QtWidgets.QComboBox(self.General)
        self.comboBox_Genotype.setObjectName("comboBox")
        self.comboBox_Genotype.addItem("")
        self.comboBox_Genotype.addItem("")
        self.comboBox_Genotype.addItem("")
        self.comboBox_Genotype.addItem("")
        self.gridLayout_3.addWidget(self.comboBox_Genotype, 0, 0, 1, 1)
        self.comboBox_N_type = QtWidgets.QComboBox(self.General)
        self.comboBox_N_type.setObjectName("comboBox_N_type")
        self.comboBox_N_type.addItem("")
        self.comboBox_N_type.addItem("")
        self.comboBox_N_type.addItem("")
        self.comboBox_N_type.addItem("")
        self.comboBox_N_type.addItem("")
        self.comboBox_N_type.addItem("")
        self.comboBox_N_type.addItem("")
        self.gridLayout_3.addWidget(self.comboBox_N_type, 0, 1, 1, 1)
        self.horizontalLayout_2.addLayout(self.gridLayout_3)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label = QtWidgets.QLabel(self.General)
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label)
        self.lineEdit_mouse_line = QtWidgets.QLineEdit(self.General)
        self.lineEdit_mouse_line.setObjectName("lineEdit")
        self.verticalLayout_3.addWidget(self.lineEdit_mouse_line)
        self.label_mouse_code = QtWidgets.QLabel(self.General)
        self.label_mouse_code.setObjectName("label_mouse_code")
        self.verticalLayout_3.addWidget(self.label_mouse_code)
        self.lineEdit_mouse_code = QtWidgets.QLineEdit(self.General)
        self.lineEdit_mouse_code.setObjectName("lineEdit_mouse_code")
        self.verticalLayout_3.addWidget(self.lineEdit_mouse_code)
        #___________________________________
        self.label_session = QtWidgets.QLabel(self.General)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_session.setFont(font)
        self.label_session.setObjectName("label session")
        self.verticalLayout_3.addWidget(self.label_session)
        self.lineEdit_session = QtWidgets.QLineEdit(self.General)
        self.lineEdit_session.setObjectName("lineEdit_session")
        self.verticalLayout_3.addWidget(self.lineEdit_session)
        #_____________________________________

        self.label_first_frame = QtWidgets.QLabel(self.General)
        self.label_first_frame.setObjectName("label_first_frame")
        self.verticalLayout_3.addWidget(self.label_first_frame)
        self.lineEdit_first_frame = QtWidgets.QLineEdit(self.General)
        self.lineEdit_first_frame.setObjectName("lineEdit_first_frame")
        self.verticalLayout_3.addWidget(self.lineEdit_first_frame)

        self.label_last_frame = QtWidgets.QLabel(self.General)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_last_frame.setFont(font)
        self.label_last_frame.setObjectName("label_last_frame")
        self.verticalLayout_3.addWidget(self.label_last_frame)

        self.lineEdit_last_frame = QtWidgets.QLineEdit(self.General)
        self.lineEdit_last_frame.setText("")
        self.lineEdit_last_frame.setObjectName("lineEdit_last_frame")
        self.verticalLayout_3.addWidget(self.lineEdit_last_frame)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        self.verticalLayout_7.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(2, -1, -1, -1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.graphicsView_3 = QtWidgets.QGraphicsView(self.General)
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.scene4 = QtWidgets.QGraphicsScene()
        self.graphicsView_3.setScene(self.scene4)
        self.horizontalLayout.addWidget(self.graphicsView_3)
        self.graphicsView_3.setFixedSize(600, 25)
        #_________________________________
        self.pushButton_co_directory = QtWidgets.QPushButton(self.General)
        self.pushButton_co_directory.setObjectName("pushButton_co_directory")
        self.pushButton_co_directory.clicked.connect(self.open_directory)
        self.horizontalLayout.addWidget(self.pushButton_co_directory)
        self.horizontalLayout.setStretch(0, 3)
        self.verticalLayout_7.addLayout(self.horizontalLayout)
        self.pushButton_OK = QtWidgets.QPushButton(self.General)
        self.pushButton_OK.setObjectName("pushButton_OK")
        self.verticalLayout_7.addWidget(self.pushButton_OK)
        self.pushButton_OK.clicked.connect(self.get_input)
        self.tabGeneral.addTab(self.General, "")
        self.gridLayout_4.addWidget(self.tabGeneral, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        self.tabGeneral.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def display_text(self, text, color):
        item = QtWidgets.QGraphicsTextItem(text)
        item.setDefaultTextColor(QtGui.QColor(color))
        self.scene4.addItem(item)

    def get_input(self):
        self.get_setting_input
        self.save_metadata = self.generate_file()
        self.session = str(self.lineEdit_session.text())
        self.first_frame = int(self.lineEdit_first_frame.text())
        self.last_frame = int(self.lineEdit_last_frame.text())
        self.mousecode = str(self.lineEdit_mouse_code.text())
        self.mouseLine = self.lineEdit_mouse_line.text()
        self.selected_neuron = self.comboBox_N_type.currentText()
        self.mouse_Genotype = self.comboBox_Genotype.currentText()
        self.selected_screen_state = self.comboBox_screen.currentText()
        self.sensor = self.comboBox_sensor.currentText()
        self.sex = self.comboBox_sex.currentText()
        self.recording_date = self.dateEdit.date().toString("yyyy-MM-dd")
        return self.recording_date

    def get_setting_input(self):
        self.F0_method = self.comboBox_F0_method.currentText()
        self.motion_filter = int(self.spinBox_motion_filter.value())
        self.skew_th = self.SpinBox_skew.value()
        self.motion_th = self.SpinBox_motion_th.value()
        self.alpha = float(self.SpinBox_alpha_factor.value())
        self.speed_threshold = float(self.SpinBox_speed_th.value())
        self.num_permutation =int(self.lineEdit_permutation.text())
        self.min_AS_win = int(self.lineEdit_min_AS.text())
        self.min_Run_win = int(self.lineEdit_min_Run_win.text())
        self.min_Rest_win = int(self.lineEdit_min_Rest_win.text())
        self.speed_filter = float(self.lineEdit_speed_filter.text())
        self.min_PM_win = int(self.lineEdit_min_PM_win.text())
        self.syn_itter = int(self.spinBox_itt.value())
    def get_Metadata(self):
        self.upload_metadata = True
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        self.meta_data_directory = QFileDialog.getExistingDirectory(None, "Select metadata file", options=options)
    def generate_file(self):
        metadata = {'Mouse_line': self.mouseLine,
                   'Mouse_Code': self.mousecode,
                   'Genotype' : self.mouse_Genotype,
                   'Sex': self.sex,
                   'Date_of_record' : self.recording_date,
                   'Neuron_type' : self.selected_neuron,
                   'Screen_state' : self.selected_screen_state,
                   'Sensor' : self.sensor,
                   'Session' : self.session,
                   }
        return metadata

    def open_directory(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        self.directory = QFileDialog.getExistingDirectory(None, "Select Directory", options=options)
        if self.directory:
            self.display_text(self.directory, "white")



    def get_face_state(self):
        return self.checkBox_face_analyse.isChecked()

    def get_pupil_state(self):
        return self.checkBox_pupil_analyse.isChecked()

    def get_generate_metadata(self):
            return self.checkBox_generate_metadata.isChecked()
    def get_lag_state(self):
            return self.checkBox_lag_analyse.isChecked()

    def get_skew_state(self):
            return self.checkBox_skew_analyse.isChecked()

    def get_blink_state(self):
            return self.checkBox_remove_blinking.isChecked()
    def get_convolve_state(self):
        return self.Convolve_checkBox.isChecked()

    def get_generate_f_state(self):
        return self.generate_figure_checkBox.isChecked()


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.lineEdit_first_frame.setText(_translate("MainWindow", f"{self.first_frame}"))
        self.lineEdit_first_frame.setStyleSheet("color: white")
        self.lineEdit_last_frame.setText(_translate("MainWindow", f"{self.last_frame}"))
        self.lineEdit_last_frame.setStyleSheet("color: white")
        self.lineEdit_mouse_line.setText(_translate("MainWindow", ""))
        self.lineEdit_min_AS.setText(_translate("MainWindow", f"{self.min_AS_win}"))
        self.lineEdit_min_AS.setStyleSheet("color: white")
        self.lineEdit_permutation.setText(_translate("MainWindow", f"{self.num_permutation}"))
        self.lineEdit_permutation.setStyleSheet("color: white")
        self.lineEdit_min_Run_win.setText(_translate("MainWindow", f"{self.min_Run_win}"))
        self.lineEdit_min_Run_win.setStyleSheet("color: white")
        self.lineEdit_min_Rest_win.setText(_translate("MainWindow", f"{self.min_Rest_win}"))
        self.lineEdit_min_Rest_win.setStyleSheet("color: white")
        self.lineEdit_min_PM_win.setText(_translate("MainWindow", f"{self.min_PM_win}"))
        self.lineEdit_min_PM_win.setStyleSheet("color: white")
        self.lineEdit_speed_filter.setText(_translate("MainWindow", f"{self.speed_filter}"))
        self.lineEdit_speed_filter.setStyleSheet("color: white")
        self.label_7.setText(_translate("MainWindow", "Motion "))
        self.label_5.setText(_translate("MainWindow", "Minimum PM window(s)"))
        self.label_speed_filter.setText(_translate("MainWindow", "Speed threshold"))
        self.label_motion_filter.setText(_translate("MainWindow", "Motion filter kernel"))
        self.label_2.setText(_translate("MainWindow", "Minimum Run window(s)"))
        self.label_3.setText(_translate("MainWindow", "Minimum Rest window(s)"))
        self.label_motionThr.setText(_translate("MainWindow", "motion threshold(*times std)"))
        self.label_filter_kernel.setText(_translate("MainWindow", "Speed filter kernel"))
        self.label_4.setText(_translate("MainWindow", "Minimum AS window(s)"))
        self.label_6.setText(_translate("MainWindow", "Skewness threshold"))
        self.label_itt.setText(_translate("MainWindow", "Number of itteration (synchrony)"))
        self.label_11.setText(_translate("MainWindow", "Number of permutations"))
        self.comboBox_F0_method.setItemText(0, _translate("MainWindow", "F0 calculation Method"))
        self.comboBox_F0_method.setItemText(1, _translate("MainWindow", "hamming"))
        self.comboBox_F0_method.setItemText(2, _translate("MainWindow", "sliding"))
        self.label_alpha_factor.setText(_translate("MainWindow", "Alpha factor"))
        self.generate_figure_checkBox.setText(_translate("MainWindow", "Generate figure"))
        self.Convolve_checkBox.setText(_translate("MainWindow", "Convolve data"))
        self.savesetting_pushButton.setText(_translate("MainWindow", "Save change"))
        self.Metadata_push.setText(_translate("MainWindow", "Upload Metadata"))
        self.tabGeneral.setTabText(self.tabGeneral.indexOf(self.tabsetting), _translate("MainWindow", "setting"))
        self.label_mean_F.setText(_translate("MainWindow", "Mean Fluorescence trace"))
        self.label_pupil.setText(_translate("MainWindow", "Pupil trace"))
        self.label_facemotion.setText(_translate("MainWindow", "facemotion trace"))
        self.checkBox_lag_analyse.setText(_translate("MainWindow", "Lag analysis"))
        self.checkBox_generate_metadata.setText(_translate("MainWindow", "Generate metadata"))
        self.checkBox_skew_analyse.setText(_translate("MainWindow", "skew analysis"))
        self.checkBox_pupil_analyse.setText(_translate("MainWindow", "Pupil analysis"))
        self.checkBox_face_analyse.setText(_translate("MainWindow", "Face analysis"))
        self.checkBox_remove_blinking.setText(_translate("MainWindow", "Remove blinking"))
        self.comboBox_sensor.setItemText(0, _translate("MainWindow", "Sensor"))
        self.comboBox_sensor.setItemText(1, _translate("MainWindow", "m8"))
        self.comboBox_sensor.setItemText(2, _translate("MainWindow", "S6"))
        self.comboBox_sensor.setItemText(3, _translate("MainWindow", "f6"))
        self.comboBox_sex.setItemText(0, _translate("MainWindow", "Sex"))
        self.comboBox_sex.setItemText(1, _translate("MainWindow", "female"))
        self.comboBox_sex.setItemText(2, _translate("MainWindow", "male"))
        self.comboBox_screen.setItemText(0, _translate("MainWindow", "Screen state"))
        self.comboBox_screen.setItemText(1, _translate("MainWindow", "Dark"))
        self.comboBox_screen.setItemText(2, _translate("MainWindow", "Screen"))
        self.comboBox_screen.setItemText(3, _translate("MainWindow", "Stimulus"))
        self.comboBox_Genotype.setItemText(0, _translate("MainWindow", "Genotype"))
        self.comboBox_Genotype.setItemText(1, _translate("MainWindow", "Knockout"))
        self.comboBox_Genotype.setItemText(2, _translate("MainWindow", "wild"))
        self.comboBox_Genotype.setItemText(3, _translate("MainWindow", "Injected"))
        self.comboBox_N_type.setItemText(0, _translate("MainWindow", "Neuronl Type"))
        self.comboBox_N_type.setItemText(1, _translate("MainWindow", "PYR"))
        self.comboBox_N_type.setItemText(2, _translate("MainWindow", "VIP"))
        self.comboBox_N_type.setItemText(3, _translate("MainWindow", "SST"))
        self.comboBox_N_type.setItemText(4, _translate("MainWindow", "NDNF"))
        self.comboBox_N_type.setItemText(5, _translate("MainWindow", "PV"))
        self.comboBox_N_type.setItemText(6, _translate("MainWindow", "SNCG"))
        self.label.setText(_translate("MainWindow", "Mouse Line"))
        self.label_mouse_code.setText(_translate("MainWindow", "Mouse Code"))
        self.label_first_frame.setText(_translate("MainWindow", "First frame"))
        self.label_last_frame.setText(_translate("MainWindow", "Last frame"))
        self.label_session.setText(_translate("MainWindow", "session"))
        self.pushButton_co_directory.setText(_translate("MainWindow", "compile directory"))
        self.pushButton_OK.setText(_translate("MainWindow", "OK"))
        self.tabGeneral.setTabText(self.tabGeneral.indexOf(self.General), _translate("MainWindow", "General"))


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self, image_file, data_file):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self,figure_path, LenData)
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    data_path = LenData
    figure_path = figure_path
    window = MyWindow(figure_path, LenData)
    window.show()
    app.exec_()