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

        #----------------------------Constants-----------------------------
        self.recording_date = None
        self.upload_metadata = False
        self.F0_method = 'sliding'
        self.first_frame = 0
        self.last_frame= LenData
        self.mouseLine = ""
        self.directory = ""
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

        #-------------------------- Display variables------------------------
        self.font = QtGui.QFont("DejaVu Sans", 9)

        #----------------------------Variables-----------------------------
        self.have_metadata = False

        #___________________________SETTING TAB______________________________
        
        self.tabsetting = QtWidgets.QWidget()
        self.tabsetting.setObjectName("tabsetting")

        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tabsetting)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")

        #Subtitle
        self.label_7 = QtWidgets.QLabel(self.tabsetting)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_7.setFont(font)
        self.label_7.setInputMethodHints(QtCore.Qt.ImhMultiLine)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_6.addWidget(self.label_7)

        #Line
        self.line = QtWidgets.QFrame(self.tabsetting)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")

        #_________________________________

        self.verticalLayout_6.addWidget(self.line)
        self.verticalLayout_2.addLayout(self.verticalLayout_6)

        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.verticalLayout_2.addLayout(self.gridLayout_6)       

        #Speed filter kernel
        self.label_filter_kernel, self.lineEdit_speed_filter = self.init_lineEdit("label_filter_kernel", "lineEdit_speed_filter", self.tabsetting)
        self.gridLayout_6.addWidget(self.label_filter_kernel, 0, 0, 1, 1)
        self.gridLayout_6.addWidget(self.lineEdit_speed_filter, 1, 0, 1, 1)

        #Motion filter kernel
        self.label_motion_filter, self.spinBox_motion_filter = self.init_spinbox("label_motion_filter", "spinBox", self.tabsetting, self.motion_filter)
        self.gridLayout_6.addWidget(self.label_motion_filter, 0, 1, 1, 1)
        self.gridLayout_6.addWidget(self.spinBox_motion_filter, 1, 1, 1, 1)

        #Speed threshold
        self.label_speed_filter, self.SpinBox_speed_th = self.init_spinbox("label_speed_filter", "SpinBox_speed_th", self.tabsetting, self.speed_threshold, double=True)
        self.gridLayout_6.addWidget(self.label_speed_filter, 2, 0, 1, 1)
        self.gridLayout_6.addWidget(self.SpinBox_speed_th, 3, 0, 1, 1)

        #motion threshold(*times std)
        self.label_motionThr, self.SpinBox_motion_th = self.init_spinbox("label_motionThr", "SpinBox_motion_th", self.tabsetting, self.motion_th, double=True)
        self.gridLayout_6.addWidget(self.label_motionThr, 2, 1, 1, 1)
        self.gridLayout_6.addWidget(self.SpinBox_motion_th, 3, 1, 1, 1)

        #Minimum Run window(s)
        self.label_2, self.lineEdit_min_Run_win = self.init_lineEdit("label_2", "lineEdit_min_Run_win", self.tabsetting)
        self.gridLayout_6.addWidget(self.label_2, 4, 0, 1, 1)
        self.gridLayout_6.addWidget(self.lineEdit_min_Run_win, 5, 0, 1, 1)

        #Minimum Rest window(s)
        self.label_3, self.lineEdit_min_Rest_win = self.init_lineEdit("label_3", "lineEdit_min_Rest_win", self.tabsetting)
        self.gridLayout_6.addWidget(self.label_3, 4, 1, 1, 1)
        self.gridLayout_6.addWidget(self.lineEdit_min_Rest_win, 5, 1, 1, 1)
        
        #Minimum AS window(s)
        self.label_4, self.lineEdit_min_AS = self.init_lineEdit("label_4", "lineEdit_min_AS", self.tabsetting)
        self.gridLayout_6.addWidget(self.label_4, 6, 0, 1, 1)
        self.gridLayout_6.addWidget(self.lineEdit_min_AS, 7, 0, 1, 1)

        #Minimum PM window(s)
        self.label_5, self.lineEdit_min_PM_win = self.init_lineEdit("label_5", "lineEdit_PM_win", self.tabsetting)
        self.gridLayout_6.addWidget(self.label_5, 6, 1, 1, 1)
        self.gridLayout_6.addWidget(self.lineEdit_min_PM_win, 7, 1, 1, 1)

        #_________________________________

        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")

        self.label_6, self.SpinBox_skew = self.init_spinbox("label_6", "SpinBox_skew", self.tabsetting, self.skew_th, self.verticalLayout_5, double=True)
        self.label_itt, self.spinBox_itt = self.init_spinbox("label_itt", "spinBox_itt", self.tabsetting, self.syn_itter, self.verticalLayout_5)
        self.label_11, self.lineEdit_permutation = self.init_lineEdit("label_11", "lineEdit_permutation", self.tabsetting, self.verticalLayout_5)
        self.comboBox_F0_method = self.init_combobox("comboBox_F0_method", self.tabsetting, 2, self.verticalLayout_5, 2)
        self.label_alpha_factor, self.SpinBox_alpha_factor = self.init_spinbox("label_alpha_factor", "SpinBox_alpha_factor", self.tabsetting, self.alpha, self.verticalLayout_5, double=True)
        self.SpinBox_alpha_factor.setRange(0., 1.)
        self.SpinBox_alpha_factor.setWrapping(True)

        self.generate_figure_checkBox = self.init_checkbox("generate_figure_checkBox", self.tabsetting, self.verticalLayout_5)
        self.Convolve_checkBox = self.init_checkbox("Convolve_checkBox", self.tabsetting, self.verticalLayout_5)
        
        #_________________________________

        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem)
        self.verticalLayout_2.addLayout(self.verticalLayout_5)

        self.savesetting_pushButton = QtWidgets.QPushButton(self.tabsetting)
        self.savesetting_pushButton.setObjectName("pushButton")
        self.savesetting_pushButton.clicked.connect(self.get_setting_input)
        self.verticalLayout_5.addWidget(self.savesetting_pushButton)

        self.Metadata_push = QtWidgets.QPushButton(self.tabsetting)
        self.Metadata_push.setObjectName("pushButton")
        self.Metadata_push.clicked.connect(self.get_Metadata)
        self.verticalLayout_5.addWidget(self.Metadata_push)

        #___________________________GENERAL TAB______________________________
        
        self.General = QtWidgets.QWidget()
        self.General.setObjectName("General")
        
        #----------------------------------------
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.General)
        self.verticalLayout_7.setObjectName("verticalLayout_7")

        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayout_7.addLayout(self.verticalLayout_4)
        
        self.label_mean_F, self.graphicsView_mean_F = self.init_traces("label_mean_F", "graphicsView_mean_F", self.mean_F_image, self.verticalLayout_4)
        self.label_pupil, self.graphicsView_pupil = self.init_traces("label_pupil", "graphicsView_pupil", self.Pupil_image, self.verticalLayout_4)
        self.label_facemotion, self.graphicsView_facemotion = self.init_traces("label_facemotion", "graphicsView_facemotion", self.facemotion_image, self.verticalLayout_4)

        #________________________________________
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2.addLayout(self.verticalLayout)

        self.checkBox_lag_analyse = self.init_checkbox("checkBox_lag_analyse", self.General, self.verticalLayout)
        self.checkBox_skew_analyse = self.init_checkbox("checkBox_skew_analyse", self.General, self.verticalLayout)
        self.checkBox_pupil_analyse = self.init_checkbox("checkBox_pupil_analyse", self.General, self.verticalLayout)
        self.checkBox_face_analyse = self.init_checkbox("checkBox_face_analyse", self.General, self.verticalLayout)
        self.checkBox_remove_blinking = self.init_checkbox("checkBox_remove_blinking", self.General, self.verticalLayout)
        self.checkBox_generate_metadata = self.init_checkbox("checkBox_generate_metadata", self.General, self.verticalLayout)

        #----------------------------------------
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)

        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setHorizontalSpacing(8)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.horizontalLayout_2.addLayout(self.gridLayout_3)

        self.comboBox_Genotype = self.init_combobox("comboBox", self.General, 3)
        self.gridLayout_3.addWidget(self.comboBox_Genotype, 0, 0, 1, 1)
        self.comboBox_N_type = self.init_combobox("comboBox_N_type", self.General, 6)
        self.gridLayout_3.addWidget(self.comboBox_N_type, 0, 1, 1, 1)
        self.comboBox_sensor = self.init_combobox("comboBox_sensor", self.General, 3)
        self.gridLayout_3.addWidget(self.comboBox_sensor, 1, 0, 1, 1)
        self.comboBox_screen = self.init_combobox("comboBox_screen", self.General, 3)
        self.gridLayout_3.addWidget(self.comboBox_screen, 1, 1, 1, 1)
        self.comboBox_sex = self.init_combobox("comboBox_sex", self.General, 2)
        self.gridLayout_3.addWidget(self.comboBox_sex, 2, 0, 1, 1)
        self.dateEdit = QtWidgets.QDateEdit(self.General)
        self.dateEdit.setObjectName("dateEdit")
        self.dateEdit.setFont(QtGui.QFont("DejaVu Sans", 10))
        self.gridLayout_3.addWidget(self.dateEdit, 2, 1, 1, 1)

        #----------------------------------------
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)

        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        
        self.label_mouse_line, self.lineEdit_mouse_line = self.init_lineEdit("label", "lineEdit", self.General, self.verticalLayout_3, self.font)
        self.label_mouse_code, self.lineEdit_mouse_code = self.init_lineEdit("label_mouse_code", "lineEdit_mouse_code", self.General, self.verticalLayout_3, self.font)  
        self.label_session, self.lineEdit_session = self.init_lineEdit("label session", "lineEdit_session", self.General, self.verticalLayout_3, self.font)
        self.label_first_frame, self.lineEdit_first_frame = self.init_lineEdit("label_first_frame", "lineEdit_first_frame", self.General, self.verticalLayout_3, self.font)
        self.label_last_frame, self.lineEdit_last_frame = self.init_lineEdit("label_last_frame", "lineEdit_last_frame", self.General, self.verticalLayout_3, self.font)  

        self.verticalLayout_7.addLayout(self.horizontalLayout_2)

        #_________________________________
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(2, -1, -1, -1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_7.addLayout(self.horizontalLayout)

        self.graphicsView_3 = QtWidgets.QGraphicsView(self.General)
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.scene4 = QtWidgets.QGraphicsScene()
        self.graphicsView_3.setScene(self.scene4)
        self.horizontalLayout.addWidget(self.graphicsView_3)
        self.graphicsView_3.setFixedSize(600, 25)

        self.pushButton_co_directory = QtWidgets.QPushButton(self.General)
        self.pushButton_co_directory.setObjectName("pushButton_co_directory")
        self.pushButton_co_directory.clicked.connect(self.open_directory)
        self.horizontalLayout.addWidget(self.pushButton_co_directory)
        self.horizontalLayout.setStretch(0, 3)

        self.pushButton_OK = QtWidgets.QPushButton(self.General)
        self.pushButton_OK.setObjectName("pushButton_OK")
        self.verticalLayout_7.addWidget(self.pushButton_OK)
        self.pushButton_OK.clicked.connect(self.get_input)

        #_________________________________

        self.tabGeneral.addTab(self.tabsetting, "")
        self.tabGeneral.addTab(self.General, "")
        
        self.gridLayout_4.addWidget(self.tabGeneral, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        self.statusbar.setStyleSheet("color: white;")
        MainWindow.setStatusBar(self.statusbar)
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        self.tabGeneral.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def display_text(self, text, color):
        for item in self.scene4.items():
            if isinstance(item, QtWidgets.QGraphicsTextItem):
                    self.scene4.removeItem(item)
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
        if self.directory == "":
            self.show_warning_popup("Do you want to analyze data without final compiling?")
        
        if not self.mouseLine.strip() and self.upload_metadata == False:
                self.statusbar.showMessage('Error: Enter mouse line or select a metadata file!', 5000)
        elif not self.mousecode.strip() and self.upload_metadata == False:
                self.statusbar.showMessage('Error: Enter mouse code or select a metadata file!', 5000)
        elif not self.session.strip() and self.upload_metadata == False:
                self.statusbar.showMessage('Error: Enter recording session or select a metadata file!', 5000)
        elif self.comboBox_sensor.currentText() == "Sensor" and self.upload_metadata == False:
                self.statusbar.showMessage('Error: Enter Sensor type or select a metadata file!', 5000)
        elif self.comboBox_sex.currentText() == "Sex" and self.upload_metadata == False:
                self.statusbar.showMessage('Error: choose a valid sex or select a metadata file', 5000)
        elif self.comboBox_screen.currentText() == "Screen state" and self.upload_metadata == False:
                self.statusbar.showMessage('Error: choose a valid screen state or select a metadata file', 5000)
        elif self.comboBox_Genotype.currentText() == "Genotype" and self.upload_metadata == False:
                self.statusbar.showMessage('Error: choose a valid Genotype or select a metadata file', 5000)
        elif self.comboBox_N_type.currentText() == "Neuronal Type" and self.upload_metadata == False:
                self.statusbar.showMessage('Error: choose a valid Neuronal Type or select a metadata file', 5000)
        else : 
            self.have_metadata = True
            self.statusbar.showMessage('Close the window', 0)
        return self.recording_date

    def show_warning_popup(self, message):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText(message)
        msg.setWindowTitle("Warning")
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        msg.buttonClicked.connect(self.handle_warning_response)
        msg.exec_()

    def handle_warning_response(self, button):
        if button.text() == "&Yes":
            self.display_text("Data will not be added to compile", "white")
            self.directory = "No_compile"
        elif button.text() == "&No":
                pass

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
        self.meta_data_directory = QFileDialog.getExistingDirectory(None, "Select metadata folder", options=options)
        self.statusbar.showMessage('Folder selected', 4000)

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

    def get_generate_svg_state(self):
        return self.generate_figure_checkBox.isChecked()
    
#Init widgets ensemble
    def init_lineEdit(self, name_label: str, name_lineedit: str, tab, vlayout=None, font=QtGui.QFont("DejaVu Sans", 10)):
        label = QtWidgets.QLabel(tab)
        label.setObjectName(name_label)
        label.setFont(font)
        lineedit = QtWidgets.QLineEdit(tab)
        lineedit.setObjectName(name_lineedit)
        lineedit.setFont(font)
        if vlayout != None : 
            vlayout.addWidget(label)
            vlayout.addWidget(lineedit)
        return label, lineedit
        
    def init_checkbox(self, name: str, tab, vlayout, font=QtGui.QFont("DejaVu Sans", 10)):
        checkbox = QtWidgets.QCheckBox(tab)
        checkbox.setFont(font)
        checkbox.setObjectName(name)
        vlayout.addWidget(checkbox)
        return checkbox
    
    def init_combobox(self, name: str, tab, nb_choice: int, vlayout=None, default_idx=0, font=QtGui.QFont("DejaVu Sans", 10)):
        combobox = QtWidgets.QComboBox(tab)
        combobox.setObjectName(name)
        combobox.setFont(font)
        for i in range(nb_choice+1):
            combobox.addItem("")
        combobox.setCurrentIndex(default_idx)
        if vlayout != None : 
            vlayout.addWidget(combobox)
        return combobox
    
    def init_spinbox(self, name_label: str, name_lineedit: str, tab, default_value=0.0 , vlayout=None, double=False, font=QtGui.QFont("DejaVu Sans", 10)):
        label = QtWidgets.QLabel(tab)
        label.setObjectName(name_label)
        label.setFont(font)
        if double :
             spinbox = QtWidgets.QDoubleSpinBox(tab)
             spinbox.setSingleStep(0.05)
        else :
             spinbox = QtWidgets.QSpinBox(tab)
        spinbox.setObjectName(name_lineedit)
        spinbox.setValue(default_value)
        spinbox.setFont(font)
        if vlayout != None : 
            vlayout.addWidget(label)
            vlayout.addWidget(spinbox)
        return label, spinbox
        
    def init_traces(self, name_label: str, name_graph: str, path, vlayout, font=QtGui.QFont("DejaVu Sans", 10)):
        label = QtWidgets.QLabel(self.General)
        label.setFont(font)
        label.setObjectName(name_label)
        vlayout.addWidget(label)
        graph = QtWidgets.QGraphicsView(self.General)
        graph.setObjectName(name_graph)
        vlayout.addWidget(graph)
        pixmap = QPixmap(path)
        scene = QGraphicsScene(graph)
        item = QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        graph.setScene(scene)
        return label, graph

#Manage texts display
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
        self.label_7.setText(_translate("MainWindow", "Motion"))
        self.label_5.setText(_translate("MainWindow", "Minimum PM window(s)"))
        self.label_speed_filter.setText(_translate("MainWindow", "Speed threshold"))
        self.label_motion_filter.setText(_translate("MainWindow", "Motion filter kernel"))
        self.label_2.setText(_translate("MainWindow", "Minimum Run window(s)"))
        self.label_3.setText(_translate("MainWindow", "Minimum Rest window(s)"))
        self.label_motionThr.setText(_translate("MainWindow", "Motion threshold(*times std)"))
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
        self.tabGeneral.setTabText(self.tabGeneral.indexOf(self.tabsetting), _translate("MainWindow", "Settings"))
        self.label_mean_F.setText(_translate("MainWindow", "Mean Fluorescence Trace"))
        self.label_pupil.setText(_translate("MainWindow", "Pupil Trace"))
        self.label_facemotion.setText(_translate("MainWindow", "Facemotion Trace"))
        self.checkBox_lag_analyse.setText(_translate("MainWindow", "Lag Analysis"))
        self.checkBox_generate_metadata.setText(_translate("MainWindow", "Generate Metadata"))
        self.checkBox_skew_analyse.setText(_translate("MainWindow", "Skew Analysis"))
        self.checkBox_pupil_analyse.setText(_translate("MainWindow", "Pupil Analysis"))
        self.checkBox_face_analyse.setText(_translate("MainWindow", "Face Analysis"))
        self.checkBox_remove_blinking.setText(_translate("MainWindow", "Remove blinking"))
        self.comboBox_sensor.setItemText(0, _translate("MainWindow", "Sensor"))
        self.comboBox_sensor.setItemText(1, _translate("MainWindow", "GCaMP8m"))
        self.comboBox_sensor.setItemText(2, _translate("MainWindow", "GCaMP6s"))
        self.comboBox_sensor.setItemText(3, _translate("MainWindow", "GCaMP6f"))
        self.comboBox_sex.setItemText(0, _translate("MainWindow", "Sex"))
        self.comboBox_sex.setItemText(1, _translate("MainWindow", "F"))
        self.comboBox_sex.setItemText(2, _translate("MainWindow", "M"))
        self.comboBox_screen.setItemText(0, _translate("MainWindow", "Screen state"))
        self.comboBox_screen.setItemText(1, _translate("MainWindow", "Dark"))
        self.comboBox_screen.setItemText(2, _translate("MainWindow", "Screen"))
        self.comboBox_screen.setItemText(3, _translate("MainWindow", "Stimulus"))
        self.comboBox_Genotype.setItemText(0, _translate("MainWindow", "Genotype"))
        self.comboBox_Genotype.setItemText(1, _translate("MainWindow", "Knockout"))
        self.comboBox_Genotype.setItemText(2, _translate("MainWindow", "wild"))
        self.comboBox_Genotype.setItemText(3, _translate("MainWindow", "Injected"))
        self.comboBox_N_type.setItemText(0, _translate("MainWindow", "Neuronal Type"))
        self.comboBox_N_type.setItemText(1, _translate("MainWindow", "PYR"))
        self.comboBox_N_type.setItemText(2, _translate("MainWindow", "VIP"))
        self.comboBox_N_type.setItemText(3, _translate("MainWindow", "SST"))
        self.comboBox_N_type.setItemText(4, _translate("MainWindow", "NDNF"))
        self.comboBox_N_type.setItemText(5, _translate("MainWindow", "PV"))
        self.comboBox_N_type.setItemText(6, _translate("MainWindow", "SNCG"))
        self.label_mouse_line.setText(_translate("MainWindow", "Mouse Line"))
        self.label_mouse_code.setText(_translate("MainWindow", "Mouse Code"))
        self.label_first_frame.setText(_translate("MainWindow", "First frame"))
        self.label_last_frame.setText(_translate("MainWindow", "Last frame"))
        self.label_session.setText(_translate("MainWindow", "Session"))
        self.pushButton_co_directory.setText(_translate("MainWindow", "Compile Directory"))
        self.pushButton_OK.setText(_translate("MainWindow", "OK"))
        self.tabGeneral.setTabText(self.tabGeneral.indexOf(self.General), _translate("MainWindow", "General"))
