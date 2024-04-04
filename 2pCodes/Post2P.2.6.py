import copy
import numpy as np
import pandas as pd
from scipy.stats import zscore
from PyQt5 import QtWidgets
import glob
import json
import os.path
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
import datetime
from scipy.stats import skew
import h5py
import figure
import motion_state
import functions
import running
import synchrony
import sys
import skewness2
import load_direction
from GUI2 import Ui_MainWindow
from openpyxl import load_workbook
# get date
current_date = datetime.date.today()
current_time = datetime.datetime.now().time()
# -----------------------------------Build directory and load data---------------------------
Base_path, save_data,save_direction1, save_direction_figure, save_direction_permutation,save_direction0_lag,\
    save_direction_skew, suite2p_path, facemap_path, xml_direction, hf = load_direction.get_directory()
cell, F, Fneu_raw, spikes, movement_file, pupil, motion = load_direction.load_data(suite2p_path, Base_path,facemap_path)
xml, channel_number, laserWavelength, objectiveLens, objectiveLensMag, opticalZoom,\
        bitDepth, dwellTime, framePeriod, micronsPerPixel, TwophotonLaserPower = load_direction.load_xml(xml_direction)
# -----------------------------------Detect Neurons between ROIs--------------------------
_ , detected_roi = functions.detect_cell(cell, F)
cell, neuron_chosen3 = functions.detect_bad_neuropils(detected_roi,Fneu_raw, F, cell,suite2p_path)
Fneu_raw, keeped_ROI = functions.detect_cell(cell, Fneu_raw)
spikes , _ = functions.detect_cell(cell, spikes)
F, _ = functions.detect_cell(cell, F)
#-----------------------------------Extract speed from movement file----------------------
movement = running.single_getspeed(movement_file[0], np.size(F, 1))
speed = np.copy(movement['speed'])
# ----------------------------------interpolating data-----------------------------------
x_inter = xml['Green']['relativeTime']
last = x_inter[-1]
xp = np.linspace(x_inter[0], last, len(pupil))
pupil = np.interp(x_inter, xp, pupil)
motion = np.interp(x_inter, xp,motion )
# ----------------------------------Making figure for GUI---------------------------------
TIM = np.arange(0,len(motion))
Mean_raw_F = np.mean(F, 0)
pupil = zscore(pupil)
len_data = len(speed)
print("mean raw F",len(Mean_raw_F))
print("TIM", len(TIM))
figure.GUIimage(TIM, motion, save_direction_figure, "raw_face_motion.png")
figure.GUIimage(TIM, pupil, save_direction_figure, "pupil.png")
figure.GUIimage(TIM, Mean_raw_F,save_direction_figure,"raw_mean_F.png")
#----------------------------------Load GUI-----------------------------------
class MyWindow(QtWidgets.QMainWindow):
    def __init__(self, image_file, data_file):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self,save_direction_figure, len_data)
if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    data_path = len_data
    figure_path = save_direction_figure
    window = MyWindow(save_direction_figure, len_data)
    window.show()
    app.exec_()
#-----------------------------Get data from GUI---------------------------------
upload_metadata = window.ui.upload_metadata
generate_metadata = window.ui.get_generate_metadata()

if upload_metadata:
    meta_data_directory = window.ui.meta_data_directory
    print(meta_data_directory)
    METADATA = glob.glob(os.path.join(meta_data_directory, '*data.txt'))[0]
    print(METADATA)
    with open( METADATA,'r') as file:
        data = file.read()
    json_data = json.loads(data)
    mouse_line = json_data["Mouse_line"]
    mouse_code = json_data["Mouse_Code"]
    mouse_Genotype = json_data["Genotype"]
    sex = json_data["Sex"]
    recording_date = json_data["Date_of_record"]
    neuron_type = json_data ["Neuron_type"]
    selected_screen_state = json_data["Screen_state"]
    sensor = json_data["Sensor"]
    session = json_data["Session"]

else:
    metadata = window.ui.save_metadata
    mouse_line = window.ui.mouseLine
    mouse_code = window.ui.mousecode
    mouse_Genotype = window.ui.mouse_Genotype
    sex = window.ui.sex
    recording_date = window.ui.get_input()
    print("recording_date",recording_date)
    neuron_type = window.ui.selected_neuron
    selected_screen_state = window.ui.selected_screen_state
    sensor = window.ui.sensor
    session = window.ui.session
    if generate_metadata:
        print(metadata)
        SaveMetadata = os.path.join(Base_path, 'metadata.txt')
        with open(SaveMetadata, 'w') as file:
            file.write(json.dumps(metadata, indent=4))

compile_directory = window.ui.directory
DO_MOTION = window.ui.get_face_state()
Do_pupil = window.ui.get_pupil_state()
convolve = window.ui.get_convolve_state()
svg = window.ui.get_generate_svg_state()
neuropil_impact_factor = window.ui.alpha
Do_lag = window.ui.get_lag_state()
Do_skew = window.ui.get_skew_state()
F0_method = window.ui.F0_method
print("F0_method", F0_method)
remove_blink = window.ui.get_blink_state()
st_FA = window.ui.first_frame
end_FA = window.ui.last_frame
speed_threshold = window.ui.speed_threshold
m_th = window.ui.motion_th
M_filter_kernel = window.ui.motion_filter
S_filter_kernel = window.ui.speed_filter
skew_threshold = window.ui.skew_th
N_iterations = window.ui.syn_itter
permutation_sample = window.ui.num_permutation
min_Run_win = window.ui.min_Run_win
min_AS_win = window.ui.min_AS_win
min_Rest_win = window.ui.min_Rest_win
min_PM_win = window.ui.min_PM_win
#-------------------------Find blinking frames blinking -------------------------
ALL_ID = np.arange(0, len_data)
if remove_blink:
    ALL_ID = functions.detect_blinking(pupil,ALL_ID,window =4)
new_id = [i for i in ALL_ID if st_FA <= i < end_FA]
#----------------Removeing bad frames neural and behavioral traces---------------
F = F[:, new_id]
motion = motion[new_id]
pupil = pupil[new_id]
TIme = x_inter[new_id]
speed = speed[new_id]
Fneu_raw1 = Fneu_raw[:, new_id]
spikes = spikes[:,new_id]
#------------------------------------Calculation alpha------------------
if neuron_type == "PYR":
    neuropil_impact_factor, remove = functions.calculate_alpha(F,Fneu_raw1)
    #-----------------Remove Neurons with negative slope---------------
    mask = np.ones(len(F), dtype=bool)
    mask[remove]= False
    F = F[mask]
    Fneu_raw1 = Fneu_raw1[mask]
    spikes = spikes[mask]
#-------------------------Calculation of F0 ----------------------
RAWF = copy.deepcopy(F)
F = F - (neuropil_impact_factor * Fneu_raw1)
Fs = 30
percentile = 10
F0 = functions.calculate_F0(F, Fs, percentile, mode= F0_method, win=60)
#-----------------Remove Neurons with F0 less than 1
zero_F0 = [i for i,val in enumerate(F0) if np.any(val < 1)]
invalid_cell_F0 = np.ones((len(F0), 2))
invalid_cell_F0[zero_F0, 0] = 0
F, _ = functions.detect_cell(invalid_cell_F0, F)
F0, _ = functions.detect_cell(invalid_cell_F0, F0)
RAWF, _ = functions.detect_cell(invalid_cell_F0, RAWF)
spikes, _ = functions.detect_cell(invalid_cell_F0, spikes)
#-------------------------Calculation of dF/F
dF = functions.deltaF_calculate(F, F0)
Time = np.arange(0, len(motion))
filtered_motion = gaussian_filter1d(motion, M_filter_kernel)
filtered_speed = gaussian_filter1d(speed,S_filter_kernel)

motion_threshold = m_th*(np.std(motion))
z_scored_dF = zscore(dF,1)
#----------------------------------save svg----------------------------------
if svg:
    figure.simple_plot_SVG(TIme, motion, save_direction_figure,"SVGface_motion.svg")
    figure.simple_plot_SVG(TIme, pupil, save_direction_figure,"SVGpupil.svg")
    figure.simple_plot_SVG(TIme, speed, save_direction_figure,"SVGspeed.svg")
#--------------------------------------------------------------------------------
Real_Time_Running, Running_window, Real_time_NABMA, NABMA_window,Real_Time_rest_window,rest_window,Real_time_AS, AS_window =\
    motion_state.split_stages(motion,speed,TIme,speed_threshold,min_Run_win,min_AS_win,min_PM_win,min_Rest_win,S_filter_kernel,M_filter_kernel)
motion_state.stage_plot(Real_Time_Running,Real_time_NABMA,Real_Time_rest_window,Real_time_AS,
        speed,motion,dF,TIme,Running_window,NABMA_window,rest_window,AS_window,save_direction_figure,pupil,S_filter_kernel,svg)
#------------------------------- Synchrony calculation------------------------------
spikes = np.reshape(spikes, (dF.shape[0], dF.shape[1]))
thr_spikes = [synchrony.sumbre_threshold(dF[roi], spikes[roi]) for roi in range(len(spikes))]
matrix_shuffled_synchrony = synchrony.get_average_rand_synchronicity(spikes, N_iterations, thr_spikes, 10)
matrix_synchro =synchrony.synchrony(spikes, thr_spikes, w_size=10)
synchrony.plot_data(matrix_synchro, matrix_shuffled_synchrony, save_direction_figure,svg, ax=None)
synchrony.plot_matrix_synchro(matrix_synchro,save_direction_figure, "synchrony.png")
synchrony.plot_matrix_synchro(matrix_shuffled_synchrony[0],save_direction_figure, "shuffeled_synchrony.png")
#-----------------------Calculatin mean dF/F for different motion states---------------------
F_running = motion_state.mean_max_interval(dF,Running_window,'mean')
F_rest = motion_state.mean_max_interval(dF,rest_window,'mean')
F_NABMA = motion_state.mean_max_interval(dF,NABMA_window,'mean')
#-----------------------------------------------------
if len(Real_Time_rest_window) ==0 :
    functions.save_data("matrix_synchro", save_data, matrix_synchro)
    functions.save_data("matrix_shuffled_synchrony", save_data, matrix_shuffled_synchrony)
    raise Exception("Rest window is zero")
if len(Real_Time_Running) ==0 :
    functions.save_data("matrix_synchro", save_data, matrix_synchro)
    functions.save_data("matrix_shuffled_synchrony", save_data, matrix_shuffled_synchrony)
    t = "ROI"
    columns = [(f'{t}{i}') for i in range(len(F_rest))]
    M_variable = pd.DataFrame(
        [F_rest],
        index=['Mean dF Rest'], columns=columns).T
    functions.save_exel('rest df.xlsx', save_data, M_variable)
    raise Exception("running window is zero")
#--------------------------------Calculating locomotion Modulotion index---------------------
valid_cell_LMI = np.zeros((len(dF), 2))
LMI = [(F_running[i] - F_rest[i]) / (F_running[i] + F_rest[i]) for i in range(len(dF))]
#--------------------------remove ROIs with LMI more than 1 or less than -1 ------------
in_RANGE_lmi = [i for i, val in enumerate(LMI) if abs(val) <= 1]
valid_cell_LMI[in_RANGE_lmi, 0] = 1
RAWF, _ = functions.detect_cell(valid_cell_LMI, RAWF)
F, _ = functions.detect_cell(valid_cell_LMI, F)
dF, _ = functions.detect_cell(valid_cell_LMI, dF)
F0, _ = functions.detect_cell(valid_cell_LMI, F0)
F_running , _ = functions.detect_cell(valid_cell_LMI, F_running)
F_NABMA , _ = functions.detect_cell(valid_cell_LMI, F_NABMA)
F_rest, _ = functions.detect_cell(valid_cell_LMI, F_rest)
z_scored_dF, _ = functions.detect_cell(valid_cell_LMI, z_scored_dF)
LMI, _ = functions.detect_cell(valid_cell_LMI, LMI)
mean_dF0 = np.mean(F0, 1)
#-----------calculate Z scored states -----------
Z_mean_F_Running = motion_state.mean_max_interval(z_scored_dF,Running_window,'mean')
Z_mean_F_rest = motion_state.mean_max_interval(z_scored_dF,rest_window,'mean')
Z_mean_F_NABMA = motion_state.mean_max_interval(z_scored_dF,NABMA_window,'mean')
#------------------creat General H5 Groups---------------
behavioral_group = hf.create_group('behavioral')
Ca_pre_group = hf.create_group('Ca_data')
mean_dF = Ca_pre_group.create_group('Mean_dF')
Zscored_Ca = Ca_pre_group.create_group('Mean_Zscored')
dF_Ca = Ca_pre_group.create_group('full_trace')
ROIs_group = hf.create_group("ROIs")
Time_group = hf.create_group("Time")
processd_group = hf.create_group('processed_data')
Sub_processd_corr = processd_group.create_group('correlation')
Sub_processd_MI = processd_group.create_group('MI')
Sub_processd_PState = processd_group.create_group('Pupil_state')
Sub_processd_syn = processd_group.create_group('synchrony')
functions.creat_H5_dataset(behavioral_group,[speed, motion, pupil],['Speed', 'FaceMotion', 'Pupil'])
functions.creat_H5_dataset(dF_Ca,[F0,mean_dF0,dF,z_scored_dF,RAWF,Fneu_raw1,spikes],['F0','Mean_F0','dF','Z_scored_F','F','Fneu','spikes'])
functions.creat_H5_dataset(mean_dF,[F_running, F_rest, F_NABMA], ['Running', 'Rest', 'paw_movement'])
functions.creat_H5_dataset(Sub_processd_syn,[matrix_synchro, matrix_shuffled_synchrony], ['data_synchrony', 'shuffel_synchrony'])
functions.creat_H5_dataset(Zscored_Ca,[Z_mean_F_Running, Z_mean_F_rest, Z_mean_F_NABMA], ['Running', 'Rest', 'paw_movement'])
Time_group.create_dataset('Time', data= TIme)
Sub_processd_MI.create_dataset('Run_MI', data = LMI)
if convolve:
     speed, motion, pupil = functions.convolve(Time, speed, motion, pupil,save_direction_figure)
########################################
print("session contains", str(len(dF)), " neurons")
num = np.arange(0, len(dF))
leN = len(dF)
valid_neurons_speed, out_neurons_speed = functions.permutation(dF, filtered_speed,"speed", save_direction_permutation,permutation_sample)
if len(valid_neurons_speed) == 0 :
    raise Exception("Zero Neuron is valid after permutation test for speed and dF ")
functions.creat_H5_dataset(ROIs_group,[keeped_ROI, valid_cell_LMI,valid_neurons_speed,out_neurons_speed]
                           , ['ROI_order_suite2p', 'Valid_LMI','Valid_Speed','Out_speed'])
percentage_valid_sp = (len(valid_neurons_speed)/len(dF)) * 100
#------------------------------------------------- Correlation Calcolating ----------------------------------
speed_corr = [pearsonr(speed, ROI)[0] for ROI in dF]
Sub_processd_corr.create_dataset('Speed_corr',data = speed_corr)
dF_speed_correlation_sorted = [dF for speed_corr, dF in sorted(zip(speed_corr, dF))]
Normal_dF_speed_correlation_sorted = functions.Normal_df(dF_speed_correlation_sorted)
#-------------------------------------------------------------Speed lag--------------------------------------
if Do_lag :
    valid_speed_lag, lag_mean_dF_speed = functions.lag(TIme,valid_neurons_speed,save_direction0_lag,dF,speed,"speed",speed_corr)

    Sub_processd_lag = processd_group.create_group('lag')
    functions.creat_H5_dataset(Sub_processd_lag,[valid_speed_lag, lag_mean_dF_speed],['speed_lag_valid_ROIs','lag_mean_dF_speed'])
if Do_pupil:
    valid_neurons_pupil, out_neurons_pupil = functions.permutation(dF, pupil,"pupil", save_direction_permutation, permutation_sample)
    ROIs_group.create_dataset('Valid_Pupil', data = valid_neurons_pupil)
    figure.pie_plot('permutation test result(pupil)', save_direction_permutation, 'correlated neuron',
                    'uncorrelated neuron' ,len(valid_neurons_pupil), len(out_neurons_pupil))
    #--------------------------------------pupil correlation------------------------------------------------
    pupil_corr = [pearsonr(pupil, i)[0] for i in dF]
    Sub_processd_corr.create_dataset('pupil_corr',data = pupil_corr)
    Mean_Running_pupil = motion_state.mean_interval(pupil, Running_window, method='mean')
    Mean_NABMA_pupil = motion_state.mean_interval(pupil, NABMA_window, method='mean')
    Mean_rest_pupil = motion_state.mean_interval(pupil, rest_window, method='mean')
    Mean_AS_pupil = motion_state.mean_interval(pupil, AS_window, method='mean')
    categories = ['Rest','paw_movement', 'Aroused_Stationary', 'Running']
    values = [Mean_rest_pupil, Mean_NABMA_pupil,Mean_AS_pupil, Mean_Running_pupil]
    functions.creat_H5_dataset(Sub_processd_PState, values, categories)
    M_active_pupil1 = leN * [Mean_Running_pupil]
    M_NABMA_pupil1 = leN * [Mean_NABMA_pupil]
    M_rest_pupil1 = leN * [Mean_rest_pupil]
    Max_AS_pupil1 = leN * [Mean_AS_pupil]
    #------------------------------------plot pupil-----------------------------------------------
    figure.pupil_state(categories, values, save_direction_figure, 'mean pupil Zscored', 'pupil states',svg)
    figure.scatter_plot(num, pupil_corr, out_neurons_pupil, save_direction_figure,
        'pupil & F Correlation','Neuron','correlation', 'pass pupil P test', 'fail pupil P test',svg)
else:
    Mean_rest_pupil= Mean_NABMA_pupil = Mean_AS_pupil = Mean_Running_pupil = None
    M_active_pupil1 = M_NABMA_pupil1 = M_rest_pupil1 = Max_AS_pupil1 = np.full(leN, np.nan)
#--------------------------------------
mean_Running_dF_valid = functions.detect_valid_neurons(valid_neurons_speed,Z_mean_F_Running)
mean_F_rest_speed_valid = functions.detect_valid_neurons(valid_neurons_speed, Z_mean_F_rest)
mean_dF_NABMA_valid = functions.detect_valid_neurons(valid_neurons_speed, Z_mean_F_NABMA)
Run_MI_valid1 = functions.detect_valid_neurons(valid_neurons_speed,LMI)
SpeedValidCorr = functions.detect_valid_neurons(valid_neurons_speed, speed_corr)
#------------------------Create H5 file for neueons which pass permutation test --------------------------
H5_valid_dir =os.path.join(save_data, "ValidROIs.h5")
valid_hf = h5py.File(H5_valid_dir, 'w')
valid_Ca_pre_group = valid_hf.create_group('Ca_data')
valid_processd_group = valid_hf.create_group('processed_data')
valid_correlation_group = valid_processd_group.create_group('correlation')
valid_Zscored_Ca = valid_Ca_pre_group.create_group('Zscored')
valid_correlation_group.create_dataset('speed_corr', data = SpeedValidCorr)
valid_MI = valid_processd_group.create_group('MI')
valid_MI.create_dataset('Running_MI', data = Run_MI_valid1)
functions.creat_H5_dataset(valid_Zscored_Ca,[mean_Running_dF_valid, mean_F_rest_speed_valid, mean_dF_NABMA_valid]
                           ,['Running', 'Rest', 'paw_movement'])
#-----------------------------------------------------
general_timing = TIme[-1] - TIme[0]
RUN_TIME = motion_state.state_duration(Real_Time_Running)
REST_TIME = motion_state.state_duration(Real_Time_rest_window)
only_paw_Time = motion_state.state_duration(Real_time_NABMA)
AS_TIME = motion_state.state_duration(Real_time_AS)
Run_percentage = (RUN_TIME/general_timing) * 100
figure.Time_pie(AS_TIME, RUN_TIME, REST_TIME, only_paw_Time, general_timing,save_direction_figure,svg)

if DO_MOTION:
    valid_neurons_face, out_neurons_face = functions.permutation(dF, filtered_motion,"motion", save_direction_permutation,permutation_sample)
    functions.creat_H5_dataset(ROIs_group, [valid_neurons_face,out_neurons_face], ['Valid_Face','out_Face'])
    #--------------------------plot picture-----------------------
    figure.pie_plot('permutation test result(facemap)', save_direction_permutation, 'correlated neuron',
                    'uncorrelated neuron', len(valid_neurons_face), len(out_neurons_face))
    if len(valid_neurons_face) == 0:
        raise Exception("Zero Neuron is valid after permutation test for face motion and dF")
    #----------------------------------------------
    face_corr = [pearsonr(motion, ROI)[0] for ROI in dF]
    faceValidCorr = functions.detect_valid_neurons(valid_neurons_face, face_corr)
    Sub_processd_corr.create_dataset('face_corr',data = face_corr)
    dF_face_correlation_sorted = [dF for face_corr, dF in sorted(zip(face_corr, dF))]
    Normal_dF_face_correlation_sorted = functions.Normal_df(dF_face_correlation_sorted)
    figure.histo_valid(valid_neurons_face, save_direction_figure, face_corr, "Face validity")
    figure.scatter_plot(num, mean_dF0, out_neurons_face, save_direction_figure,
                        'F0_face', 'Neuron','mean F0', 'pass face P test', 'fail face P test',svg)
    #----------------------------------------------------------
    F_AS = motion_state.mean_max_interval(dF, AS_window, method='mean')
    Z_mean_F_AS = motion_state.mean_max_interval(z_scored_dF, AS_window, method='mean')
    mean_dF.create_dataset('AS', data=F_AS)
    Whisking_TIME = AS_TIME + RUN_TIME + only_paw_Time
    Whisking_percentage = (Whisking_TIME / TIme[-1])*100
    #--------------------------------------------------------------------------
    if Do_lag:
        valid_daceMo_lag, lag_mean_dF_facemotion = functions.lag(TIme, valid_neurons_face,
                                save_direction0_lag, dF,motion, "face motion", face_corr)
        functions.creat_H5_dataset(Sub_processd_lag, [valid_daceMo_lag, lag_mean_dF_facemotion],
                                   ['FaceMotion_lag_valid_ROIs', 'lag_mean_dF_facemotion'])
    #----------------------------------------------------------------------------
    if len(AS_window)>0 :
        AS_MI = [((F_AS[i] - F_rest[i]) / (F_AS[i] + F_rest[i])) for i in range(len(F_AS))]
        num_AS = np.arange(0, len(AS_MI))
        AS_MI_valid1 = functions.detect_valid_neurons(valid_neurons_face,AS_MI)
        mean_ZF_AS_valid1 = functions.detect_valid_neurons(valid_neurons_face, Z_mean_F_AS)
        mean_F_rest_face_valid = functions.detect_valid_neurons(valid_neurons_face, Z_mean_F_rest)
        functions.creat_H5_dataset(valid_Zscored_Ca,[mean_ZF_AS_valid1,mean_F_rest_face_valid], ['Aroused_stationary','Rest_face'])
        Sub_processd_MI.create_dataset('AS_MI', data=AS_MI)
        valid_MI.create_dataset('AS_MI', data = AS_MI_valid1)
        Zscored_Ca.create_dataset('Aroused_stationary',data = Z_mean_F_AS)
        figure.HistoPlot(AS_MI, 'Histo AS MI', save_direction_figure)
        figure.scatter_plot(num_AS, AS_MI, out_neurons_speed, save_direction_figure, 'AS MI',
                'Neuron', 'Aroused_stationary MI', 'pass speed P test', 'fail speed P test',svg)
    #-----------------------------------------plot face------------------------------------------
    figure.fit_plot(speed_corr, face_corr, save_direction_figure, 'Speed & facemotion','facemotion correlation','speed correlation')
    figure.scatter_plot(num, face_corr, out_neurons_face, save_direction_figure,
                    'Facemotion & F Correlation', 'Neuron', 'correlation', 'pass face P test', 'fail face P test',svg)
else:
    Z_mean_F_AS = np.full(leN, np.nan)
    face_corr = np.full(leN, np.nan)
    Whisking_TIME = "whisking was not analyzed"
    Whisking_percentage = "whisking was not analyzed"
Mean__dF = np.mean(dF, 0)
num_LMI = np.arange(0, len(LMI))
#------------------------------------Plot figure---------------------------------
figure.plot_running(TIme, speed, Real_Time_Running,filtered_speed,save_direction_figure,'Speed(cm/s)','Time(S)','speed.png',svg)
figure.histo_valid(valid_neurons_speed,save_direction_figure,speed_corr,"Speed validity")
figure.HistoPlot(LMI, "H_Running LMI", save_direction_figure)
figure.scatter_plot(num,mean_dF0,out_neurons_speed,save_direction_figure,'F0_speed','Neuron', 'mean F0', 'pass speed P test', 'fail speed P test',svg)
figure.double_trace_plot(TIme,Mean__dF,speed, save_direction_figure,"Time(S)","Mean dF", "Speed","mean dF vs speed",svg)
figure.double_trace_plot(TIme,Mean__dF,motion, save_direction_figure,"Time(S)","Mean dF", "motion","mean dF vs facemotion",svg)
figure.double_trace_plot(TIme,Mean__dF,pupil, save_direction_figure,"Time(S)","Mean dF", "pupil","mean dF vs pupil",svg)
figure.power_plot(Mean__dF,Fs,save_direction_figure)
figure.general_figure(TIme, pupil, speed, motion, Normal_dF_speed_correlation_sorted, save_direction_figure,"General.png")
figure.scatter_plot(num,speed_corr, out_neurons_speed,save_direction_figure,'Speed & F Correlation','Neuron','correlation',
                    'pass speed P test', 'fail speed P test',svg)
figure.pie_plot('permutation test result(speed)', save_direction_permutation, 'correlated neuron',
                'uncorrelated neuron',len(valid_neurons_speed), len(out_neurons_speed))
figure.scatter_plot(F_rest,F_running, out_neurons_speed,save_direction_figure,
                    'dF Run & dF Rest(LMI)', 'mean dF rest', 'mean dF run', 'pass speed P test', 'fail speed P test',svg)
figure.scatter_plot(num_LMI, LMI, out_neurons_speed, save_direction_figure,
                    'Running LMI', 'Neuron', 'LMI', 'pass speed P test', 'fail speed P test',svg)

sp = np.copy(speed)
sp[sp == 0] = 'nan'
mean_speed = np.nanmean(sp)
speed_mean = len(dF) * [mean_speed]
skewness = skew(dF, 1)
t = "ROI"
columns = [(f'{t}{i}') for i in range(len(F))]
RUN_TIME1 = leN *[RUN_TIME]
REST_TIME1 = leN *[REST_TIME]
only_paw_Time1 = leN *[only_paw_Time]
AS_TIME1 = leN*[AS_TIME]
Run_percentage1 = leN * [Run_percentage]

df = pd.DataFrame(
    [LMI, speed_mean, Z_mean_F_Running,Z_mean_F_NABMA, Z_mean_F_rest,
     Z_mean_F_AS, speed_corr, face_corr, skewness,
     M_active_pupil1, M_NABMA_pupil1, M_rest_pupil1,
     Max_AS_pupil1, F_NABMA, F_rest, F_running,RUN_TIME1,REST_TIME1,only_paw_Time1, AS_TIME1,Run_percentage1],
    index=['Run MI', 'Mean speed','Mean z run', 'Mean z paw movement', 'Mean z rest',
           'Mean z AS','Speed corr', 'Face corr', 'Skewness',
           'Max pupil Run', 'Max pupil paw movement','Max pupil rest',
           'Max pupil AS', 'Mean F paw movement', 'Mean F rest', 'Mean F Running'
        ,'Run time', 'Rest time','PM time', 'AS Time', 'Run percentage'], columns=columns).T
functions.save_exel('variable.xlsx', save_data, df)
parameters = f"Date = {current_date}\n" \
             f"Time = {current_time}\n" \
             f"relative time base on xml:\nFirst Frame = {st_FA}\nLast frame = {end_FA}\n" \
             f"first frame{TIme[0]}(s)\n last frame {TIme[-1]}(s)\n" \
             f"Runnig Time = {RUN_TIME}\n Rest Time =  {REST_TIME}\n" \
             f"Run percentage = {Run_percentage}\n Whisking Time =  {Whisking_TIME}\n" \
             f"Whisking percentage = {Whisking_percentage}\n neuropil impact factor = {neuropil_impact_factor}\n" \
             f"F0 calculating method = {F0_method}\nThis session has {len(pupil)} frames\n" \
             f"motion filter kernel = {M_filter_kernel}\nSpeed filter kernel = {S_filter_kernel}\n"\
             f"save path = {save_direction1}\n suite2p path = {Base_path}\n 2Photon setting:\n"\
             f"channel number = {channel_number}\nLaser Wavelength = {laserWavelength}\nObjective Lens = {objectiveLens}" \
             f"Objective Lens Mag = {objectiveLensMag}\nOptical Zoom = {opticalZoom}\nBit Depth = {bitDepth}\nDwell Time = {dwellTime}" \
             f"Frame Period{framePeriod}\nMicrons Per Pixel = {micronsPerPixel}\n Two Photon Laser Power{TwophotonLaserPower}"
save_direction_text = os.path.join(save_data , "parameters.text")
with open(save_direction_text, 'a') as file:
        file.write(parameters + '\n')
#motion_state.kruskal_test(z_scored_dF, Running_window, NABMA_window, rest_window, AS_window, save_direction_figure)
if Do_skew:
    mean_high_skew, mean_low_skew = skewness2.skewness(dF,skew_threshold, save_direction_skew, ROIs_group, LMI,Z_mean_F_Running, Z_mean_F_NABMA,
             Z_mean_F_AS,Z_mean_F_rest,speed_corr,face_corr,TIme, pupil, speed, motion,svg)
else:
    mean_high_skew = mean_low_skew = None
hf.close()
#------------------------------------- Compile data --------------------------------
M_Run_LMI = np.nanmean(LMI)
M_Zscored_F_Run = np.nanmean(Z_mean_F_Running)
M_Zscored_F_PM = np.nanmean(Z_mean_F_NABMA)
M_Zscored_F_Rest = np.nanmean(Z_mean_F_rest)
M_Zscored_F_AS = np.nanmean(Z_mean_F_AS)
M_speed_corr = np.nanmean(speed_corr)
M_face_corr = np.nanmean(face_corr)
M_skewness = np.nanmean(skewness)
M_F_NABMA = np.nanmean(F_NABMA)
M_F_rest = np.nanmean(F_rest)
M_F_Run = np.nanmean(F_running)
compile_data = [M_Run_LMI,mean_speed,M_Zscored_F_Run, M_Zscored_F_PM,
                             M_Zscored_F_Rest, M_Zscored_F_AS,M_speed_corr,M_face_corr,M_skewness,
                             Mean_Running_pupil,Mean_NABMA_pupil,Mean_rest_pupil,Mean_AS_pupil,
                             M_F_NABMA,M_F_rest,M_F_Run,RUN_TIME,REST_TIME,only_paw_Time, AS_TIME,Run_percentage,mean_high_skew, mean_low_skew,percentage_valid_sp]
compile_name = neuron_type + "_"+ mouse_line +"_"+mouse_Genotype +"_"+ sensor +"_"+ sex +"_"+ selected_screen_state
compile_name2 = compile_name+".xlsx"
column_name = compile_name +"_"+mouse_code +"_"+recording_date+"_"+session
column = [column_name]
path_compile = os.path.join(compile_directory, compile_name2)
isexsist = os.path.exists(path_compile)
if isexsist:
    print("exist")
    workbook = load_workbook(path_compile)
    worksheet = workbook.active
    new_row = column + compile_data
    print(new_row)
    first_column_values = [cell.value for cell in worksheet['A'] if cell.value is not None]

    if column_name not in first_column_values:
        # Append the new row data to the worksheet
        worksheet.append(new_row)
        workbook.save(path_compile)
        print("New row added successfully.")
    else:
        print("Row with the same value in the first column already exists in the Excel file.")

else:

    data_compile = pd.DataFrame(compile_data,
                                index=['Run MI', 'Mean speed','Mean z run', 'Mean z paw movement', 'Mean z rest',
               'Mean z AS','Speed corr', 'Face corr', 'Skewness',
               'mean pupil Run', 'mean pupil paw movement','mean pupil rest',
               'mean pupil AS', 'Mean F paw movement', 'Mean F rest', 'Mean F Running','Run time',
                                       'Rest time','PM time', 'AS Time', 'Run percentage', 'mean_high_skew', 'mean_low_skew','valid_sp'],columns=column).T
    functions.save_exel(compile_name2, compile_directory, data_compile)
    print("End of program")
