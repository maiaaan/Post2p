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

#DO_SYNCHRONY = False
#DO_SPEED = True
Fs = 30

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
motion = np.interp(x_inter, xp, motion)
# ----------------------------------Making figure for GUI---------------------------------
frames = np.arange(0,len(motion))
Mean_raw_F = np.mean(F, 0)
pupil = zscore(pupil)
len_data = len(speed)
print("mean raw F",len(Mean_raw_F))
print("TIM", len(frames))
figure.GUIimage(frames, motion, save_direction_figure, "raw_face_motion.png")
figure.GUIimage(frames, pupil, save_direction_figure, "pupil.png")
figure.GUIimage(frames, Mean_raw_F,save_direction_figure,"raw_mean_F.png")

#----------------------------------Load GUI-----------------------------------
class MyWindow(QtWidgets.QMainWindow):
    def __init__(self, image_file, data_file):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self, Base_path, save_direction_figure, len_data)

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    data_path = len_data
    figure_path = save_direction_figure
    window = MyWindow(save_direction_figure, len_data)
    window.show()
    app.exec_()

LAUNCH_PROCESSING = window.ui.have_metadata

if LAUNCH_PROCESSING : 

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
    DO_PUPIL = window.ui.get_pupil_state()
    DO_CONVOLUTION = window.ui.get_convolve_state()
    SAVE_SVG = window.ui.get_generate_svg_state()
    DO_LAG = window.ui.get_lag_state()
    Do_skew = window.ui.get_skew_state()
    REMOVE_BLINK = window.ui.get_blink_state()
    
    neuropil_impact_factor = window.ui.alpha
    F0_method = window.ui.F0_method
    st_FA = window.ui.first_frame
    end_FA = window.ui.last_frame
    N_iterations = window.ui.syn_itter
    permutation_sample = window.ui.num_permutation

    min_states_window = {'run' : window.ui.min_Run_win, 
                        'AS' : window.ui.min_AS_win, 
                        'NABMA' : window.ui.min_PM_win, 
                        'rest' : window.ui.min_Rest_win}
    
    filter_kernel = {'speed' : window.ui.speed_filter, 
                     'motion' : window.ui.motion_filter}
    
    threshold = {'speed' : window.ui.speed_threshold,
                 'motion' : window.ui.motion_th,
                 'skew' : window.ui.skew_th}
    
    #----------------Removing bad frames of neural and behavioral traces---------------
    ALL_ID = np.arange(0, len_data)
    if REMOVE_BLINK:
        ALL_ID = functions.detect_blinking(pupil,ALL_ID,window =4)
    new_id = [i for i in ALL_ID if st_FA <= i < end_FA]
    
    F = F[:, new_id]
    Fneu_raw1 = Fneu_raw[:, new_id]
    spikes = spikes[:,new_id]
    
    motion = motion[new_id]
    pupil = pupil[new_id]
    real_time = x_inter[new_id]
    speed = speed[new_id]

    nvar = {'F': F, 'Fneu' : Fneu_raw1, 'spikes' : spikes} #to stock neural variables
    #------------------------------Calculate alpha------------------------
    if neuron_type == "PYR":
        neuropil_impact_factor, remove = functions.calculate_alpha(nvar['F'], nvar['Fneu'])
        #-----------------Remove Neurons with negative slope---------------
        nvar = functions.remove_roi(remove, nvar)

    #-------------------------Calculation of F0 ----------------------
    RAWF = copy.deepcopy(nvar['F'])
    nvar['F'] = nvar['F'] - (neuropil_impact_factor * nvar['Fneu'])
    F0 = functions.calculate_F0(nvar['F'], Fs, 10, mode=F0_method, win=60)
    variables_add = {"RAWF" : RAWF, 'F0' : F0}
    nvar.update(variables_add)

    #-----------------Remove Neurons with F0 less than 1
    remove_F0 = [i for i,val in enumerate(nvar['F0']) if np.any(val < 1)]
    nvar = functions.remove_roi(remove_F0, nvar)

    #-------------------------Calculation of dF/F
    dF = functions.deltaF_calculate(nvar['F'], nvar['F0'])
    z_scored_dF = zscore(dF,1)
    variables_add = {"dF" : dF, 'dF z-score' : z_scored_dF}
    nvar.update(variables_add)

    #----------------------------------save svg----------------------------------
    if SAVE_SVG:
        figure.simple_plot_SVG(real_time, motion, save_direction_figure,"SVGface_motion.svg")
        figure.simple_plot_SVG(real_time, pupil, save_direction_figure,"SVGpupil.svg")
        figure.simple_plot_SVG(real_time, speed, save_direction_figure,"SVGspeed.svg")

    #-----------------------------Define mouse state and plot-----------------------------------
    Real_Time_states, states_window = motion_state.split_stages(motion, speed, real_time, threshold, min_states_window, filter_kernel)
    motion_state.stage_plot(motion, speed, pupil, nvar['dF'], real_time, Real_Time_states, states_window, filter_kernel, save_direction_figure, SAVE_SVG, threshold)
    
    #------------------------------- Synchrony calculation------------------------------
    nvar['spikes'] = np.reshape(nvar['spikes'], (nvar['dF'].shape[0], nvar['dF'].shape[1]))
    thr_spikes = [synchrony.sumbre_threshold(nvar['dF'][roi], nvar['spikes'][roi]) for roi in range(len(nvar['spikes']))]
    matrix_shuffled_synchrony = synchrony.get_average_rand_synchronicity(nvar['spikes'], N_iterations, thr_spikes, 10)
    matrix_synchro = synchrony.synchrony(nvar['spikes'], thr_spikes, w_size=10)
    synchrony.plot_data(matrix_synchro, matrix_shuffled_synchrony, save_direction_figure,SAVE_SVG, ax=None)
    synchrony.plot_matrix_synchro(matrix_synchro,save_direction_figure, "synchrony.png")
    synchrony.plot_matrix_synchro(matrix_shuffled_synchrony[0],save_direction_figure, "synchrony_shuffled.png")

    #-----------------------Calculatin mean dF/F for different motion states---------------------
    F_running = motion_state.mean_max_interval(nvar['dF'], states_window['run'], 'mean')
    F_rest = motion_state.mean_max_interval(nvar['dF'], states_window['rest'], 'mean')
    F_NABMA = motion_state.mean_max_interval(nvar['dF'], states_window['NABMA'], 'mean')
    F_AS = motion_state.mean_max_interval(nvar['dF'], states_window['AS'], method='mean')
    variables_add = {"F_run" : F_running, 'F_rest' : F_rest, 'F_NABMA' : F_NABMA, "F_AS" : F_AS}
    nvar.update(variables_add)

    #-----------------------------------------------------
    """ QUESTION """
    if len(Real_Time_states['rest']) == 0 :
        functions.save_data("matrix_synchro", save_data, matrix_synchro)
        functions.save_data("matrix_shuffled_synchrony", save_data, matrix_shuffled_synchrony)
        raise Exception("Rest window is zero")
    if len(Real_Time_states['run']) ==0 :
        functions.save_data("matrix_synchro", save_data, matrix_synchro)
        functions.save_data("matrix_shuffled_synchrony", save_data, matrix_shuffled_synchrony)
        t = "ROI"
        columns = [(f'{t}{i}') for i in range(len(nvar['F_rest']))]
        M_variable = pd.DataFrame(
            [nvar['F_rest']],
            index=['Mean dF Rest'], columns=columns).T
        functions.save_exel('rest df.xlsx', save_data, M_variable)
        raise Exception("running window is zero")
    
    #--------------------------------Calculating locomotion Modulation index---------------------
    LMI = [(nvar['F_run'][i] - nvar['F_rest'][i]) / (nvar['F_run'][i] + nvar['F_rest'][i]) for i in range(len(nvar['dF']))]
    AS_MI = [((nvar['F_AS'][i] - nvar['F_rest'][i]) / (nvar['F_AS'][i] + nvar['F_rest'][i])) for i in range(len(nvar['F_AS']))]
    variables_add = {"run_LMI" : LMI, "AS_MI" : AS_MI}
    nvar.update(variables_add)

    #--------------------------remove ROIs with LMI more than 1 or less than -1 ------------
    not_in_RANGE_lmi = [i for i, val in enumerate(nvar['run_LMI']) if abs(val) > 1]
    nvar = functions.remove_roi(not_in_RANGE_lmi, nvar)

    valid_cell_LMI = np.ones((len(dF), 2))
    valid_cell_LMI[not_in_RANGE_lmi, 0] = 0
    
    #-----------calculate F0 mean -----------
    mean_F0 = np.mean(nvar['F0'], 1)
    variables_add = {"mean_F0" : mean_F0}
    nvar.update(variables_add)

    #-----------calculate Z scored states -----------
    mean_zdF_run = motion_state.mean_max_interval(nvar['dF z-score'], states_window['run'],'mean')
    mean_zdF_rest = motion_state.mean_max_interval(nvar['dF z-score'], states_window['rest'],'mean')
    mean_zdF_NABMA = motion_state.mean_max_interval(nvar['dF z-score'], states_window['NABMA'],'mean')
    mean_zdF_AS = motion_state.mean_max_interval(nvar['dF z-score'], states_window['AS'], method='mean')
    variables_add = {"mean_zdF_run" : mean_zdF_run, "mean_zdF_rest" : mean_zdF_rest, "mean_zdF_NABMA" : mean_zdF_NABMA, "mean_zdF_AS" : mean_zdF_AS}
    nvar.update(variables_add)

    #------------------Create General H5 Groups---------------
    behavioral_group = hf.create_group('behavioral')
    Ca_pre_group = hf.create_group('Ca_data')
    ROIs_group = hf.create_group("ROIs")
    Time_group = hf.create_group("Time")
    processd_group = hf.create_group('processed_data')

    dF_Ca = Ca_pre_group.create_group('full_trace')
    mean_dF = Ca_pre_group.create_group('Mean_dF')
    Zscored_Ca = Ca_pre_group.create_group('Mean_Zscored')
    Sub_processd_syn = processd_group.create_group('synchrony')
    Sub_processd_MI = processd_group.create_group('MI')
    Sub_processd_corr = processd_group.create_group('correlation')
    Sub_processd_PState = processd_group.create_group('Pupil_state')
    Sub_processd_lag = processd_group.create_group('lag')

    functions.creat_H5_dataset(behavioral_group, [speed, motion, pupil], ['Speed', 'FaceMotion', 'Pupil'])
    functions.creat_H5_dataset(dF_Ca, [nvar['F0'], nvar['mean_F0'], nvar['dF'], nvar['dF z-score'], nvar['RAWF'], nvar['Fneu'], nvar['spikes']], ['F0','Mean_F0','dF','Z_scored_F','F','Fneu','spikes'])
    functions.creat_H5_dataset(mean_dF, [nvar['F_run'], nvar['F_rest'], nvar['F_NABMA'], nvar['F_AS']], ['Running', 'Rest', 'NABMA', 'AS'])
    functions.creat_H5_dataset(Zscored_Ca, [nvar['mean_zdF_run'], nvar['mean_zdF_rest'], nvar['mean_zdF_NABMA'], nvar['mean_zdF_AS']], ['Running', 'Rest', 'NABMA', 'AS'])
    Time_group.create_dataset('Time', data=real_time)
    functions.creat_H5_dataset(Sub_processd_syn, [matrix_synchro, matrix_shuffled_synchrony], ['data_synchrony', 'shuffle_synchrony'])
    functions.creat_H5_dataset(Sub_processd_MI, [nvar['run_LMI'], nvar['AS_MI']], ['Run_MI', 'AS_MI'])

    #------------------ Convolve ---------------
    if DO_CONVOLUTION :
        Time = np.arange(0, len(motion))
        speed, motion, pupil = functions.convolve(Time, speed, motion, pupil, save_direction_figure)
    
    
    ########################################

    print("session contains", str(len(nvar['dF'])), " neurons")
    num = np.arange(0, len(nvar['dF']))
    leN = len(nvar['dF'])

    #----------------------------------------Speed Permutation Test--------------------------------------
    filtered_speed = gaussian_filter1d(speed,filter_kernel['speed'])
    valid_neurons_speed, out_neurons_speed = functions.permutation(nvar['dF'], filtered_speed, "speed", real_time, save_direction_permutation, permutation_sample)
    if len(valid_neurons_speed) == 0 :
        raise Exception("Zero Neuron is valid after permutation test for speed and dF ")
    functions.creat_H5_dataset(ROIs_group,[keeped_ROI, valid_cell_LMI, valid_neurons_speed, out_neurons_speed]
                            , ['ROI_order_suite2p', 'Valid_LMI','Valid_Speed','Out_speed'])
    percentage_valid_sp = (len(valid_neurons_speed)/len(nvar['dF'])) * 100

    #----------------------------------------Speed Correlation ------------------------------------------
    speed_corr = [pearsonr(speed, ROI)[0] for ROI in nvar['dF']]
    Sub_processd_corr.create_dataset('Speed_corr',data = speed_corr)
    dF_speed_correlation_sorted = [dF for _, dF in sorted(zip(speed_corr, nvar['dF']))]
    Normal_dF_speed_correlation_sorted = functions.Normal_df(dF_speed_correlation_sorted)

    #---------------------------------COMPUTE OTHER MERTICS---------------------------------
    mean_zdF_run_speed_valid = functions.detect_valid_neurons(valid_neurons_speed, nvar['mean_zdF_run'])
    mean_zdF_rest_speed_valid = functions.detect_valid_neurons(valid_neurons_speed, nvar['mean_zdF_rest'])
    mean_zdF_NABMA_speed_valid = functions.detect_valid_neurons(valid_neurons_speed, nvar['mean_zdF_NABMA'])
    Run_MI_speed_valid = functions.detect_valid_neurons(valid_neurons_speed, nvar['run_LMI'])
    SpeedValidCorr = functions.detect_valid_neurons(valid_neurons_speed, speed_corr)

    #------------------------------------------Speed Lag--------------------------------------
    if DO_LAG :
        valid_speed_lag, lag_mean_dF_speed = functions.lag(real_time,valid_neurons_speed,save_direction0_lag,nvar['dF'],speed,"speed",speed_corr)
        functions.creat_H5_dataset(Sub_processd_lag,[valid_speed_lag, lag_mean_dF_speed],['speed_lag_valid_ROIs','lag_mean_dF_speed'])
    
    #------------------------Create H5 file for neurons which pass permutation test--------------------------
    H5_valid_dir = os.path.join(save_data, "ValidROIs.h5")
    valid_hf = h5py.File(H5_valid_dir, 'w')
    
    valid_Ca_pre_group = valid_hf.create_group('Ca_data')
    valid_processd_group = valid_hf.create_group('processed_data')
    
    valid_Zscored_Ca = valid_Ca_pre_group.create_group('Zscored')
    valid_MI = valid_processd_group.create_group('MI')
    valid_correlation_group = valid_processd_group.create_group('correlation')

    functions.creat_H5_dataset(valid_Zscored_Ca,[mean_zdF_run_speed_valid, mean_zdF_rest_speed_valid, mean_zdF_NABMA_speed_valid]
                            ,['mean_Run_speed', 'mean_Rest_speed', 'mean_NABMA_speed'])
    valid_MI.create_dataset('Running_MI', data=Run_MI_speed_valid)
    valid_correlation_group.create_dataset('speed_corr', data=SpeedValidCorr)
    
    #---------------------------------COMPUTE STATES DURATION---------------------------------
    duration = real_time[-1] - real_time[0]
    RUN_TIME = motion_state.state_duration(Real_Time_states['run'])
    REST_TIME = motion_state.state_duration(Real_Time_states['rest'])
    NABMA_Time = motion_state.state_duration(Real_Time_states['NABMA'])
    AS_TIME = motion_state.state_duration(Real_Time_states['AS'])
    Run_percentage = (RUN_TIME/duration) * 100
    figure.Time_pie(AS_TIME, RUN_TIME, REST_TIME, NABMA_Time, duration, save_direction_figure, SAVE_SVG)

    #-------------------------------------------PUPIL COMPUTATION------------------------------------------------------
    if DO_PUPIL:
        valid_neurons_pupil, out_neurons_pupil = functions.permutation(nvar['dF'], pupil, "pupil", real_time, save_direction_permutation, permutation_sample)
        if len(valid_neurons_pupil) == 0:
            raise Exception("Zero valid neuron after permutation test for the correlation between pupil and dF/F")
        functions.creat_H5_dataset(ROIs_group, [valid_neurons_pupil, out_neurons_pupil], ['Valid_Pupil','out_Pupil'])
        #--------------------------------------pupil correlation------------------------------------------------
        pupil_corr = [pearsonr(pupil, i)[0] for i in nvar['dF']]
        Sub_processd_corr.create_dataset('pupil_corr', data=pupil_corr)
        
        Mean_Running_pupil = motion_state.mean_interval(pupil, states_window['run'], method='mean')
        Mean_NABMA_pupil = motion_state.mean_interval(pupil, states_window['NABMA'], method='mean')
        Mean_rest_pupil = motion_state.mean_interval(pupil, states_window['rest'], method='mean')
        Mean_AS_pupil = motion_state.mean_interval(pupil, states_window['AS'], method='mean')
        categories = ['Rest','NABMA', 'Aroused_Stationary', 'Running']
        values = [Mean_rest_pupil, Mean_NABMA_pupil, Mean_AS_pupil, Mean_Running_pupil]
        functions.creat_H5_dataset(Sub_processd_PState, values, categories)
        
        pupil_run = motion_state.get_interval_array(pupil, states_window['run'])
        pupil_AS = motion_state.get_interval_array(pupil, states_window['AS'])
        pupil_NABMA = motion_state.get_interval_array(pupil, states_window['NABMA'])
        pupil_rest = motion_state.get_interval_array(pupil, states_window['rest'])

        M_active_pupil1 = leN * [Mean_Running_pupil]
        M_NABMA_pupil1 = leN * [Mean_NABMA_pupil]
        M_rest_pupil1 = leN * [Mean_rest_pupil]
        Max_AS_pupil1 = leN * [Mean_AS_pupil]
        #------------------------------------plot pupil-----------------------------------------------
        figure.pie_plot('permutation test result(pupil)', save_direction_permutation, 'correlated neuron',
                        'uncorrelated neuron', len(valid_neurons_pupil), len(out_neurons_pupil))
        figure.pupil_state(categories, values, save_direction_figure, 'mean of pupil z-score', 'pupil states', SAVE_SVG)
        figure.box_plot(pupil_run, pupil_AS, pupil_NABMA, pupil_rest, 
                        'Run', 'AS', 'NABMA', "Rest", 
                        "pupil_boxplot_per_state", 'pupil z-score', '', save_direction_figure, SAVE_SVG)
        figure.fit_plot(speed_corr, pupil_corr, save_direction_figure, 'Speed & Pupil','Pupil correlation','speed correlation')
        figure.histo_valid(valid_neurons_pupil, save_direction_figure, pupil_corr, "Pupil validity", "Passed permutation test")
        figure.scatter_plot(num, pupil_corr, valid_neurons_pupil, save_direction_figure,
            'pupil & F Correlation', "Neuron's ID",'correlation', 'passed pupil P test', 'failed pupil P test', SAVE_SVG)
        figure.colormap_perm_test(real_time, nvar['dF'], pupil, valid_neurons_pupil, 'pupil', save_direction_figure)
        #--------------------------------------------------------------------------
        if DO_LAG :
            valid_pupil_lag, lag_mean_dF_pupil = functions.lag(real_time, valid_neurons_pupil, save_direction0_lag, nvar['dF'], pupil, "pupil", pupil_corr)
            functions.creat_H5_dataset(Sub_processd_lag,[valid_pupil_lag, lag_mean_dF_pupil],['pupil_lag_valid_ROIs','lag_mean_dF_pupil'])
    else:
        Mean_rest_pupil= Mean_NABMA_pupil = Mean_AS_pupil = Mean_Running_pupil = None
        M_active_pupil1 = M_NABMA_pupil1 = M_rest_pupil1 = Max_AS_pupil1 = np.full(leN, np.nan)
    
    #-------------------------------------------MOTION COMPUTATION------------------------------------------------------
    if DO_MOTION:
        filtered_motion = gaussian_filter1d(motion, filter_kernel['motion'])
        valid_neurons_face, out_neurons_face = functions.permutation(nvar['dF'], filtered_motion, "motion", real_time, save_direction_permutation,permutation_sample)
        if len(valid_neurons_face) == 0:
            raise Exception("Zero valid neuron after permutation test for the correlation between face motion and dF/F")
        functions.creat_H5_dataset(ROIs_group, [valid_neurons_face,out_neurons_face], ['Valid_Face','out_Face'])
        #----------------------------------------------
        face_corr = [pearsonr(motion, ROI)[0] for ROI in nvar['dF']]
        Sub_processd_corr.create_dataset('face_corr', data=face_corr)

        facemotion_run = motion_state.get_interval_array(filtered_motion, states_window['run'])
        facemotion_AS = motion_state.get_interval_array(filtered_motion, states_window['AS'])
        facemotion_NABMA = motion_state.get_interval_array(filtered_motion, states_window['NABMA'])
        facemotion_rest = motion_state.get_interval_array(filtered_motion, states_window['rest'])

        Whisking_TIME = AS_TIME + RUN_TIME + NABMA_Time
        Whisking_percentage = (Whisking_TIME / real_time[-1])*100

        #------------------------ H5 file for neurons which passed the face permutation test--------------------------
        if len(states_window['AS'])>0 :
            AS_MI_valid = functions.detect_valid_neurons(valid_neurons_face, nvar['AS_MI'])
            mean_zdF_AS_face_valid = functions.detect_valid_neurons(valid_neurons_face, nvar['mean_zdF_AS'])
            mean_zdF_rest_face_valid = functions.detect_valid_neurons(valid_neurons_face, nvar['mean_zdF_rest'])
            functions.creat_H5_dataset(valid_Zscored_Ca,[mean_zdF_AS_face_valid, mean_zdF_rest_face_valid], ['mean_AS_face','mean_Rest_face'])
            valid_MI.create_dataset('AS_MI', data = AS_MI_valid)

        #-----------------------------------------plot face------------------------------------------
        figure.pie_plot('permutation test result(facemap)', save_direction_permutation, 'correlated neuron',
                        'uncorrelated neuron', len(valid_neurons_face), len(out_neurons_face))
        figure.box_plot(facemotion_run, facemotion_AS, facemotion_NABMA, facemotion_rest, 
                        'Run', 'AS', 'NABMA', "Rest", 
                        "facemotion_boxplot_per_state", 'filtered facemotion', '', save_direction_figure, SAVE_SVG)
        figure.fit_plot(speed_corr, face_corr, save_direction_figure, 'Speed & facemotion','facemotion correlation','speed correlation')
        figure.histo_valid(valid_neurons_face, save_direction_figure, face_corr, "Face validity", "Passed permutation test")
        figure.scatter_plot(num, face_corr, valid_neurons_face, save_direction_figure,
                            'Facemotion & F Correlation', "Neuron's ID", 'correlation', 'passed face P test', 'failed face P test',SAVE_SVG)
        figure.scatter_plot(num, nvar['mean_F0'], valid_neurons_face, save_direction_figure,
                            'F0_face', "Neuron's ID",'mean F0', 'passed face P test', 'failed face P test', SAVE_SVG)
        figure.colormap_perm_test(real_time, nvar['dF'], motion, valid_neurons_face, 'facemotion', save_direction_figure)
        #--------------------------------------------------------------------------
        if DO_LAG:
            valid_faceMo_lag, lag_mean_dF_facemotion = functions.lag(real_time, valid_neurons_face, save_direction0_lag, nvar['dF'],motion, "face motion", face_corr)
            functions.creat_H5_dataset(Sub_processd_lag, [valid_faceMo_lag, lag_mean_dF_facemotion], ['FaceMotion_lag_valid_ROIs', 'lag_mean_dF_facemotion'])
    else:
        face_corr = np.full(leN, np.nan)
        Whisking_TIME = "whisking was not analyzed"
        Whisking_percentage = "whisking was not analyzed"
    
    #-------------------------------------------PLOT FIGURES------------------------------------------------------
    Mean__dF = np.mean(nvar['dF'], 0)
    num_LMI = np.arange(0, len(nvar['run_LMI']))
    
    figure.plot_running(real_time, speed, Real_Time_states['run'], filtered_speed, threshold['speed'], save_direction_figure, 'Speed (cm/s)', 'Time (s)', 'speed.png', SAVE_SVG)
    figure.histo_valid(valid_neurons_speed, save_direction_figure, speed_corr, "Speed validity", "Passed permutation test")
    figure.HistoPlot(nvar['run_LMI'], 'Running LMI', "Running_LMI_hist", save_direction_figure)
    figure.scatter_plot(num, nvar['mean_F0'], valid_neurons_speed, save_direction_figure, 'F0_speed', "Neuron's ID", 'mean F0', 'passed speed P test', 'failed speed P test',SAVE_SVG)
    figure.double_trace_plot(real_time, Mean__dF, speed, save_direction_figure,"Time (s)","Mean dF/F", "Speed (cm/s)", "mean dF vs speed", SAVE_SVG)
    figure.double_trace_plot(real_time, Mean__dF, motion, save_direction_figure,"Time (s)","Mean dF/F", "motion", "mean dF vs facemotion",SAVE_SVG)
    figure.double_trace_plot(real_time, Mean__dF, pupil, save_direction_figure,"Time (s)","Mean dF/F", "pupil z-score", "mean dF vs pupil",SAVE_SVG)
    figure.power_plot(Mean__dF, Fs, save_direction_figure)
    figure.general_figure(real_time, pupil, speed, motion, Normal_dF_speed_correlation_sorted, save_direction_figure,"General.png")
    figure.scatter_plot(num, speed_corr, valid_neurons_speed, save_direction_figure, 'Speed & F Correlation', "Neuron's ID", 'correlation',
                        'passed speed P test', 'failed speed P test', SAVE_SVG)
    figure.pie_plot('permutation test result(speed)', save_direction_permutation, 'correlated neuron',
                    'uncorrelated neuron', len(valid_neurons_speed), len(out_neurons_speed))
    figure.scatter_plot(nvar['F_rest'], nvar['F_run'], valid_neurons_speed, save_direction_figure,
                        'dF Run & dF Rest(LMI)', 'mean dF rest', 'mean dF run', 'passed speed P test', 'failed speed P test',SAVE_SVG)
    figure.scatter_plot(num_LMI, nvar['run_LMI'], valid_neurons_speed, save_direction_figure,
                        'Running LMI', "Neuron's ID", 'LMI', 'passed speed P test', 'failed speed P test',SAVE_SVG)
    if len(states_window['AS']) > 0 :
        num_AS = np.arange(0, len(nvar['AS_MI']))
        figure.HistoPlot(nvar['AS_MI'], 'Aroused stationnary LMI', 'AS_MI_hist', save_direction_figure)
        figure.scatter_plot(num_AS, nvar['AS_MI'], valid_neurons_speed, save_direction_figure, 'AS MI',
                        "Neuron's ID", 'Aroused_stationary MI', 'passed speed P test', 'failed speed P test', SAVE_SVG)
    figure.colormap_perm_test(real_time, nvar['dF'], speed, valid_neurons_speed, 'speed', save_direction_figure)
    #---------------------------------------------VARIABLES EXCEL SHEET---------------------------------------------------
    sp = np.copy(speed)
    sp[sp == 0] = 'nan'
    mean_speed = np.nanmean(sp)
    speed_mean = len(nvar['dF']) * [mean_speed]
    skewness = skew(nvar['dF'], 1)
    t = "ROI"
    columns = [(f'{t}{i}') for i in range(len(nvar['F']))]
    RUN_TIME1 = leN *[RUN_TIME]
    REST_TIME1 = leN *[REST_TIME]
    only_paw_Time1 = leN *[NABMA_Time]
    AS_TIME1 = leN*[AS_TIME]
    Run_percentage1 = leN * [Run_percentage]

    df = pd.DataFrame(
        [nvar['run_LMI'], speed_mean, nvar['mean_zdF_run'],nvar['mean_zdF_NABMA'], nvar['mean_zdF_rest'],
        nvar['mean_zdF_AS'], speed_corr, face_corr, skewness,
        M_active_pupil1, M_NABMA_pupil1, M_rest_pupil1,
        Max_AS_pupil1, nvar['F_NABMA'], nvar['F_rest'], nvar['F_run'],RUN_TIME1,REST_TIME1,only_paw_Time1, AS_TIME1,Run_percentage1],
        index=['Run MI', 'Mean speed','Mean z run', 'Mean z paw movement', 'Mean z rest',
            'Mean z AS','Speed corr', 'Face corr', 'Skewness',
            'Max pupil Run', 'Max pupil paw movement','Max pupil rest',
            'Max pupil AS', 'Mean F paw movement', 'Mean F rest', 'Mean F Running'
            ,'Run time', 'Rest time','PM time', 'AS Time', 'Run percentage'], columns=columns).T
    functions.save_exel('variable.xlsx', save_data, df)
    parameters = f"Date = {current_date}\n" \
                f"Time = {current_time}\n" \
                f"relative time base on xml:\nFirst Frame = {st_FA}\nLast frame = {end_FA}\n" \
                f"first frame{real_time[0]}(s)\n last frame {real_time[-1]}(s)\n" \
                f"Runnig Time = {RUN_TIME}\n Rest Time =  {REST_TIME}\n" \
                f"Run percentage = {Run_percentage}\n Whisking Time =  {Whisking_TIME}\n" \
                f"Whisking percentage = {Whisking_percentage}\n neuropil impact factor = {neuropil_impact_factor}\n" \
                f"F0 calculating method = {F0_method}\nThis session has {len(pupil)} frames\n" \
                f"motion filter kernel = {filter_kernel['motion']}\nSpeed filter kernel = {filter_kernel['speed']}\n"\
                f"save path = {save_direction1}\n suite2p path = {Base_path}\n 2Photon setting:\n"\
                f"channel number = {channel_number}\nLaser Wavelength = {laserWavelength}\nObjective Lens = {objectiveLens}" \
                f"Objective Lens Mag = {objectiveLensMag}\nOptical Zoom = {opticalZoom}\nBit Depth = {bitDepth}\nDwell Time = {dwellTime}" \
                f"Frame Period{framePeriod}\nMicrons Per Pixel = {micronsPerPixel}\n Two Photon Laser Power{TwophotonLaserPower}"
    save_direction_text = os.path.join(save_data , "parameters.text")
    with open(save_direction_text, 'a') as file:
            file.write(parameters + '\n')
    #motion_state.kruskal_test(nvar['dF z-score'], Running_window, NABMA_window, rest_window, AS_window, save_direction_figure)
    if Do_skew:
        mean_high_skew, mean_low_skew = skewness2.skewness(nvar['dF'], threshold['skew'], save_direction_skew, ROIs_group, nvar['run_LMI'],nvar['mean_zdF_run'], nvar['mean_zdF_NABMA'],
                nvar['mean_zdF_AS'],nvar['mean_zdF_rest'],speed_corr,face_corr,real_time, pupil, speed, motion,SAVE_SVG)
    else:
        mean_high_skew = mean_low_skew = None
    hf.close()

    #---------------------------------------------COMPILE DATA----------------------------------------------------
    M_Run_LMI = np.nanmean(nvar['run_LMI'])
    M_Zscored_F_Run = np.nanmean(nvar['mean_zdF_run'])
    M_Zscored_F_PM = np.nanmean(nvar['mean_zdF_NABMA'])
    M_Zscored_F_Rest = np.nanmean(nvar['mean_zdF_rest'])
    M_Zscored_F_AS = np.nanmean(nvar['mean_zdF_AS'])
    M_speed_corr = np.nanmean(speed_corr)
    M_face_corr = np.nanmean(face_corr)
    M_skewness = np.nanmean(skewness)
    M_F_NABMA = np.nanmean(nvar['F_NABMA'])
    M_F_rest = np.nanmean(nvar['F_rest'])
    M_F_Run = np.nanmean(nvar['F_run'])
    compile_data = [
                    M_Run_LMI, mean_speed, 
                    M_Zscored_F_Run, M_Zscored_F_PM,  M_Zscored_F_Rest, M_Zscored_F_AS, 
                    M_speed_corr, M_face_corr, M_skewness,
                    Mean_Running_pupil, Mean_NABMA_pupil, Mean_rest_pupil, Mean_AS_pupil,
                    M_F_NABMA, M_F_rest, M_F_Run, 
                    RUN_TIME, REST_TIME, NABMA_Time, AS_TIME, Run_percentage, 
                    mean_high_skew, mean_low_skew, percentage_valid_sp
                    ]
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