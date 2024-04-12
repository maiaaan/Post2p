import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import functions
from itertools import chain
from matplotlib.collections import LineCollection
from scipy.stats import zscore
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import kruskal
import os.path
from matplotlib.lines import Line2D
import seaborn as sns

def split_stages(motion, speed, real_time, threshold, min_states_window, filter_kernel):
    """
    October 2023 - Bacci Lab - faezeh.rabbani97@gmail.com

    .......................................................................

    This function splits fluorescence activity to different time windows based
    on duration of running, speed of running and standard deviation of whisking
    final results contains maximum 4 movement states:
    1. Active movement: running with the speed of above 0.5 cm/s for more than 2 seconds
    2. intermediate movement: any kind of movement which is not considered as active movement
    3. only Whisking: time periods which mouse is only whisking without any movement
    4. Rest: time periods which there are no whisking or movements
    \n.....................................................................
    :param motion: facemotion data or whisking
    :param speed: running speed
    :param real_time: real time stamps extracted from xml file

    :return Real_Time_Aroused_Running: windows of real timestamps for running activity(s)
    :return Aroused_Running_window: windows of index for running activity(frame)
    :return Real_time_Not_aroused_activity2: windows of real timestamps for Not aroused, basal motor activity(s)
    :return Not_aroused_activity_window2: windows of index for Not aroused, basal motor activity (frame)
    :return Real_Time_rest_window2: windows of real timestamps for rest period(s)
    :return rest_window2: windows of index for rest period(frame)
    :return Real_time_Aroused_stationary2: windows of real timestamps for time windows that mouse is not activly running but whisking and pupil is aroused(s)
    :return Aroused_stationary_window2: windows of index for Aroused, stationary period(frame)
    """

    undefined_state_idx = np.arange(0,len(speed))
    ids = np.arange(0, len(speed))

    ###------------------------------ Calculate active movement state -----------------------------###
    # duration > 60 frames and speed > 0.5 s/m
    filtered_speed = gaussian_filter1d(speed, filter_kernel['speed'])
    id_above_thr_speed = np.extract(filtered_speed >= threshold['speed'], ids)
    Aroused_Running_index, Aroused_Running_window, Real_Time_Aroused_Running = functions.find_intervals(id_above_thr_speed, min_states_window['run'], real_time, 45)

    Aroused_Running_index_check1 = []
    for i in Aroused_Running_index:
        if i[0]>45:
            ST = np.arange(i[0] - 45, i[-1]+1)
        else:
            ST = np.arange(0, i[-1]+1)
        Aroused_Running_index_check1.append(ST)
    Aroused_Running_index_check = list(chain(*Aroused_Running_index_check1))

    ###------------------------------Calculate preliminary rest state-----------------------------###
    binary_movement = []
    binary_movement_threshold = 0.01
    for i in filtered_speed:
        if i < binary_movement_threshold:
            binary_movement.append(0)
        else:
            binary_movement.append(1)
    binary_movement = np.array(binary_movement)
    id_rest = np.extract(binary_movement < 1, ids)
    P_rest_index, _, _ = functions.find_intervals(id_rest, min_states_window['rest'], real_time)
    P_rest_index = list(chain(*P_rest_index))
    ###------------------------------Calculate Aroused stationary state-----------------------------###
    filtered_motion = gaussian_filter1d(motion, filter_kernel['motion'])
    motion_threshold = threshold['motion']*(np.std(motion))
    id_above_thr_motion = np.extract(filtered_motion >= motion_threshold, ids)
    Aroused_stationary_index_temp, _, _  = functions.find_intervals(id_above_thr_motion, 30, real_time)
    #to extract running index use the function with bigger interval
    
    delet_Running_IDX = []
    Aroused_stationary_index_check = list(chain(*Aroused_stationary_index_temp))
    for i in Aroused_stationary_index_check:
        if i in Aroused_Running_index_check:
            delet_Running_IDX.append(i)
    mask = np.isin(Aroused_stationary_index_check, delet_Running_IDX, invert=True)
    result = np.extract(mask, Aroused_stationary_index_check)

    Aroused_stationary_index, Aroused_stationary_window, Real_time_Aroused_stationary = functions.find_intervals(result, min_states_window['AS'], real_time, 45)
    ###------------------------------Calculate only whisking state-----------------------------###
    only_whisking = []
    for i in id_above_thr_motion:
        if i in P_rest_index:
            only_whisking.append(i)
    only_whisking_index, _, _ = functions.find_intervals(only_whisking, 30, real_time)
    only_whisking = list(chain(*only_whisking_index))
    ###------------------------------Calculate final rest state-----------------------------###
    rest_index_temp = [x for x in P_rest_index if x not in only_whisking]
    rest_index, rest_window, Real_Time_rest = functions.find_intervals(rest_index_temp, 2, real_time)
    ###------------------------------Not aroused, basal motor activity-----------------------------###
    id_above_thr_binary = np.extract(binary_movement == 1, ids)
    delete_activity = []
    for id in id_above_thr_binary:
        if id in result:
            delete_activity.append(0)
        elif id in Aroused_Running_index_check:
            delete_activity.append(0)
        else:
            delete_activity.append(1)
    Not_aroused_activity = np.extract(delete_activity, id_above_thr_binary)
    Not_aroused_activity_index, Not_aroused_activity_window, Real_time_Not_aroused_activity = functions.find_intervals(Not_aroused_activity, min_states_window['NABMA'], real_time, 45)

    ###------------------------------Undefined states-----------------------------###
    to_delete__idx = []
    for i in undefined_state_idx:
        if i in list(chain(*Aroused_stationary_index)) or\
                i in list(chain(*Aroused_Running_index)) or\
                i in list(chain(*rest_index)) or\
                i in list(chain(*Not_aroused_activity_index)):
            to_delete__idx.append(i)
    mask = np.isin(undefined_state_idx, to_delete__idx, invert=True)
    result = np.extract(mask, undefined_state_idx)
    _,  Undefined_state_window, Real_time_Undefined_state = functions.find_intervals(result, 0, real_time)

    Real_Time_states = {'run' : Real_Time_Aroused_Running, 
                        'AS' : Real_time_Aroused_stationary, 
                        'NABMA' : Real_time_Not_aroused_activity, 
                        'rest' : Real_Time_rest,
                        'undefined' : Real_time_Undefined_state}
    states_window = {'run' : Aroused_Running_window, 
                     'AS' : Aroused_stationary_window, 
                     'NABMA' : Not_aroused_activity_window, 
                     'rest' : rest_window,
                     'undefined' : Undefined_state_window}

    return Real_Time_states, states_window

def mean_max_interval(dF, interval, method):
    if method == 'max':
        All_cells = []
        for ROI in dF:
            max_i = []
            for window in interval:
                start = window[0]
                end = window[-1]
                max_interval = np.max(ROI[start: end])
                max_i.append((max_interval))
            All_cells.append(np.mean(max_i))
    elif method == 'mean':
        All_cells = []
        for ROI in dF:
            ROI_i = []
            for window in interval:
                dF_interval = ROI[window[0]: window[-1]]
                ROI_i.append(dF_interval)
            ROI_i = list(chain(*ROI_i))
            mean_ROI = np.mean(ROI_i)
            All_cells.append(mean_ROI)
    return All_cells

def state_duration(state_real_time):
    general_time = 0
    for window in state_real_time:
        interval = window[-1]-window[0]
        general_time = general_time + interval
    return general_time

def mean_interval (X, interval, method):
    pupil_interval = []
    for window in interval:
        start = window[0]
        end = window[-1]
        if method == 'max':
            max_interval = np.max(X[start: end])
            pupil_interval.append(max_interval)
        elif method == 'mean':
            mean_interval = np.mean(X[start: end])
            pupil_interval.append(mean_interval)
    Max_interval_pupil= np.mean(pupil_interval)
    return Max_interval_pupil

def stage_plot(motion, speed, pupil, F, real_time, Real_Time_states, states_window, filter_kernel, save_dir, svg, threshold=None):
    speed = gaussian_filter1d(speed, filter_kernel['speed'])
    motion = gaussian_filter1d(motion, filter_kernel['motion'])
    pupil = gaussian_filter1d(pupil, 10)

    colors_list = ['crimson', 'darkorange', 'gold', 'c', 'gray']
    colors_list2 = ['red', 'green', 'navajowhite', 'plum']
    c = 'gray'
    alphas_list = [[0.2,1], [0,1], [0,1], [0,1], [0,1]]
    states_names = ['Running', 'NABMA', 'As', 'Rest', 'Undefined']

    ###------------------------------Plotting_states2-----------------------------###
    Mean_F = np.mean(F, 0)
    filtered_F = gaussian_filter1d(Mean_F, 10)
    marker_idx1 = list(chain(*states_window['run']))
    marker_idx2 = list(chain(*states_window['NABMA']))
    marker_idx3 = list(chain(*states_window['AS']))
    marker_idx4 = list(chain(*states_window['rest']))
    marker_idx5 = list(chain(*states_window['undefined']))
    marker_idx = [marker_idx1, marker_idx2, marker_idx3, marker_idx4, marker_idx5]
    line_seg_list = [np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]

    #------------------
    fig2, ax = plt.subplots(4, 1, figsize=(17, 7))

    xy_vals = np.transpose([real_time, filtered_F])
    for i in range(len(marker_idx)) :
        line_seg_list[i] = np.split(xy_vals, marker_idx[i])
    for i in range(len(line_seg_list)) :
        ax[0].add_collection(LineCollection(line_seg_list[i],  colors=(colors_list[i]), alpha=alphas_list[i]))

    ax[0].set_ylim(min(filtered_F), max(filtered_F))
    ax[0].set_ylabel(r'$\Delta$F/F')
    ax[0].set_xticks([])
    ax[0].margins(x=0)
    
    #------------------
    zscored_speed = zscore(speed)
    xy_vals = np.transpose([real_time, zscored_speed])
    for i in range(len(marker_idx)) :
        line_seg_list[i] = np.split(xy_vals, marker_idx[i])
    for i in range(len(line_seg_list)) :
        ax[1].add_collection(LineCollection(line_seg_list[i],  colors=(colors_list[i]), alpha=alphas_list[i]))

    ax[1].set_ylim(min(zscored_speed), max(zscored_speed))
    ax[1].set_ylabel('Speed z-score')
    ax[1].set_xticks([])
    ax[1].margins(x=0)

    #------------------
    zscored_pupil = zscore(pupil)
    xy_vals = np.transpose([real_time, zscored_pupil])
    for i in range(len(marker_idx)) :
        line_seg_list[i] = np.split(xy_vals, marker_idx[i])
    for i in range(len(line_seg_list)) :
        ax[2].add_collection(LineCollection(line_seg_list[i],  colors=(colors_list[i]), alpha=alphas_list[i]))

    ax[2].set_ylim(min(pupil), max(pupil))
    ax[2].set_ylabel('Pupil z-score')
    ax[2].set_xticks([])
    ax[2].margins(x=0)

    #--------------------
    zscored_motion = zscore(motion)
    xy_vals = np.transpose([real_time, zscored_motion])
    for i in range(len(marker_idx)) :
        line_seg_list[i] = np.split(xy_vals, marker_idx[i])
    for i in range(len(line_seg_list)) :
        ax[3].add_collection(LineCollection(line_seg_list[i],  colors=(colors_list[i]), alpha=alphas_list[i]))

    ax[3].set_ylim(min(zscored_motion), max(zscored_motion))
    ax[3].set_ylabel('Motion z-score')
    ax[3].set_xlabel('Time(s)')
    ax[3].margins(x=0)

    #------------------
    Running = Line2D([0], [0], color=colors_list[0], linewidth=5)
    N_aroused_motor_activity = Line2D([0], [0], color=colors_list[1], linewidth=5)
    Aroused_stationary = Line2D([0], [0], color=colors_list[2], linewidth=5)
    rest = Line2D([0], [0], color=colors_list[3], linewidth=5)
    undefined = Line2D([0], [0], color=colors_list[4], linewidth=5)
    ax[0].legend([Running, N_aroused_motor_activity, Aroused_stationary, rest, undefined],
                 states_names,
                 loc='center left', bbox_to_anchor=(1.0, 1.0) ,prop={'size': 6})
    
    #------------------
    if svg == True:
        save_direction_svg = os.path.join(save_dir, "activity_states2.svg")
        fig2.savefig(save_direction_svg,format = 'svg')
    functions.save_fig("activity_states2", save_dir, fig2)

    ##------------------------------------Plotting_states-----------------------------###
    min_val = np.min(motion)
    max_val = np.max(motion)
    min_speed = np.min(speed)
    max_speed = np.max(speed)

    # Apply Min-Max Scaling
    normalizedmotion = (motion - min_val) / (max_val - min_val)
    normalizespeed = (speed - min_speed) / (max_speed - min_speed)

    fig, axs = plt.subplots(3, 1, figsize=(15, 5))
    # Create a basic plot
    axs[0].plot(real_time, filtered_F, color=c)
    axs[0].set_ylabel(r'$\Delta$F/F')
    axs[0].margins(x=0)
    axs[1].set_xticks([])

    axs[1].plot(real_time, normalizespeed, color=c)
    axs[1].set_ylabel('speed')
    axs[1].margins(x=0)
    axs[1].set_xticks([])

    axs[2].plot(real_time, normalizedmotion, color=c)
    axs[2].set_ylabel('face motion')
    axs[2].margins(x=0)
    axs[2].set_xlabel('Time(s)')

    if threshold != None :
        tr = len(F[0])*[(threshold['speed'] - min_speed) / (max_speed - min_speed)]
        axs[1].plot(real_time, tr, color = 'r', linestyle='--')
        tr = len(F[0])*[(0.01 - min_speed) / (max_speed - min_speed)]
        axs[1].plot(real_time, tr, color = 'peru', linestyle='--')
        tr = len(F[0])*[(threshold['motion']*(np.std(motion)) - min_val) / (max_val - min_val)]
        axs[2].plot(real_time, tr, color = 'r', linestyle='--')

    for k in range(3):
        for i in range(len(Real_Time_states['run'])):
            axs[k].axvspan(Real_Time_states['run'][i][0], Real_Time_states['run'][i][-1], color=colors_list[0], alpha=0.5)
        for i in range(len(Real_Time_states['NABMA'])):
            axs[k].axvspan(Real_Time_states['NABMA'][i][0], Real_Time_states['NABMA'][i][-1], color=colors_list[1], alpha=0.5)
        for i in range(len(Real_Time_states['rest'])):
            axs[k].axvspan(Real_Time_states['rest'][i][0], Real_Time_states['rest'][i][-1], color=colors_list[3], alpha=0.5)
        for i in range(len(Real_Time_states['AS'])):
            axs[k].axvspan(Real_Time_states['AS'][i][0], Real_Time_states['AS'][i][-1], color=colors_list[2], alpha=0.5)
    
    Running = Line2D([0], [0], color=colors_list[0], linewidth=5)
    N_aroused_motor_activity = Line2D([0], [0], color=colors_list[1], linewidth=5)
    Aroused_stationary = Line2D([0], [0], color=colors_list[2], linewidth=5)
    rest = Line2D([0], [0], color=colors_list[3], linewidth=5)
    axs[0].legend([Running, N_aroused_motor_activity, Aroused_stationary, rest],
                 states_names[:-1],
                 loc='center left', bbox_to_anchor=(1.0, 1.0) ,prop={'size': 6})
    
    if svg == True:
        save_direction_svg = os.path.join(save_dir, "activity_states.svg")
        fig.savefig(save_direction_svg,format = 'svg')
    functions.save_fig("activity_states",save_dir,fig)


# stage_plot(Real_Time_Aroused_Running, Real_time_Not_aroused_activity2, Real_Time_rest_window2,
#                 Real_time_Aroused_stationary2, speed, motion, F, real_time,
#                 Aroused_Running_window, Not_aroused_activity_window2, rest_window2, Aroused_stationary_window2, save_dir, pupil)

def kruskal_test(valid_neuron,active_movement_window2, inactive_window, rest_window2, only_whisking_window, save_direction,svg):
    mean_active = mean_max_interval(valid_neuron, active_movement_window2, method='mean')
    mean_Intermediate = mean_max_interval(valid_neuron, inactive_window, method='mean')
    mean_rest = mean_max_interval(valid_neuron, rest_window2, method='mean')
    mean_OnlyWhisking = mean_max_interval(valid_neuron, only_whisking_window, method='mean')
    # Combine the data into a single array
    if len(only_whisking_window) >= 1:
        K = [mean_active, mean_Intermediate, mean_rest, mean_OnlyWhisking]
        data = np.concatenate(K)
        labels = ['Running'] * len(mean_active) + ['Paw movement'] * len(mean_Intermediate) + ['Rest'] * len(mean_rest) + ['As'] * len(mean_OnlyWhisking)
        # Perform ANOVA
        h_statistic, p_value = kruskal(mean_active, mean_Intermediate, mean_rest, mean_OnlyWhisking)

        print("Kruskal-Wallis test results:")
        print("H-statistic:", h_statistic)
        print("p-value:", p_value)
    else:
        K = [mean_active, mean_Intermediate, mean_rest]
        data = np.concatenate(K)
        labels = ['Running'] * len(mean_active) + ['Paw movement'] * len(mean_Intermediate) + ['Rest'] * len(
            mean_rest)
        h_statistic, p_value = kruskal(mean_active, mean_Intermediate, mean_rest)

        print("Kruskal-Wallis test results:")
        print("H-statistic:", h_statistic)
        print("p-value:", p_value)
    if p_value < 0.05:
        tukey_results = pairwise_tukeyhsd(data, labels)
        summary_results = tukey_results.summary()
        print("summary_results")
        print(summary_results)
        df_results = pd.DataFrame(data=summary_results.data[1:], columns=summary_results.data[0])

    else:
        df_results = pd.DataFrame(['No significant difference'], index = [''], columns = ['']).T
    file_name = "tukey_results.xlsx"
    save_direction2 = os.path.join(save_direction, file_name)
    df_results.to_excel(save_direction2, index=False)
    fig, ax = plt.subplots(figsize=(6, 8))
    boxplot1 = ax.boxplot(mean_rest, positions=[1], widths=0.6, patch_artist=True)
    boxplot2 = ax.boxplot(mean_Intermediate, positions=[2], widths=0.6, patch_artist=True)
    boxplot3 = ax.boxplot(mean_OnlyWhisking, positions=[3], widths=0.6, patch_artist=True)
    boxplot4 = ax.boxplot(mean_active, positions=[4], widths=0.6, patch_artist=True)
    for box in boxplot1['boxes']:
        box.set(color='lightsteelblue', facecolor='lightsteelblue')

    for box in boxplot2['boxes']:
        box.set(color='slategray', facecolor='slategray')

    for box in boxplot3['boxes']:
        box.set(color='rosybrown', facecolor='rosybrown')

    for box in boxplot4['boxes']:
        box.set(color='darkseagreen', facecolor='darkseagreen')

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['Rest','paw Movement', 'As', 'Running'])
    ax.set_ylabel('mean')
    ax.set_title(r'z-scored $\Delta$F/F')
    if svg == True:
        save_direction_svg = os.path.join(save_direction,"activity_B_states.svg")
        fig.savefig(save_direction_svg,format = 'svg')
    functions.save_fig("activity_B_states", save_direction, fig)
