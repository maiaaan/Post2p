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

def split_stages(motion,speed, real_time,speed_threshold,min_run_win,min_AS_win,min_PM_win,min_Rest_win,S_filter_kernel,M_filter_kernel):
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
    ###------------------------------ Calculate active movement state -----------------------------###

    ids = np.arange(0,len(speed))
    # duration > 60 frames and speed > 0.5 s/m
    filtered_speed = gaussian_filter1d(speed, S_filter_kernel)
    id_above_thr_speed = np.extract(filtered_speed >= speed_threshold, ids)
    Aroused_Running_index, Aroused_Running_window, Real_Time_Aroused_Running = functions.find_intervals(id_above_thr_speed, min_run_win, real_time, 45)
    # to extract running index use the function with bigger interval
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
    for i in filtered_speed:
        if i < 0.01:
            binary_movement.append(0)
        else:
            binary_movement.append(1)
    binary_movement = np.array(binary_movement)
    id_rest = np.extract(binary_movement < 1, ids)
    P_rest_index, _, _ = functions.find_intervals(id_rest, min_Rest_win, real_time)
    P_rest_index = list(chain(*P_rest_index))
    ###------------------------------Calculate Aroused stationary state-----------------------------###
    filtered_motion = gaussian_filter1d(motion, M_filter_kernel)
    motion_threshold = 2*(np.std(motion))
    id_above_thr_motion = np.extract(filtered_motion >= motion_threshold, ids)
    Aroused_stationary_index, Aroused_stationary_window, Real_time_Aroused_stationary  = functions.find_intervals(id_above_thr_motion, 30 , real_time)
    #to extract running index use the function with bigger interval
    delet_Running_IDX = []
    Aroused_stationary_index_check = list(chain(*Aroused_stationary_index))
    for i in Aroused_stationary_index_check:
        if i in Aroused_Running_index_check:
            delet_Running_IDX.append(i)
    mask = np.isin(Aroused_stationary_index_check, delet_Running_IDX, invert=True)
    result = np.extract(mask, Aroused_stationary_index_check)
    Aroused_stationary_index2,Aroused_stationary_window2,Real_time_Aroused_stationary2 = functions.find_intervals(result,min_AS_win, real_time, 45)
    ###------------------------------Calculate only whisking state-----------------------------###
    only_whisking = []
    for i in id_above_thr_motion:
        if i in P_rest_index:
            only_whisking.append(i)
    only_whisking_index, only_whisking_window, Real_time_only_whisking_window = functions.find_intervals(only_whisking, 30, real_time)
    only_whisking = list(chain(*only_whisking_index))
    ###------------------------------Calculate final rest state-----------------------------###
    rest_index3 = [x for x in P_rest_index if x not in only_whisking]
    rest_index, rest_window2, Real_Time_rest_window2 = functions.find_intervals(rest_index3, 2, real_time)
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
    Not_aroused_activity1, Not_aroused_activity_window2, Real_time_Not_aroused_activity2 = functions.find_intervals(Not_aroused_activity,min_PM_win, real_time, 45)

    return Real_Time_Aroused_Running, Aroused_Running_window,\
        Real_time_Not_aroused_activity2, Not_aroused_activity_window2,\
        Real_Time_rest_window2, rest_window2,\
        Real_time_Aroused_stationary2, Aroused_stationary_window2

# Real_Time_Aroused_Running, Aroused_Running_window,\
#         Real_time_Not_aroused_activity2, Not_aroused_activity_window2,\
#         Real_Time_rest_window2, rest_window2,\
#         Real_time_Aroused_stationary2, Aroused_stationary_window2 = split_stages(motion, speed, real_time)
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

##------------------------------------Plotting_states2-----------------------------###
def stage_plot(Real_Time_Aroused_Running, Real_time_Not_aroused_activity2, Real_Time_rest_window2,
                Real_time_Aroused_stationary2, speed, motion, F, real_time,
                Aroused_Running_window, Not_aroused_activity_window2, rest_window2,
               Aroused_stationary_window2, save_dir, pupil,S_filter_kernel,svg):
    speed = gaussian_filter1d(speed,S_filter_kernel)
    min_val = np.min(motion)
    max_val = np.max(motion)
    min_speed = np.min(speed)
    max_speed = np.max(speed)
    # Apply Min-Max Scaling
    normalizedmotion = (motion - min_val) / (max_val - min_val)
    normalizespeed = (speed - min_speed) / (max_speed - min_speed)
    ###------------------------------Plotting_states-----------------------------###
    Mean_F = np.mean(F, 0)
    filtered_F = gaussian_filter1d(Mean_F, 10)
    marker_idx = list(chain(*Aroused_Running_window))
    marker_idx2 = list(chain(*Not_aroused_activity_window2))
    marker_idx3 = list(chain(*rest_window2))
    marker_idx4 = list(chain(*Aroused_stationary_window2))
    xy_vals = np.transpose([real_time, filtered_F])
    line_segments = np.split(xy_vals, marker_idx)
    line_segments2 = np.split(xy_vals, marker_idx2)
    line_segments3 = np.split(xy_vals, marker_idx3)
    line_segments4 = np.split(xy_vals, marker_idx4)

    fig2, ax = plt.subplots(4, 1, figsize=(17, 7))
    ax[0].add_collection(LineCollection(line_segments,  colors=('crimson','gray'), alpha=[0.2,1]))
    ax[0].add_collection(LineCollection(line_segments2, colors=('darkorange', 'gray'), alpha=[0,1]))
    ax[0].add_collection(LineCollection(line_segments4, colors=('gold', 'gray'), alpha=[0,1]))
    ax[0].add_collection(LineCollection(line_segments3, colors=('c', 'gray'), alpha=[0, 1]))

    ax[0].set_ylim(min(filtered_F), max(filtered_F))
    ax[0].set_ylabel('dF/F')
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    ax[0].margins(x=0)
    Running = Line2D([0], [0], color='crimson', linewidth=5)
    N_aroused_motor_activity = Line2D([0], [0], color='darkorange', linewidth=5)
    Aroused_stationary = Line2D([0], [0], color='gold', linewidth=5)
    rest = Line2D([0], [0], color='c', linewidth=5)
    ax[1].legend([Running, N_aroused_motor_activity, Aroused_stationary, rest],
                 ['Running', 'Paw movement', 'As', 'Rest'],
                 bbox_to_anchor=(1.0, 1.0) ,prop={'size': 6})
    #------------------

    xy_vals = np.transpose([real_time, pupil])
    line_segments = np.split(xy_vals, marker_idx)
    line_segments2 = np.split(xy_vals, marker_idx2)
    line_segments3 = np.split(xy_vals, marker_idx3)
    line_segments4 = np.split(xy_vals, marker_idx4)

    ax[2].add_collection(LineCollection(line_segments, colors=('crimson','gray'), alpha=[0.1,1]))
    ax[2].add_collection(LineCollection(line_segments2, colors=('darkorange', 'gray'), alpha=[0,1]))
    ax[2].add_collection(LineCollection(line_segments4, colors=('gold', 'gray'), alpha=[0,1]))
    ax[2].add_collection(LineCollection(line_segments3, colors=('c', 'gray'), alpha=[0, 1]))

    ax[2].set_ylim(min(pupil), max(pupil))
    ax[2].set_ylabel('Z scored pupil')
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].margins(x=0)
    #--------------------
    zscored_motion = zscore(motion)
    xy_vals = np.transpose([real_time, zscored_motion])
    line_segments = np.split(xy_vals, marker_idx)
    line_segments2 = np.split(xy_vals, marker_idx2)
    line_segments3 = np.split(xy_vals, marker_idx3)
    line_segments4 = np.split(xy_vals, marker_idx4)

    ax[3].add_collection(LineCollection(line_segments, colors=('crimson','gray'), alpha=[0.2,1]))
    ax[3].add_collection(LineCollection(line_segments2, colors=('darkorange', 'gray'), alpha=[0,1]))
    ax[3].add_collection(LineCollection(line_segments4, colors=('gold', 'gray'), alpha=[0,1]))
    ax[3].add_collection(LineCollection(line_segments3, colors=('c', 'gray'), alpha=[0, 1]))

    ax[3].set_ylim(min(zscored_motion), max(zscored_motion))
    ax[3].set_ylabel('Z scored motion')
    ax[3].set_xlabel('Time(s)')
    ax[3].set_yticks([])
    ax[3].margins(x=0)

    zscored_speed = gaussian_filter1d(speed, S_filter_kernel)
    xy_vals = np.transpose([real_time, zscored_speed])
    line_segments = np.split(xy_vals, marker_idx)
    line_segments2 = np.split(xy_vals, marker_idx2)
    line_segments3 = np.split(xy_vals, marker_idx3)
    line_segments4 = np.split(xy_vals, marker_idx4)

    ax[1].add_collection(LineCollection(line_segments, colors=('crimson','gray'), alpha=[0.2,1]))
    ax[1].add_collection(LineCollection(line_segments2, colors=('darkorange', 'gray'), alpha=[0,1]))
    ax[1].add_collection(LineCollection(line_segments4, colors=('gold', 'gray'), alpha=[0,1]))
    ax[1].add_collection(LineCollection(line_segments3, colors=('c', 'gray'), alpha=[0, 1]))
    ax[1].set_ylim(min(zscored_speed), max(zscored_speed))
    ax[1].set_ylabel('Z scored speed')
    #ax[1].set_yticks([])
    ax[1].set_xticks([])
    ax[1].margins(x=0)
    if svg == True:
        save_direction_svg = os.path.join(save_dir, "activity_states2.svg")
        fig2.savefig(save_direction_svg,format = 'svg')
    functions.save_fig("activity_states2", save_dir, fig2)

    fig, axs = plt.subplots(3, 1, figsize=(15, 5))
    # Create a basic plot
    axs[0].plot(real_time,normalizedmotion )
    axs[0].set_ylabel('whisking')
    axs[0].set_yticks([])
    axs[0].margins(x=0)
    axs[1].plot(real_time,normalizespeed)
    axs[1].set_ylabel('speed(cm/s)')
    tr = len(F[0])*[0.5]
    axs[1].plot(real_time, tr, color = 'r')
    #axs[1].set_yticks([])
    axs[1].margins(x=0)
    axs[1].set_xticks([])
    axs[2].plot(real_time, filtered_F)
    axs[2].set_ylabel('dF/F')
    axs[2].set_yticks([])
    axs[2].set_xlabel('Time(S)')
    axs[2].margins(x=0)
    for i in range(len(Real_Time_Aroused_Running)):
        axs[0].axvspan(Real_Time_Aroused_Running[i][0], Real_Time_Aroused_Running[i][-1], color='red', alpha=0.5)
        axs[1].axvspan(Real_Time_Aroused_Running[i][0], Real_Time_Aroused_Running[i][-1], color='red', alpha=0.5)
        axs[2].axvspan(Real_Time_Aroused_Running[i][0], Real_Time_Aroused_Running[i][-1], color='red', alpha=0.5)
    for i in range(len(Real_time_Not_aroused_activity2)):
        axs[0].axvspan(Real_time_Not_aroused_activity2[i][0], Real_time_Not_aroused_activity2[i][-1], color='green', alpha=0.5)
        axs[1].axvspan(Real_time_Not_aroused_activity2[i][0], Real_time_Not_aroused_activity2[i][-1], color='green', alpha=0.5)
        axs[2].axvspan(Real_time_Not_aroused_activity2[i][0], Real_time_Not_aroused_activity2[i][-1], color='green', alpha=0.5)
    for i in range(len(Real_Time_rest_window2)):
        axs[0].axvspan(Real_Time_rest_window2[i][0], Real_Time_rest_window2[i][-1], color='plum', alpha=0.5)
        axs[1].axvspan(Real_Time_rest_window2[i][0], Real_Time_rest_window2[i][-1], color='plum', alpha=0.5)
        axs[2].axvspan(Real_Time_rest_window2[i][0], Real_Time_rest_window2[i][-1], color='plum', alpha=0.5)
    for i in range(len(Real_time_Aroused_stationary2)):
        axs[0].axvspan(Real_time_Aroused_stationary2[i][0], Real_time_Aroused_stationary2[i][-1], color='navajowhite', alpha=0.5)
        axs[1].axvspan(Real_time_Aroused_stationary2[i][0], Real_time_Aroused_stationary2[i][-1], color='navajowhite', alpha=0.5)
        axs[2].axvspan(Real_time_Aroused_stationary2[i][0], Real_time_Aroused_stationary2[i][-1], color='navajowhite', alpha=0.5)
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
    ax.set_title('z-scored dF/F')
    if svg == True:
        save_direction_svg = os.path.join(save_direction,"activity_B_states.svg")
        fig.savefig(save_direction_svg,format = 'svg')
    functions.save_fig("activity_B_states", save_direction, fig)
