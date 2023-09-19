import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os.path
import easygui
import random
from scipy.ndimage import filters
from scipy.signal import correlate
from matplotlib import gridspec
import copy
from matplotlib.gridspec import GridSpec
from scipy.signal.windows import hamming
from scipy.signal import convolve
# Removing ROIs which are not cell
def detect_cell(cell,F):
    removed_ROI=[]
    keeped_ROI=[]
    k = 0
    for i in range(len(cell)):
        if cell[i][0]==0:
            removed_ROI.append(i)
        else:
            keeped_ROI.append([i, k])
            k += 1

    if len(F) != len(keeped_ROI):
        removed_ROI = sorted(removed_ROI, reverse=True)
        for idx in  removed_ROI:
            if idx < len(F):
                F.pop(idx)
        # print(len(removed_ROI), "Non cell ROI were removed")
    else:
        print("non cell ROIs are already removed")
    F = np.array(F)
    return F, keeped_ROI

# def detect_cell(cell,F):
#     removed_ROI=[]
#     keeped_ROI=[]
#     for i in range(len(cell)):
#         if cell[i][0]==0:
#             removed_ROI.append(i)
#         else:
#             keeped_ROI.append(i)
#     # print(len(removed_ROI), "non cell ROI were detected")
#
#     #removing non cells ROIs
#
#     if len(F) != len(keeped_ROI):
#         removed_ROI = sorted(removed_ROI, reverse=True)
#         for idx in  removed_ROI:
#             if idx < len(F):
#                 F.pop(idx)
#         # print(len(removed_ROI), "Non cell ROI were removed")
#     else:
#         print("non cell ROIs are already removed")
#     F = np.array(F)
#     return F

#detecting f0
def sliding_window(F, fs=30, sig= 60, win = 60):
    Flow = filters.gaussian_filter(F, [0., sig])
    Flow = filters.minimum_filter1d(Flow, win * fs, mode='wrap')
    Flow = filters.maximum_filter1d(Flow, win * fs, mode='wrap')
    return Flow

#df/f
def deltaF_calculate(F, F0):
    normalized_F = np.copy(F)
    for i in range(0, np.size(F, 0)):
        normalized_F[i] = (F[i]-F0[i])/F0[i]
    return normalized_F


''' this function will calculate if is there any activity or not for face motion
input:
threshold_motion ---> calculated threshold for considering a value in the facemap motion output as moving or not
motion ---> motion values in the facemap for whisking

output:
whisk_motion ---->values for whisker motion 
whisk_No_motion ---->values for whisker No motion
index_motion ----> coresponding index in the list for whisker mition values
index_nomotion ----> coresponding index in the list for whisker No mition values '''

def faceMotion_calculate(motion, threshold_motion):
    whisk_motion=[]
    whisk_No_motion=[]
    index_motion=[]
    index_nomotion=[]
    for i in range(len(motion)):
        if motion[i]>threshold_motion:
            whisk_motion.append(motion[i])
            index_motion.append(i)
        else:
            whisk_No_motion.append(motion[i])
            index_nomotion.append(i)
    
    return whisk_motion, whisk_No_motion, index_motion, index_nomotion

############################################################

def dF_faceMotion(index_motion, index_nomotion,dF ):
    F_motion=[]
    for j in range(len(dF)):
        f_motion=[]
        for k in range(len(index_motion)):
            v=index_motion[k]
            f_motion.append(dF[j][v])
        F_motion.append(f_motion)

    F_NoMotion=[]
    for p in range(len(dF)):
        f_NoMotion=[]
        for r in range(len(index_nomotion)):
            v=index_nomotion[r]
            f_NoMotion.append(dF[p][v])
        F_NoMotion.append(f_NoMotion)
    return F_NoMotion, F_motion


'''Calculating mean '''

def mean_fluo(dF):
    mean_dF=[]
    for i in range(len(dF)):
        average_dF =np.mean(dF[i])
        mean_dF.append(average_dF)
    return mean_dF
#######################################################
#calculating absolut run and rest index

def RunRest_calculate(speed,th):
    run=[]
    rest=[]
    index_run=[]
    index_rest=[]
    for i in range(len(speed)):
        if speed[i]>th:
            run.append(speed[i])
            index_run.append(i)
        else:
            rest.append(speed[i])
            index_rest.append(i)
    
    return run, rest, index_run, index_rest

## normalized df

def Normal_df(dF):
    Normal_df=[]
    for i in range(len(dF)):
        p = max([abs(b) for b in dF[i]])
        NORMAL_dF=[]
        for j in range(len(dF[i])):
            normal_dF=dF[i][j]/p
            NORMAL_dF.append(normal_dF)
        Normal_df.append(NORMAL_dF)
        
    return Normal_df

####################################################
# get all data in the folder and combine them
def get_shuffling_data(data_type):
    path=easygui.diropenbox(title='select folder contaning '+data_type+' for shuffling')
    #read WIs
    TOTAL__data = []
    for i in os.listdir(path):
        data = np.load(path+'\\'+i)
        TOTAL__data.append(data)
    return TOTAL__data
######################################################
#merging data for shuffling and creat a subset of data
def subset_shuffling_data(total__data,data_for_session):
    #merge all together
    total_data =(total__data[0]).tolist()
    for i in range(1, len(total__data)):
        total__data[i]=total__data[i].tolist()
        total_data= total_data + total__data[i]
    total_data=total_data+data_for_session
    #Shuffling and creat a random subset of data
    Shuffled_data=random.sample(total_data, len(data_for_session))
    #Total_LMI=np.concatenate((Total_LMI,LMI_P))
    return Shuffled_data  
###################################################

def interpolation1(sampel_data_for_interpolation, data_for_interpolating):
    h=str([k for k, v in globals().items() if v is data_for_interpolating][0])
    step=len(data_for_interpolating)/len(sampel_data_for_interpolation[0])
    fp= data_for_interpolating
    xp=(np.arange(0, len(data_for_interpolating), 1, dtype=float))
    x=(np.arange(0, len(data_for_interpolating), step, dtype=float))
    data_for_interpolating = np.interp(x, xp, fp)
    print("data for ",h," were interpolated")
    return data_for_interpolating
################################################
#Plots
def HistoPlot(X,xLabel,save_direction1):
    # ploting
    fig14 = plt.figure(figsize=(16, 7))
    plt.hist(X, weights=(np.ones(len(X)) / len(X)) * 100, edgecolor="black", color="gray", bins=20)
    plt.ylabel('percentage', size=20, labelpad=10)
    plt.xlabel(xLabel, size=20, labelpad=10)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    median_absolut = np.median(X)
    plt.axvline(x=median_absolut, label='median', linewidth=4)
    plt.legend(bbox_to_anchor=(1.0, 1), prop={'size': 14}, )
    file_name14 = xLabel
    save_direction14 = os.path.join(save_direction1, file_name14)
    isExist = os.path.exists(save_direction14)
    if isExist:
        fig14.savefig(save_direction14)
    else:
        fig14.savefig(save_direction14)

def lag(t_imaging,valid_neurons, save_direction03, dF, X, label):
    length = len(X)
    step = t_imaging[-1] / length
    freq = length / t_imaging[-1]
    l = -length + 1
    time_corr = np.arange(l / freq, length / freq, step)
    Time = np.linspace(0, t_imaging[-1], length)
    plot_duration = int(20 / step)
    start = length - plot_duration
    end = length + plot_duration
    all_lag = []
    for i in valid_neurons:
        correlation = correlate(dF[i], X)
        correlation = correlation.tolist()
        max_lagI = max(correlation)
        Max_index = correlation.index(max_lagI)
        lagI = (Max_index - length) / freq
        all_lag.append(lagI)
        gs = gridspec.GridSpec(6, 1)
        fig1 = plt.figure(figsize=(14, 7))
        ax2 = plt.subplot(gs[0:3, 0])
        ax1 = plt.subplot(gs[3, 0])
        ax3 = plt.subplot(gs[4, 0])
        ax3.plot(Time, X, label=label, color="teal")
        ax3.set_xlabel('Time(s)')
        ax3.set_yticks([])
        ax1.plot(Time, dF[i], label="dF ROI " + str(i), color="pink")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.plot(time_corr[start:end], correlation[start:end], label="correlation")
        ax2.set_title("dF and " + label + " correlation", fontsize=11)
        ax2.axvline(0, color='plum', linestyle='dashed', linewidth=1.5, label='0')
        ax2.annotate(f'lag(s) =  {lagI:.3f}', xy=(0.01, 0.98), xycoords='axes fraction', fontsize=9, va='top',
                     ha='left')
        ax2.margins(x=0)
        ax1.margins(x=0)
        ax3.margins(x=0)
        gs.update(hspace=0.6)
        ax1.legend(loc="upper right", fontsize="x-small")
        ax2.legend(loc="upper right", fontsize="x-small")
        ax3.legend(loc="upper right", fontsize="x-small")
        file_name = "ROI " + str(i) + " " + label + " correlation"
        save_direction = os.path.join(save_direction03, file_name)
        isExist = os.path.exists(save_direction)
        if isExist:
            pass
        else:
            plt.savefig(save_direction)
        plt.close(fig1)

    dF_list = dF.tolist()
    valid_dF = []
    for i in valid_neurons:
        valid_dF.append(dF_list[i])
    valid_dF = np.array(valid_dF)

    Mean_dF_corr = np.mean(valid_dF, 0)
    correlation_mean = correlate(Mean_dF_corr, X)
    correlation_mean = correlation_mean.tolist()
    max_mean_lag = max(correlation_mean)
    Max_index_mean = correlation_mean.index(max_mean_lag)
    lag_mean = (Max_index_mean - length) / freq

    fig = plt.figure(figsize=(9, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax.hist(all_lag, weights=(np.ones(len(all_lag)) / len(all_lag)) * 100, bins=25, alpha=0.5, label='all ROIs lag')
    median = np.median(all_lag)
    ax.axvline(median, color= "teal", linestyle='dashed', linewidth=1.5, label='median')
    ax.set_xlabel('lag for mean dF all ROIs')
    ax.set_ylabel('Percentage')
    ax.set_title("dF- " + label + " lag")
    ax.legend()
    ax2.plot(time_corr[start:end], correlation_mean[start:end], alpha=0.7, label='mean dF lag')
    ax2.axvline(0, color='red', linestyle='dashed', linewidth=1.5, label='zero')
    ax2.set_yticks([])
    ax.set_xlabel('TIME(s)')
    ax2.margins(x=0)
    ax2.annotate(f'mean lag (s) =  {lag_mean:.3f}', xy=(0.01, 0.98), xycoords='axes fraction', fontsize=9, va='top',
                 ha='left')
    ax2.legend(loc='upper right', fontsize='small')
    file_name003 = "all ROIs lag( " + label + " )"
    save_direction003 = os.path.join(save_direction03, file_name003)
    isExist = os.path.exists(save_direction003)
    if isExist:
        pass
    else:
        plt.savefig(save_direction003)
    plt.close(fig)

    return all_lag, lag_mean
#Permutation

def permutation(dF, speed,label, save_direction202,sampels = 1000):
    dF_p = copy.deepcopy(dF)
    speed_p = speed/max(speed)
    for i in range(len(dF_p)):
        dF_p[i]= dF_p[i]/max(dF_p[i])
    dF_p= dF_p.tolist()
    out_neurons = []
    valid_neurons = []
    for s in range(len(dF_p)):
        random_list = (np.random.choice(np.arange(0, len(dF_p[s]) - 2), sampels, replace=False))
        df_shuffeled = []
        for i in random_list:
            df_shuffeled_i = dF_p[s][i:] + dF_p[s][0:i]
            df_shuffeled.append(df_shuffeled_i)
        real_corr, _= pearsonr(speed_p, dF_p[s])

        permuted_corrs = []
        for i in range(len(df_shuffeled)):
            correlation, _ = pearsonr(speed_p, df_shuffeled[i])
            permuted_corrs.append(correlation)

        p_value = np.sum(np.abs(permuted_corrs) >= np.abs(real_corr)) / sampels


        fig = plt.figure(figsize=(9, 7))
        gs = GridSpec(2, 1, height_ratios=[3, 1])
        ax = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        ax.hist(permuted_corrs,weights=(np.ones(len(permuted_corrs)) / len(permuted_corrs)) * 100, bins=30, alpha=0.5, label='Permutations')
        ax.axvline(real_corr, color='red', linestyle='dashed', linewidth=2, label='Observed')
        ax.set_xlabel('Correlation Coefficient')
        ax.set_ylabel('Percentage')
        ax.set_title('Permutation Test(' + label + ')')
        ax.legend(loc='upper right', fontsize='small')
        ax.annotate(f'p-value = {p_value:.3f}', xy=(0.02, 0.98), xycoords='axes fraction', fontsize=9,
                     va='top', ha='left')
        ax2.plot(dF_p[s], label='dF')
        ax2.set_ylabel("ROI"+ str(s))
        ax2.plot(speed_p, alpha=0.7, label= label)
        ax2.set_yticks([])
        ax2.margins(x=0)
        ax2.legend(loc='upper right', fontsize='small')
        file_name = "ROI " + str(s) +" " + label + " Permutation"
        save_direction = os.path.join(save_direction202, file_name)
        isExist = os.path.exists(save_direction)
        if isExist:
            pass
        else:
            fig.savefig(save_direction)
        plt.close(fig)

        if p_value > 0.05:
            out_neurons.append(s)
        else:
            valid_neurons.append(s)
    return valid_neurons, out_neurons

def calculate_F0(F, fs, percentile, mode = 'hamming', win = 60, sig = 60):
    if mode == 'hamming':
        F0 = []
        window_duration = 0.5  # Duration of the Hamming window in seconds
        window_size = int(window_duration * fs)
        hamming_window = hamming(window_size)
        for i in range(len(F)):
            F_smooth = convolve(F[i], hamming_window, mode='same') / sum(hamming_window)
            roi_percentile = np.percentile(F_smooth, percentile)
            F_below_percentile = np.extract(F_smooth <= roi_percentile, F_smooth)
            f0 = np.mean(F_below_percentile)
            f0 = [f0]*len(F[i])
            F0.append(f0)
        F0 = np.array(F0)
    elif mode == 'sliding':
        F0 = filters.gaussian_filter(F, [0., sig])
        F0 = filters.minimum_filter1d(F0, win * fs, mode='wrap')
        F0 = filters.maximum_filter1d(F0, win * fs, mode='wrap')
    return F0

def find_intervals(id_below_thr_speed, interval):
    speed_window =[]
    window = []
    for i in range(len(id_below_thr_speed)):
        if (id_below_thr_speed[i] + 1) in id_below_thr_speed:
            window.append(id_below_thr_speed[i])
        elif (id_below_thr_speed[i] - 1) in id_below_thr_speed:
            window.append(id_below_thr_speed[i])
            if len(window)>interval:
                speed_window.append(window)
                window = []
            else:
                window = []
        else:
            pass
    return speed_window

def save_data(file_name, save_dir, data):
    save_direction = os.path.join(save_dir, file_name)
    isExist = os.path.exists(save_direction)
    if isExist:
        pass
    else:
        np.save(save_direction, data, allow_pickle=True)
def save_fig(fig_name, save_dir, fig):
    save_direction = os.path.join(save_dir, fig_name)
    isExist = os.path.exists(save_direction)
    if isExist:
        pass
    else:
        fig.savefig(save_direction)
    plt.close(fig)

def plot_matrix_synchro(synchro, settings,path, show=True):
    plt.ioff()
    plt.matshow(synchro)
    plt.colorbar()

    plt.xlabel('ROI')
    plt.ylabel('ROI')
    #    plt.title('Synchrony')

    plt.savefig('{}/synchrony.pdf'.format(path))
    plt.savefig('{}/synchrony'.format(path))

    if show == True:
        plt.show()
    else:
        plt.close()