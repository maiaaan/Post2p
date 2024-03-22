import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os.path
from scipy.ndimage import filters
from scipy.signal import correlate
from matplotlib import gridspec
import copy
from scipy.signal.windows import hamming
from sklearn.linear_model import LinearRegression
from scipy.signal import convolve
from tqdm import tqdm
import bottleneck as bn


def detect_cell(cell, F):
    removed_ROI = [i for i, c in enumerate(cell) if c[0] == 0]
    keeped_ROI = [j for j, i in enumerate(cell) if i[0] != 0]
    if len(F) != len(keeped_ROI):
        F = np.delete(F, removed_ROI, axis=0)
    return F, keeped_ROI

def creat_H5_dataset(group, variable, variable_name):
    for name, value in zip(variable_name, variable):
        group.create_dataset(name, data=value)

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

'''Calculating mean '''

def mean_fluo(dF):
    mean_dF=[(np.mean(dF[i])) for i in range(len(dF))]
    return mean_dF

## normalized df
def Normal_df(dF):
    Normal_df=[]
    for i in range(len(dF)):
        p = max([abs(b) for b in dF[i]])
        NORMAL_dF=[dF[i][j]/p for j in range(len(dF[i]))]
        Normal_df.append(NORMAL_dF)
        
    return Normal_df

def interpolation1(sampel_data_for_interpolation, data_for_interpolating):
    h=str([k for k, v in globals().items() if v is data_for_interpolating][0])
    step=len(data_for_interpolating)/len(sampel_data_for_interpolation[0])
    fp= data_for_interpolating
    xp=(np.arange(0, len(data_for_interpolating), 1, dtype=float))
    x=(np.arange(0, len(data_for_interpolating), step, dtype=float))
    data_for_interpolating = np.interp(x, xp, fp)
    print("data for ",h," were interpolated")
    return data_for_interpolating

def lag(t_imaging, valid_neurons, save_direction03, dF, X, label, speed_corr):
    length = len(X)
    step = t_imaging[-1] / length
    freq = length / t_imaging[-1]
    l = -length + 1
    time_corr = np.arange(l / freq, length / freq, step)
    Time = np.linspace(0, t_imaging[-1], length)
    plot_duration = int(20 / step)
    corr_interval = 150  # frame
    sstart = length - corr_interval
    eend = length + corr_interval
    start = length - plot_duration
    end = length + plot_duration
    all_lag = []
    positive_dF = []
    if len(valid_neurons)>0:
        for i in tqdm(valid_neurons, desc= 'calculating lag'):
            correlation = correlate(dF[i], X)
            correlation = correlation.tolist()
            interested_zone = correlation[sstart:eend]
            if speed_corr[i] >= 0:
                positive_dF.append(dF[i])
                max_lagI = max(interested_zone)
                Max_index = interested_zone.index(max_lagI)

                lagI = (Max_index - corr_interval-1) / freq
                all_lag.append(lagI)
                gs = gridspec.GridSpec(6, 1)
                fig1 = plt.figure(figsize=(14, 7))
                ax2 = plt.subplot(gs[0:4, 0])
                ax1 = plt.subplot(gs[4, 0])
                ax3 = plt.subplot(gs[5, 0])
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
            else:
                pass
    if positive_dF == []:
        raise Exception("No neuron has positive correlation")
    positive_dF = np.array(positive_dF)

    #---------------------------------------------------
    mean_posetive_dF = np.mean(positive_dF, 0)
    positive_correlation_mean = correlate(mean_posetive_dF, X)
    positive_correlation_mean = positive_correlation_mean.tolist()
    positive_interested_zone = positive_correlation_mean[sstart:eend]
    max_mean_lag_pos = max(positive_interested_zone)
    max_index_mean_pos = positive_interested_zone.index(max_mean_lag_pos)
    lag_mean_pos = (max_index_mean_pos - corr_interval-1) / freq

    fig = plt.figure(figsize=(11, 11))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax.hist(all_lag, weights=(np.ones(len(all_lag)) / len(all_lag)) * 100, bins=25, alpha=0.5, label='all ROIs lag')
    median = np.median(all_lag)
    ax.axvline(median, color="teal", linestyle='dashed', linewidth=1.5, label='median')
    ax.set_xlabel('lag for mean dF all ROIs')
    ax.set_ylabel('Percentage')
    ax.set_title("dF- " + label + " lag")
    ax.legend()
    ax2.plot(time_corr[start:end], positive_correlation_mean[start:end], alpha=0.7, label='mean dF lag')
    ax2.axvline(lag_mean_pos, color='red', linestyle='dashed', linewidth=1.5, label='mean lag')
    ax2.set_yticks([])

    ax2.margins(x=0)
    ax2.annotate(f'mean dF correlation (s) =  {lag_mean_pos:.3f}', xy=(0.01, 0.98), xycoords='axes fraction', fontsize=9, va='top',
                 ha='left')
    ax2.legend(loc='upper right', fontsize='small')
    ax2.set_ylabel('correlation')

    file_name003 = "all ROIs lag( " + label + " )"
    save_direction003 = os.path.join(save_direction03, file_name003)
    plt.savefig(save_direction003)
    plt.close(fig)
    return all_lag, lag_mean_pos

def permutation(dF, speed,label, save_direction202,samples = 1000):
    label2 = label + " permutation Processing"
    dF_p = copy.deepcopy(dF)
    speed_p = speed/np.max(speed)
    weights = np.ones(samples) / samples

    random_list = (np.random.choice(np.arange(0, len(speed_p) - 2), samples, replace=False))
    speed_shuffeled = [np.roll(speed_p, i) for i in random_list]
    dF_p = [i/np.max(i) for i in dF_p]

    out_neurons = []
    valid_neurons = []
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={'height_ratios': [3, 1]})
    for s in tqdm(range(len(dF_p)) ,desc=label2):
        real_corr, _= pearsonr(speed_p, dF_p[s])
        permuted_corrs = np.array([pearsonr(speed_shuffeled[i], dF_p[s])[0] for i in range(samples)])
        p_value = np.sum(np.abs(permuted_corrs) >= np.abs(real_corr)) / samples

        ax.clear()
        ax2.clear()
        ax.hist(permuted_corrs, weights=weights * 100, bins=30, alpha=0.5, label='Permutations')
        ax.axvline(real_corr, color='red', linestyle='dashed', linewidth=2, label='Observed')
        ax.set_xlabel('Correlation Coefficient')
        ax.set_ylabel('Percentage')
        ax.set_title(f'Permutation Test({label})')
        ax.legend(loc='upper right', fontsize='small')
        ax.annotate(f'p-value = {p_value:.3f}', xy=(0.02, 0.98), xycoords='axes fraction', fontsize=9,
                    va='top', ha='left')

        ax2.plot(dF_p[s], label='dF')
        ax2.set_ylabel(f'ROI {s}')
        ax2.plot(speed_p, alpha=0.7, label=label)
        ax2.set_yticks([])
        ax2.margins(x=0)
        ax2.legend(loc='upper right', fontsize='small')

        file_name = f'ROI {s} {label} Permutation'
        save_direction = os.path.join(save_direction202, file_name)
        is_exist = os.path.exists(save_direction)
        if not is_exist:
            fig.savefig(save_direction)

        if p_value > 0.05:
            out_neurons.append(s)
        else:
            valid_neurons.append(s)

    plt.close(fig)
    return valid_neurons, out_neurons
def calculate_alpha (F, Fneu):
    Slope = []
    per = np.arange(5,101,5)
    for k in range(len(F)):
        b = 0
        All_F, percentile_Fneu = [], []
        for i in per:
            percentile_before = np.percentile(Fneu[k], b)
            percentile_now = np.percentile(Fneu[k], i)
            index_percentile_i = np.where((percentile_before <= Fneu[k]) & (Fneu[k] < percentile_now))
            b = i
            F_percentile_i = F[k][index_percentile_i]
            perc_F_i = np.percentile(F_percentile_i, 5)
            percentile_Fneu.append(percentile_now)
            All_F.append(perc_F_i)

        #fitting a linear regression model
        x = np.array(percentile_Fneu).reshape(-1, 1)
        y = np.array(All_F)
        model = LinearRegression()
        model.fit(x, y)
        slope = model.coef_[0]
        Slope.append(slope)
    remove,alpha = [],[]
    for i in range(len(Slope)):
        if Slope[i] <= 0:
            remove.append(i)
        else:
            alpha.append(Slope[i])
    alpha = np.mean(alpha)
    return alpha, remove

def calculate_F0(F, fs, percentile, mode = 'sliding', win = 60, sig = 60):
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

def find_intervals(selected_ids, interval, RealTime, exclude_S = 0):
    #interval: minimum number of frames for each interval
    #exclude_S: frames that should be removed at the beginning of window
    Real_TIME_W = []
    motion_window = []
    motion_index =[]
    window = []
    for i in range(len(selected_ids)):
        if (selected_ids[i] + 1) in selected_ids:
            window.append(selected_ids[i])
        elif (selected_ids[i] - 1) in selected_ids:
            window.append(selected_ids[i])
            if len(window)>interval:
                motion_index.append(window[exclude_S:])
                window = []
            else:
                window = []
        else:
            pass
    for i in motion_index:
        rael_time_W = []
        S_E = []
        S_E.append(i[0])
        S_E.append(i[-1])
        rael_time_W.append(RealTime[i[0]])
        rael_time_W.append(RealTime[i[-1]])
        motion_window.append(S_E)
        Real_TIME_W.append(rael_time_W)
    return motion_index, motion_window, Real_TIME_W

def save_data(file_name, save_dir, data):
    save_direction = os.path.join(save_dir, file_name)
    isExist = os.path.exists(save_direction)
    if isExist:
        pass
    else:
        np.save(save_direction, data, allow_pickle=True)
def save_fig(fig_name, save_dir, fig):
    save_direction = os.path.join(save_dir, fig_name)
    fig.savefig(save_direction)
    plt.close(fig)

def detect_valid_neurons(valid_neurons, X):
    valid_X = [X[i] for i in valid_neurons]
    valid_X = np.array(valid_X)
    return valid_X

def make_dir (Base_path, file_name1):
    save_direction1 = os.path.join(Base_path, file_name1)
    isExist1 = os.path.exists(save_direction1)
    if isExist1:
        pass
    else:
        os.mkdir(save_direction1)
    return save_direction1
def save_exel(file_name, save_direction, exel):
    save_xl_skew = os.path.join(save_direction, file_name)
    exel.to_excel(save_xl_skew)
def convolve(Time, speed, motion, pupil,save_direction):
    tau=1.3 #characteristic decay time
    kernel_size=10 #size of the kernel in units of tau
    dt=np.mean(np.diff(Time)) #spacing between successive timepoints
    n_points=int(kernel_size*tau/dt)
    kernel_times=np.linspace(-n_points*dt,n_points*dt,2*n_points+1) #linearly spaced array from -n_points*dt to n_points*dt with spacing dt
    kernel=np.exp(-kernel_times/tau) #define kernel as exponential decay
    kernel[kernel_times<0]=0 #set to zero for negative times
    fig,ax=plt.subplots()
    ax.plot(kernel_times,kernel)
    ax.set_xlabel('time (s)')
    save_fig("convolotion_kernel",save_direction, fig)
    speed = convolve(speed,kernel,mode='same')*dt
    motion = convolve(motion,kernel,mode='same')*dt
    pupil = convolve(pupil,kernel,mode='same')*dt
    return speed, motion, pupil
def detect_blinking(pupil,pupil_id,window =4):
    """ March 2024 - Bacci Lab - faezeh.rabbani97@gmail.com
    ...........................................................................
    TOverall, this code calculates the moving variance of the pupil data using a
    window size of 4, which is used in this blink detection algorithms to identify
    sudden changes or fluctuations in pupil size that might indicate blinks.
    - - - - - - - - - - - - - - - Method  - - - - - - - - - - - - - - - - - - -
    This is a function provided by the Bottleneck library in Python.
    The move_var function calculates the moving variance along a given axis
     using a specified window size. The function divides the range of fluctuation
      by 20 to determine a threshold value. Any value above threshold is considered as blinking.
    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -
    pupil                       pupil trace extracted from facemap
    pupil_id                    the ids correspond to pupil frames made by np.arra
    nge
    window                      Size of window to calculate variance inside
    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - -
    ID_without_blinking         pupil ids after blinking was removed
    ...........................................................................
    """
    blink_detection = bn.move_var(pupil, window=window, min_count=1)
    threshold = (np.max(blink_detection) - np.min(blink_detection)) / 20
    blink_indices = {i for i, val in enumerate(blink_detection) if val > threshold}
    blink_indices.update({i + j for i in blink_indices for j in range(-5, 6)})
    ID_without_blinking = np.setdiff1d(pupil_id, sorted(blink_indices))
    return ID_without_blinking
