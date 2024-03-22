
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, resample
import pyabf
import easygui

def get_data():
    path = easygui.diropenbox(title='Select folder containing traces')
    files = easygui.fileopenbox(title='Select files to analyze', multiple=True)
    return path, files

def set_settings():
    settings = {
        'sampling frequency':2000,
        'final_sampling_frequency': 30,
        'binary conversion threshold':1.5,
        'cpr':1000,
        'perimeter_cm':29,
        'holding samples':10000,
        'excess samples':20000,
        'speed threshold':0.1,
        'time rec':300,
        'time_before':0.5,
        'time_after':2
        }
    return settings

def import_data(settings, path):
    '''
    Imports data from rotary encoder
    ---------------
    Input:
        path to .abf file 
    Output:
        numpy array with each channel
    '''
    abf = pyabf.ABF(path)
    abf.setSweep(sweepNumber=0, channel=0)
    ch_A = abf.sweepY
    abf.setSweep(sweepNumber=0, channel=1)
    ch_B = abf.sweepY
    ch_A = ch_A[(settings['holding samples']):(len(ch_A)-settings['holding samples']-settings['excess samples'])]
    ch_B = ch_B[settings['holding samples']:len(ch_B)-settings['holding samples']-settings['excess samples']]
    return ch_A, ch_B

def convert_binary(trace, thr):
    '''
    Converts Analog Files to Binary (0, 1) using a given threshold (thr)
    ---------------
    Input:
        trace - trace to convert
        thr - threshold to use
    
    Output:
        Converted trace
    '''
    binary = []
    for x in trace:
        if x >= thr:
            binary.append(1)
        else:
            binary.append(0)
    return binary

def position(A, B, settings):
    '''
    Takes traces A and B and converts it to a trace that has the same number of
    points but with positions points.
    ---------------
    Input:
        A, B - traces to convert
    
    Output:
        Positions through time
    '''
    positions = [0]
    a_last = 0
    b_last = 0
    for nA, nB in zip(A, B):
        if nA != a_last and nA == 1 and nB == 0:
            positions.append(positions[-1]+1)
        elif nB != b_last and nB == 1 and nA == 0:
            positions.append(positions[-1]+1)    
        else:
            positions.append(positions[-1])
        a_last = nA
        b_last = nB
    positions.pop(0)
    for i, v in enumerate(positions):
        positions[i] = v * settings['perimeter_cm']/settings['cpr']
    if positions[-1] <= positions[0]:
        for i in positions:
            positions[i] = positions[i]*-1
    return positions

def calc_speed(positions, settings, n_samples):
    '''
    Takes the positions through time and calculate the change between every
    n_samples.
    ---------------
    Input:
        positions - list containing the position at every time point. 
        n_samples - time points to jump when calculating the delta position.
        
    Output:
        speed - speed of the mouse at every time point in cm/s
        t - time distribution in seconds
    '''
    #factor = round(settings['sampling frequency']/settings['final_sampling_frequency']-0.5)
    #target_down = settings['final_sampling_frequency'] * settings['time rec']
    #positions_downsampled = positions[0:len(positions):factor]
    #speed = np.gradient(positions_downsampled)
    #speed = resample(speed, target_down)
    #speed_filt = savgol_filter(speed, 11, 1)
    #t = np.linspace(0,settings['time rec'],len(speed))
    
    if settings['sampling frequency'] % 2 > 0:
        window = settings['sampling frequency']
    else:
        window = settings['sampling frequency']+1
    #positions_smooth = savgol_filter(positions, window, 5)
    positions_smooth = positions
    
    
    t_positions = np.linspace(0,settings['time rec'],len(positions))
    speed_full = np.gradient(positions_smooth, t_positions)
    speed_downsampled = resample(speed_full, n_samples)
    #speed_smooth = savgol_filter(speed_downsampled, 31, 5)
    speed_smooth = speed_downsampled

    
    
    speed = np.where(speed_smooth<0, 0, speed_smooth)
    t = np.linspace(0,settings['time rec'],len(speed))
    
    return speed, t

def move_nomove(data_speed, settings):
    '''
    Converts the raw data into an array of moving/no moving 
    determined by a minimum speed.
    ---------------        
    Output:
        binary_movement - array with 0 or 1 corresponding to still or moving.
    '''
    binary_movement = (data_speed > settings['speed threshold']) * 1
#    binary_movement = []
#    for x in data_speed:
#        if abs(x) >= settings['speed threshold']:
#            binary_movement.append(1)
#        else:
#            binary_movement.append(0)
    return binary_movement

def extend_movenomove(binary_movement, settings):
    original = np.copy(binary_movement)
    ext_move = np.copy(binary_movement)
    before = original[0]
    n_before = int(settings['time_before'] * settings['final_sampling_frequency'])
    array_before = np.full((1,n_before),1)
    n_after = int(settings['time_after'] * settings['final_sampling_frequency'])
    array_after = np.full((1,n_after),1)
    for i, v in enumerate(original):
        if v == before:
            before = v
        elif v > before and i < n_before:
            ext_move[0:i] = array_before[0,0:i]
            before = v
        elif v > before:
            ext_move[(i-n_before):i] = array_before
            before = v
        elif v < before and i >= (len(ext_move)-n_after):
            ext_move[i:len(ext_move)] = array_after[0,0:(len(ext_move)-i)]
            before = v
        elif v < before:
            ext_move[i:i+n_after] = array_after
            before = v
        else:
            pass
    return ext_move

def plot_positions(positions, speed, binary_movement, t, settings):
    t_positions = np.linspace(0,len(positions)/settings['sampling frequency'],len(positions))
    plt.subplot(3, 1, 1)
    plt.plot(t_positions, positions)
    plt.subplot(3, 1, 2)
    plt.plot(t, speed)
    plt.subplot(3, 1, 3)
    plt.plot(t, binary_movement)
    return

def single_getspeed(file_path, n_samples):
    settings = set_settings()
    #directory = os.path.dirname(file_path)
    #file = file_path.split('\\')[-1]
    trace_A, trace_B = import_data(settings, file_path)
    bi_A = convert_binary(trace_A,settings['binary conversion threshold'])
    bi_B = convert_binary(trace_B, settings['binary conversion threshold'])
    positions = position(bi_A, bi_B, settings)
    speed, t = calc_speed(positions, settings, n_samples)
    binary_movement = np.array(move_nomove(speed, settings))
    extended_binary_movement = extend_movenomove(binary_movement, settings)
    total_samples = np.size(binary_movement)
    running_samples = np.count_nonzero(binary_movement == 1)
    percentage_running = running_samples/total_samples
    results = {'binary_movement':binary_movement,
               'extended_binary_movement':extended_binary_movement,
               'positions':positions, 'settings':settings, 'speed':speed,
               'speed':speed, 't':t,
               'percentage':percentage_running,'trace_A':trace_A , 'trace_B':trace_B }
    #save_session.save_variable(file_path, results)
    return results


def calc_speed2(positions, settings, n_samples):
    '''
    Takes the positions through time and calculate the change between every
    n_samples.
    ---------------
    Input:
        positions - list containing the position at every time point.
        n_samples - time points to jump when calculating the delta position.

    Output:
        speed - speed of the mouse at every time point in cm/s
        t - time distribution in seconds
    '''
    # factor = round(settings['sampling frequency']/settings['final_sampling_frequency']-0.5)
    # target_down = settings['final_sampling_frequency'] * settings['time rec']
    # positions_downsampled = positions[0:len(positions):factor]
    # speed = np.gradient(positions_downsampled)
    # speed = resample(speed, target_down)
    # speed_filt = savgol_filter(speed, 11, 1)
    # t = np.linspace(0,settings['time rec'],len(speed))

    if settings['sampling frequency'] % 2 > 0:
        window = settings['sampling frequency']
    else:
        window = settings['sampling frequency'] + 1
    # positions_smooth = savgol_filter(positions, window, 5)
    positions_smooth = positions

    t_positions = np.linspace(0, settings['time rec'], len(positions))
    speed_full = np.gradient(positions_smooth, t_positions)
    speed_downsampled = resample(speed_full, n_samples)
    # speed_smooth = savgol_filter(speed_downsampled, 31, 5)
    speed_smooth = speed_downsampled

    speed = np.where(speed_smooth < 0, 0, speed_smooth)
    t = np.linspace(0, settings['time rec'], len(speed))

    return speed, t
def single_getspeed2(file_path, n_samples):
    settings = set_settings()
    #directory = os.path.dirname(file_path)
    #file = file_path.split('\\')[-1]
    trace_A, trace_B = import_data(settings, file_path)
    bi_A = convert_binary(trace_A,settings['binary conversion threshold'])
    bi_B = convert_binary(trace_B, settings['binary conversion threshold'])
    positions = position(bi_A, bi_B, settings)
    speed, t = calc_speed2(positions, settings, n_samples)
    binary_movement = np.array(move_nomove(speed, settings))
    extended_binary_movement = extend_movenomove(binary_movement, settings)
    total_samples = np.size(binary_movement)
    running_samples = np.count_nonzero(binary_movement == 1)
    percentage_running = running_samples/total_samples
    results = {'binary_movement':binary_movement,
               'extended_binary_movement':extended_binary_movement,
               'positions':positions, 'settings':settings, 'speed':speed,
               'speed':speed, 't':t,
               'percentage':percentage_running,'trace_A':trace_A , 'trace_B':trace_B }
    #save_session.save_variable(file_path, results)
    return results
