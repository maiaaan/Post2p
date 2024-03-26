from scipy.stats import skew
import matplotlib.pyplot as plt
import numpy as np
import functions
import h5py
import pandas as pd
import figure
import os.path
from scipy.stats import zscore

def skewness(dF, threshold:float, save_direction_skew, ROIs_group, LMI,Z_mean_Running, Z_mean_paw,
             Z_mean_AS,Z_mean_rest,speed_corr,face_corr,TIme, pupil, speed, motion,svg):
    """ March 2024 - Bacci Lab - faezeh.rabbani97@gmail.com

    ...........................................................................

    This function compute the skewness of each ROI and separate ROIs base on the given threshold to
    skew and not skew ROIs.
    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -
    dF                       dF/F
    threshold                skewness threshold
    save_direction_skew      direction to save outpt
    ROIs_group               a group created within an HDF5 file to save index of ROIs
    LMI                      Locomotion Modulation index
    Z_mean_Running           mean of z scored dF in the running period
    Z_mean_paw               mean of z scored dF in the paw movement period
    Z_mean_AS                mean of z scored dF in the Aroused stationary period
    Z_mean_rest              mean of z scored dF in the rest period
    speed_corr               Speed and df correlation
    face_corr                Facemotion and dF correlation
    pupil                    pupil trace
    speed                    speed trace
    motion                   motion tace
    TIme                     timestamps extracted from xml file

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - -
    out put of function are excel and Hdf5 file all saved in skewness folder. Plus some
     figures and result of statistical test between skewed and not skewed and skewed and all cells
    ...........................................................................
    """
    normal_df = functions.Normal_df(dF)
    z_scored_dF = zscore(dF, 1)
    skewness = skew(dF, axis=1)
    high_skew = np.where(skewness > threshold)[0]
    low_skew = np.where(skewness <= threshold)[0]
    skewness_high_skew = [skewness[i] for i in high_skew]
    skewness_low_skew = [skewness[i] for i in low_skew]
    mean_high_skew = np.mean(skewness_high_skew)
    mean_low_skew = np.mean(skewness_low_skew)


    #plot skewness Histogram
    for i in high_skew:
        fig2 = plt.figure(figsize=(8, 4))
        plt.hist(dF[i], bins=30, density=True, alpha=0.7, color='blue', label=str(skewness[i]))
        plt.legend()
        functions.save_fig(f"mean_stage_activity{i}.png", save_direction_skew, fig2)
    ROIs_group.create_dataset('High_skew', data = high_skew)
    #------------------choose neurons with high skewness -------------
    variables = [LMI, Z_mean_Running, Z_mean_paw, Z_mean_AS, Z_mean_rest, speed_corr, face_corr, normal_df,
                 z_scored_dF]
    skew_variables = {}
    for var_name, var in zip(
            ['LMI', 'Z_mean_Running', 'Z_mean_paw', 'Z_mean_AS', 'Z_mean_rest', 'speed_corr', 'face_corr',
             'normal_df', 'z_scored_dF'], variables):
        skew_variables[var_name] = functions.detect_valid_neurons(high_skew, var)
    # ------------------choose neurons with low skewness -------------
    variable2 = [Z_mean_Running,Z_mean_paw,Z_mean_AS,Z_mean_rest]
    not_skew_variable = {}
    for var_name, var in zip(
            ['Z_mean_Running','Z_mean_paw', 'Z_mean_AS', 'Z_mean_rest'], variable2):
        not_skew_variable[var_name] = functions.detect_valid_neurons(low_skew, var)
    #_______________________save skew data h5_________________
    H5_dir_skew =os.path.join(save_direction_skew, "skewdata.h5")
    hf_skew = h5py.File(H5_dir_skew, 'w')
    skew_Ca_pre_group = hf_skew.create_group('Ca_data')
    skew_processd_group = hf_skew.create_group('processed_data')
    skew_full_trace_Ca_group = skew_Ca_pre_group.create_group('Full_trace')
    skew_Sub_processd_corr = skew_processd_group.create_group('correlation')

    skew_Sub_processd_corr.create_dataset('speed_corr',data =skew_variables['speed_corr'])
    skew_Sub_processd_corr.create_dataset('face_corr',data =skew_variables['face_corr'])
    skew_full_trace_Ca_group.create_dataset('dF',data = skew_variables['normal_df'])
    skew_full_trace_Ca_group.create_dataset('Z_scored_F', data = skew_variables['z_scored_dF'])

    skew_Zscored_Ca = skew_Ca_pre_group.create_group('Mean_Zscored')
    variable = [skew_variables['Z_mean_AS'],skew_variables['Z_mean_rest'],skew_variables['Z_mean_Running'],skew_variables['Z_mean_paw']]
    for name, value in zip(['Aroused_stationary','Rest','Running','paw_movement'], variable):
        skew_Zscored_Ca.create_dataset(name, data=value)
    hf_skew.close()
    #_______________________save not skew data h5_________________
    H5_dir_noskew =os.path.join(save_direction_skew, "noskewdata.h5")
    hf_noskew = h5py.File(H5_dir_noskew, 'w')
    Noskew_Ca_pre_group = hf_noskew.create_group('Ca_data')
    Noskew_Zscored_Ca = Noskew_Ca_pre_group.create_group('Zscored')
    Noskew_Zscored_Ca.create_dataset('Aroused_stationary',data = not_skew_variable['Z_mean_AS'])
    Noskew_Zscored_Ca.create_dataset('Rest',data = not_skew_variable['Z_mean_rest'])
    Noskew_Zscored_Ca.create_dataset('Running', data = not_skew_variable['Z_mean_Running'])
    Noskew_Zscored_Ca.create_dataset('paw_movement',data = not_skew_variable['Z_mean_paw'])
    hf_noskew.close()
    #_______________________save skew data in the excel__________________
    Num_highSkew = len(high_skew)
    t = "ROI"
    columns_skew = [(f'{t}{i}') for i in high_skew]
    sp = np.copy(speed)
    sp[sp == 0] = 'nan'
    speed_mean_skew = Num_highSkew* [np.nanmean(sp)]
    df_skew = pd.DataFrame(
        [skew_variables['LMI'], speed_mean_skew, skew_variables['Z_mean_Running'],skew_variables['Z_mean_paw'],
         skew_variables['Z_mean_rest'], skew_variables['speed_corr'], skew_variables['face_corr'], high_skew],
        index=['Run MI', 'Mean speed','Mean z run', 'Mean z paw movement', 'Mean Z rest',
               'Speed corr', 'Face corr', 'Skewness'], columns=columns_skew).T
    functions.save_exel('skew_variable.xlsx', save_direction_skew, df_skew)
    #_______________________plot skew Data __________________________
    Label1 = 'Skew < ' + str(threshold)
    Label2 = 'Skew > ' + str(threshold)
    figure.box_plot(skew_variables['Z_mean_Running'], skew_variables['Z_mean_rest'], not_skew_variable['Z_mean_Running'],
                    not_skew_variable['Z_mean_rest'], 'Srun', 'Srest', 'NSrun',
                    "NSrest", "skew_notSkew_activity", 'z-scored dF/F', '', save_direction_skew,svg)

    NUM_cell =  np.arange(0, len(dF))
    figure.scatter_plot(NUM_cell,speed_corr, high_skew,save_direction_skew, 'Speed & F Correlation','Neuron', 'correlation',
                 Label1, Label2,svg, color1='red', color2 = 'mediumpurple')

    NUM_LMI = np.arange(0, len(LMI))
    figure.scatter_plot(NUM_LMI,speed_corr, high_skew,save_direction_skew, 'LMI','Neuron', 'LMI',
                        Label1,Label2,svg, color1='red', color2 = 'mediumpurple')

    not_skew = len(dF)- Num_highSkew
    figure.pie_plot('high skewed percentage', save_direction_skew, Label1,Label2 , Num_highSkew, not_skew)
    if len(skew_variables['normal_df']) > 0:
        figure.general_figure(TIme, pupil, speed, motion, skew_variables['normal_df'], save_direction_skew, "General_skew.png")
        figure.box_plot(skew_variables['Z_mean_rest'], skew_variables['Z_mean_paw'], skew_variables['Z_mean_Running'], skew_variables['Z_mean_AS'], 'Rest', 'paw Movement','AS',
        'Running', "skew_stage_mean", 'mean', 'z-scored dF/F', save_direction_skew,svg)
    return mean_high_skew, mean_low_skew


