import functions
import figure
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd
import os.path
from scipy.stats import zscore
def red_analyze(RG_cells, OnlyGreen_cells, RG_direction, dF, LMI,Z_mean_Running, Z_mean_paw,
             Z_mean_AS,Z_mean_rest,speed_corr,face_corr,pupil_corr,TIme, pupil, speed, motion,svg, DO_MOTION, Do_pupil):
    normal_df = functions.Normal_df(dF)
    z_scored_dF = zscore(dF, 1)
    #------------------------------ Choose Common Neurons -------------------------
    variables = [LMI, Z_mean_Running, Z_mean_paw, Z_mean_AS, Z_mean_rest, speed_corr, face_corr, normal_df,
                 z_scored_dF]
    RG_variables = {}
    for var_name, var in zip(
            ['LMI', 'Z_mean_Running', 'Z_mean_paw', 'Z_mean_AS', 'Z_mean_rest', 'speed_corr', 'face_corr',
             'normal_df', 'z_scored_dF'], variables):
        RG_variables[var_name] = functions.detect_valid_neurons(RG_cells, var)

    # ---------------------------- Choose Only Green Neurons -------------------------

    variables = [LMI, Z_mean_Running, Z_mean_paw, Z_mean_AS, Z_mean_rest, speed_corr, face_corr, normal_df,
                 z_scored_dF]
    OnlyGreen_variables = {}
    for var_name, var in zip(
            ['LMI', 'Z_mean_Running', 'Z_mean_paw', 'Z_mean_AS', 'Z_mean_rest', 'speed_corr', 'face_corr',
             'normal_df', 'z_scored_dF'], variables):
        OnlyGreen_variables[var_name] = functions.detect_valid_neurons(OnlyGreen_cells, var)

    #____________________________save RG Neurons data h5________________________________
    H5_dir_RG = os.path.join(RG_direction, "RGCell.h5")
    hf_RG = h5py.File(H5_dir_RG, 'w')
    RG_Ca_pre_group = hf_RG.create_group('Ca_data')
    RG_processd_group = hf_RG.create_group('processed_data')
    RG_full_trace_Ca_group = RG_Ca_pre_group.create_group('Full_trace')
    RG_Sub_processd_corr = RG_processd_group.create_group('correlation')

    RG_Sub_processd_corr.create_dataset('speed_corr',data =RG_variables['speed_corr'])
    RG_Sub_processd_corr.create_dataset('face_corr',data =RG_variables['face_corr'])
    RG_full_trace_Ca_group.create_dataset('dF',data = RG_variables['normal_df'])
    RG_full_trace_Ca_group.create_dataset('Z_scored_F', data = RG_variables['z_scored_dF'])

    RG_Zscored_Ca = RG_Ca_pre_group.create_group('Mean_Zscored')
    variable = [RG_variables['Z_mean_AS'],RG_variables['Z_mean_rest'],RG_variables['Z_mean_Running'],RG_variables['Z_mean_paw']]
    for name, value in zip(['Aroused_stationary','Rest','Running','paw_movement'], variable):
        RG_Zscored_Ca.create_dataset(name, data=value)
    hf_RG.close()

    #____________________________save Only Green data h5________________________________
    H5_dir_OnlyGreen = os.path.join(RG_direction, "Only_Green.h5")
    hf_OnlyGreen = h5py.File(H5_dir_OnlyGreen, 'w')
    OnlyGreen_Ca_pre_group = hf_OnlyGreen.create_group('Ca_data')
    OnlyGreen_processd_group = hf_OnlyGreen.create_group('processed_data')
    OnlyGreen_full_trace_Ca_group = OnlyGreen_Ca_pre_group.create_group('Full_trace')
    OnlyGreen_Sub_processd_corr = OnlyGreen_processd_group.create_group('correlation')

    OnlyGreen_Sub_processd_corr.create_dataset('speed_corr',data =RG_variables['speed_corr'])
    OnlyGreen_Sub_processd_corr.create_dataset('face_corr',data =RG_variables['face_corr'])
    OnlyGreen_full_trace_Ca_group.create_dataset('dF',data = RG_variables['normal_df'])
    OnlyGreen_full_trace_Ca_group.create_dataset('Z_scored_F', data = RG_variables['z_scored_dF'])

    OnlyGreen_Zscored_Ca = OnlyGreen_Ca_pre_group.create_group('Mean_Zscored')
    variable = [RG_variables['Z_mean_AS'],RG_variables['Z_mean_rest'],RG_variables['Z_mean_Running'],RG_variables['Z_mean_paw']]
    for name, value in zip(['Aroused_stationary','Rest','Running','paw_movement'], variable):
        OnlyGreen_Zscored_Ca.create_dataset(name, data=value)
    hf_OnlyGreen.close()

    #__________________________________ save RG data in the excel _______________________________
    Num_RG_cells = len(RG_cells)
    t = "ROI"
    columns_RG_cells = [(f'{t}{i}') for i in RG_cells]
    sp = np.copy(speed)
    sp[sp == 0] = 'nan'
    speed_mean_RG = Num_RG_cells * [np.nanmean(sp)]
    df_RG = pd.DataFrame(
        [RG_variables['LMI'], speed_mean_RG, RG_variables['Z_mean_Running'],RG_variables['Z_mean_paw'],
         RG_variables['Z_mean_rest'], RG_variables['speed_corr'], RG_variables['face_corr']],
        index=['Run MI', 'Mean speed','Mean z run', 'Mean z paw movement', 'Mean Z rest',
               'Speed corr', 'Face corr'], columns=columns_RG_cells).T
    functions.save_exel('RG_variable.xlsx', RG_direction, df_RG)

    #__________________________________ save OnlyGreen data in the excel _______________________________
    Num_OnlyGreen_cells = len(OnlyGreen_cells)
    t = "ROI"
    columns_OnlyGreen_cells = [(f'{t}{i}') for i in OnlyGreen_cells]
    sp = np.copy(speed)
    sp[sp == 0] = 'nan'
    speed_mean_OnlyGreen = Num_OnlyGreen_cells * [np.nanmean(sp)]
    df_OnlyGreen = pd.DataFrame(
        [OnlyGreen_variables['LMI'], speed_mean_OnlyGreen, OnlyGreen_variables['Z_mean_Running'],OnlyGreen_variables['Z_mean_paw'],
         OnlyGreen_variables['Z_mean_rest'], OnlyGreen_variables['speed_corr'], OnlyGreen_variables['face_corr']],
        index=['Run MI', 'Mean speed','Mean z run', 'Mean z paw movement', 'Mean Z rest',
               'Speed corr', 'Face corr'], columns=columns_OnlyGreen_cells).T
    functions.save_exel('OnlyGreen_variable.xlsx', RG_direction, df_OnlyGreen)

    # __________________________________________ plot Red-Green Data ___________________________________
    Label1 = 'Only Green'
    Label2 = 'Red-Green'
    figure.box_plot(OnlyGreen_variables['Z_mean_Running'], OnlyGreen_variables['Z_mean_rest'], RG_variables['Z_mean_Running'],
                    RG_variables['Z_mean_rest'], 'Green run', 'Green rest', 'Red/Green Srun',
                    "Red/Green rest", "Red_Green_activity", 'z-scored dF/F', '', RG_direction,svg)

    NUM_cell = np.arange(0, len(dF))
    figure.scatter_plot(NUM_cell,speed_corr, RG_cells, RG_direction, 'Speed & F Correlation','Neuron', 'correlation',
                 Label1, Label2,svg, color1='green', color2 = 'red')

    NUM_LMI = np.arange(0, len(LMI))
    figure.scatter_plot(NUM_LMI,LMI, RG_cells, RG_direction,'LMI','Neuron', 'LMI',
                        Label1,Label2,svg, color1='green', color2 = 'red')

    figure.pie_plot('Red_Green percentage', RG_direction, Label1, Label2,  len(OnlyGreen_cells), len(RG_cells), 'darkseagreen','tomato')

    if DO_MOTION:
        figure.scatter_plot(NUM_cell, face_corr, RG_cells, RG_direction, 'facemotion & F Correlation', 'Neuron',
                            'correlation',
                            Label1, Label2, svg, color1='green', color2='red')
    if Do_pupil:
        figure.scatter_plot(NUM_cell, pupil_corr, RG_cells, RG_direction, 'pupil & F Correlation', 'Neuron',
                            'correlation',
                            Label1, Label2, svg, color1='green', color2='red')
