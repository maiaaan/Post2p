import matplotlib.pyplot as plt
import functions
import numpy as np
import os
from scipy import signal
import pandas as pd
import plotly.express as px
import scipy.stats
from scipy.ndimage import gaussian_filter1d
import seaborn as sns

def Time_pie(Aroused_stat_T, Running_T, Rest_T, NABMA_T, total_duration, save_direction, svg):
    not_used = total_duration - (Aroused_stat_T + Running_T + Rest_T + NABMA_T)
    labels = ["Running/high whisking", "Stationnary/high whisking", "Stationnary/low whisking", "Resting/no whisking",  "Undefined"]
    sizes = [Running_T , Aroused_stat_T, NABMA_T, Rest_T, not_used]
    colors = ['crimson', 'darkorange', 'gold', 'c', 'gray']
    explode = (0.05, 0.05, 0.05, 0.05, 0.05)

    fig = plt.figure(figsize=(10, 5))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%.1f%%', labeldistance=None, pctdistance=1.2,
            textprops={'fontsize': 12, 'fontweight':'bold'})
    plt.legend(labels=labels, loc='center left', bbox_to_anchor=(0.8, 0.3), frameon=False)
    plt.axis('equal')
    plt.title("Motion states duration")
    if svg == True:
        svg_name = "Motion_time.svg"
        save_direction_svg = os.path.join(save_direction, svg_name)
        fig.savefig(save_direction_svg)
    functions.save_fig("Motion_time.png",save_direction,fig)

def general_figure(TIme, normal_pupil, speed, normal_motion, dF, save_direction_figure, label):
    fig = plt.figure(figsize=(36, 20))
    y2 = dF[-1]
    mean_dF = np.mean(dF, 0)
    gs = fig.add_gridspec(26, 45)

    # Speed
    ax0 = fig.add_subplot(gs[0:2, :43])
    ax0.set_title('Running Speed', fontsize=25, y=1.)
    ax0.plot(TIme, speed, linewidth=4)
    ax0.set_xticks([])
    ax0.margins(x=0)
    ax0.set_ylabel('cm/s', fontsize=20)
    ax0.set_facecolor("white")
    plt.yticks(fontsize=20)

    # Pupil
    ax00 = fig.add_subplot(gs[3:5, :43])
    ax00.set_title('Pupil z-score', fontsize=25, y=1.)
    ax00.plot(TIme, normal_pupil, linewidth=4)
    ax00.set_xticks([])
    ax00.margins(x=0)
    ax00.set_facecolor("white")
    plt.yticks(fontsize=20)

    # Facemotion
    ax3 = fig.add_subplot(gs[6:8, :43])
    ax3.set_title('Face Motion', fontsize=25, y=1.0, horizontalalignment='center')
    ax3.plot(TIme, normal_motion, linewidth=4)
    ax3.set_xticks([])
    ax3.margins(x=0)
    ax3.set_facecolor("white")
    plt.yticks(fontsize=20)

    # mean dF/F
    ax2 = fig.add_subplot(gs[9:11, :43])
    ax2.set_title('Normalized dF/F mean', fontsize=25, y=1.0, horizontalalignment='center')
    ax2.plot(TIme, mean_dF, linewidth=4)
    ax2.set_xticks([])
    ax2.margins(x=0)
    ax2.set_facecolor("white")
    plt.yticks(fontsize=20)

    # Neuronal activity map
    ax1 = fig.add_subplot(gs[12:23, :43])
    ax1.set_title('Normalized neuronal activity (sorted by speed correlation)', fontsize=25, y=1.0, horizontalalignment='center')
    ax1.pcolormesh(dF, cmap='viridis')
    ax1.set_xticks([])
    ax1.margins(x=0)
    ax1.set_ylabel('Neuron', fontsize=20)
    plt.yticks(fontsize=20)

    m = ax1.pcolormesh(dF, cmap='viridis')
    ax2 = fig.add_subplot(gs[12:23, 44:45])
    fig.colorbar(m, cax=ax2)
    plt.yticks(fontsize=20)

    # dF/F
    ax4 = fig.add_subplot(gs[24:26, :43])
    ax4.set_title("Most speed correlated neuron's normalized dF/F", fontsize=25, y=1.0, horizontalalignment='center')
    ax4.plot(TIme, y2)
    ax4.set_xlabel('Time (s)', fontsize=20)
    ax4.margins(x=0)
    ax4.set_facecolor("white")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    functions.save_fig(label, save_direction_figure, fig)

def box_plot(data1, data2, data3, data4, label1, label2, label3, label4, fig_label, y_label, Title, save_direction, svg):
    fig, ax = plt.subplots(figsize=(7, 10))
    boxplot1 = ax.boxplot(data1, positions=[1], widths=0.6, patch_artist=True)
    boxplot2 = ax.boxplot(data2, positions=[2], widths=0.6, patch_artist=True)
    boxplot3 = ax.boxplot(data3, positions=[3], widths=0.6, patch_artist=True)
    boxplot4 = ax.boxplot(data4, positions=[4], widths=0.6, patch_artist=True)
    for box in boxplot1['boxes']:
        box.set(color='lightsteelblue', facecolor='lightsteelblue')
    for box in boxplot2['boxes']:
        box.set(color='slategray', facecolor='slategray')
    for box in boxplot3['boxes']:
        box.set(color='rosybrown', facecolor='rosybrown')
    for box in boxplot4['boxes']:
        box.set(color='darkseagreen', facecolor='darkseagreen')
    # Add labels and title
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([label1,label2,label3,label4])
    ax.set_ylabel(y_label)
    ax.set_title(Title)
    if svg == True:
        svg_name = "activity_state.svg"
        save_direction_svg = os.path.join(save_direction, svg_name)
        fig.savefig(save_direction_svg)
    functions.save_fig(fig_label,save_direction,fig)

def simple_plot(time, y,save_direction, label ):
    fig = plt.figure(figsize=(7, 2))
    plt.plot(time, y)
    functions.save_fig(label, save_direction, fig)

def scatter_plot(NUM, y, chosen_neurons, save_direction_figure, label:str, xlabel:str, ylabel:str,
                 condition_label1:str, condition_label2:str, svg, color1='lightgreen', color2='red'):
    sns.set_theme()
    NUM_color = np.arange(0, len(NUM))
    fig_label = label + ".png"
    fig = plt.figure(figsize=(8, 6))
    colors = np.where(np.in1d(NUM_color, chosen_neurons), color1, color2)
    #plt.scatter(np.zeros(len(NUM)), y, s=100, c=colors)
    plt.scatter(NUM, y, s=100, c=colors)
    plt.xlabel(xlabel, fontsize=15, labelpad=10)
    plt.ylabel(ylabel, fontsize=15, labelpad=10)
    plt.title(label=label, y=1.05, fontsize=18)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=condition_label1, markerfacecolor=color1, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label=condition_label2, markerfacecolor=color2, markersize=10)]
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(handles=legend_elements)
    if svg == True:
        svg_name = label + ".svg"
        save_direction_svg = os.path.join(save_direction_figure, svg_name)
        fig.savefig(save_direction_svg)
    functions.save_fig(fig_label, save_direction_figure, fig)

def GUIimage(time, trace, figure_path, title):
    fig, ax = plt.subplots(figsize=(9,1))
    ax.plot(time, trace)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.margins(x=0)
    fig_path = os.path.join(figure_path,title)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    fig.savefig(fig_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)

def pie_plot(fig_title, save_direction, label1, label2, data_size1, data_size2):
    title = fig_title + ".png"
    fig = plt.figure(figsize=(7, 5))
    labels = [label1, label2]
    sizes = [data_size1, data_size2]
    colors = ['mediumpurple', 'plum']
    plt.pie(sizes, labels=[f'{label} ({size})' for label, size in zip(labels, sizes)], colors=colors,
            autopct='%1.1f%%')
    plt.title(fig_title)
    functions.save_fig(title, save_direction, fig)

def pupil_state(categories, values,save_direction_figure,ylabel,title,svg):
    save_name = title + ".png"
    fig_p, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(categories, values, color='yellowgreen', s=120)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if svg == True:
        svg_name = title + ".svg"
        save_direction_svg = os.path.join(save_direction_figure, svg_name)
        fig_p.savefig(save_direction_svg,format = 'svg')
    functions.save_fig(save_name, save_direction_figure, fig_p)

def double_trace_plot(TIme, Mean__dF, speed, save_direction_figure, x_label, y_label1, y_label2, title, svg):
    custom_params = {"axes.spines.top": False}
    sns.set_theme(style="ticks", palette=None, rc=custom_params)
    fig, ax = plt.subplots(figsize=(17, 10))
    save_title = title + ".png"
    ax2 = ax.twinx()
    ax.plot(TIme, Mean__dF, color="red", alpha=0.5)
    ax2.plot(TIme, speed, color="green", alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label1, color="red")
    ax2.set_ylabel(y_label2, color="green")
    plt.tight_layout()
    plt.title(title)
    if svg == True:
        svg_name = title + ".svg"
        save_direction_svg = os.path.join(save_direction_figure, svg_name)
        fig.savefig(save_direction_svg ,format = 'svg')
    functions.save_fig(save_title, save_direction_figure, fig)

def plot_running(TIme, speed, Real_Time_Running, filtered_speed, threshold, save_direction_figure, ylabel, xlabel, title, svg):
    sns.set_theme()
    max_val = max(speed)
    y = np.arange(0, max_val, 0.1)
    fig25, ax = plt.subplots(figsize=(17, 6))
    plt.plot(TIme, speed, label='speed')
    ax.plot(TIme, filtered_speed, color='paleturquoise', label='filtered speed')
    ax.margins(x=0.01)
    plt.ylabel(ylabel, labelpad=10)
    plt.xlabel(xlabel, labelpad=10)
    ax.set_facecolor("white")
    ax.axhline(threshold, color='orange', lw=2, label='Speed threshold')
    for i in Real_Time_Running:
        plt.fill_betweenx(y, i[0], i[-1], color='crimson', alpha=.5, label='running state window')
    plt.legend()
    if svg == True:
        svg_name = "running.svg"
        save_direction_svg = os.path.join(save_direction_figure, svg_name)
        fig25.savefig(save_direction_svg,format = 'svg')
    functions.save_fig(title, save_direction_figure, fig25)

def power_plot(Mean__dF, Fs, save_direction_figure):
    fig = plt.figure(figsize=(18, 8))
    f, t, Sxx = signal.spectrogram(Mean__dF, Fs)
    c = plt.pcolormesh(t, f, np.log(Sxx), shading='gouraud', cmap='viridis')
    plt.colorbar(c)
    plt.ylabel('Frequency (Hz)', fontsize=19)
    plt.xlabel('Time (s)', fontsize=19)
    plt.title('Power spectrum of the mean dF/F (in log scale)', fontsize=19)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    functions.save_fig("Power_Spectrum.png", save_direction_figure, fig)

def histo_valid(valid_neurons, save_direction_figure, pearson_corr, title, label):
    Valid_NotValid = ["YES" if i in valid_neurons else "NO" for i in range(len(pearson_corr))]
    pearson_corr_histo_data = {"Correlation": pearson_corr, label: Valid_NotValid}
    color_map = {"YES": "palevioletred", "NO": "silver"}
    pearson_corr_histo_data = pd.DataFrame(pearson_corr_histo_data)
    fig = px.histogram(pearson_corr_histo_data, x="Correlation", color=label, marginal="violin",
                       hover_data=pearson_corr_histo_data.columns, nbins=15, barmode='relative', opacity=0.5,
                       color_discrete_map=color_map)
    fig.update_layout(plot_bgcolor='white')
    fig.update_traces(marker_line_color='white', marker_line_width=2)
    save_direction_histo = os.path.join(save_direction_figure, title+".png")
    fig.write_image(save_direction_histo)

def fit_plot(xcorr, ycorr, save_path, title, ylabel, xlabel):
    p = np.polyfit(xcorr, ycorr, 1)
    fig = plt.figure(figsize=(11, 6))
    plt.title(label=title, y=1.05, fontsize=18)
    plt.plot(np.unique(xcorr), np.poly1d(p)(np.unique(xcorr)), linewidth=3,
             color='salmon')
    plt.scatter(xcorr, ycorr, s=80, color='gray')
    plt.ylabel(ylabel, fontsize=15, labelpad=10)
    plt.xlabel(xlabel, fontsize=15, labelpad=10)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.annotate(f'slope = {p[0]}', xy=(0.01, 0.98), xycoords='axes fraction', fontsize=9, va='top', ha='left')
    functions.save_fig(title +".png", save_path, fig)

def HistoPlot(X, xlabel, filename, save_direction1):
    fig14 = plt.figure(figsize=(16, 7))
    plt.hist(X, weights=(np.ones(len(X)) / len(X)) * 100, edgecolor="black", bins=20)
    plt.ylabel('Count (in %)', size=20, labelpad=10)
    plt.xlabel(xlabel, size=20, labelpad=10)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    median_absolut = np.median(X)
    plt.axvline(x=median_absolut, color='black', label='median', linewidth=4)
    plt.legend(bbox_to_anchor=(1.0, 1), prop={'size': 14}, )
    file_name14 = filename
    save_direction14 = os.path.join(save_direction1, file_name14)
    fig14.savefig(save_direction14)

def simple_plot_SVG(time, trace, figure_path, title):
    fig, ax = plt.subplots(figsize=(10,2))
    ax.plot(time, trace)
    ax.margins(x=0)
    fig_path2 = os.path.join(figure_path,title)
    fig.savefig(fig_path2, format = 'svg')
    plt.close(fig)

def colormap_perm_test(time, dF, var, valid_neurons, label:str=None, save_path=None):
    dF = functions.detect_valid_neurons(valid_neurons, dF)
    var_corr = [scipy.stats.pearsonr(var, x)[0] for x in dF]
    dF_corr_sorted = [x for _, x in sorted(zip(var_corr, dF))]
    norm_dF_corr_sorted = functions.Normal_df(dF_corr_sorted)
    mean_dF = np.mean(dF, 0)
    
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(2, 2, height_ratios=[2,3], width_ratios=[22,1], hspace=0.3, wspace=0.05)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax1b = fig.add_subplot(gs[1, 1])
    
    ax0.plot(time, gaussian_filter1d(mean_dF, 10), label=r'$\Delta$F/F')
    ax0.set_xticks([])
    ax0.margins(x=0)
    ax0.set_facecolor("white")
    ax0.set_title(r'Normalized $\Delta$F/F mean on neurons which passed the ' + label + ' permutation test')

    c = ax1.pcolormesh(norm_dF_corr_sorted, cmap='viridis')
    ax1.set_ylabel('Sorted neurons')
    ax1.set_xlabel('Frame')
    ax1.margins(x=0)
    ax1.set_facecolor("white")
    ax1.set_title('Normalized neuronal activity (sorted by ' + label + ' correlation)')
    fig.colorbar(c, cax=ax1b)

    if save_path != None : 
        functions.save_fig(label +"_permT_neural_activity.png", save_path, fig)