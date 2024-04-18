import matplotlib.pyplot as plt
import functions
import numpy as np
import os
from scipy import signal
import pandas as pd
import plotly.express as px

def Time_pie(whisking_T, Running_T, Rest_T, Only_paw, general_timing,save_direction,svg):
    not_used = general_timing -(whisking_T + Running_T + Rest_T + Only_paw)
    labels = ["Resting/no whisking","Resting/whisking", "Running/whisking","Paw movment/whisking",  "not analyzed"]
    fig = plt.figure(figsize=(10, 5))
    sizes = [Rest_T,whisking_T , Running_T , Only_paw, not_used]
    colors = ['lightpink','plum','steelblue','chocolate','lightslategrey']
    explode = (0.05, 0.05, 0.05, 0.05, 0.05)
    plt.pie(sizes,labels=labels,
            colors=colors, autopct='%1.1f%%', labeldistance=None, explode = explode)
    plt.legend(labels = labels, loc='center left', bbox_to_anchor=(0.8, 0.3), frameon=False)
    plt.axis('equal')
    plt.title("Motion timing")
    if svg == True:
        svg_name = "Motion_time.svg"
        save_direction_svg = os.path.join(save_direction, svg_name)
        fig.savefig(save_direction_svg)
    functions.save_fig("Motion_time.png",save_direction,fig)

def general_figure(TIme, normal_pupil,speed, normal_motion,dF, save_direction_figure,label):
    fig = plt.figure(figsize=(36, 20))
    y2 = dF[-1]
    gs = fig.add_gridspec(26, 45)
    ax00 = fig.add_subplot(gs[3:5, :43])
    ax00.set_title('Pupil', fontsize=25, y=1.)
    ax00.plot(TIme, normal_pupil, linewidth=4)
    ax00.set_xticks([])
    ax00.set_yticks([])
    ax00.margins(x=0)
    ax00.set_facecolor("white")
    plt.yticks(fontsize=20)
    ax0 = fig.add_subplot(gs[0:2, :43])
    ax0.set_title('Running Speed', fontsize=25, y=1)
    ax0.plot(TIme, speed, linewidth=4)
    ax0.set_xticks([])
    plt.yticks(fontsize=20)
    ax0.margins(x=0)
    ax0.set_ylabel('cm/S', fontsize=20)
    ax0.set_facecolor("white")
    # Plot1
    ax1 = fig.add_subplot(gs[10:22, :43])
    ax1.set_title('color code', fontsize=25, y=1.0, horizontalalignment='center')
    ax1.pcolormesh(dF)
    ax1.set_xticks([])
    ax1.margins(x=0)
    ax1.set_ylabel('Neuron', fontsize=25)
    plt.yticks(fontsize=20)
    # Plot2
    m = ax1.pcolormesh(dF)
    ax2 = fig.add_subplot(gs[10:22, 44:45])
    fig.colorbar(m, cax=ax2)
    plt.yticks(fontsize=20)
    # Plot3
    ax3 = fig.add_subplot(gs[7:9, :43])
    ax3.set_title('face motion', fontsize=25, y=1.0, horizontalalignment='center')
    ax3.plot(TIme, normal_motion)
    ax3.margins(x=0)
    ax3.set_facecolor("white")
    ax3.set_yticks([])
    ax3.set_xticks([])
    plt.yticks(fontsize=20)
    # Plot4
    ax4 = fig.add_subplot(gs[23:25, :43])
    ax4.set_title('dF/F', fontsize=25, y=1.0, horizontalalignment='center')
    ax4.plot(TIme, y2)
    ax4.margins(x=0)
    ax4.set_facecolor("white")
    plt.yticks(fontsize=20)
    functions.save_fig(label, save_direction_figure, fig)

def box_plot(data1, data2, data3, data4, label1, label2, label3, label4, fig_label, y_labe, Title, save_direction,svg):
    fig, ax = plt.subplots(figsize=(4, 5))
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
    ax.set_ylabel(y_labe)
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

def scatter_plot(NUM,y, chosen_neurons,save_direction_figure, label:str, xlabel:str, ylabel:str,
                 condition_label1:str, condition_label2:str,svg, color1='red', color2='lightgreen'):
    NUM_color = np.arange(0, len(NUM))
    fig_label = label + ".png"
    fig = plt.figure(figsize=(14, 6))
    colors = np.where(np.in1d(NUM_color, chosen_neurons), color1, color2)
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

def pie_plot(fig_title, save_direction, label1, label2, data_size1, data_size2, color1 = 'mediumpurple', color2 = 'plum'):
    title = fig_title + ".png"
    fig = plt.figure(figsize=(7, 5))
    labels = [label1, label2]
    sizes = [data_size1, data_size2]
    colors = [color1, color2]
    plt.pie(sizes, labels=[f'{label} ({size})' for label, size in zip(labels, sizes)], colors=colors,
            autopct='%1.1f%%')
    plt.title(fig_title)
    functions.save_fig(title, save_direction, fig)
def pupil_state(categories, values,save_direction_figure,ylabel,title,svg):
    save_name = title + ".png"
    fig_p, ax = plt.subplots(figsize=(5, 6))
    ax.scatter(categories, values, color='yellowgreen', s=120)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if svg == True:
        svg_name = title + ".svg"
        save_direction_svg = os.path.join(save_direction_figure, svg_name)
        fig_p.savefig(save_direction_svg,format = 'svg')
    functions.save_fig(save_name, save_direction_figure, fig_p)

def double_trace_plot(TIme,Mean__dF,speed, save_direction_figure,x_label,y_label1, y_label2,title,svg):
    fig, ax = plt.subplots(figsize=(10, 7))
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
def plot_running(TIme, speed, Real_Time_Running,filtered_speed,save_direction_figure,ylabel,xlabel,title,svg):
    max_val = max(filtered_speed)
    y = np.arange(0, max_val, 0.1)
    fig25, ax = plt.subplots(figsize=(17, 6))
    plt.plot(TIme, speed)
    ax.plot(TIme, filtered_speed, color='gold')
    ax.margins(x=0.01)
    threshold = 0.5
    plt.ylabel(ylabel, labelpad=10)
    plt.xlabel(xlabel, labelpad=10)
    ax.set_facecolor("white")
    ax.axhline(threshold, color='orange', lw=2)
    for i in Real_Time_Running:
        plt.fill_betweenx(y, i[0], i[-1], color='lightpink', alpha=.5, label='runing>2(S)')
    if svg == True:
        svg_name = "running.svg"
        save_direction_svg = os.path.join(save_direction_figure, svg_name)
        fig25.savefig(save_direction_svg,format = 'svg')
    functions.save_fig(title, save_direction_figure, fig25)

def power_plot(Mean__dF,Fs,save_direction_figure):
    fig5 = plt.figure(figsize=(17, 8))
    f, t, Sxx = signal.spectrogram(Mean__dF, Fs)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]', fontsize=19)
    plt.xlabel('Time [sec]', fontsize=19)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    functions.save_fig("Power_Spectrum.png", save_direction_figure, fig5)
def histo_valid(valid_neurons_speed,save_direction_figure,speed_corr,label):
    Valid_NotValid_speed = ["yes" if i in valid_neurons_speed else "NO" for i in range(len(speed_corr))]
    speed_corr_histo_data = {"Correlation": speed_corr, label: Valid_NotValid_speed}
    color_map = {"NO": "silver", "yes": "palevioletred"}
    Speed_corr_histo_data = pd.DataFrame(speed_corr_histo_data)
    fig = px.histogram(Speed_corr_histo_data, x="Correlation", color=label, marginal="violin",
                       hover_data=Speed_corr_histo_data.columns, nbins=15, barmode='relative', opacity=0.5,
                       color_discrete_map=color_map)
    fig.update_layout(plot_bgcolor='white')
    fig.update_traces(marker_line_color='white', marker_line_width=2)
    save_direction_histo = os.path.join(save_direction_figure, label+".png")
    fig.write_image(save_direction_histo)

def fit_plot(speed_corr, face_corr,save_direction_figure,title, ylabel, xlabel):
    fig = plt.figure(figsize=(11, 6))
    plt.title(label=title, y=1.05, fontsize=18)
    plt.plot(np.unique(speed_corr), np.poly1d(np.polyfit(speed_corr, face_corr, 1))(np.unique(speed_corr)), linewidth=3,
             color='salmon')
    plt.scatter(speed_corr, face_corr, s=80, color='gray')
    plt.ylabel(ylabel, fontsize=15, labelpad=10)
    plt.xlabel(xlabel, fontsize=15, labelpad=10)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    functions.save_fig(title +".png", save_direction_figure, fig)

def HistoPlot(X,xLabel,save_direction1):
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
    fig14.savefig(save_direction14)

def simple_plot_SVG(time, trace, figure_path, title):
    fig, ax = plt.subplots(figsize=(10,2))
    ax.plot(time, trace)
    ax.margins(x=0)
    fig_path2 = os.path.join(figure_path,title)
    fig.savefig(fig_path2, format = 'svg')
    plt.close(fig)