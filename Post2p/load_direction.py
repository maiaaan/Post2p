import easygui
import glob
from pathlib import Path
import os
import h5py
import numpy as np
import functions
import xml_parser
def get_directory():
    Base_path = easygui.diropenbox(title='Select folder containing data')
    print("Base path =", Base_path)
    tail = Path(Base_path).name
    tail2 = f"Results2.7_{tail}"
    save_direction1 = functions.make_dir(Base_path, tail2)
    save_direction_figure = functions.make_dir(save_direction1, "Figures")
    save_data = functions.make_dir(save_direction1, "data")
    H5_dir = os.path.join(save_data, "AllData.h5")
    hf = h5py.File(H5_dir, 'w')
    save_direction_permutation = functions.make_dir(save_direction1, "permutation_test")
    save_direction0_lag = functions.make_dir(save_direction1, "lag")
    save_direction_skew = functions.make_dir(save_direction1, "skewness")
    suite2p_path = os.path.join(Base_path, "suite2p", "plane0")
    xml_direction = glob.glob(os.path.join(Base_path, '*.xml'))[0]
    facemap_path = glob.glob(os.path.join(Base_path, '*proc.npy'))[0]
    return Base_path,save_data, save_direction1, save_direction_figure,save_direction_permutation\
        , save_direction0_lag, save_direction_skew, suite2p_path, facemap_path, xml_direction, hf

def load_data(suite2p_path, Base_path,facemap_path):
    # Load suite2p data
    cell = np.load(os.path.join(suite2p_path, "iscell.npy"), allow_pickle=True)
    F = np.load(os.path.join(suite2p_path, "F.npy"), allow_pickle=True)
    Fneu_raw = np.load(os.path.join(suite2p_path, "Fneu.npy"), allow_pickle=True)
    spikes = np.load(os.path.join(suite2p_path, "spks.npy"), allow_pickle=True)
    # Load speed data
    movement_file = glob.glob(os.path.join(Base_path, '*.abf'))
    # Load facemap data
    facemap_data = np.load(facemap_path, allow_pickle=True)
    pupil = facemap_data.item().get('pupil', [{}])[0].get('area', np.array([]))
    motion = facemap_data.item().get('motion', [])[1]
    np.nan_to_num(pupil, copy=False)
    np.nan_to_num(motion, copy=False)
    return cell, F, Fneu_raw, spikes, movement_file, pupil, motion
def load_xml(xml_direction):
    xml = xml_parser.bruker_xml_parser(xml_direction)
    channel_number = xml['Nchannels']
    laserWavelength = xml['settings']['laserWavelength']
    objectiveLens = xml['settings']['objectiveLens']
    objectiveLensMag = xml['settings']['objectiveLensMag']
    opticalZoom = xml['settings']['opticalZoom']
    bitDepth = xml['settings']['bitDepth']
    dwellTime = xml['settings']['dwellTime']
    framePeriod = xml['settings']['framePeriod']
    micronsPerPixel = xml['settings']['micronsPerPixel']
    TwophotonLaserPower = xml['settings']['twophotonLaserPower']
    return xml, channel_number, laserWavelength, objectiveLens, objectiveLensMag, opticalZoom,\
        bitDepth, dwellTime, framePeriod, micronsPerPixel, TwophotonLaserPower
