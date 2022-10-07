import biorbd
import numpy as np
import ezc3d
from casadi import MX, Function, sum1
import pickle
from scipy.io import loadmat
from bioviz import Viz
import os
import sys
from pathlib import Path
from matplotlib import pyplot
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.mplot3d import Axes3D
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'optimal_gravity_ocp'))
from load_data_filename import load_data_filename
from adjust_number_shooting_points import adjust_number_shooting_points
from reorder_markers import reorder_markers
from adjust_Kalman import correct_Kalman, check_Kalman, shift_by_2pi


if __name__ == "__main__":
    # subject = 'DoCi'
    # subject = 'JeCh'
    # subject = 'BeLa'
    # subject = 'GuSe'
    subject = 'SaMi'
    number_shooting_points = 100
    trial = '821_seul_1'

    data_path = '/home/andre/Optimisation/data/' + subject + '/'
    model_path = data_path + 'Model/'
    c3d_path = data_path + 'Essai/'

    data_filename = load_data_filename(subject, trial)
    model_name = data_filename['model']
    c3d_name = data_filename['c3d']

    load_path = 'Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_stats_noOGE'
    load_variables_name = load_path + '.pkl'
    with open(load_variables_name, 'rb') as handle:
        data = pickle.load(handle)

    mocap = data['markers_mocap']
    mocap_reshape = mocap.reshape(-1, mocap.shape[-1])
    q_EKF = data['q_EKF']
    q_OGE = data['q_OGE']
    q_OE = data['q_OE']

    print(q_OE)

    b = Viz(model_path=model_path+model_name)
    b.load_movement(q_EKF)
    b.exec()
