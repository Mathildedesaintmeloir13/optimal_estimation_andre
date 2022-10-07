import biorbd
import numpy as np
import ezc3d
from scipy.io import loadmat
import time
from casadi import MX, Function
import os
import pickle
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'optimal_gravity_ocp'))
from load_data_filename import load_data_filename
from adjust_number_shooting_points import adjust_number_shooting_points
from reorder_markers import reorder_markers
from adjust_Kalman import correct_Kalman, check_Kalman, shift_by_2pi


if __name__ == "__main__":
    # subject = 'DoCi'
    subject = 'JeCh'
    # subject = 'BeLa'
    # subject = 'GuSe'
    # subject = 'SaMi'
    number_shooting_points = 10000
    trial = '821_1'

    data_path = '/home/andre/Optimisation/data/' + subject + '/'
    model_path = data_path + 'Model/'
    c3d_path = data_path + 'Essai/'

    data_filename = load_data_filename(subject, trial)
    model_name = data_filename['model']
    c3d_name = data_filename['c3d']

    biorbd_model = biorbd.Model(model_path + model_name)
    c3d = ezc3d.c3d(c3d_path + c3d_name)

    frames = range(3419, c3d['data']['points'].shape[2])
    adjusted_number_shooting_points, step_size = adjust_number_shooting_points(number_shooting_points, frames)
    markers_reordered, _ = reorder_markers(biorbd_model, c3d, frames, step_size)

    # load_path = 'Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_stats_noOGE'
    # load_variables_name = load_path + '.pkl'
    # with open(load_variables_name, 'rb') as handle:
    #     data = pickle.load(handle)

    markers_c3d = markers_reordered
    # markers_mocap = data['markers_mocap']
    # markers_EKF = data['markers_EKF']
    # markers_OGE = data['markers_OGE']
    # markers_OE = data['markers_OE']

    fig = pyplot.figure()
    ax = Axes3D(fig)
    # for frame in range(markers_mocap.shape[2]):
    for frame in range(markers_c3d.shape[2]):
        ax.scatter(markers_c3d[0, :, frame], markers_c3d[1, :, frame], markers_c3d[2, :, frame], color='red', marker='x')
        # ax.scatter(markers_mocap[0, :, frame], markers_mocap[1, :, frame], markers_mocap[2, :, frame], color='blue', marker='x')
        # ax.scatter(markers_EKF[0, :, frame], markers_EKF[1, :, frame], markers_EKF[2, :, frame], color='orange')
        # ax.scatter(markers_OGE[0, :, frame], markers_OGE[1, :, frame], markers_OGE[2, :, frame], color='green')
        # ax.scatter(markers_OE[0, :, frame], markers_OE[1, :, frame], markers_OE[2, :, frame], color='purple')
        ax.set_xlim3d(0, 3)
        ax.set_ylim3d(0, 3)
        ax.set_zlim3d(0, 2)
        pyplot.pause(1)
        pyplot.draw()
        ax.clear()
        print(frame)

