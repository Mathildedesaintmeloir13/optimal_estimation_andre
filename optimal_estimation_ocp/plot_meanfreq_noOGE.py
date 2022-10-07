import biorbd
import matlab.engine
import numpy as np
from casadi import MX
import ezc3d
import pickle
from scipy.io import loadmat
import os
import sys
from pathlib import Path
from matplotlib import pyplot
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'optimal_gravity_ocp'))
from load_data_filename import load_data_filename
from adjust_number_shooting_points import adjust_number_shooting_points
from adjust_Kalman import shift_by_2pi


if __name__ == "__main__":
    subjects_trials = [('DoCi', '822', 100), ('DoCi', '44_1', 100), ('DoCi', '44_2', 100), ('DoCi', '44_3', 100),
                       ('BeLa', '44_1', 100), ('BeLa', '44_2', 80), ('BeLa', '44_3', 100),
                       ('GuSe', '44_2', 80), ('GuSe', '44_3', 100), ('GuSe', '44_4', 100),
                       ('SaMi', '821_contact_1', 100), ('SaMi', '821_contact_2', 100), ('SaMi', '821_contact_3', 100), ('SaMi', '822_contact_1', 100),
                       ('SaMi', '821_seul_1', 100), ('SaMi', '821_seul_2', 100), ('SaMi', '821_seul_3', 100), ('SaMi', '821_seul_4', 100), ('SaMi', '821_seul_5', 100),
                       ('SaMi', '821_822_2', 100), ('SaMi', '821_822_3', 100),
                       ('JeCh', '833_1', 100), ('JeCh', '833_2', 100), ('JeCh', '833_3', 100), ('JeCh', '833_4', 100), ('JeCh', '833_5', 100),
                      ]

    load_path = 'Solutions/'
    load_name = load_path + 'meanfreq_noOGE'
    load_variables_name = load_name + ".pkl"
    with open(load_variables_name, 'rb') as handle:
        data = pickle.load(handle)

    MNF_EKF_biorbd_sum = data['MNF']['EKF_biorbd']
    MNF_OGE_sum = data['MNF']['OGE']
    MNF_OE_sum = data['MNF']['OE']

    MNF_EKF_biorbd_mean = np.mean(MNF_EKF_biorbd_sum)
    MNF_EKF_biorbd_std = np.std(MNF_EKF_biorbd_sum)
    MNF_OGE_mean = np.mean(MNF_OGE_sum)
    MNF_OGE_std = np.std(MNF_OGE_sum)
    MNF_OE_mean = np.mean(MNF_OE_sum)
    MNF_OE_std = np.std(MNF_OE_sum)

    # load_variables_name = load_name + "_third_node.pkl"
    # with open(load_variables_name, 'rb') as handle:
    #     data = pickle.load(handle)
    #
    # MNF_EKF_biorbd_third_node_sum = data['MNF']['EKF_biorbd']
    #
    # MNF_EKF_biorbd_third_node_mean = np.mean(MNF_EKF_biorbd_third_node_sum)
    # MNF_EKF_biorbd_third_node_std = np.std(MNF_EKF_biorbd_third_node_sum)

    # Create dataset
    MNF = np.array(MNF_EKF_biorbd_sum + MNF_OE_sum + MNF_OGE_sum)
    recons_type = np.array(['EKF'] * len(MNF_EKF_biorbd_sum) + ['Marker\ntracking'] * len(MNF_OE_sum) + ['Joint angle\ntracking'] * len(MNF_OGE_sum))
    subjects = np.array([subject_trial[0] for subject_trial in subjects_trials] * 3)
    trials = np.array([subject_trial[1] for subject_trial in subjects_trials] * 3)
    number_shooting_points = np.array([subject_trial[2] for subject_trial in subjects_trials] * 3)

    data = pd.DataFrame(data=MNF, index=range(MNF.size), columns=['MNF'])
    data['recons_type'] = recons_type
    data['subjects'] = subjects
    data['trials'] = trials
    data['number_shooting_points'] = number_shooting_points

    # Draw swarmplot
    fig = pyplot.figure(figsize=(10, 10))
    sns.set(rc={'axes.facecolor': 'white', 'figure.facecolor': 'white'}, font_scale=2)
    ax = sns.swarmplot(data=data, x='recons_type', y='MNF', size=10)
    ax.set(xlabel="", ylabel='Mean power frequency (Hz)')
    # ax.set_ylim([0, 21])
    lm_EKF = Line2D([0, 1], [0, 1], linestyle='-', color='blue', marker="o", markersize=8)
    lm_QT = Line2D([0, 1], [0, 1], linestyle='-', color='red', marker="o", markersize=8)
    lm_MT = Line2D([0, 1], [0, 1], linestyle='-', color='green', marker="o", markersize=8)
    ax.legend([lm_EKF, lm_QT, lm_MT], ['EKF', 'Marker tracking', 'Joint angle tracking'], fontsize=20, handlelength=1)

    save_path = 'Solutions/'
    fig.tight_layout
    save_name = save_path + 'MNF_controls_noOGE' + '.eps'
    fig.savefig(save_name, bbox_inches='tight', pad_inches=0, dpi=300)
    save_name = save_path + 'MNF_controls_noOGE' + '.png'
    fig.savefig(save_name, bbox_inches='tight', pad_inches=0, dpi=300)

    pyplot.show()
