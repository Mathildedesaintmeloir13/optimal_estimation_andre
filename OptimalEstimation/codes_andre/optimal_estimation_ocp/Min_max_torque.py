import biorbd
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

    controls_OE = []
    controls_OGE = []
    controls_EKF = []

    for subject, trial, _ in subjects_trials:
        data_filename = load_data_filename(subject, trial)
        c3d_name = data_filename['c3d']

        load_path = 'Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_stats_noOGE'
        load_variables_name = load_path + '.pkl'
        with open(load_variables_name, 'rb') as handle:
            data = pickle.load(handle)

        controls_OE.append(data['controls_OE'])
        controls_OGE.append(data['controls_OGE'])
        controls_EKF.append(data['controls_EKF'])

    min_controls_OE = np.min([np.min(trial['tau'], axis=1) for trial in controls_OE], axis=0)
    max_controls_OE = np.max([np.max(trial['tau'], axis=1) for trial in controls_OE], axis=0)
    min_controls_OGE = np.min([np.min(trial['tau'], axis=1) for trial in controls_OGE], axis=0)
    max_controls_OGE = np.max([np.max(trial['tau'], axis=1) for trial in controls_OGE], axis=0)
    min_controls_EKF = np.min([np.min(trial['tau'], axis=1) for trial in controls_EKF], axis=0)
    max_controls_EKF = np.max([np.max(trial['tau'], axis=1) for trial in controls_EKF], axis=0)

    print(min_controls_OE)
