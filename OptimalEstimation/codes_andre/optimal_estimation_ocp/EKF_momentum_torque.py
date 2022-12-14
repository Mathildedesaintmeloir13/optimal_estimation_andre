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

    states_EKF = []
    controls_EKF = []
    qddot_EKF = []
    momentum_EKF = []
    RMSE_momentum_EKF = []
    linear_momentum_EKF = []
    RMSE_linear_momentum_EKF = []

    for subject, trial, _ in subjects_trials:
        data_filename = load_data_filename(subject, trial)
        c3d_name = data_filename['c3d']

        load_path = 'Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_stats_noOGE'
        load_variables_name = load_path + '.pkl'
        with open(load_variables_name, 'rb') as handle:
            data = pickle.load(handle)

        controls_OE.append(data['controls_OE'])
        controls_OGE.append(data['controls_OGE'])

        states_EKF.append(data['states_EKF'])
        controls_EKF.append(data['controls_EKF'])
        qddot_EKF.append(data['qddot_EKF'])
        momentum_EKF.append(data['momentum_EKF'])
        RMSE_momentum_EKF.append(data['RMSE_momentum_EKF'])
        linear_momentum_EKF.append(data['linear_momentum_EKF'])
        RMSE_linear_momentum_EKF.append(data['RMSE_linear_momentum_EKF'])

    peak_to_peak_momentum_EKF = [np.max(np.abs(trial), axis=1) - np.min(np.abs(trial), axis=1) for trial in momentum_EKF]
    peak_to_peak_momentum_EKF_max = np.max(peak_to_peak_momentum_EKF)
    peak_to_peak_momentum_EKF_mean = np.mean(peak_to_peak_momentum_EKF)
    peak_to_peak_momentum_EKF_std = np.std(peak_to_peak_momentum_EKF)
    RMSE_momentum_EKF_mean = np.mean(RMSE_momentum_EKF)
    RMSE_momentum_EKF_std = np.std(RMSE_momentum_EKF)
    RMSE_linear_momentum_EKF_mean = np.mean(RMSE_linear_momentum_EKF)
    RMSE_linear_momentum_EKF_std = np.std(RMSE_linear_momentum_EKF)

    mean_angular_momentum_EKF = [np.abs(np.mean(trial, axis=1)) for trial in momentum_EKF]
    max_mean_angular_momentum_EKF = np.max(mean_angular_momentum_EKF)
    min_mean_angular_momentum_EKF = np.min(mean_angular_momentum_EKF)
    mean_horizontal_linear_momentum_EKF = [np.abs(np.mean(trial[0:2], axis=1)) for trial in linear_momentum_EKF]
    max_mean_horizontal_linear_momentum_EKF = np.max(mean_horizontal_linear_momentum_EKF)
    min_mean_horizontal_linear_momentum_EKF = np.min(mean_horizontal_linear_momentum_EKF)

    mean_angular_momentum_x_EKF = np.mean([trial[1] if idx in range(7, 21) else trial[0] for idx, trial in enumerate(mean_angular_momentum_EKF)])
    mean_angular_momentum_y_EKF = np.mean([trial[0] if idx in range(7, 21) else trial[1] for idx, trial in enumerate(mean_angular_momentum_EKF)])
    mean_angular_momentum_z_EKF = np.mean([trial[2] for trial in mean_angular_momentum_EKF])

    std_angular_momentum_x_EKF = np.std([trial[1] if idx in range(7, 21) else trial[0] for idx, trial in enumerate(mean_angular_momentum_EKF)])
    std_angular_momentum_y_EKF = np.std([trial[0] if idx in range(7, 21) else trial[1] for idx, trial in enumerate(mean_angular_momentum_EKF)])
    std_angular_momentum_z_EKF = np.std([trial[2] for trial in mean_angular_momentum_EKF])

    mean_horizontal_linear_momentum_x_EKF = np.mean([trial[1] if idx in range(7, 21) else trial[0] for idx, trial in enumerate(mean_horizontal_linear_momentum_EKF)])
    mean_horizontal_linear_momentum_y_EKF = np.mean([trial[0] if idx in range(7, 21) else trial[1] for idx, trial in enumerate(mean_horizontal_linear_momentum_EKF)])

    std_horizontal_linear_momentum_x_EKF = np.std([trial[1] if idx in range(7, 21) else trial[0] for idx, trial in enumerate(mean_horizontal_linear_momentum_EKF)])
    std_horizontal_linear_momentum_y_EKF = np.std([trial[0] if idx in range(7, 21) else trial[1] for idx, trial in enumerate(mean_horizontal_linear_momentum_EKF)])

    abs_max_controls_root_EKF_trans = np.max([np.max(np.abs(trial['tau'].full()[:3, :])) for trial in controls_EKF])
    abs_max_controls_root_EKF_rot = np.max([np.max(np.abs(trial['tau'].full()[3:6, :])) for trial in controls_EKF])

    trial_idx_abs_max_controls_root_EKF_trans = np.argmax([np.max(np.abs(trial['tau'].full()[:3, :])) for trial in controls_EKF])
    trial_idx_abs_max_controls_root_EKF_rot = np.argmax([np.max(np.abs(trial['tau'].full()[3:6, :])) for trial in controls_EKF])
    idx_abs_max_controls_root_EKF_trans = np.unravel_index(np.argmax(np.abs(controls_EKF[trial_idx_abs_max_controls_root_EKF_trans]['tau'].full()[:3, :])), controls_EKF[trial_idx_abs_max_controls_root_EKF_trans]['tau'].full()[:3, :].shape)
    idx_abs_max_controls_root_EKF_rot = np.unravel_index(np.argmax(np.abs(controls_EKF[trial_idx_abs_max_controls_root_EKF_rot]['tau'].full()[3:6, :])), controls_EKF[trial_idx_abs_max_controls_root_EKF_rot]['tau'].full()[3:6, :].shape)
    max_qddot_EKF_trans = qddot_EKF[trial_idx_abs_max_controls_root_EKF_trans][:3, :][idx_abs_max_controls_root_EKF_trans]
    max_qddot_EKF_rot = qddot_EKF[trial_idx_abs_max_controls_root_EKF_rot][3:6, :][idx_abs_max_controls_root_EKF_rot]
    subjects_trials[trial_idx_abs_max_controls_root_EKF_trans]
    subjects_trials[trial_idx_abs_max_controls_root_EKF_rot]

    RMS_controls_EKF_root_trans = [np.mean(np.sqrt(np.mean(trial_EKF['tau'][:3, :] ** 2, axis=1))) for trial_EKF in controls_EKF]
    RMS_controls_EKF_root_trans_mean = np.mean(RMS_controls_EKF_root_trans)
    RMS_controls_EKF_root_trans_std = np.std(RMS_controls_EKF_root_trans)

    RMS_controls_EKF_root_rot = [np.mean(np.sqrt(np.mean(trial_EKF['tau'][3:6, :] ** 2, axis=1))) for trial_EKF in controls_EKF]
    RMS_controls_EKF_root_rot_mean = np.mean(RMS_controls_EKF_root_rot)
    RMS_controls_EKF_root_rot_std = np.std(RMS_controls_EKF_root_rot)

    RMS_controls_EKF = [np.mean(np.sqrt(np.mean(trial_EKF['tau'][6:, :] ** 2, axis=1))) for trial_EKF in controls_EKF]
    RMS_controls_EKF_mean = np.mean(RMS_controls_EKF)
    RMS_controls_EKF_std = np.std(RMS_controls_EKF)

    RMS_controls_OE = [np.mean(np.sqrt(np.mean(trial_OE['tau'][6:, :] ** 2, axis=1))) for trial_OE in controls_OE]
    RMS_controls_OE_mean = np.mean(RMS_controls_OE)
    RMS_controls_OE_std = np.std(RMS_controls_OE)

    RMS_controls_OGE = [np.mean(np.sqrt(np.mean(trial_OGE['tau'][6:, :] ** 2, axis=1))) for trial_OGE in controls_OGE]
    RMS_controls_OGE_mean = np.mean(RMS_controls_OGE)
    RMS_controls_OGE_std = np.std(RMS_controls_OGE)

    # RMS_controls_EKF_OE = [np.mean(np.sqrt(np.mean((trial_EKF['tau'].full()[6:, :] - trial_OE['tau'][6:, :]) ** 2, axis=1))) for trial_EKF, trial_OE in zip(controls_EKF, controls_OE)]
    RMS_controls_EKF_OE = np.array(RMS_controls_EKF) - np.array(RMS_controls_OE)
    RMS_controls_EKF_OE_mean = np.mean(RMS_controls_EKF_OE)
    RMS_controls_EKF_OE_std = np.std(RMS_controls_EKF_OE)

    # RMS_controls_EKF_OGE = [np.mean(np.sqrt(np.mean((trial_EKF['tau'].full()[6:, :] - trial_OGE['tau'][6:, :]) ** 2, axis=1))) for trial_EKF, trial_OGE in zip(controls_EKF, controls_OGE)]
    RMS_controls_EKF_OGE = np.array(RMS_controls_EKF) - np.array(RMS_controls_OGE)
    RMS_controls_EKF_OGE_mean = np.mean(RMS_controls_EKF_OGE)
    RMS_controls_EKF_OGE_std = np.std(RMS_controls_EKF_OGE)


    print('% EKF vs OE: ', (RMS_controls_EKF_OE_mean/RMS_controls_OE_mean)*100)
    print('% EKF vs OGE: ', (RMS_controls_EKF_OGE_mean/RMS_controls_OGE_mean)*100)
