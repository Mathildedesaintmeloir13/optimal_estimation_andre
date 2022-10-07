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
    # subjects_trials = [('SaMi', '821_contact_1', 100), ('SaMi', '821_contact_2', 100),
    #                    ('SaMi', '821_seul_2', 100), ('SaMi', '821_seul_4', 100),
    #                   ]

    segment_Q_EKF_OE = []
    segment_Q_EKF_OGE = []

    segment_marker_error_OE = []
    segment_marker_error_OGE = []
    segment_marker_error_EKF = []

    segment_marker_error_OE_all = []
    segment_marker_error_OGE_all = []
    segment_marker_error_EKF_all = []

    for subject, trial, _ in subjects_trials:
        data_filename = load_data_filename(subject, trial)
        c3d_name = data_filename['c3d']

        load_path = 'Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_stats_noOGE'
        load_variables_name = load_path + '.pkl'
        with open(load_variables_name, 'rb') as handle:
            data = pickle.load(handle)

        segment_Q_EKF_OE.append(data['RMSE_difference_between_Q_OE_segment'])
        segment_Q_EKF_OGE.append(data['RMSE_difference_between_Q_OGE_segment'])

        segment_marker_error_OE.append(data['segment_marker_error_OE'])
        segment_marker_error_OGE.append(data['segment_marker_error_OGE'])
        segment_marker_error_EKF.append(data['segment_marker_error_EKF_biorbd'])

        segment_marker_error_OE_all.append(data['segment_marker_error_OE_all'])
        segment_marker_error_OGE_all.append(data['segment_marker_error_OGE_all'])
        segment_marker_error_EKF_all.append(data['segment_marker_error_EKF_biorbd_all'])

    mean_segment_Q_EKF_OE = {}
    mean_segment_Q_EKF_OGE = {}

    mean_segment_marker_error_OE = {}
    mean_segment_marker_error_OGE = {}
    mean_segment_marker_error_EKF = {}

    mean_segment_marker_error_OE_all = {}
    mean_segment_marker_error_OGE_all = {}
    mean_segment_marker_error_EKF_all = {}

    std_segment_marker_error_OE_all = {}
    std_segment_marker_error_OGE_all = {}
    std_segment_marker_error_EKF_all = {}
    for segment_name in segment_marker_error_OE[0].keys():
        mean_segment_Q_EKF_OE[segment_name] = np.nanmean([trial[segment_name] for trial in segment_Q_EKF_OE])
        mean_segment_Q_EKF_OGE[segment_name] = np.nanmean([trial[segment_name] for trial in segment_Q_EKF_OGE])

        mean_segment_marker_error_OE[segment_name] = np.nanmean([trial[segment_name] for trial in segment_marker_error_OE])
        mean_segment_marker_error_OGE[segment_name] = np.nanmean([trial[segment_name] for trial in segment_marker_error_OGE])
        mean_segment_marker_error_EKF[segment_name] = np.nanmean([trial[segment_name] for trial in segment_marker_error_EKF])

        mean_segment_marker_error_OE_all[segment_name] = np.nanmean(np.concatenate([trial[segment_name] for trial in segment_marker_error_OE_all]))
        mean_segment_marker_error_OGE_all[segment_name] = np.nanmean(np.concatenate([trial[segment_name] for trial in segment_marker_error_OGE_all]))
        mean_segment_marker_error_EKF_all[segment_name] = np.nanmean(np.concatenate([trial[segment_name] for trial in segment_marker_error_EKF_all]))

        std_segment_marker_error_OE_all[segment_name] = np.nanstd(np.concatenate([trial[segment_name] for trial in segment_marker_error_OE_all]))
        std_segment_marker_error_OGE_all[segment_name] = np.nanstd(np.concatenate([trial[segment_name] for trial in segment_marker_error_OGE_all]))
        std_segment_marker_error_EKF_all[segment_name] = np.nanstd(np.concatenate([trial[segment_name] for trial in segment_marker_error_EKF_all]))

    # --- Plots --- #
    segments_all = [
                ['Pelvis', 'Thorax', 'Tete'],
                ['Pelvis', 'Thorax', 'EpauleD', 'BrasD', 'ABrasD', 'MainD'],
                ['Pelvis', 'Thorax', 'EpauleG', 'BrasG', 'ABrasG', 'MainG'],
                ['Pelvis', 'CuisseD', 'JambeD', 'PiedD'],
                ['Pelvis', 'CuisseG', 'JambeG', 'PiedG'],
                ]

    fig_diff_mean = []
    # for segments in segments_all:
        # fig_diff_Q = pyplot.figure(figsize=(20, 10))
        # clrs_bright = sns.color_palette("bright", 2)
        #
        # for trial_OE, trial_OGE in zip(segment_Q_EKF_OE, segment_Q_EKF_OGE):
        #     for idx, segment in enumerate(segments):
        #         pyplot.plot(idx, trial_OE[segment], '-s', markersize=11, color=clrs_bright[0])
        #         pyplot.plot(idx, trial_OGE[segment], '-s', markersize=11, color=clrs_bright[1])
        # # pyplot.xlabel('Segment', fontsize=16)
        # pyplot.ylabel('Joint angle difference (°)', fontsize=16)
        # pyplot.xticks(np.arange(len(segments)), segments, fontsize=14)
        # pyplot.yticks(fontsize=14)
        # fig_diff_Q.gca().legend(['Marker tracking', 'Joint angle tracking'], fontsize=15)
        #
        # fig_diff_marker_error = pyplot.figure(figsize=(20, 10))
        # clrs_bright = sns.color_palette("bright", 2)
        #
        # for trial_OE, trial_OGE in zip(segment_marker_error_OE, segment_marker_error_OGE):
        #     for idx, segment in enumerate(segments):
        #         pyplot.plot(idx, trial_OE[segment], '-s', markersize=11, color=clrs_bright[0])
        #         pyplot.plot(idx, trial_OGE[segment], '-s', markersize=11, color=clrs_bright[1])
        # # pyplot.xlabel('Segment', fontsize=16)
        # pyplot.ylabel('Marker error (mm)', fontsize=16)
        # pyplot.xticks(np.arange(len(segments)), segments, fontsize=14)
        # pyplot.yticks(fontsize=14)
        # fig_diff_marker_error.gca().legend(['Marker tracking', 'Joint angle tracking'], fontsize=15)


        # # Mean
        # fig_diff_mean.append(pyplot.subplots(nrows=1, ncols=2, figsize=(20, 10)))
        # axs = fig_diff_mean[-1][1]
        #
        # clrs_bright = sns.color_palette("bright", 2)
        #
        # for idx, segment in enumerate(segments):
        #     axs[0].plot(idx, mean_segment_Q_EKF_OE[segment], '-s', markersize=11, color=clrs_bright[0])
        #     axs[0].plot(idx, mean_segment_Q_EKF_OGE[segment], '-s', markersize=11, color=clrs_bright[1])
        # # axs[0].xlabel('Segment', fontsize=16)
        # axs[0].set_ylabel('Joint angle difference (°)', fontsize=16)
        # axs[0].set_xticks(np.arange(len(segments)))
        # axs[0].set_xticklabels(segments, fontsize=14)
        # axs[0].tick_params(axis='y', labelsize=14)
        # # axs[0].legend(['Marker tracking', 'Joint angle tracking'], fontsize=15)
        #
        # clrs_bright = sns.color_palette("bright", 3)
        #
        # for idx, segment in enumerate(segments):
        #     axs[1].plot(idx, mean_segment_marker_error_OE[segment], '-s', markersize=11, color=clrs_bright[0])
        #     axs[1].plot(idx, mean_segment_marker_error_OGE[segment], '-s', markersize=11, color=clrs_bright[1])
        #     axs[1].plot(idx, mean_segment_marker_error_EKF[segment], '-s', markersize=11, color=clrs_bright[2])
        # # axs[1].xlabel('Segment', fontsize=16)
        # axs[1].set_ylabel('Marker error (mm)', fontsize=16)
        # axs[1].set_xticks(np.arange(len(segments)))
        # axs[1].set_xticklabels(segments, fontsize=14)
        # axs[1].tick_params(axis='y', labelsize=14)
        # axs[1].legend(['Marker tracking', 'Joint angle tracking', 'EKF'], fontsize=15)



    segments_topo = [
                        ['Pelvis'],
                        ['Thorax', 'CuisseD', 'CuisseG'],
                        ['Tete', 'EpauleD', 'EpauleG', 'JambeD', 'JambeG'],
                        ['BrasD', 'BrasG', 'PiedD', 'PiedG'],
                        ['ABrasD', 'ABrasG'],
                        ['MainD', 'MainG']
                    ]

    mean_topo_segment_marker_error_OE = []
    mean_topo_segment_marker_error_OGE = []
    mean_topo_segment_marker_error_EKF = []

    std_topo_segment_marker_error_OE = []
    std_topo_segment_marker_error_OGE = []
    std_topo_segment_marker_error_EKF = []
    for topo in segments_topo:
        mean_topo_segment_marker_error_OE.append(np.nanmean(np.concatenate([trial[segment_name] for trial in segment_marker_error_OE_all for segment_name in topo])))
        mean_topo_segment_marker_error_OGE.append(np.nanmean(np.concatenate([trial[segment_name] for trial in segment_marker_error_OGE_all for segment_name in topo])))
        mean_topo_segment_marker_error_EKF.append(np.nanmean(np.concatenate([trial[segment_name] for trial in segment_marker_error_EKF_all for segment_name in topo])))

        std_topo_segment_marker_error_OE.append(np.nanstd(np.concatenate([trial[segment_name] for trial in segment_marker_error_OE_all for segment_name in topo])))
        std_topo_segment_marker_error_OGE.append(np.nanstd(np.concatenate([trial[segment_name] for trial in segment_marker_error_OGE_all for segment_name in topo])))
        std_topo_segment_marker_error_EKF.append(np.nanstd(np.concatenate([trial[segment_name] for trial in segment_marker_error_EKF_all for segment_name in topo])))

    markersize = 10
    linewidth = 3
    capsize = 0
    elinewidth = 3

    fig_diff_topo = pyplot.figure(figsize=(10, 10))
    ax_topo = fig_diff_topo.gca()
    clrs_bright = sns.color_palette("bright", 3)
    ax_topo.errorbar(np.arange(len(mean_topo_segment_marker_error_EKF))-0.1, mean_topo_segment_marker_error_EKF, yerr=std_topo_segment_marker_error_EKF, fmt='-o', markersize=markersize, linewidth=linewidth, color=clrs_bright[0], capsize=capsize, elinewidth=elinewidth)
    ax_topo.errorbar(np.arange(len(mean_topo_segment_marker_error_OE)), mean_topo_segment_marker_error_OE, yerr=std_topo_segment_marker_error_OE, fmt='-o', markersize=markersize, linewidth=linewidth, color=clrs_bright[1], capsize=capsize, elinewidth=elinewidth)
    ax_topo.errorbar(np.arange(len(mean_topo_segment_marker_error_OGE))+0.1, mean_topo_segment_marker_error_OGE, yerr=std_topo_segment_marker_error_OGE, fmt='-o', markersize=markersize, linewidth=linewidth, color=clrs_bright[2], capsize=capsize, elinewidth=elinewidth)
    ax_topo.set_xlabel('Segment topological distance from the root (pelvis)', fontsize=20)
    ax_topo.set_ylabel('Markers RMSD (mm)', fontsize=20)
    # ax_topo.set_xticks(np.arange(len(segments)))
    # ax_topo.set_xticklabels(segments, fontsize=14)
    ax_topo.tick_params(axis='both', labelsize=20)
    ax_topo.spines["top"].set_visible(False)
    ax_topo.spines["bottom"].set_visible(False)
    ax_topo.spines["right"].set_visible(False)
    ax_topo.spines["left"].set_visible(False)
    # ax_topo.legend(['EKF', 'MT', 'QT'], fontsize=20, loc=2)

    print('EKF hands mean RMS: ', np.nanmean([trial['MainD'] for trial in segment_marker_error_EKF]), ' ± ', np.nanstd([trial['MainD'] for trial in segment_marker_error_EKF]))
    print('Joint angle tracking hands mean RMS: ', np.nanmean([trial['MainD'] for trial in segment_marker_error_OGE]), ' ± ', np.nanstd([trial['MainD'] for trial in segment_marker_error_OGE]))
    print('Marker tracking hands mean RMS: ', np.nanmean([trial['MainD'] for trial in segment_marker_error_OE]), ' ± ', np.nanstd([trial['MainD'] for trial in segment_marker_error_OE]))

    # save_path = 'Solutions/'
    # save_name = save_path + "End_chain_Q_diff" + '.png'
    # fig_diff_Q.savefig(save_name, bbox_inches='tight', pad_inches=0)

    # save_path = 'Solutions/'
    # save_name = save_path + "End_chain_marker_error" + '.png'
    # fig_diff_marker_error.savefig(save_name, bbox_inches='tight', pad_inches=0)

    # save_path = 'Solutions/'
    # for idx, (fig, _) in enumerate(fig_diff_mean):
    #     save_name = save_path + "End_chain_Q_marker_diff_" + segments_all[idx][-1] + '_.png'
    #     fig.savefig(save_name, bbox_inches='tight', pad_inches=0)

    save_path = 'Solutions/'
    save_name = save_path + "End_chain_marker_error_topo" + '.eps'
    fig_diff_topo.savefig(save_name, bbox_inches='tight', pad_inches=0, dpi=300)
    save_name = save_path + "End_chain_marker_error_topo" + '.png'
    fig_diff_topo.savefig(save_name, bbox_inches='tight', pad_inches=0, dpi=300)

    pyplot.show()

