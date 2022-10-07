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
import matplotlib.ticker as mticker
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

    save_path = 'Solutions/'

    q_OE = []
    q_OGE = []
    q_EKF = []

    controls_OE = []
    controls_OGE = []
    controls_EKF = []

    lm_kalman = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
    lm_OGE = Line2D([0, 1], [0, 1], linestyle='-', color='red')
    lm_OE = Line2D([0, 1], [0, 1], linestyle='-', color='green')

    for subject, trial, _ in subjects_trials:
        Path(save_path + subject + '/Plots/Kinematics').mkdir(parents=True, exist_ok=True)
        Path(save_path + subject + '/Plots/Kinetics').mkdir(parents=True, exist_ok=True)

        data_filename = load_data_filename(subject, trial)
        c3d_name = data_filename['c3d']

        load_path = 'Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_stats_noOGE'
        load_variables_name = load_path + '.pkl'
        with open(load_variables_name, 'rb') as handle:
            data = pickle.load(handle)

        time_per_node = data['step_size'] / data['frequency']

        q_OE.append(data['q_OE'])
        q_OGE.append(data['q_OGE'])
        q_EKF.append(data['q_EKF'])

        controls_OE.append(data['controls_OE'])
        controls_OGE.append(data['controls_OGE'])
        controls_EKF.append(data['controls_EKF'])

        dofs = [range(0, 3), range(3, 6), range(6, 9), range(9, 12),
                range(12, 14), range(14, 17), range(17, 19), range(19, 21),
                range(21, 23), range(23, 26), range(26, 28), range(28, 30),
                range(30, 33), range(33, 34), range(34, 36),
                range(36, 39), range(39, 40), range(40, 42),
                ]
        dofs_kinematics_name = \
                    ['Pelvis kinematics: XYZ translation', 'Pelvis kinematics: XYZ rotation', 'Thorax kinematics: XYZ rotation', 'Head kinematics: XYZ rotation',
                     'Right shoulder kinematics: YZ rotation', 'Right arm kinematics: XYZ rotation', 'Right forearm kinematics: XZ rotation', 'Right hand kinematics: XY rotation',
                     'Left shoulder kinematics: YZ rotation', 'Left arm kinematics: XYZ rotation', 'Left forearm kinematics: XZ rotation', 'Left hand kinematics: XY rotation',
                     'Right thigh kinematics: XYZ rotation', 'Right leg kinematics: X rotation', 'Right foot kinematics: XZ rotation',
                     'Left thigh kinematics: XYZ rotation', 'Left leg kinematics: X rotation', 'Left foot kinematics: XZ rotation',
                     ]
        dofs_kinetics_name = \
            ['Pelvis kinetics: XYZ translation', 'Pelvis kinetics: XYZ rotation', 'Thorax kinetics: XYZ rotation', 'Head kinetics: XYZ rotation',
             'Right shoulder kinetics: YZ rotation', 'Right arm kinetics: XYZ rotation', 'Right forearm kinetics: XZ rotation', 'Right hand kinetics: XY rotation',
             'Left shoulder kinetics: YZ rotation', 'Left arm kinetics: XYZ rotation', 'Left forearm kinetics: XZ rotation', 'Left hand kinetics: XY rotation',
             'Right thigh kinetics: XYZ rotation', 'Right leg kinetics: X rotation', 'Right foot kinetics: XZ rotation',
             'Left thigh kinetics: XYZ rotation', 'Left leg kinetics: X rotation', 'Left foot kinetics: XZ rotation',
             ]
        dofs_save_name = ['Pelvis_Trans', 'Pelvis_Rot', 'Thorax', 'Head',
                     'Right_Shoulder', 'Right_Arm', 'Right_Forearm', 'Right_Hand',
                     'Left_Shoulder', 'Left_Arm', 'Left_Forearm', 'Left_Hand',
                     'Right_Thigh', 'Right_Leg', 'Right_Foot',
                     'Left_Thigh', 'Left_Leg', 'Left_Foot',
                     ]
        fig_model_dof = [(3, 2), (4, 2), (1, 2), (0, 2),
                         (1, 0), (2, 0), (3, 0), (4, 0),
                         (1, 4), (2, 4), (3, 4), (4, 4),
                         (5, 1), (6, 1), (7, 1),
                         (5, 3), (6, 3), (7, 3)]
        fig_model_Q = pyplot.figure(figsize=(20, 10))
        fig_model_U = pyplot.figure(figsize=(20, 10))
        gs_model = gridspec.GridSpec(8, 6)
        for idx_dof, dof in enumerate(dofs):
            # --- Each DoF --- #
            fig_q = pyplot.figure()
            pyplot.plot(q_EKF[-1][dof, :].T, color='blue')
            pyplot.plot(q_OGE[-1][dof, :].T, color='red')
            pyplot.plot(q_OE[-1][dof, :].T, color='green')

            fig_q.suptitle(dofs_kinematics_name[idx_dof], size=9)
            ax_q = fig_q.axes[0]
            ax_q.set_xlabel("Aerial time (s)", size=9)
            ymin, ymax = ax_q.get_ylim()
            if not (ymin <= 0 <= ymax):
                if ymin > 0:
                    ax_q.set_ylim([0, ymax])
                elif ymax < 0:
                    ax_q.set_ylim([ymin, 0])
                    ax_q.xaxis.set_label_coords(0.5, -0.025)
            else:
                ax_q.xaxis.set_label_coords(0.5, -0.025)
            if idx_dof == 0:
                ax_q.set_ylabel(r"$\mathregular{m}$", size=9)
            else:
                ax_q.set_ylabel(r"$\mathregular{rad}$", size=9)
            ticks_loc = ax_q.get_xticks().tolist()
            ax_q.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            ax_q.set_xticklabels(["{:.1f}".format(tick*time_per_node) for tick in ticks_loc])
            ax_q.tick_params(axis="y", direction='in', labelsize=9)
            ax_q.spines['right'].set_visible(False)
            ax_q.spines['top'].set_visible(False)
            ax_q.spines['bottom'].set_position('zero')
            ax_q.legend([lm_kalman, lm_OGE, lm_OE], ['EKF', 'QT', 'MT'], handlelength=1)

            fig_q.tight_layout
            save_name = save_path + subject + '/Plots/Kinematics/' + os.path.splitext(c3d_name)[0] + '_' + dofs_save_name[idx_dof] + '.png'
            fig_q.savefig(save_name)

            fig_u = pyplot.figure()
            pyplot.plot(controls_EKF[-1]['tau'][dof, :].T, color='blue')
            pyplot.plot(controls_OGE[-1]['tau'][dof, :].T, color='red')
            pyplot.plot(controls_OE[-1]['tau'][dof, :].T, color='green')

            fig_u.suptitle(dofs_kinetics_name[idx_dof], size=9)
            ax_u = fig_u.axes[0]
            ax_u.set_xlabel("Aerial time (s)", size=9)
            ymin, ymax = ax_u.get_ylim()
            if not (ymin <= 0 <= ymax):
                if ymin > 0:
                    ax_u.set_ylim([0, ymax])
                elif ymax < 0:
                    ax_u.set_ylim([ymin, 0])
                    ax_u.xaxis.set_label_coords(0.5, -0.025)
            else:
                ax_u.xaxis.set_label_coords(0.5, -0.025)
            if idx_dof == 0:
                ax_u.set_ylabel(r"$\mathregular{N}$", size=9)
            else:
                ax_u.set_ylabel(r"$\mathregular{Nm}$", size=9)
            ticks_loc = ax_u.get_xticks().tolist()
            ax_u.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            ax_u.set_xticklabels(["{:.1f}".format(tick*time_per_node) for tick in ticks_loc])
            ax_u.tick_params(axis="y", direction='in', labelsize=9)
            ax_u.spines['right'].set_visible(False)
            ax_u.spines['top'].set_visible(False)
            ax_u.spines['bottom'].set_position('zero')
            ax_u.legend([lm_kalman, lm_OGE, lm_OE], ['EKF', 'QT', 'MT'], handlelength=1)

            fig_u.tight_layout
            save_name = save_path + subject + '/Plots/Kinetics/' + os.path.splitext(c3d_name)[0] + '_' + dofs_save_name[idx_dof] + '.png'
            fig_u.savefig(save_name)

            # --- Model subplots --- #
            gs_model_subplot = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_model[fig_model_dof[idx_dof][0],
                                                                                   fig_model_dof[idx_dof][1]:
                                                                                   fig_model_dof[idx_dof][1] + 2])

            ax_model_box = fig_model_Q.add_subplot(
                gs_model[fig_model_dof[idx_dof][0], fig_model_dof[idx_dof][1]:fig_model_dof[idx_dof][1] + 2])
            ax_model_box.tick_params(axis='both', which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
            ax_model_box.patch.set_alpha(0.3)

            ax_model_Q = fig_model_Q.add_subplot(gs_model_subplot[0])
            ax_model_Q.plot(q_EKF[-1][dof, :].T, color='blue')
            ax_model_Q.plot(q_OGE[-1][dof, :].T, color='red')
            ax_model_Q.plot(q_OE[-1][dof, :].T, color='green')

            ax_model_Q.set_title(dofs_kinematics_name[idx_dof], size=9)
            ax_model_Q.set_ylabel(r"$\mathregular{rad}$", size=9)
            ax_model_Q.tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
            ax_model_Q.tick_params(axis="y", direction='in', labelsize=9)
            ax_model_Q.spines['right'].set_visible(False)
            ax_model_Q.spines['top'].set_visible(False)
            ax_model_Q.spines['bottom'].set_visible(False)

            ax_model_box = fig_model_U.add_subplot(
                gs_model[fig_model_dof[idx_dof][0], fig_model_dof[idx_dof][1]:fig_model_dof[idx_dof][1] + 2])
            ax_model_box.tick_params(axis='both', which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
            ax_model_box.patch.set_alpha(0.3)

            ax_model_U = fig_model_U.add_subplot(gs_model_subplot[0])
            ax_model_U.plot(controls_EKF[-1]['tau'][dof, :].T, color='blue')
            ax_model_U.plot(controls_OGE[-1]['tau'][dof, :].T, color='red')
            ax_model_U.plot(controls_OE[-1]['tau'][dof, :].T, color='green')

            ax_model_U.set_title(dofs_kinetics_name[idx_dof], size=9)
            if idx_dof == 0:
                ax_model_U.set_ylabel(r"$\mathregular{N}$", size=9)
            else:
                ax_model_U.set_ylabel(r"$\mathregular{Nm}$", size=9)
            ax_model_U.tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
            ax_model_U.tick_params(axis="y", direction='in', labelsize=9)
            ax_model_U.spines['right'].set_visible(False)
            ax_model_U.spines['top'].set_visible(False)
            ax_model_U.spines['bottom'].set_visible(False)

        fig_model_Q.legend([lm_kalman, lm_OGE, lm_OE], ['EKF', 'QT', 'MT'])
        fig_model_U.legend([lm_kalman, lm_OGE, lm_OE], ['EKF', 'QT', 'MT'])

        fig_model_Q.subplots_adjust(wspace=0.3, hspace=0.3)
        fig_model_U.subplots_adjust(wspace=0.3, hspace=0.3)

        fig_model_Q.tight_layout
        save_name = save_path + subject + '/Plots/' + os.path.splitext(c3d_name)[0] + '_model_Q' + '.png'
        fig_model_Q.savefig(save_name)

        fig_model_U.tight_layout
        save_name = save_path + subject + '/Plots/' + os.path.splitext(c3d_name)[0] + '_model_U' + '.png'
        fig_model_U.savefig(save_name)

        pyplot.close('all')
