import biorbd
import numpy as np
import ezc3d
from casadi import MX, Function, sum1
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
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'optimal_gravity_ocp'))
from load_data_filename import load_data_filename
from adjust_number_shooting_points import adjust_number_shooting_points
from reorder_markers import reorder_markers
from adjust_Kalman import correct_Kalman, check_Kalman, shift_by_2pi



def states_to_markers(biorbd_model, ocp, states):
    q = states['q']
    n_q = ocp.nlp[0]["model"].nbQ()
    n_mark = ocp.nlp[0]["model"].nbMarkers()
    n_frames = q.shape[1]

    markers = np.ndarray((3, n_mark, q.shape[1]))
    symbolic_states = MX.sym("x", n_q, 1)
    markers_func = Function(
        "markers_func", [symbolic_states], [biorbd_model.markers(symbolic_states)], ["q"], ["markers"]
    ).expand()
    for i in range(n_frames):
        markers[:, :, i] = markers_func(q[:, i])

    return markers


def states_to_markers_velocity(biorbd_model, ocp, states):
    q = states['q']
    qdot = states['q_dot']
    n_q = ocp.nlp[0]["model"].nbQ()
    n_qdot = ocp.nlp[0]["model"].nbQdot()
    n_mark = ocp.nlp[0]["model"].nbMarkers()
    n_frames = q.shape[1]

    markers_velocity = np.ndarray((3, n_mark, q.shape[1]))
    symbolic_q = MX.sym("q", n_q, 1)
    symbolic_qdot = MX.sym("qdot", n_qdot, 1)
    # This doesn't work for some mysterious reasons
    # markers_func = Function(
    #     "markers_func", [symbolic_q, symbolic_qdot], [biorbd_model.markersVelocity(symbolic_q, symbolic_qdot)], ["q", "q_dot"], ["markers_velocity"]
    # ).expand()
    for j in range(n_mark):
        markers_func = biorbd.to_casadi_func('markers_func', biorbd_model.markerVelocity, symbolic_q, symbolic_qdot, j)
        for i in range(n_frames):
            markers_velocity[:, j, i] = markers_func(q[:, i], qdot[:, i]).full().squeeze()

    return markers_velocity



def rot_to_EulerAngle(rot, seqR):
    conversion_function = Function(
        "conversion_function", [], [biorbd.Rotation.toEulerAngles(rot, seqR).to_mx()], [], []
    ).expand()

    return conversion_function()['o0'].full().squeeze()


def view(subject, trial, number_shooting_points):
    print('Subject: ', subject)
    print('Trial: ', trial)

    data_path = '/home/andre/Optimisation/data/' + subject + '/'
    model_path = data_path + 'Model/'
    c3d_path = data_path + 'Essai/'

    data_filename = load_data_filename(subject, trial)
    model_name = data_filename['model']
    c3d_name = data_filename['c3d']
    frames = data_filename['frames']

    biorbd_model = biorbd.Model(model_path + model_name)
    biorbd_model.setGravity(biorbd.Vector3d(0, 0, -9.80639))
    c3d = ezc3d.c3d(c3d_path + c3d_name)

    # --- Adjust number of shooting points --- #
    adjusted_number_shooting_points, step_size = adjust_number_shooting_points(number_shooting_points, frames)

    frequency = c3d['header']['points']['frame_rate']
    duration = len(frames) / frequency

    # --- Functions --- #
    q = MX.sym("Q", biorbd_model.nbQ(), 1)
    qdot = MX.sym("Qdot", biorbd_model.nbQdot(), 1)
    qddot = MX.sym("Qddot", biorbd_model.nbQddot(), 1)
    tau = MX.sym("Tau", biorbd_model.nbQddot(), 1)

    id = biorbd.to_casadi_func("id", biorbd_model.InverseDynamics, q, qdot, qddot)

    # --- Load --- #
    load_path = '/home/andre/BiorbdOptim/examples/optimal_estimation_ocp/Solutions/'
    load_name = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + '_stats_noOGE' + '.pkl'
    with open(load_name, 'rb') as handle:
        data = pickle.load(handle)

    q_OGE = data['q_OGE']
    q_OE = data['q_OE']

    states_OGE = data['states_OGE']
    states_OE = data['states_OE']

    controls_OGE = data['controls_OGE']
    controls_OE = data['controls_OE']

    markers_mocap = data['markers_mocap']
    frames = data['frames']
    step_size = data['step_size']

    load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
    load_variables_name = load_path + subject + '/Kalman/' + os.path.splitext(c3d_name)[0] + ".pkl"
    with open(load_variables_name, 'rb') as handle:
        kalman_states = pickle.load(handle)

    # q_kalman_biorbd = kalman_states['q'][:, ::step_size]
    # q_kalman_biorbd[12, :] -= np.pi
    # q_kalman_biorbd[13, :] -= np.pi
    # q_kalman_biorbd[13, :] *= -1
    #
    # tmp = q_kalman_biorbd[12, :]
    # q_kalman_biorbd[12, :] = q_kalman_biorbd[13, :]
    # q_kalman_biorbd[13, :] = tmp
    #
    # q_kalman_biorbd[14, :] += np.pi
    # q_kalman_biorbd[15, :] *= -1

    q_kalman_biorbd = shift_by_2pi(biorbd_model, kalman_states['q'][:, ::step_size])
    # q_kalman_biorbd = kalman_states['q'][:, ::step_size]
    qdot_kalman_biorbd = kalman_states['qd'][:, ::step_size]
    qddot_kalman_biorbd = kalman_states['qdd'][:, ::step_size]

    root_EKF_shifts = [(((q_kalman_biorbd[3, :] - states_OE['q'][3, :])/(2*np.pi)).mean().round(), 3),
                       (((q_kalman_biorbd[5, :] - states_OE['q'][5, :])/(2*np.pi)).mean().round(), 5)]
    for root_EKF_shift, idx in root_EKF_shifts:
        q_kalman_biorbd[idx, :] -= root_EKF_shift * (2*np.pi)

    # counter = 3
    # for segment_idx in range(biorbd_model.nbSegment()):
    #     seqR = biorbd_model.segment(segment_idx).seqR().to_string()
    #     prev_counter = counter
    #     counter += len(seqR)
    #     if segment_idx == 0:
    #         for node in range(adjusted_number_shooting_points + 1):
    #             rot = biorbd.Rotation.fromEulerAngles(q_kalman_biorbd[prev_counter:counter, node], seqR)
    #             q_kalman_biorbd[prev_counter+1, node] = rot_to_EulerAngle(rot, seqR)[1]
    #     else:
    #         for node in range(adjusted_number_shooting_points+1):
    #             rot = biorbd.Rotation.fromEulerAngles(q_kalman_biorbd[prev_counter:counter, node], seqR)
    #             q_kalman_biorbd[prev_counter:counter, node] = rot_to_EulerAngle(rot, seqR)

    states_kalman_biorbd = {'q': q_kalman_biorbd, 'q_dot': qdot_kalman_biorbd}
    controls_kalman_biorbd = {'tau': id(q_kalman_biorbd, qdot_kalman_biorbd, qddot_kalman_biorbd)}

    dofs = [range(0, 6), range(6, 9), range(9, 12),
            range(12, 14), range(14, 17), range(17, 19), range(19, 21),
            range(21, 23), range(23, 26), range(26, 28), range(28, 30),
            range(30, 33), range(33, 34), range(34, 36),
            range(36, 39), range(39, 40), range(40, 42),
            ]
    dofs_name = ['Pelvis', 'Thorax', 'Head',
                 'Right shoulder', 'Right arm', 'Right forearm', 'Right hand',
                 'Left shoulder', 'Left arm', 'Left forearm', 'Left hand',
                 'Right thigh', 'Right leg', 'Right foot',
                 'Left thigh', 'Left leg', 'Left foot',
                 ]
    # dofs = range(0, 6)
    fig_model_dof = [(4, 2), (1, 2), (0, 2),
                     (1, 0), (2, 0), (3, 0), (4, 0),
                     (1, 4), (2, 4), (3, 4), (4, 4),
                     (5, 1), (6, 1), (7, 1),
                     (5, 3), (6, 3), (7, 3)]
    fig_model_Q = pyplot.figure(figsize=(20, 10))
    # fig_model_U = pyplot.figure(figsize=(20, 10))
    gs_model = gridspec.GridSpec(8, 6)
    for idx_dof, dof in enumerate(dofs):
        # --- Model subplots --- #
        gs_model_subplot = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_model[fig_model_dof[idx_dof][0], fig_model_dof[idx_dof][1]:fig_model_dof[idx_dof][1]+2])

        ax_model_box = fig_model_Q.add_subplot(gs_model[fig_model_dof[idx_dof][0], fig_model_dof[idx_dof][1]:fig_model_dof[idx_dof][1] + 2])
        ax_model_box.tick_params(axis='both', which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
        ax_model_box.patch.set_alpha(0.3)

        ax_model_Q = fig_model_Q.add_subplot(gs_model_subplot[0])
        ax_model_Q.plot(states_kalman_biorbd['q'][dof, :].T, color='blue')
        ax_model_Q.plot(states_OGE['q'][dof, :].T, color='red')
        ax_model_Q.plot(states_OE['q'][dof, :].T, color='green')

        ax_model_Q.set_title(dofs_name[idx_dof], size=9)
        ax_model_Q.set_ylabel(r"$\mathregular{rad}$", size=9)
        ax_model_Q.tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
        ax_model_Q.tick_params(axis="y", direction='in', labelsize=9)
        ax_model_Q.spines['right'].set_visible(False)
        ax_model_Q.spines['top'].set_visible(False)
        ax_model_Q.spines['bottom'].set_visible(False)

        # ax_model_box = fig_model_U.add_subplot(gs_model[fig_model_dof[idx_dof][0], fig_model_dof[idx_dof][1]:fig_model_dof[idx_dof][1] + 2])
        # ax_model_box.tick_params(axis='both', which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
        # ax_model_box.patch.set_alpha(0.3)
        #
        # ax_model_U = fig_model_U.add_subplot(gs_model_subplot[0])
        # ax_model_U.plot(controls_kalman_biorbd['tau'][dof, :].T, color='blue')
        # ax_model_U.plot(controls_OGE['tau'][dof, :].T, color='red')
        # ax_model_U.plot(controls_OE['tau'][dof, :].T, color='green')
        #
        # ax_model_U.set_title(dofs_name[idx_dof], size=9)
        # ax_model_U.set_ylabel(r"$\mathregular{rad}$", size=9)
        # ax_model_U.tick_params(axis="x", direction='in', which='both', bottom=False, top=False, labelbottom=False)
        # ax_model_U.tick_params(axis="y", direction='in', labelsize=9)
        # ax_model_U.spines['right'].set_visible(False)
        # ax_model_U.spines['top'].set_visible(False)
        # ax_model_U.spines['bottom'].set_visible(False)

    lm_kalman = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
    lm_OGE = Line2D([0, 1], [0, 1], linestyle='-', color='red')
    lm_OE = Line2D([0, 1], [0, 1], linestyle='-', color='green')
    fig_model_Q.legend([lm_kalman, lm_OGE, lm_OE], ['Kalman', 'OGE', 'OE'])
    # fig_model_U.legend([lm_kalman, lm_OGE, lm_OE], ['Kalman', 'OGE', 'OE'])

    fig_model_Q.subplots_adjust(wspace=0.3, hspace=0.3)
    # fig_model_U.subplots_adjust(wspace=0.3, hspace=0.3)

    pyplot.show()


if __name__ == "__main__":
    # subject = 'DoCi'
    subject = 'JeCh'
    # subject = 'BeLa'
    # subject = 'GuSe'
    # subject = 'SaMi'
    number_shooting_points = 100
    trial = '821_1'

    view(subject, trial, number_shooting_points)