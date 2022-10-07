import biorbd
import numpy as np
import ezc3d
from scipy.io import loadmat
import time
import casadi as cas
import pickle
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(1, '/home/lim/Documents/Stage_Mathilde/Mathidle_Andre/Andre_bioptim_gravity_optimization/BiorbdOptim-gravity_optimization/examples/optimal_gravity_ocp/')
from load_data_filename import load_data_filename
from x_bounds import x_bounds
from adjust_number_shooting_points import adjust_number_shooting_points
from reorder_markers import reorder_markers
from adjust_Kalman import shift_by_2pi

sys.path.insert(1, '/home/lim/Documents/Stage_Mathilde/Mathidle_Andre/Andre_bioptim_gravity_optimization/BiorbdOptim-gravity_optimization/')
from biorbd_optim import (
    OptimalControlProgram,
    ObjectiveList,
    Objective,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    Bounds,
    InitialConditionsList,
    InitialConditions,
    ShowResult,
    InterpolationType,
    Data,
    Solver,
)


#
# def root_explicit_dynamic(states, controls, parameters, nlp,):
#     DynamicsFunctions.apply_parameters(parameters, nlp)
#     q = DynamicsFunctions.get(nlp.states["q"], states)
#     qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
#     nb_root = nlp.model.nbRoot()
#     qddot_joints = DynamicsFunctions.get(nlp.controls["qddot_joint"], controls)
#     mass_matrix_nl_effects = nlp.model.InverseDynamics(q, qdot, cas.vertcat(cas.MX.zeros((nb_root, 1)), qddot_joints)).to_mx()[:6]
#     mass_matrix = nlp.model.massMatrix(q).to_mx()
#     mass_matrix_nl_effects_func = cas.Function("mass_matrix_nl_effects_func", [q, qdot, qddot_joints], [mass_matrix_nl_effects[:nb_root]]).expand()
#     M_66 = mass_matrix[:nb_root, :nb_root]
#     M_66_func = cas.Function("M66_func", [q], [M_66]).expand()
#     qddot_root = cas.solve(M_66_func(q), -mass_matrix_nl_effects_func(q, qdot, qddot_joints), "ldl")
#     return qdot, cas.vertcat(qddot_root, qddot_joints)
#
#
# def custom_configure_root_explicit(ocp, nlp):
#     ConfigureProblem.configure_q(nlp, as_states=True, as_controls=False)
#     ConfigureProblem.configure_qdot(nlp, as_states=True, as_controls=False)
#     configure_qddot_joint(nlp, as_states=False, as_controls=True)
#     ConfigureProblem.configure_dynamics_function(ocp, nlp, root_explicit_dynamic, expand=False)
#
#
# def configure_qddot_joint(nlp, as_states, as_controls):
#     nb_root = nlp.model.nbRoot()
#     name_qddot_joint = [str(i + nb_root) for i in range(nlp.model.nbQddot() - nb_root)]
#     ConfigureProblem.configure_new_variable("qddot_joint", name_qddot_joint, nlp, as_states, as_controls)
#
# def dynamics_root(m, X, Qddot_J):
#     Q = X[:m.nbQ()]
#     Qdot = X[m.nbQ():]
#     Qddot = np.hstack((np.zeros((6,)), Qddot_J)) #qddot2
#     NLEffects = m.InverseDynamics(Q, Qdot, Qddot).to_array()
#     mass_matrix = m.massMatrix(Q).to_array()
#     Qddot_R = np.linalg.solve(mass_matrix[:6, :6], -NLEffects[:6])
#     Xdot = np.hstack((Qdot, Qddot_R, Qddot_J))
#     return Xdot
#
#
# # def custom_func_track_markers(all_pn):
# #     markers = BiorbdInterface.mx_to_cx("markers", all_pn.nlp.model.markers, all_pn.nlp.states["q"])
# #     return val
#
# def plot_dynamic_variables(m, q, qdot, qddot, time, root=True, save_name='oups.png'):
#     Q_sym = cas.MX.sym('Q_sym', m.nbQ())
#     Qdot_sym = cas.MX.sym('Qdot_sym', m.nbQ())
#     Qddot_sym = cas.MX.sym('Qddot_sym', m.nbQ())
#     CoM_func = cas.Function('CoM_func', [Q_sym], [m.CoM(Q_sym).to_mx()])
#     CoM_dot_func = cas.Function('CoM_dot_func', [Q_sym, Qdot_sym], [m.CoMdot(Q_sym, Qdot_sym).to_mx()])
#     CoM_ddot_func = cas.Function('CoM_ddot_func', [Q_sym, Qdot_sym,  Qddot_sym], [m.CoMddot(Q_sym, Qdot_sym, Qddot_sym).to_mx()])
#     AngMom_func = cas.Function('AngMom_func', [Q_sym, Qdot_sym], [m.CalcAngularMomentum(Q_sym, Qdot_sym, True).to_mx()])
#     Mass_func = cas.Function('Mass_func', [], [m.mass().to_mx()])
#
#     N = np.shape(q)[1]
#     CoM = np.zeros((3, N))
#     CoM_dot = np.zeros((3, N))
#     CoM_ddot = np.zeros((3, N))
#     AngMom = np.zeros((3, N))
#     LinMom = np.zeros((3, N))
#     for i in range(N):
#         CoM[:, i] = np.reshape(CoM_func(q[:, i]), (3,))
#         CoM_dot[:, i] = np.reshape(CoM_dot_func(q[:, i], qdot[:, i]), (3,))
#         if root:
#             CoM_ddot[:, i] = np.reshape(CoM_ddot_func(q[:, i], qdot[:, i], qddot[:, i]), (3,))
#         AngMom[:, i] = np.reshape(AngMom_func(q[:, i], qdot[:, i]), (3,))
#         LinMom[:, i] = np.reshape(Mass_func()['o0'] * CoM_dot[:, i], (3,))
#
#     AngMom_norm = np.linalg.norm(AngMom, axis=0)
#     LinMom_norm = np.linalg.norm(LinMom, axis=0)
#
#     if root:
#         CoM_dot_0 = np.zeros((3, N))
#         c0 = CoM_dot[0:2, 0]
#         CoM_dot_0[0:2, :] = np.repeat(c0[:, np.newaxis], N, axis=1)
#         CoM_dot_0[2, :] = CoM_ddot[2, 0] * time + CoM_dot[2, 0]
#
#         LinMom_0 = np.linalg.norm(Mass_func()['o0'] * CoM_dot_0, axis=0)
#         a0 = AngMom_func(q[:, 0], qdot[:, 0])
#         AngMom_0 = np.linalg.norm(np.repeat(a0, N, axis=1), axis=0)
#
#
#
#     fig, axs = plt.subplots(3)
#     for i in range(3):
#         axs[i].plot(CoM[i, :], '-r', label='CoM')
#         if root:
#             axs[i].plot(CoM_dot_0[i, :], '--g', label='CoM dot supposé')
#         axs[i].plot(CoM_dot[i, :], '-g', label='CoM dot')
#     plt.legend()
#     plt.savefig(save_name[:-4] + "_CoMdot" + ".png")
#     # plt.show()
#
#     plt.figure()
#     plt.plot(AngMom_norm, '-m', label='Angular momentum')
#     plt.plot(LinMom_norm, '-b', label='Linear momentum')
#     if root:
#         plt.plot(AngMom_0, '--m', label='Angular momentum')
#         plt.plot(LinMom_0, '--b', label='Linear momentum')
#     plt.legend()
#     plt.savefig(save_name[:-4] + "_momentum" + ".png")
#     # plt.show()
#

def plot_markers_diff(m, markers, q, save_name):

    import seaborn as sns

    def markers_fun(m, nb_markers):
        Q_sym = cas.MX.sym('Q_sym', m.nbQ())
        Markers_sym = cas.MX(3, nb_markers)
        for i in range(nb_markers):
            Markers_sym[:, i] = m.markers(Q_sym)[i]
        markers_func = cas.Function('makers_func', [Q_sym], [Markers_sym])
        return markers_func

    nb_markers = m.nbMarkers()
    N = markers.shape[2]
    colors = sns.color_palette("gist_rainbow", nb_markers)
    markers_q = np.zeros((3, nb_markers, N))
    markers_func = markers_fun(m, nb_markers)
    for i in range(N):
        markers_q[:, :, i] = markers_func(q[:, i])

    diff = np.linalg.norm(markers_q - markers, axis=0)
    fig = plt.figure()
    for i in range(nb_markers):
        plt.plot(diff[i, :], color=colors[i], label=f'{m.markerNames()[i].to_string()}')
    plt.legend(ncol=10, bbox_to_anchor=(0.1, 1.1))

    fig.subplots_adjust(top=0.6)
    plt.savefig(save_name)
    # plt.show()

    track_markers_start = np.nansum((markers_q[:, :, 0] - markers[:, :nb_markers, 0]) ** 2)
    track_markers_all = np.nansum((markers_q - markers[:, :nb_markers, :]) ** 2)
    track_markers_end = np.nansum((markers_q[:, :, -1] - markers[:, :nb_markers, -1]) ** 2)
    print(save_name)
    # print(f"START: {track_markers_start}")
    # print(f"ALL: {track_markers_all}")
    # print(f"END: {track_markers_end}")



def inverse_dynamics(biorbd_model, q_ref, qd_ref, qdd_ref):
    q = cas.MX.sym("Q", biorbd_model.nbQ(), 1)
    qdot = cas.MX.sym("Qdot", biorbd_model.nbQdot(), 1)
    qddot = cas.MX.sym("Qddot", biorbd_model.nbQddot(), 1)
    id = biorbd.to_casadi_func("id", biorbd_model.InverseDynamics, q, qdot, qddot)

    return id(q_ref, qd_ref, qdd_ref)[:, :-1]


def prepare_ocp(biorbd_model, final_time, number_shooting_points, markers_ref, q_init, qdot_init, tau_init, xmin, xmax, min_torque_diff=False):
    # --- Options --- #
    torque_min, torque_max = -300, 300
    n_q = biorbd_model.nbQ()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Add objective functions
    state_ref = np.concatenate((q_init, qdot_init))
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=1, target=markers_ref)
    objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=1e-5, target=state_ref)
    objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=1e-5, states_idx=range(6, n_q))
    control_weight_segments = [0   , 0   , 0   ,  # pelvis trans
                               0   , 0   , 0   ,  # pelvis rot
                               1e-7, 1e-7, 1e-6,  # thorax
                               1e-5, 1e-5, 1e-4,  # head
                               1e-5, 1e-4,        # right shoulder
                               1e-5, 1e-5, 1e-4,  # right arm
                               1e-4, 1e-3,        # right forearm
                               1e-4, 1e-3,        # right hand
                               1e-5, 1e-4,        # left shoulder
                               1e-5, 1e-5, 1e-4,  # left arm
                               1e-4, 1e-3,        # left forearm
                               1e-4, 1e-3,        # left hand
                               1e-7, 1e-7, 1e-6,  # right thigh
                               1e-6,              # right leg
                               1e-4, 1e-3,        # right foot
                               1e-7, 1e-7, 1e-6,  # left thigh
                               1e-6,              # left leg
                               1e-4, 1e-3,        # left foot
                               ]
    for idx in range(n_tau):
      objective_functions.add(Objective.Lagrange.TRACK_TORQUE, weight=control_weight_segments[idx], target=tau_init, controls_idx=idx)
      objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=control_weight_segments[idx], controls_idx=idx)
    if min_torque_diff:
        objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=1e-5)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN)

    # Path constraint
    X_bounds = BoundsList()
    X_bounds.add(Bounds(min_bound=xmin, max_bound=xmax))

    # Initial guess
    X_init = InitialConditionsList()
    # q_init = np.zeros(q_init.shape)
    # qdot_init = np.zeros(qdot_init.shape)
    X_init.add(np.concatenate([q_init, qdot_init]), interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    U_bounds = BoundsList()
    U_bounds.add(Bounds(min_bound=[torque_min] * n_tau, max_bound=[torque_max] * n_tau))
    U_bounds[0].min[:6, :] = 0
    U_bounds[0].max[:6, :] = 0

    U_init = InitialConditionsList()
    tau_init = np.zeros(tau_init.shape)
    U_init.add(tau_init, interpolation=InterpolationType.EACH_FRAME)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        # constraints,
        nb_integration_steps=4,
        nb_threads=8,
    )


if __name__ == "__main__":

    GENERATE_OPTIM = True  # False
    GENERATE_GRAPHS = True

    subject = 'DoCi'
    # subject = 'JeCh'
    # subject = 'BeLa'
    # subject = 'GuSe'
    # subject = 'SaMi'
    number_shooting_points = 100
    trial = '44_3'
    print('Subject: ', subject, ', Trial: ', trial)

    trial_needing_min_torque_diff = {
                                     'DoCi': ['44_2'],
                                     'BeLa': ['44_2', '44_3'],
                                     'SaMi': [
                                              '821_822_2', '821_822_3',
                                              '821_contact_1', '821_contact_2', '821_contact_3'
                                             ],
                                     'JeCh': ['821_1', '833_5']
                                     }
    min_torque_diff = False
    if subject in trial_needing_min_torque_diff.keys():
        if trial in trial_needing_min_torque_diff[subject]:
            min_torque_diff = True

    data_path = '/home/lim/Documents/Stage_Mathilde/Mathidle_Andre/OptimalEstimation/Andre_data/'
    model_path = data_path + 'Model/'
    c3d_path = data_path + 'Essai/'
    kalman_path = data_path + 'Q/'

    data_filename = load_data_filename(subject, trial)
    model_name = data_filename['model']
    c3d_name = data_filename['c3d']
    q_name = data_filename['q']
    qd_name = data_filename['qd']
    qdd_name = data_filename['qdd']
    frames = data_filename['frames']

    biorbd_model = biorbd.Model(model_path + model_name)
    c3d = ezc3d.c3d(c3d_path + c3d_name)

    biorbd_model.setGravity(biorbd.Vector3d(0, 0, -9.80639))

    # --- Adjust number of shooting points --- #
    adjusted_number_shooting_points, step_size = adjust_number_shooting_points(number_shooting_points, frames)
    print('Adjusted number of shooting points: ', adjusted_number_shooting_points)
    print('Node step size: ', step_size)

    frequency = c3d['header']['points']['frame_rate']
    duration = len(frames) / frequency

    # optimal_gravity_filename = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_optimal_gravity_N' + str(adjusted_number_shooting_points) + '_mixed_EKF_noOGE' + ".bo"
    # ocp_optimal_gravity, sol_optimal_gravity = OptimalControlProgram.load(optimal_gravity_filename)
    # states_optimal_gravity, controls_optimal_gravity, params_optimal_gravity_part = Data.get_data(ocp_optimal_gravity, sol_optimal_gravity, get_parameters=True)

    load_variables_name = data_path + '/Kalman/' + os.path.splitext(c3d_name)[0] + ".pkl"
    with open(load_variables_name, 'rb') as handle:
        kalman_states = pickle.load(handle)

    q_kalman_biorbd = shift_by_2pi(biorbd_model, kalman_states['q'][:, ::step_size])
    # q_kalman_biorbd = kalman_states['q'][:, ::step_size]
    qdot_kalman_biorbd = kalman_states['qd'][:, ::step_size]
    qddot_kalman_biorbd = kalman_states['qdd'][:, ::step_size]
    tau_kalman_biorbd = inverse_dynamics(biorbd_model, q_kalman_biorbd, qdot_kalman_biorbd, qddot_kalman_biorbd).full()

    # q_ref = states_optimal_gravity['q']
    # qdot_ref = states_optimal_gravity['q_dot']
    # tau_ref = controls_optimal_gravity['tau'][:, :-1]
    q_ref = q_kalman_biorbd
    qdot_ref = qdot_kalman_biorbd
    tau_ref = tau_kalman_biorbd

    xmin, xmax = x_bounds(biorbd_model)

    markers_reordered, _ = reorder_markers(biorbd_model, c3d, frames, step_size)

    # markers_rotated = np.zeros(markers.shape)
    # for frame in range(markers.shape[2]):
    #     markers_rotated[..., frame] = refential_matrix(subject).T.dot(markers_reordered[..., frame])
    markers_rotated = markers_reordered

    if GENERATE_GRAPHS:
        figure_save_path =  "/home/lim/Documents/Stage_Mathilde/Mathidle_Andre/OptimalEstimation/figures/Figures_vieuxCodesAndre/"
        plot_markers_diff(biorbd_model, markers_reordered, q_ref, figure_save_path + os.path.splitext(c3d_name)[0] + "_Kalman")

    save_path = 'Solutions/'
    save_name = save_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(
        adjusted_number_shooting_points) + '_noOGE'
    save_variables_name = save_name + ".pkl"

    if GENERATE_OPTIM:
        ocp = prepare_ocp(
            biorbd_model=biorbd_model, final_time=duration, number_shooting_points=adjusted_number_shooting_points,
            markers_ref=markers_rotated, q_init=q_ref, qdot_init=qdot_ref, tau_init=tau_ref,
            xmin=xmin, xmax=xmax, min_torque_diff=min_torque_diff,
        )

        # --- Solve the program --- #
        start = time.time()
        options = {"max_iter": 3000, "tol": 1e-4, "constr_viol_tol": 1e-2, "linear_solver": "ma57"}
        sol = ocp.solve(solver=Solver.IPOPT, solver_options=options, show_online_optim=False)
        stop = time.time()
        print('Number of shooting points: ', adjusted_number_shooting_points)
        print(stop - start)

        # --- Get the results --- #
        states, controls = Data.get_data(ocp, sol)

        # --- Save --- #
        ocp.save(sol, save_name + ".bo")

        get_gravity = cas.Function('print_gravity', [], [biorbd_model.getGravity().to_mx()], [], ['gravity'])
        gravity = get_gravity()['gravity'].full().squeeze()

        with open(save_variables_name, 'wb') as handle:
            pickle.dump({'mocap': markers_rotated, 'duration': duration, 'frames': frames, 'step_size': step_size,
                         'states': states, 'controls': controls, 'gravity': gravity},
                        handle, protocol=3)

    else:
        file = open(save_variables_name, 'rb')
        data = pickle.load(file)
        file.close()

        states = data["states"]
        controls = data["controls"]
        gravity = data["gravity"]

    # --- Show results --- #
    # ShowResult(ocp, sol).animate(nb_frames=adjusted_number_shooting_points)


    # import bioviz
    # b = bioviz.Viz(model_path + model_name)
    # b.load_movement(states["q"])
    # b.exec()

    if GENERATE_GRAPHS:
        plot_markers_diff(biorbd_model, markers_reordered, states["q"], figure_save_path + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_noOGE')


