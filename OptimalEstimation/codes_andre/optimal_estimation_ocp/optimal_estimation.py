import biorbd
import numpy as np
import ezc3d
from scipy.io import loadmat
import time
from casadi import MX, Function
import pickle
import os
import sys
sys.path.insert(1, '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp')
from load_data_filename import load_data_filename
from x_bounds import x_bounds
from adjust_number_shooting_points import adjust_number_shooting_points
from reorder_markers import reorder_markers
from adjust_Kalman import shift_by_2pi

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


def rotating_gravity(biorbd_model, value):
    # The pre dynamics function is called right before defining the dynamics of the system. If one wants to
    # modify the dynamics (e.g. optimize the gravity in this case), then this function is the proper way to do it
    # `biorbd_model` and `value` are mandatory. The former is the actual model to modify, the latter is the casadi.MX
    # used to modify it,  the size of which decribed by the value `size` in the parameter definition.
    # The rest of the parameter are defined by the user in the parameter
    gravity = biorbd_model.getGravity()
    gravity.applyRT(
        biorbd.RotoTrans.combineRotAndTrans(biorbd.Rotation.fromEulerAngles(value, 'zx'), biorbd.Vector3d()))
    biorbd_model.setGravity(gravity)


def inverse_dynamics(biorbd_model, q_ref, qd_ref, qdd_ref):
    q = MX.sym("Q", biorbd_model.nbQ(), 1)
    qdot = MX.sym("Qdot", biorbd_model.nbQdot(), 1)
    qddot = MX.sym("Qddot", biorbd_model.nbQddot(), 1)
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
    # tau_init = np.zeros(tau_init.shape)
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
        nb_threads=4,
    )


if __name__ == "__main__":
    start = time.time()
    # subject = 'DoCi'
    # subject = 'JeCh'
    # subject = 'BeLa'
    # subject = 'GuSe'
    subject = 'SaMi'
    number_shooting_points = 100
    trial = '821_822_3'
    print('Subject: ', subject, ', Trial: ', trial)

    trial_needing_min_torque_diff = {
                                     'JeCh': '833_4',
                                     # 'SaMi': ['821_822_2',
                                     #          '821_contact_2',
                                     #          '821_seul_3', '821_seul_4']
                                     }
    min_torque_diff = False
    if subject in trial_needing_min_torque_diff.keys():
        if trial in trial_needing_min_torque_diff[subject]:
            min_torque_diff = True

    data_path = '/home/andre/Optimisation/data/' + subject + '/'
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

    optimal_gravity_filename = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_optimal_gravity_N' + str(adjusted_number_shooting_points) + '_mixed_EKF' + ".bo"
    ocp_optimal_gravity, sol_optimal_gravity = OptimalControlProgram.load(optimal_gravity_filename)
    states_optimal_gravity, controls_optimal_gravity, params_optimal_gravity_part = Data.get_data(ocp_optimal_gravity, sol_optimal_gravity, get_parameters=True)

    optimal_gravity_filename_full = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_optimal_gravity_N' + str(frames.stop - frames.start - 1) + '_mixed_EKF' + ".bo"
    ocp_optimal_gravity_full, sol_optimal_gravity_full = OptimalControlProgram.load(optimal_gravity_filename_full)
    states_optimal_gravity_full, controls_optimal_gravity_full, params_optimal_gravity_full = Data.get_data(ocp_optimal_gravity_full, sol_optimal_gravity_full, get_parameters=True)

    params_optimal_gravity = params_optimal_gravity_full


    load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
    load_variables_name = load_path + subject + '/Kalm' \
                                                'an/' + os.path.splitext(c3d_name)[0] + ".pkl"
    with open(load_variables_name, 'rb') as handle:
        kalman_states = pickle.load(handle)

    q_kalman_biorbd = shift_by_2pi(biorbd_model, kalman_states['q'][:, ::step_size])
    # q_kalman_biorbd = kalman_states['q'][:, ::step_size]
    qdot_kalman_biorbd = kalman_states['qd'][:, ::step_size]
    qddot_kalman_biorbd = kalman_states['qdd'][:, ::step_size]
    tau_kalman_biorbd = inverse_dynamics(biorbd_model, q_kalman_biorbd, qdot_kalman_biorbd, qddot_kalman_biorbd).full()

    q_ref = states_optimal_gravity['q']
    qdot_ref = states_optimal_gravity['q_dot']
    tau_ref = controls_optimal_gravity['tau'][:, :-1]
    # q_ref = q_kalman_biorbd
    # qdot_ref = qdot_kalman_biorbd
    # tau_ref = tau_kalman_biorbd

    angle = params_optimal_gravity["gravity_angle"].squeeze()
    rotating_gravity(biorbd_model, angle)
    # print_gravity = Function('print_gravity', [], [biorbd_model.getGravity().to_mx()], [], ['gravity'])
    # print(print_gravity()['gravity'].full())

    xmin, xmax = x_bounds(biorbd_model)

    markers_reordered, _ = reorder_markers(biorbd_model, c3d, frames, step_size)

    # markers_rotated = np.zeros(markers.shape)
    # for frame in range(markers.shape[2]):
    #     markers_rotated[..., frame] = refential_matrix(subject).T.dot(markers_reordered[..., frame])
    markers_rotated = markers_reordered

    ocp = prepare_ocp(
        biorbd_model=biorbd_model, final_time=duration, number_shooting_points=adjusted_number_shooting_points,
        markers_ref=markers_rotated, q_init=q_ref, qdot_init=qdot_ref, tau_init=tau_ref,
        xmin=xmin, xmax=xmax, min_torque_diff=min_torque_diff,
    )

    # --- Solve the program --- #
    options = {"max_iter": 3000, "tol": 1e-4, "constr_viol_tol": 1e-2, "linear_solver": "ma57"}
    sol = ocp.solve(solver=Solver.IPOPT, solver_options=options, show_online_optim=False)

    # --- Get the results --- #
    states, controls = Data.get_data(ocp, sol)

    # --- Save --- #
    save_path = 'Solutions/'
    save_name = save_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points)
    ocp.save(sol, save_name + ".bo")

    get_gravity = Function('print_gravity', [], [biorbd_model.getGravity().to_mx()], [], ['gravity'])
    gravity = get_gravity()['gravity'].full().squeeze()

    save_variables_name = save_name + ".pkl"
    with open(save_variables_name, 'wb') as handle:
        pickle.dump({'mocap': markers_rotated, 'duration': duration, 'frames': frames, 'step_size': step_size,
                     'states': states, 'controls': controls, 'gravity': gravity, 'gravity_angle': angle},
                    handle, protocol=3)

    stop = time.time()
    print('Number of shooting points: ', adjusted_number_shooting_points)
    print(stop - start)

    # --- Show results --- #
    # ShowResult(ocp, sol).animate(nb_frames=adjusted_number_shooting_points)
