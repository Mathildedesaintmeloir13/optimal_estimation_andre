import biorbd
import numpy as np
import ezc3d
from scipy.io import loadmat
from casadi import MX, Function
import pickle
import os
from load_data_filename import load_data_filename
from adjust_number_shooting_points import adjust_number_shooting_points

from biorbd_optim import (
    OptimalControlProgram,
    Problem,
    Bounds,
    InitialConditions,
    ShowResult,
    Objective,
    InterpolationType,
    Data,
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


if __name__ == "__main__":
    # subject = 'DoCi'
    # subject = 'JeCh'
    # subject = 'BeLa'
    # subject = 'GuSe'
    subject = 'SaMi'
    number_shooting_points = 100
    trial = '821_822_3'

    data_filename = load_data_filename(subject, trial)
    model_name = data_filename['model']
    c3d_name = data_filename['c3d']
    frames = data_filename['frames']

    data_path = '/home/andre/Optimisation/data/' + subject + '/'
    model_path = data_path + 'Model/'
    biorbd_model = biorbd.Model(model_path + model_name)
    biorbd_model.setGravity(biorbd.Vector3d(0, 0, -9.80639))

    # --- Adjust number of shooting points --- #
    adjusted_number_shooting_points, step_size = adjust_number_shooting_points(number_shooting_points, frames)

    # --- Load --- #
    load_path = 'Solutions/'
    load_name = load_path + subject + '/' + os.path.splitext(c3d_name)[0] + "_optimal_gravity_N" + str(adjusted_number_shooting_points) + '_mixed_EKF'
    ocp, sol = OptimalControlProgram.load(load_name + '.bo')

    # --- Get the results --- #
    states, controls, params = Data.get_data(ocp, sol, get_parameters=True)
    angle = params["gravity_angle"]/np.pi*180
    print('Number of shooting points: ', number_shooting_points)
    print('Gravity rotation: ', angle)

    load_variables_name = load_name + ".pkl"
    with open(load_variables_name, 'rb') as handle:
        data = pickle.load(handle)

    rotating_gravity(biorbd_model, params["gravity_angle"].squeeze())
    print_gravity = Function('print_gravity', [], [biorbd_model.getGravity().to_mx()], [], ['gravity'])
    gravity = print_gravity()['gravity'].full().squeeze()
    print('Gravity: ', gravity)

    # save_variables_name = load_name + ".pkl"
    # with open(save_variables_name, 'wb') as handle:
    #     pickle.dump({'states': states, 'controls': controls, 'params': params, 'gravity': gravity},
    #                 handle, protocol=3)

    # --- Show results --- #
    # ShowResult(ocp, sol).animate(nb_frames=adjusted_number_shooting_points)
