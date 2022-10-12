import numpy as np
import scipy.optimize
import biorbd
# import BiorbdViz
import ezc3d
import os
import pickle
from x_bounds import x_bounds
from load_data_filename import load_data_filename
from reorder_markers import reorder_markers
from adjust_Kalman import shift_by_2pi


def rot_to_EulerAngle(rot, seqR):
    return biorbd.Rotation.toEulerAngles(rot, seqR).to_array()

# subject = 'DoCi'
subject = 'JeCh'
# subject = 'BeLa'
# subject = 'GuSe'
# subject = 'SaMi'
trial = '833_2'
backward_frames = 15
forward_frames = 0
step_size = 3
shift = 0

scaled_backward_frames = int(backward_frames/step_size)
scaled_forward_frames = int(forward_frames/step_size)

data_path = '/home/andre/Optimisation/data/' + subject + '/'
model_path = data_path + 'Model/'
c3d_path = data_path + 'Essai/'
kalman_path = data_path + 'Q/'

data_filename = load_data_filename(subject, trial)
model_name = data_filename['model']
c3d_name = data_filename['c3d']
frames = data_filename['frames']

frames = range(frames.start-backward_frames, frames.stop, frames.step)

biorbd_model = biorbd.Model(model_path + model_name)
c3d = ezc3d.c3d(c3d_path + c3d_name)

markers_reordered, _ = reorder_markers(biorbd_model, c3d, frames)

markers_reordered[np.isnan(markers_reordered)] = 0.0  # Remove NaN
markers_reordered = markers_reordered[:, :, ::step_size]

# Dispatch markers in biorbd structure so EKF can use it
markersOverFrames = []
for i in range(markers_reordered.shape[2]):
    markersOverFrames.append([biorbd.NodeSegment(m) for m in markers_reordered[:, :, i].T])

# Create a Kalman filter structure
frequency = c3d['header']['points']['frame_rate']  # Hz
# params = biorbd.KalmanParam(frequency=frequency, noiseFactor=1e-10, errorFactor=1e-5)
params = biorbd.KalmanParam(frequency=frequency/step_size)
kalman = biorbd.KalmanReconsMarkers(biorbd_model, params)

# Find an initial state to initialize Kalman
def distance_markers(q, *args):
    distances_ignoring_missing_markers = 0
    markers_estimated = np.array([marker.to_array() for marker in biorbd_model.markers(q)]).T
    for i in range(markers_reordered.shape[1]):
        if markers_reordered[0, i, scaled_forward_frames] != 0:
            distances_ignoring_missing_markers += np.sum((markers_estimated[:, i] - markers_reordered[:, i, shift+scaled_forward_frames]) ** 2)
    return distances_ignoring_missing_markers

Q_init = np.zeros(biorbd_model.nbQ())
bounds = x_bounds(biorbd_model)
bounds_pair = [(bounds[0][i], bounds[1][i]) for i in range(biorbd_model.nbQ())]
res = scipy.optimize.minimize(distance_markers, Q_init, bounds=bounds_pair)
Q_init = res.x

# load_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
# load_variables_name = load_path + subject + '/Kalman/' + os.path.splitext('Sa_821_seul_2.c3d')[0] + "_third_node.pkl"
# with open(load_variables_name, 'rb') as handle:
#     kalman_states = pickle.load(handle)
# Q_init = kalman_states['q'][:, shift]
# Qd_init = kalman_states['qd'][:, shift]
# Qdd_init = kalman_states['qdd'][:, shift]

kalman.setInitState(Q_init, np.zeros(biorbd_model.nbQ()), np.zeros(biorbd_model.nbQ()))
# kalman.setInitState(Q_init, Qd_init, Qdd_init)

Q = biorbd.GeneralizedCoordinates(biorbd_model)
Qdot = biorbd.GeneralizedVelocity(biorbd_model)
Qddot = biorbd.GeneralizedAcceleration(biorbd_model)

q_recons = np.ndarray((biorbd_model.nbQ(), len(markersOverFrames)))
qd_recons = np.ndarray((biorbd_model.nbQdot(), len(markersOverFrames)))
qdd_recons = np.ndarray((biorbd_model.nbQddot(), len(markersOverFrames)))
for i, targetMarkers in enumerate(markersOverFrames[shift:]):
    i += shift
    kalman.reconstructFrame(biorbd_model, targetMarkers, Q, Qdot, Qddot)
    q_recons[:, i] = Q.to_array()
    qd_recons[:, i] = Qdot.to_array()
    qdd_recons[:, i] = Qddot.to_array()

kalman = biorbd.KalmanReconsMarkers(biorbd_model, params)
kalman.setInitState(Q_init, np.zeros(biorbd_model.nbQ()), np.zeros(biorbd_model.nbQ()))
for i, targetMarkers in enumerate(markersOverFrames[:shift][::-1]):
    j = shift - i - 1
    kalman.reconstructFrame(biorbd_model, targetMarkers, Q, Qdot, Qddot)
    q_recons[:, j] = Q.to_array()
    qd_recons[:, j] = Qdot.to_array()
    qdd_recons[:, j] = Qddot.to_array()

q_recons[:, :shift] = shift_by_2pi(biorbd_model, q_recons[:, :shift])
q_recons[:, shift:] = shift_by_2pi(biorbd_model, q_recons[:, shift:])

counter = 3
for segment_idx in range(biorbd_model.nbSegment()):
    seqR = biorbd_model.segment(segment_idx).seqR().to_string()
    prev_counter = counter
    counter += len(seqR)
    if segment_idx == 0:
        for node in range(q_recons.shape[1]):
            rot = biorbd.Rotation.fromEulerAngles(q_recons[prev_counter:counter, node], seqR)
            q_recons[prev_counter+1, node] = rot_to_EulerAngle(rot, seqR)[1]
    else:
        for node in range(q_recons.shape[1]):
            rot = biorbd.Rotation.fromEulerAngles(q_recons[prev_counter:counter, node], seqR)
            q_recons[prev_counter:counter, node] = rot_to_EulerAngle(rot, seqR)

save_path = '/home/andre/BiorbdOptim/examples/optimal_gravity_ocp/Solutions/'
save_variables_name = save_path + subject + '/Kalman/' + os.path.splitext(c3d_name)[0] + "_third_node.pkl"
with open(save_variables_name, 'wb') as handle:
    pickle.dump({'q': q_recons[:, scaled_backward_frames:], 'qd': qd_recons[:, scaled_backward_frames:], 'qdd': qdd_recons[:, scaled_backward_frames:]},
                handle, protocol=3)

# Animate the results if biorbd viz is installed
# b = BiorbdViz.BiorbdViz(loaded_model=biorbd_model)
# b.load_movement(q_recons)
# b.exec()