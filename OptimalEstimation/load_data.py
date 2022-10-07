
import matplotlib.pyplot as plt
import biorbd_casadi as biorbd
import numpy as np
import ezc3d
import time
import casadi as cas
import pickle
import os


subject = 'DoCi'
number_shooting_points = 83
trial = '44_3'
data_path = 'data/optimizations/Do_44_mvtPrep_3_N83.pkl'
biorbd_model_path = data_path + model_name


markers_reordered, _ = reorder_markers(biorbd_model, c3d, frames, step_size)


with open(data_path, 'rb') as handle:
    data_optim = pickle.load(handle)
    q_optim = data_optim['q']

import bioviz
b = bioviz.Viz(biorbd_model_path)
b.load_experimental_markers(markers_reordered)
b.load_movement(q_sol)
b.exec()
