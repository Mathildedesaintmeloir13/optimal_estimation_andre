import numpy as np
import ezc3d
import pickle
import bioviz
from IPython import embed

data_name = "Do_44_mvtPrep_3_optimal_gravity_N83_noOGE"
data_path = "/home/mickaelbegon/Documents/Stage_Mathilde/programation/optimal_estimation_andre/optimal_estimation_ocp/Solutions/DoCi"
model_path = "/home/mickaelbegon/Documents/Stage_Mathilde/programation/optimal_estimation_andre/OptimalEstimation/Andre_data/Model/DoCi.s2mMod"

with open(f'{data_path}/{data_name}.pkl', 'rb') as file:
    data = pickle.load(file)
Q = data["states"]['q']

b = bioviz.Viz(model_path)
b.load_movement(Q)
b.load_experimental_markers(data['mocap'])
b.exec()


















