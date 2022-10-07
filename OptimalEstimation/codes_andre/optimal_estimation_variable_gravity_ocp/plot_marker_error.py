import numpy as np
import os
import sys
import pickle
from matplotlib import pyplot
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'optimal_gravity_ocp'))
from load_data_filename import load_data_filename


if __name__ == "__main__":
    trials = [('GuSe', '44_2'), ('SaMi', '821_seul_5'), ('DoCi', '822_contact'), ('JeCh', '833_1')]

    induced_gravity = np.array([0, 1, 2, 3, 4, 5, 10, 15, 20, 25])

    marker_error = []

    for subject, trial in trials:
        data_path = '/home/andre/Optimisation/data/' + subject + '/'
        model_path = data_path + 'Model/'
        c3d_path = data_path + 'Essai/'

        data_filename = load_data_filename(subject, trial)
        model_name = data_filename['model']
        c3d_name = data_filename['c3d']

        marker_error.append([])

        for testing_angle in induced_gravity:
            load_path = 'Solutions/' + subject + '/' + os.path.splitext(c3d_name)[0] + '_stats_test_angle_' + str(testing_angle)
            load_variables_name = load_path + '.pkl'
            if os.path.isfile(load_variables_name):
                with open(load_variables_name, 'wb') as handle:
                    data = pickle.load(handle)

                marker_error[-1].append(data['markers_error_OE'])
            else:
                break


    # --- Plots --- #

    fig = pyplot.figure(figsize=(20, 10))

    pyplot.plot(induced_gravity, marker_error.T, '-s', markersize=11)
    pyplot.xlabel('Induced gravity deviation (°)', fontsize=16)
    pyplot.ylabel('Marker error (mm)', fontsize=16)
    pyplot.xticks(fontsize=14)
    pyplot.yticks(fontsize=14)
    fig.gca().legend(['One somersault, two twist', 'Two somersaults, one and a half twists (pike)', 'Two somersaults, two twists', 'Two somersaults, three twists'], fontsize=15, loc="lower right")

    save_path = 'Solutions/'
    save_name = save_path + "induced_gravity_marker_error" + '.png'
    fig.savefig(save_name, bbox_inches='tight', pad_inches=0)

    pyplot.show()
