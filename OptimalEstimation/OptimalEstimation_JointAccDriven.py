
import matplotlib.pyplot as plt
import biorbd_casadi as biorbd
import numpy as np
import ezc3d
import time
import casadi as cas
import pickle
import os

import sys
sys.path.append('/home/lim/Documents/Programmation/bioptim/bioptim')

from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    InitialGuessList,
    InterpolationType,
    Solver,
    OdeSolver,
    Node,
    ConfigureProblem,
    DynamicsFunctions,
    ConstraintList,
    ConstraintFcn,
    BiorbdInterface,
    InterpolationType,
    Dynamics,
    BiMappingList,
)


def shift_by_2pi(biorbd_model, q, error_margin=0.35):
    n_q = biorbd_model.nbQ()
    q[4, :] = q[4, :] - ((2 * np.pi) * (np.mean(q[4, :]) / (2 * np.pi)).astype(int))
    for dof in range(6, n_q):
        q[dof, :] = q[dof, :] - ((2 * np.pi) * (np.mean(q[dof, :]) / (2 * np.pi)).astype(int))
        if ((2 * np.pi) * (1 - error_margin)) < np.mean(q[dof, :]) < ((2 * np.pi) * (1 + error_margin)):
            q[dof, :] = q[dof, :] - (2 * np.pi)
        elif ((2 * np.pi) * (1 - error_margin)) < -np.mean(q[dof, :]) < ((2 * np.pi) * (1 + error_margin)):
            q[dof, :] = q[dof, :] + (2 * np.pi)
    return q

def shift_by_pi(q, error_margin):
    if ((np.pi)*(1-error_margin)) < np.mean(q) < ((np.pi)*(1+error_margin)):
        q = q - np.pi
    elif ((np.pi)*(1-error_margin)) < -np.mean(q) < ((np.pi)*(1+error_margin)):
        q = q + np.pi
    return q

def reorder_markers(biorbd_model, c3d, frames, step_size=1, broken_dofs=None):
    markers = c3d['data']['points'][:3, :95, frames.start:frames.stop:step_size] / 1000

    c3d_labels = c3d['parameters']['POINT']['LABELS']['value'][:95]
    model_labels = [label.to_string() for label in biorbd_model.markerNames()]

    labels_index = []
    missing_markers_index = []
    for index_model, model_label in enumerate(model_labels):
        missing_markers_bool = True
        for index_c3d, c3d_label in enumerate(c3d_labels):
            if model_label in c3d_label:
                labels_index.append(index_c3d)
                missing_markers_bool = False
        if missing_markers_bool:
            labels_index.append(index_model)
            missing_markers_index.append(index_model)

    markers_reordered = np.zeros((3, markers.shape[1], markers.shape[2]))
    for index, label_index in enumerate(labels_index):
        if index in missing_markers_index:
            markers_reordered[:, index, :] = np.nan
        else:
            markers_reordered[:, index, :] = markers[:, label_index, :]

    model_segments = {
        'pelvis': {'markers': ['EIASD', 'CID', 'EIPSD', 'EIPSG', 'CIG', 'EIASG'], 'dofs': range(0, 6)},
        'thorax': {'markers': ['MANU', 'MIDSTERNUM', 'XIPHOIDE', 'C7', 'D3', 'D10'], 'dofs': range(6, 9)},
        'head': {'markers': ['ZYGD', 'TEMPD', 'GLABELLE', 'TEMPG', 'ZYGG'], 'dofs': range(9, 12)},
        'right_shoulder': {'markers': ['CLAV1D', 'CLAV2D', 'CLAV3D', 'ACRANTD', 'ACRPOSTD', 'SCAPD'], 'dofs': range(12, 14)},
        'right_arm': {'markers': ['DELTD', 'BICEPSD', 'TRICEPSD', 'EPICOND', 'EPITROD'], 'dofs': range(14, 17)},
        'right_forearm': {'markers': ['OLE1D', 'OLE2D', 'BRACHD', 'BRACHANTD', 'ABRAPOSTD', 'ABRASANTD', 'ULNAD', 'RADIUSD'], 'dofs': range(17, 19)},
        'right_hand': {'markers': ['METAC5D', 'METAC2D', 'MIDMETAC3D'], 'dofs': range(19, 21)},
        'left_shoulder': {'markers': ['CLAV1G', 'CLAV2G', 'CLAV3G', 'ACRANTG', 'ACRPOSTG', 'SCAPG'], 'dofs': range(21, 23)},
        'left_arm': {'markers': ['DELTG', 'BICEPSG', 'TRICEPSG', 'EPICONG', 'EPITROG'], 'dofs': range(23, 26)},
        'left_forearm': {'markers': ['OLE1G', 'OLE2G', 'BRACHG', 'BRACHANTG', 'ABRAPOSTG', 'ABRANTG', 'ULNAG', 'RADIUSG'], 'dofs': range(26, 28)},
        'left_hand': {'markers': ['METAC5G', 'METAC2G', 'MIDMETAC3G'], 'dofs': range(28, 30)},
        'right_thigh': {'markers': ['ISCHIO1D', 'TFLD', 'ISCHIO2D', 'CONDEXTD', 'CONDINTD'], 'dofs': range(30, 33)},
        'right_leg': {'markers': ['CRETED', 'JAMBLATD', 'TUBD', 'ACHILED', 'MALEXTD', 'MALINTD'], 'dofs': range(33, 34)},
        'right_foot': {'markers': ['CALCD', 'MIDMETA4D', 'MIDMETA1D', 'SCAPHOIDED', 'METAT5D', 'METAT1D'], 'dofs': range(34, 36)},
        'left_thigh': {'markers': ['ISCHIO1G', 'TFLG', 'ISCHIO2G', 'CONEXTG', 'CONDINTG'], 'dofs': range(36, 39)},
        'left_leg': {'markers': ['CRETEG', 'JAMBLATG', 'TUBG', 'ACHILLEG', 'MALEXTG', 'MALINTG', 'CALCG'], 'dofs': range(39, 40)},
        'left_foot': {'markers': ['MIDMETA4G', 'MIDMETA1G', 'SCAPHOIDEG', 'METAT5G', 'METAT1G'], 'dofs': range(40, 42)},
    }

    markers_idx_broken_dofs = []
    if broken_dofs is not None:
        for dof in broken_dofs:
            for segment in model_segments.values():
                if dof in segment['dofs']:
                    marker_positions = [index_model for marker_label in segment['markers'] for index_model, model_label in enumerate(model_labels) if marker_label in model_label]
                    if range(min(marker_positions), max(marker_positions) + 1) not in markers_idx_broken_dofs:
                        markers_idx_broken_dofs.append(range(min(marker_positions), max(marker_positions) + 1))

    return markers_reordered, markers_idx_broken_dofs

def adjust_number_shooting_points(number_shooting_points, frames):
    list_adjusted_number_shooting_points = []
    for frame_num in range(1, (abs(frames.stop - frames.start) - 1) // abs(frames.step) + 1):
        list_adjusted_number_shooting_points.append((abs(frames.stop - frames.start) - 1) // frame_num + 1)
    diff_shooting_points = [abs(number_shooting_points - point) for point in list_adjusted_number_shooting_points]
    step_size = diff_shooting_points.index(min(diff_shooting_points)) + 1
    adjusted_number_shooting_points = ((abs(frames.stop - frames.start) - 1) // step_size + 1) - 1

    return adjusted_number_shooting_points, step_size

def x_bounds(biorbd_model):
    pi = np.pi
    inf = 50000
    n_qdot = biorbd_model.nbQdot()

    qmin_base = [-3, -3, -1, -6*np.pi, -pi / 2.1, -8*np.pi]
    qmax_base = [3, 3, 7, 6*np.pi, pi / 2.1, 8*np.pi]
    qmin_thorax = [-pi / 2, -pi / 2.1, -pi / 2]
    qmax_thorax = [pi / 2, pi / 2.1, pi / 2]
    qmin_tete = [-pi / 2, -pi / 2.1, -pi / 2]
    qmax_tete = [pi / 2, pi / 2.1, pi / 2]
    qmin_epaule_droite = [-pi / 2, -pi / 2]
    qmax_epaule_droite = [pi / 2, pi / 2]
    qmin_bras_droit = [-pi, -pi / 2.1, -pi]
    qmax_bras_droit = [pi, pi / 2.1, pi]
    qmin_avantbras_droit = [-3*pi/4, -pi/2]
    qmax_avantbras_droit = [pi, pi]
    qmin_main_droite = [-pi / 2, -pi / 2]
    qmax_main_droite = [pi / 2, pi / 2]
    qmin_epaule_gauche = [-pi / 2, -pi / 2]
    qmax_epaule_gauche = [pi / 2, pi / 2]
    qmin_bras_gauche = [-pi, -pi / 2.1, -pi]
    qmax_bras_gauche = [pi, pi / 2.1, pi]
    qmin_avantbras_gauche = [-3*pi/4, -pi]
    qmax_avantbras_gauche = [pi, pi/2]
    qmin_main_gauche = [-3 * pi / 2, -3 * pi / 2]
    qmax_main_gauche = [3 * pi / 2, 3 * pi / 2]
    qmin_cuisse_droite = [-pi, -pi / 2.1, -pi / 2]
    qmax_cuisse_droite = [pi, pi / 2.1, pi / 2]
    qmin_jambe_droite = [-pi]
    qmax_jambe_droite = [pi/4]
    qmin_pied_droit = [-pi / 2, -pi / 2]
    qmax_pied_droit = [pi / 2, pi / 2]
    qmin_cuisse_gauche = [-pi, -pi / 2.1, -pi / 2]
    qmax_cuisse_gauche = [pi, pi / 2.1, pi / 2]
    qmin_jambe_gauche = [-pi]
    qmax_jambe_gauche = [pi/4]
    qmin_pied_gauche = [-pi / 2, -pi / 2]
    qmax_pied_gauche = [pi / 2, pi / 2]

    qdotmin_base = [-inf, -inf, -inf, -inf, -inf, -inf]
    qdotmax_base = [inf, inf, inf, inf, inf, inf]

    xmin = (qmin_base +  # q
            qmin_thorax +
            qmin_tete +
            qmin_epaule_droite +
            qmin_bras_droit +
            qmin_avantbras_droit +
            qmin_main_droite +
            qmin_epaule_gauche +
            qmin_bras_gauche +
            qmin_avantbras_gauche +
            qmin_main_gauche +
            qmin_cuisse_droite +
            qmin_jambe_droite +
            qmin_pied_droit +
            qmin_cuisse_gauche +
            qmin_jambe_gauche +
            qmin_pied_gauche +
            qdotmin_base +  # qdot
            [-200] * (n_qdot - 6))

    xmax = (qmax_base +
            qmax_thorax +
            qmax_tete +
            qmax_epaule_droite +
            qmax_bras_droit +
            qmax_avantbras_droit +
            qmax_main_droite +
            qmax_epaule_gauche +
            qmax_bras_gauche +
            qmax_avantbras_gauche +
            qmax_main_gauche +
            qmax_cuisse_droite +
            qmax_jambe_droite +
            qmax_pied_droit +
            qmax_cuisse_gauche +
            qmax_jambe_gauche +
            qmax_pied_gauche +
            qdotmax_base +  # qdot
            [200] * (n_qdot - 6))

    return xmin, xmax


def load_data_filename(subject, trial):
    if subject == 'DoCi':
        model_name = 'DoCi.s2mMod'
        # model_name = 'DoCi_SystemesDaxesGlobal_surBassin_rotAndre.s2mMod'
        if trial == '822':
            c3d_name = 'Do_822_contact_2.c3d'
            q_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_Q.mat'
            qd_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_V.mat'
            qdd_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_A.mat'
            frames = range(3089, 3360)
        if trial == '822_short':
            c3d_name = 'Do_822_contact_2_short.c3d'
            q_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_Q.mat'
            qd_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_V.mat'
            qdd_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_A.mat'
            frames = range(3119, 3330)
        if trial == '822_time_inverted':
            c3d_name = 'Do_822_contact_2_time_inverted.c3d'
            q_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_Q.mat'
            qd_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_V.mat'
            qdd_name = 'Do_822_contact_2_MOD200.00_GenderF_DoCig_A.mat'
            frames = range(3089, 3360)
        elif trial == '44_1':
            c3d_name = 'Do_44_mvtPrep_1.c3d'
            q_name = 'Do_44_mvtPrep_1_MOD200.00_GenderF_DoCig_Q.mat'
            qd_name = 'Do_44_mvtPrep_1_MOD200.00_GenderF_DoCig_V.mat'
            qdd_name = 'Do_44_mvtPrep_1_MOD200.00_GenderF_DoCig_A.mat'
            frames = range(2449, 2700)
        elif trial == '44_2':
            c3d_name = 'Do_44_mvtPrep_2.c3d'
            q_name = 'Do_44_mvtPrep_2_MOD200.00_GenderF_DoCig_Q.mat'
            qd_name = 'Do_44_mvtPrep_2_MOD200.00_GenderF_DoCig_V.mat'
            qdd_name = 'Do_44_mvtPrep_2_MOD200.00_GenderF_DoCig_A.mat'
            frames = range(2599, 2850)
        elif trial == '44_3':
            c3d_name = 'Do_44_mvtPrep_3.c3d'
            q_name = 'Do_44_mvtPrep_3_MOD200.00_GenderF_DoCig_Q.mat'
            qd_name = 'Do_44_mvtPrep_3_MOD200.00_GenderF_DoCig_V.mat'
            qdd_name = 'Do_44_mvtPrep_3_MOD200.00_GenderF_DoCig_A.mat'
            frames = range(4099, 4350)
    elif subject == 'JeCh':
        model_name = 'JeCh_201.s2mMod'
        # model_name = 'JeCh_SystemeDaxesGlobal_surBassin'
        if trial == '821_1':
            model_name = 'JeCh_201_bras_modifie.s2mMod'
            c3d_name = 'Je_821_821_1.c3d'
            q_name = ''
            qd_name = ''
            qdd_name = ''
            frames = range(2339, 2659)
        if trial == '821_821_1':
            model_name = 'JeCh_201_bras_modifie.s2mMod'
            c3d_name = 'Je_821_821_1.c3d'
            q_name = ''
            qd_name = ''
            qdd_name = ''
            frames = range(3129, 3419)
        if trial == '821_2':
            c3d_name = 'Je_821_821_2.c3d'
            q_name = ''
            qd_name = ''
            qdd_name = ''
            frames = range(2299, 2590)
        if trial == '821_3':
            c3d_name = 'Je_821_821_3.c3d'
            q_name = ''
            qd_name = ''
            qdd_name = ''
            frames = range(2299, 2590)
        if trial == '821_5':
            c3d_name = 'Je_821_821_5.c3d'
            q_name = ''
            qd_name = ''
            qdd_name = ''
            frames = range(2299, 2590)
        if trial == '833_1':
            c3d_name = 'Je_833_1.c3d'
            q_name = 'Je_833_1_MOD201.00_GenderM_JeChg_Q.mat'
            qd_name = 'Je_833_1_MOD201.00_GenderM_JeChg_V.mat'
            qdd_name = 'Je_833_1_MOD201.00_GenderM_JeChg_A.mat'
            # frames = range(1919, 2220)
            frames = range(2299, 2590)
        if trial == '833_2':
            c3d_name = 'Je_833_2.c3d'
            q_name = 'Je_833_2_MOD201.00_GenderM_JeChg_Q.mat'
            qd_name = 'Je_833_2_MOD201.00_GenderM_JeChg_V.mat'
            qdd_name = 'Je_833_2_MOD201.00_GenderM_JeChg_A.mat'
            # frames = range(1899, 2210)
            frames = range(2289, 2590)
        if trial == '833_3':
            c3d_name = 'Je_833_3.c3d'
            q_name = 'Je_833_3_MOD201.00_GenderM_JeChg_Q.mat'
            qd_name = 'Je_833_3_MOD201.00_GenderM_JeChg_V.mat'
            qdd_name = 'Je_833_3_MOD201.00_GenderM_JeChg_A.mat'
            # frames = range(2179, 2490)
            frames = range(2569, 2880)
        if trial == '833_4':
            c3d_name = 'Je_833_4.c3d'
            q_name = 'Je_833_4_MOD201.00_GenderM_JeChg_Q.mat'
            qd_name = 'Je_833_4_MOD201.00_GenderM_JeChg_V.mat'
            qdd_name = 'Je_833_4_MOD201.00_GenderM_JeChg_A.mat'
            # frames = range(2269, 2590)
            frames = range(2669, 2970)
        if trial == '833_5':
            c3d_name = 'Je_833_5.c3d'
            q_name = 'Je_833_5_MOD201.00_GenderM_JeChg_Q.mat'
            qd_name = 'Je_833_5_MOD201.00_GenderM_JeChg_V.mat'
            qdd_name = 'Je_833_5_MOD201.00_GenderM_JeChg_A.mat'
            # frames = range(2279, 2600)
            frames = range(2669, 2980)
    elif subject == 'BeLa':
        model_name = 'BeLa.s2mMod'
        # model_name = 'BeLa_SystemeDaxesGlobal_surBassin.s2mMod'
        if trial == '44_1':
            c3d_name = 'Ben_44_mvtPrep_1.c3d'
            q_name = 'Ben_44_mvtPrep_1_MOD202.00_GenderM_BeLag_Q.mat'
            qd_name = 'Ben_44_mvtPrep_1_MOD202.00_GenderM_BeLag_V.mat'
            qdd_name = 'Ben_44_mvtPrep_1_MOD202.00_GenderM_BeLag_A.mat'
            frames = range(1799, 2050)
        elif trial == '44_2':
            c3d_name = 'Ben_44_mvtPrep_2.c3d'
            q_name = 'Ben_44_mvtPrep_2_MOD202.00_GenderM_BeLag_Q.mat'
            qd_name = 'Ben_44_mvtPrep_2_MOD202.00_GenderM_BeLag_V.mat'
            qdd_name = 'Ben_44_mvtPrep_2_MOD202.00_GenderM_BeLag_A.mat'
            frames = range(2149, 2350)
        elif trial == '44_3':
            c3d_name = 'Ben_44_mvtPrep_3.c3d'
            q_name = 'Ben_44_mvtPrep_3_MOD202.00_GenderM_BeLag_Q.mat'
            qd_name = 'Ben_44_mvtPrep_3_MOD202.00_GenderM_BeLag_V.mat'
            qdd_name = 'Ben_44_mvtPrep_3_MOD202.00_GenderM_BeLag_A.mat'
            frames = range(2449, 2700)
    elif subject == 'GuSe':
        model_name = 'GuSe.s2mMod'
        # model_name = 'GuSe_SystemeDaxesGlobal_surBassin.s2mMod'
        if trial == '44_2':
            c3d_name = 'Gui_44_mvt_Prep_2.c3d'
            q_name = 'Gui_44_mvt_Prep_2_MOD200.00_GenderM_GuSeg_Q.mat'
            qd_name = 'Gui_44_mvt_Prep_2_MOD200.00_GenderM_GuSeg_V.mat'
            qdd_name = 'Gui_44_mvt_Prep_2_MOD200.00_GenderM_GuSeg_A.mat'
            frames = range(1649, 1850)
        elif trial == '44_3':
            c3d_name = 'Gui_44_mvt_Prep_3.c3d'
            q_name = 'Gui_44_mvt_Prep_3_MOD200.00_GenderM_GuSeg_Q.mat'
            qd_name = 'Gui_44_mvt_Prep_3_MOD200.00_GenderM_GuSeg_V.mat'
            qdd_name = 'Gui_44_mvt_Prep_3_MOD200.00_GenderM_GuSeg_A.mat'
            frames = range(1699, 1950)
        elif trial == '44_4':
            c3d_name = 'Gui_44_mvtPrep_4.c3d'
            q_name = 'Gui_44_mvtPrep_4_MOD200.00_GenderM_GuSeg_Q.mat'
            qd_name = 'Gui_44_mvtPrep_4_MOD200.00_GenderM_GuSeg_V.mat'
            qdd_name = 'Gui_44_mvtPrep_4_MOD200.00_GenderM_GuSeg_A.mat'
            frames = range(1599, 1850)
    elif subject == 'SaMi':
        model_name = 'SaMi.s2mMod'
        if trial == '821_822_2':
            c3d_name = 'Sa_821_822_2.c3d'
            q_name = 'Sa_821_822_2_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_822_2_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_822_2_MOD200.00_GenderF_SaMig_A.mat'
            # frames = range(2909, 3220)
            frames = range(3299, 3590)
            # frames = range(3659, 3950)
        elif trial == '821_822_3':
            c3d_name = 'Sa_821_822_3.c3d'
            q_name = 'Sa_821_822_3_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_822_3_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_822_3_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3139, 3440)
        # elif trial == '821_822_4':
        #     c3d_name = 'Sa_821_822_4.c3d'
        #     q_name = 'Sa_821_822_4_MOD200.00_GenderF_SaMig_Q.mat'
        #     qd_name = 'Sa_821_822_4_MOD200.00_GenderF_SaMig_V.mat'
        #     qdd_name = 'Sa_821_822_4_MOD200.00_GenderF_SaMig_A.mat'
        #     # frames = range(3509, 3820)
        #     frames = range(3909, 4190)
        # elif trial == '821_822_5':
        #     c3d_name = 'Sa_821_822_5.c3d'
        #     q_name = 'Sa_821_822_5_MOD200.00_GenderF_SaMig_Q.mat'
        #     qd_name = 'Sa_821_822_5_MOD200.00_GenderF_SaMig_V.mat'
        #     qdd_name = 'Sa_821_822_5_MOD200.00_GenderF_SaMig_A.mat'
        #     frames = range(3339, 3630)
        elif trial == '821_contact_1':
            c3d_name = 'Sa_821_contact_1.c3d'
            q_name = 'Sa_821_contact_1_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_contact_1_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_contact_1_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3019, 3330)
        elif trial == '821_contact_2':
            c3d_name = 'Sa_821_contact_2.c3d'
            q_name = 'Sa_821_contact_2_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_contact_2_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_contact_2_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3569, 3880)
        elif trial == '821_contact_3':
            c3d_name = 'Sa_821_contact_3.c3d'
            q_name = 'Sa_821_contact_3_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_contact_3_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_contact_3_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3309, 3620)
        elif trial == '822_contact_1':
            c3d_name = 'Sa_822_contact_1.c3d'
            q_name = 'Sa_822_contact_1_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_822_contact_1_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_822_contact_1_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(5009+3*4, 5310)
        elif trial == '821_seul_1':
            c3d_name = 'Sa_821_seul_1.c3d'
            q_name = 'Sa_821_seul_1_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_seul_1_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_seul_1_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3349, 3650)
        elif trial == '821_seul_2':
            c3d_name = 'Sa_821_seul_2.c3d'
            q_name = 'Sa_821_seul_2_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_seul_2_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_seul_2_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3429, 3740)
        elif trial == '821_seul_3':
            c3d_name = 'Sa_821_seul_3.c3d'
            q_name = 'Sa_821_seul_3_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_seul_3_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_seul_3_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3209, 3520)
        elif trial == '821_seul_4':
            c3d_name = 'Sa_821_seul_4.c3d'
            q_name = 'Sa_821_seul_4_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_seul_4_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_seul_4_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(3309, 3620)
        elif trial == '821_seul_5':
            c3d_name = 'Sa_821_seul_5.c3d'
            q_name = 'Sa_821_seul_5_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_821_seul_5_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_821_seul_5_MOD200.00_GenderF_SaMig_A.mat'
            frames = range(2689, 3000)
        elif trial == 'bras_volant_1':
            c3d_name = 'Sa_bras_volant_1.c3d'
            q_name = 'Sa_bras_volant_1_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_bras_volant_1_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_bras_volant_1_MOD200.00_GenderF_SaMig_A.mat'
            # frames = range(0, 4657)
            # frames = range(649, 3950)
            # frames = range(649, 1150)
            # frames = range(1249, 1950)
            # frames = range(2549, 3100)
            frames = range(3349, 3950)
        elif trial == 'bras_volant_2':
            c3d_name = 'Sa_bras_volant_2.c3d'
            q_name = 'Sa_bras_volant_2_MOD200.00_GenderF_SaMig_Q.mat'
            qd_name = 'Sa_bras_volant_2_MOD200.00_GenderF_SaMig_V.mat'
            qdd_name = 'Sa_bras_volant_2_MOD200.00_GenderF_SaMig_A.mat'
            # frames = range(0, 3907)
            # frames = range(0, 3100)
            # frames = range(49, 849)
            # frames = range(1599, 2200)
            frames = range(2249, 3100)
    else:
        raise Exception(subject + ' is not a valid subject')

    data_filename = {
        'model': model_name,
        'c3d': c3d_name,
        'q': q_name,
        'qd': qd_name,
        'qdd': qdd_name,
        'frames': frames,
    }

    return data_filename

def inverse_dynamics(biorbd_model, q_ref, qd_ref, qdd_ref):
    q = cas.MX.sym("Q", biorbd_model.nbQ(), 1)
    qdot = cas.MX.sym("Qdot", biorbd_model.nbQdot(), 1)
    qddot = cas.MX.sym("Qddot", biorbd_model.nbQddot(), 1)
    id = biorbd.to_casadi_func("id", biorbd_model.InverseDynamics, q, qdot, qddot)

    return id(q_ref, qd_ref, qdd_ref)[:, :-1]



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


def root_explicit_dynamic(states, controls, parameters, nlp,):
    DynamicsFunctions.apply_parameters(parameters, nlp)
    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    nb_root = nlp.model.nbRoot()
    qddot_joints = DynamicsFunctions.get(nlp.controls["qddot_joint"], controls)
    mass_matrix_nl_effects = nlp.model.InverseDynamics(q, qdot, cas.vertcat(cas.MX.zeros((nb_root, 1)), qddot_joints)).to_mx()[:6]
    mass_matrix = nlp.model.massMatrix(q).to_mx()
    mass_matrix_nl_effects_func = cas.Function("mass_matrix_nl_effects_func", [q, qdot, qddot_joints], [mass_matrix_nl_effects[:nb_root]]).expand()
    M_66 = mass_matrix[:nb_root, :nb_root]
    M_66_func = cas.Function("M66_func", [q], [M_66]).expand()
    qddot_root = cas.solve(M_66_func(q), -mass_matrix_nl_effects_func(q, qdot, qddot_joints), "ldl")
    return qdot, cas.vertcat(qddot_root, qddot_joints)


def custom_configure_root_explicit(ocp, nlp):
    ConfigureProblem.configure_q(nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(nlp, as_states=True, as_controls=False)
    configure_qddot_joint(nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, root_explicit_dynamic, expand=False)


def configure_qddot_joint(nlp, as_states, as_controls):
    nb_root = nlp.model.nbRoot()
    name_qddot_joint = [str(i + nb_root) for i in range(nlp.model.nbQddot() - nb_root)]
    ConfigureProblem.configure_new_variable("qddot_joint", name_qddot_joint, nlp, as_states, as_controls)

def dynamics_root(m, X, Qddot_J):
    Q = X[:m.nbQ()]
    Qdot = X[m.nbQ():]
    Qddot = np.hstack((np.zeros((6,)), Qddot_J)) #qddot2
    NLEffects = m.InverseDynamics(Q, Qdot, Qddot).to_array()
    mass_matrix = m.massMatrix(Q).to_array()
    Qddot_R = np.linalg.solve(mass_matrix[:6, :6], -NLEffects[:6])
    Xdot = np.hstack((Qdot, Qddot_R, Qddot_J))
    return Xdot


# def custom_func_track_markers(all_pn):
#     markers = BiorbdInterface.mx_to_cx("markers", all_pn.nlp.model.markers, all_pn.nlp.states["q"])
#     return val

def plot_dynamic_variables(m, q, qdot, qddot, time, root=True, save_name='oups.png'):
    Q_sym = cas.MX.sym('Q_sym', m.nbQ())
    Qdot_sym = cas.MX.sym('Qdot_sym', m.nbQ())
    Qddot_sym = cas.MX.sym('Qddot_sym', m.nbQ())
    CoM_func = cas.Function('CoM_func', [Q_sym], [m.CoM(Q_sym).to_mx()])
    CoM_dot_func = cas.Function('CoM_dot_func', [Q_sym, Qdot_sym], [m.CoMdot(Q_sym, Qdot_sym).to_mx()])
    CoM_ddot_func = cas.Function('CoM_ddot_func', [Q_sym, Qdot_sym,  Qddot_sym], [m.CoMddot(Q_sym, Qdot_sym, Qddot_sym).to_mx()])
    AngMom_func = cas.Function('AngMom_func', [Q_sym, Qdot_sym], [m.CalcAngularMomentum(Q_sym, Qdot_sym, True).to_mx()])
    Mass_func = cas.Function('Mass_func', [], [m.mass().to_mx()])

    N = np.shape(q)[1]
    CoM = np.zeros((3, N))
    CoM_dot = np.zeros((3, N))
    CoM_ddot = np.zeros((3, N))
    AngMom = np.zeros((3, N))
    LinMom = np.zeros((3, N))
    for i in range(N):
        CoM[:, i] = np.reshape(CoM_func(q[:, i]), (3,))
        CoM_dot[:, i] = np.reshape(CoM_dot_func(q[:, i], qdot[:, i]), (3,))
        if root:
            CoM_ddot[:, i] = np.reshape(CoM_ddot_func(q[:, i], qdot[:, i], qddot[:, i]), (3,))
        AngMom[:, i] = np.reshape(AngMom_func(q[:, i], qdot[:, i]), (3,))
        LinMom[:, i] = np.reshape(Mass_func()['o0'] * CoM_dot[:, i], (3,))

    AngMom_norm = np.linalg.norm(AngMom, axis=0)
    LinMom_norm = np.linalg.norm(LinMom, axis=0)

    if root:
        CoM_dot_0 = np.zeros((3, N))
        c0 = CoM_dot[0:2, 0]
        CoM_dot_0[0:2, :] = np.repeat(c0[:, np.newaxis], N, axis=1)
        CoM_dot_0[2, :] = CoM_ddot[2, 0] * time + CoM_dot[2, 0]

        LinMom_0 = np.linalg.norm(Mass_func()['o0'] * CoM_dot_0, axis=0)
        a0 = AngMom_func(q[:, 0], qdot[:, 0])
        AngMom_0 = np.linalg.norm(np.repeat(a0, N, axis=1), axis=0)



    fig, axs = plt.subplots(3)
    for i in range(3):
        axs[i].plot(CoM[i, :], '-r', label='CoM')
        if root:
            axs[i].plot(CoM_dot_0[i, :], '--g', label='CoM dot supposé')
        axs[i].plot(CoM_dot[i, :], '-g', label='CoM dot')
    plt.legend()
    plt.savefig(save_name[:-4] + "_CoMdot" + ".png")
    # plt.show()

    plt.figure()
    plt.plot(AngMom_norm, '-m', label='Angular momentum')
    plt.plot(LinMom_norm, '-b', label='Linear momentum')
    if root:
        plt.plot(AngMom_0, '--m', label='Angular momentum')
        plt.plot(LinMom_0, '--b', label='Linear momentum')
    plt.legend()
    plt.savefig(save_name[:-4] + "_momentum" + ".png")
    # plt.show()


def plot_markers_diff(m, markers, q, save_name):

    import seaborn as sns

    def markers_fun(m, nb_markers):
        Q_sym = cas.MX.sym('Q_sym', m.nbQ())
        Markers_sym = cas.MX(3, nb_markers)
        for i in range(nb_markers):
            Markers_sym[:, i] = m.markers(Q_sym)[i].to_mx()
        markers_func = cas.Function('makers_func', [Q_sym], [Markers_sym])
        return markers_func

    nb_markers = m.nbMarkers()
    N = markers.shape[2]
    colors = sns.color_palette("gist_rainbow", nb_markers)
    markers_q = np.zeros((3, nb_markers, N))
    markers_func = markers_fun(m, nb_markers)
    for i in range(N):
        markers_q[:, :, i] = markers_func(q[:, i])

    diff = np.linalg.norm(markers_q - markers[:, :nb_markers, :], axis=0)
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


def prepare_ocp(
        biorbd_model_path,
        final_time,
        number_shooting_points,
        markers_ref,
        q_init,
        qdot_init,
        qddot_init,
        tau_init,
        xmin,
        xmax,
        track_q,
        dynamics_type):


    model = biorbd.Model(biorbd_model_path)
    model.setGravity(biorbd.Vector3d(0, 0, -9.80639))
    biorbd_model = (model)

    control_min, control_max = -1000, 1000
    n_q = biorbd_model.nbQ()
    n_controls = n_q - biorbd_model.nbRoot()

    objective_functions = ObjectiveList()
    if track_q:
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, node=Node.ALL, weight=1, key="q", target=q_init, multi_thread=False)
    else:
        # objective_functions.add(ObjectiveFcn.Mayer.TRACK_MARKERS, node=Node.ALL, weight=1000, target=markers_ref)
        # # objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MARKERS, node=Node.ALL, weight=1000, target=markers_ref)
        # objective_functions.add(ObjectiveFcn.Mayer.TRACK_MARKERS, node=Node.START, weight=1000000, target=markers_ref)
        # objective_functions.add(ObjectiveFcn.Mayer.TRACK_MARKERS, node=Node.END, weight=100000, target=markers_ref)
        # objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, node=Node.ALL, weight=1e-5, key="q", target=q_init, multi_thread=False)
        # objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, node=Node.ALL, weight=1e-5, key="qdot", target=qdot_init, multi_thread=False)
        # if dynamics_type == "qddot_joints":
        #     objective_functions.add(ObjectiveFcn.Lagrange.TRACK_CONTROL, node=Node.ALL_SHOOTING, weight=1e-5, key="qddot_joints", target=qddot_init, multi_thread=False)
        # elif dynamics_type == "tau":
        #     objective_functions.add(ObjectiveFcn.Lagrange.TRACK_CONTROL, node=Node.ALL_SHOOTING, weight=1e-5, key="tau", target=tau_init, multi_thread=False)

        objective_functions.add(ObjectiveFcn.Mayer.TRACK_MARKERS, node=Node.ALL, weight=1, target=markers_ref)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, node=Node.ALL, weight=1e-5, key='q', target=q_init[6:n_q, :], multi_thread=False, index=range(6, n_q))
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, node=Node.ALL, weight=1e-5, key='qdot', target=qdot_init[6:n_q, :], multi_thread=False, index=range(6, n_q))
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, weight=1e-5, key='q', index=range(6, n_q))
        control_weight_segments = [1e-7, 1e-7, 1e-6,  # thorax
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
                                   # 0   , 0   , 0   ,  # pelvis trans
                                   #  0   , 0   , 0   ,  # pelvis rot

        for idx in range(n_controls):
            objective_functions.add(ObjectiveFcn.Lagrange.TRACK_CONTROL, key='tau', weight=control_weight_segments[idx], target=tau_init[idx, :], index=[idx], multi_thread=False)
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key=dynamics_type, weight=control_weight_segments[idx], index=[idx], multi_thread=False)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key=dynamics_type, derivative=True, weight=1e-5)

    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, node=Node.END, weight=1e-5, min_bound=duration-0.2,  max_bound=duration+0.2)

    # # # Constraints
    # # constraints = ConstraintList()
    # # # constraints.add(ConstraintFcn.TRACK_MARKERS, node=Node.START, min_bound=markers_ref[:, 0]-0.1, max_bound=markers_ref[:, 0]+0.1)
    # # constraints.add(ConstraintFcn.TRACK_MARKERS, node=Node.START, min_bound=-0.01, max_bound=0.01, target=markers_ref[:, 0])
    # # # constraints.add(ConstraintFcn.TRACK_MARKERS, node=Node.MID, min_bound=markers_ref[:, 0]-1, max_bound=markers_ref[:, 0]+1)
    # # # for i in range(biorbd_model.nbMarkers()):
    # # #     constraints.add(custom_func_track_markers, node=Node.ALL, min_bound=markers_ref-0.5, max_bound=markers_ref+0.5)
    # # # MID et vérrifier que c'est le bon indice

    # Dynamics
    if dynamics_type == "qddot_joint":
        dynamics = DynamicsList()
        dynamics.add(custom_configure_root_explicit, dynamic_function=root_explicit_dynamic)
    elif dynamics_type == "tau":
        dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)
    else:
        print("Dynamics not implemented")

    # Path constraint
    X_bounds = BoundsList()
    X_bounds.add(min_bound=xmin, max_bound=xmax)
    # X_bounds[0].min[:, 0] = np.hstack((q_init[:, 0] - 0.05, qdot_init[:, 0] - 0.1))
    # X_bounds[0].max[:, 0] = np.hstack((q_init[:, 0] + 0.05, qdot_init[:, 0] + 0.1))
    # X_bounds[0].min[:biorbd_model.nbQ(), 0] = q_init[:, 0]
    # X_bounds[0].max[:biorbd_model.nbQ(), 0] = q_init[:, 0]

    # Initial guess
    X_init = InitialGuessList()
    X_init.add(np.concatenate([q_init, qdot_init]), interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    U_bounds = BoundsList()
    U_bounds.add(min_bound=[control_min] * n_controls, max_bound=[control_max] * n_controls)

    U_init = InitialGuessList()
    if dynamics_type == "qddot_joint":
        U_init.add(qddot_init, interpolation=InterpolationType.EACH_FRAME)
    elif dynamics_type == "tau":
        # U_init.add(np.random.random((n_controls, number_shooting_points)), interpolation=InterpolationType.EACH_FRAME)
        U_init.add(tau_init, interpolation=InterpolationType.EACH_FRAME)

    variable_mappings = BiMappingList()
    if dynamics_type == "tau":
        variable_mappings.add("tau",
                         to_second=[None, None, None, None, None, None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                    35],
                         to_first=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                                   28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41])

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
        variable_mappings=variable_mappings,
        ode_solver=OdeSolver.RK4(n_integration_steps=4),
        n_threads=32,
    )

if __name__ == "__main__":

    generate_track_Q = True #
    generate_track_markers = True #False #
    reduced_nb_nodes = True # False
    dynamics_type = 'tau' # 'qddot_joint'

    start = time.time()
    subject = 'DoCi'
    number_shooting_points = 100
    trial = '44_3'
    print('Subject: ', subject, ', Trial: ', trial)

    data_path = 'data/' + subject + '/'

    data_filename = load_data_filename(subject, trial)
    model_name = data_filename['model']
    c3d_name = data_filename['c3d']
    frames = data_filename['frames']

    biorbd_model_path = data_path + model_name
    biorbd_model = biorbd.Model(biorbd_model_path)
    c3d = ezc3d.c3d(data_path + c3d_name)

    # --- Adjust number of shooting points --- #
    if reduced_nb_nodes:
        adjusted_number_shooting_points, step_size = adjust_number_shooting_points(number_shooting_points, frames)
    else:
        adjusted_number_shooting_points = len(frames)-1
        step_size = 1

    print('Node step size: ', step_size)
    print('Adjusted number of shooting points: ', adjusted_number_shooting_points)
    print('Dynamics type : ', dynamics_type)

    frequency = c3d['header']['points']['frame_rate']
    duration = len(frames) / frequency

    load_path = f'data/{subject}/EKF/'
    load_name = load_path + os.path.splitext(c3d_name)[0] + ".pkl"
    with open(load_name, 'rb') as handle:
        EKF = pickle.load(handle)
    if reduced_nb_nodes:
        q_ref = shift_by_2pi(biorbd_model, EKF['q'][:, ::step_size])
        qdot_ref = EKF['qd'][:, ::step_size]
        qddot_ref = EKF['qdd'][:, ::step_size]
    else:
        q_ref = shift_by_2pi(biorbd_model, EKF['q'])
        qdot_ref = EKF['qd']
        qddot_ref = EKF['qdd']

    tau_ref = inverse_dynamics(biorbd_model, q_ref, qdot_ref, qddot_ref).full()

    xmin, xmax = x_bounds(biorbd_model)

    # Organize the markers in the same order as in the model
    markers_reordered, _ = reorder_markers(biorbd_model, c3d, frames, step_size)

    # plot_dynamic_variables(biorbd_model, q_ref, qdot_ref, qddot_ref, np.linspace(0, 1/200*step_size*adjusted_number_shooting_points, adjusted_number_shooting_points+1), save_name=f"figures/{subject}_{trial}_Kalman_dynamics.png")
    # plot_markers_diff(biorbd_model, markers_reordered, q_ref, save_name=f"figures/{subject}_{trial}_Kalman.png")

    # import bioviz
    # b = bioviz.Viz(biorbd_model_path)
    # b.load_experimental_markers(markers_reordered)
    # b.load_movement(q_ref)
    # b.exec()


    save_path = 'data/optimizations/'
    save_name = save_path + os.path.splitext(c3d_name)[0] + "_N" + str(adjusted_number_shooting_points) + "_trackQ"
    save_variables_name = save_name + ".pkl"

    if generate_track_Q:
        ocp = prepare_ocp(
            biorbd_model_path=biorbd_model_path, final_time=duration, number_shooting_points=adjusted_number_shooting_points,
            markers_ref=markers_reordered, q_init=q_ref, qdot_init=qdot_ref, qddot_init=qddot_ref[6:, :-1], tau_init=tau_ref[6:, :],
            xmin=xmin, xmax=xmax, track_q=True, dynamics_type=dynamics_type,
        )

        # --- Solve the program --- #
        ocp.add_plot_penalty()

        solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
        solver.set_maximum_iterations(3000)
        solver.set_linear_solver("ma57")
        solver.set_tol(1e-4) # 1e-4
        solver.set_constr_viol_tol(1e-2) # 1e-2
        sol = ocp.solve(solver)
        stop = time.time()
        print('Runtime: ', stop - start)
        print('Number of shooting points: ', adjusted_number_shooting_points)

        # # --- Get the results --- #
        q_sol = sol.states['q']
        qdot_sol = sol.states['qdot']
        control_sol = sol.controls[dynamics_type]


        with open(save_variables_name, 'wb') as handle:
            pickle.dump({'mocap': markers_reordered, 'duration': duration, 'frames': frames, 'step_size': step_size,
                         'q': q_sol, 'qdot': qdot_sol, dynamics_type: control_sol},
                        handle, protocol=3)

        import bioviz
        b = bioviz.Viz(biorbd_model_path)
        b.load_experimental_markers(markers_reordered)
        b.load_movement(q_sol)
        b.exec()

        q_ref = q_sol
        qdot_ref = qdot_sol
        if dynamics_type == "tau":
            tau_ref = control_sol
        else:
            qddot_ref = control_sol

        # sol.graphs(show_bounds=True)

    else:
        file = open(save_variables_name, 'rb')
        data_plk = pickle.load(file)
        q_ref = data_plk["q"]
        qdot_ref = data_plk["qdot"]
        qddot_ref = data_plk["qddot_joint"][:, :-1]

    # plot_dynamic_variables(biorbd_model,
    #                        q_ref,
    #                        qdot_ref,
    #                        qddot_ref,
    #                        np.linspace(0, 1 / 200 * step_size * adjusted_number_shooting_points, adjusted_number_shooting_points + 1),
    #                        root=False, save_name=f"figures/{subject}_{trial}_Track_Q_dynamics.png")
    # plot_markers_diff(biorbd_model, markers_reordered, q_ref,  save_name=f"figures/{subject}_{trial}_Track_Q.png")


    # ----------------------------------------------------------------------------------------------------------

    save_path = 'data/optimizations/'
    save_name = save_path + os.path.splitext(c3d_name)[0] + "_N" + str(adjusted_number_shooting_points)
    save_variables_name = save_name + ".pkl"

    if generate_track_markers:

        ocp = prepare_ocp(
            biorbd_model_path=biorbd_model_path, final_time=duration, number_shooting_points=adjusted_number_shooting_points,
            markers_ref=markers_reordered, q_init=q_ref, qdot_init=qdot_ref, qddot_init=qddot_ref, tau_init=tau_ref[6:, :],
            xmin=xmin, xmax=xmax, track_q=False, dynamics_type=dynamics_type,
        )

        # --- Solve the program --- #
        ocp.add_plot_penalty()
        solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
        solver.set_maximum_iterations(3000)
        solver.set_linear_solver("ma57")
        solver.set_tol(1e-4) # 1e-4
        solver.set_constr_viol_tol(1e-2) # 1e-2
        sol = ocp.solve(solver)
        stop = time.time()

        print('Runtime: ', stop - start)
        print('Number of shooting points: ', adjusted_number_shooting_points)

        # --- Get the results --- #
        q_sol = sol.states['q']
        qdot_sol = sol.states['qdot']
        control_sol = sol.controls[dynamics_type]
        # t_sol = sol.parameters['time']

        # --- Save --- #
        with open(save_variables_name, 'wb') as handle:
            pickle.dump({'mocap': markers_reordered, 'duration': duration, 'frames': frames, 'step_size': step_size,
                         'q': q_sol, 'qdot': qdot_sol, 'control': control_sol, 'dynamics_type': dynamics_type},
                        handle, protocol=3)

        # sol.graphs(show_bounds=True)
        sol.detailed_cost_values()
        print(sol.detailed_cost)
        #
        # dt = duration / adjusted_number_shooting_points
        # biorbd_model = biorbd.Model(biorbd_model_path)
        # symbolic_states = cas.MX.sym("q", biorbd_model.nbQ(), 1)
        # markers_fun = biorbd.to_casadi_func("ForwardKin", biorbd_model.markers, symbolic_states)
        # markers_opt = np.zeros((3, biorbd_model.nbMarkers(), adjusted_number_shooting_points + 1))
        # for i in range(adjusted_number_shooting_points + 1):
        #     markers_opt[:, :, i] = markers_fun(q_sol[:, i])
        #
        # markers_cost_sum = np.nansum((markers_reordered - markers_opt) ** 2 * dt) * 1000
        # print(markers_cost_sum)

    else:
        file = open(save_variables_name, 'rb')
        data_plk = pickle.load(file)
        q_sol = data_plk["q"]
        qdot_sol = data_plk["qdot"]
        # qddot_sol = data_plk["qddot_joint"][:, :-1]
        # qddot_sol = data_plk["tau"][:, :-1]

    # plot_dynamic_variables(biorbd_model, q_sol, qdot_sol, qddot_sol, np.linspace(0, 1/200*step_size*adjusted_number_shooting_points, adjusted_number_shooting_points+1), root=False, save_name=f"figures/{subject}_{trial}_Track_markers_dynamics.png")
    plot_markers_diff(biorbd_model, markers_reordered, q_sol, save_name=f"figures/{subject}_{trial}_Track_markers.png")

    import bioviz
    b = bioviz.Viz(biorbd_model_path)
    b.load_experimental_markers(markers_reordered)
    b.load_movement(q_sol)
    b.exec()
