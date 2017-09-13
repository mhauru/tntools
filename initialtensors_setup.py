import numpy as np
import initialtensors
from tensors import TensorCommon
from ncon import ncon

version = 1.0

twodee_models = {"ising", "potts3", "sixvertex"}
threedee_models = {"ising3d", "potts33d"}

parinfo = {
    # Generic parameters
    "model": {
        "default": "",
        "idfunc":  lambda dataname, pars: True
    },
    "dtype": {
        "default": np.float_,
        "idfunc":  lambda dataname, pars: True
    },
    "initial2x2": {
        "default": False,
        "idfunc":  lambda dataname, pars: pars["model"] in twodee_models
    },
    "initial4x4": {
        "default": False,
        "idfunc":  lambda dataname, pars: pars["model"] in twodee_models
    },
    "initial2x2x2": {
        "default": False,
        "idfunc":  lambda dataname, pars: pars["model"] in threedee_models
    },
    "initial2z": {
        "default": False,
        "idfunc":  lambda dataname, pars: pars["model"] in threedee_models
    },
    "symmetry_tensors": {
        "default": False,
        "idfunc":  lambda dataname, pars: True
    },

    # Model dependent parameters.
    # Ising and 3-state Potts
    "beta": {
        "default": 1.,
        "idfunc":  lambda dataname, pars: (
            pars["model"] in {"ising", "potts3", "ising3d", "potts33d"}
        )
    },

    "J": {
        "default": 1.,
        "idfunc":  lambda dataname, pars: pars["model"] in {"ising", "potts3"}
    },
    "H": {
        "default": 0.,
        "idfunc":  lambda dataname, pars: pars["model"] == "ising"
    },

    # Sixvertex model
    "sixvertex_a": {
        "default": 1.,
        "idfunc":  lambda dataname, pars: pars["model"] == "sixvertex"
    },
    "sixvertex_b": {
        "default": 1.,
        "idfunc":  lambda dataname, pars: pars["model"] == "sixvertex"
    },
    "sixvertex_c": {
        "default": np.sqrt(2),
        "idfunc":  lambda dataname, pars: pars["model"] == "sixvertex"
    },

    # Impurity parameters
    "impurity": {
        "default": None,
        "idfunc":  lambda dataname, pars: "impure" in dataname
    },
}


def prereq_pairs(dataname, pars):
    if dataname in {"A", "As", "As_impure", "As_impure111",
                    "As_impure333", "A_impure"}:
        res = []
    else:
        raise ValueError("Unknown dataname: {}".format(dataname))
    return res


def generate(dataname, *args, pars=dict(), filelogger=None):
    if dataname == "A":
        res = generate_A(*args, pars=pars)
    elif dataname == "As":
        res = generate_As(*args, pars=pars)
    elif dataname == "A_impure":
        res = generate_A_impure(*args, pars=pars)
    elif dataname == "As_impure":
        res = generate_As_impure(*args, pars=pars)
    elif dataname == "As_impure111":
        res = generate_As_impure111(*args, pars=pars)
    elif dataname == "As_impure333":
        res = generate_As_impure333(*args, pars=pars)
    else:
        raise ValueError("Unknown dataname: {}".format(dataname))
    return res



def contract2x2(A_list):
    """ Takes an iterable of rank 4 tensors and contracts a square made
    of them to a single rank 4 tensor. If only a single tensor is given
    4 copies of the same tensor are used.
    """
    if isinstance(A_list, (np.ndarray, TensorCommon)):
        A = A_list
        A_list = [A]*4
    else:
        A_list = list(A_list)
    A4 = ncon((A_list[0], A_list[1], A_list[2], A_list[3]),
              ([-2,-3,1,3], [1,-4,-6,4], [-1,3,2,-7], [2,4,-5,-8]))
    A4 = A4.join_indices((0,1), (2,3), (4,5), (6,7), dirs=[1,1,-1,-1])
    return A4

def contract2x2_ndarray(T_list, vert_flip=False):
    if vert_flip:
        def flip(T):
            return np.transpose(T.conjugate(), (0,3,2,1))
        T_list[2] = flip(T_list[2])
        T_list[3] = flip(T_list[3])
    T4 = ncon((T_list[0], T_list[1], T_list[2], T_list[3]),
              ([-2,-3,1,3], [1,-4,-6,4], [-1,3,2,-7], [2,4,-5,-8]))
    sh = T4.shape
    S = np.reshape(T4, (sh[0]*sh[1], sh[2]*sh[3], sh[4]*sh[5], sh[6]*sh[7]))
    return S


def contract2x2x2(A_list):
    """ Takes an iterable of rank 6 tensors and contracts a cube made
    of them to a single rank 6 tensor. If only a single tensor is given
    8 copies of the same tensor are used.
    """
    if isinstance(A_list, (np.ndarray, TensorCommon)):
        A = A_list
        A_list = [A]*8
    else:
        A_list = list(A_list)
    Acube = ncon((A_list[0], A_list[1], A_list[2], A_list[3], A_list[4],
                  A_list[5], A_list[6], A_list[7]),
                 ([-2,-5,7,3,-18,1], [7,-8,-10,9,-19,6],
                  [12,9,-9,-14,20,10], [-1,3,12,-13,-17,4],
                  [-3,-6,5,2,1,-22], [5,-7,-11,8,6,-23],
                  [11,8,-12,-15,10,-24], [-4,2,11,-16,4,-21]))
    S = Acube.join_indices((0,1,2,3), (4,5,6,7), (8,9,10,11),
                           (12,13,14,15), (16,17,18,19), (20,21,22,23),
                           dirs=[1,1,-1,-1,1,-1])
    return Acube


def generate_A(*args, pars=dict()):
    A = initialtensors.get_initial_tensor(pars)
    log_fact = 0
    if pars["initial4x4"]:
        A = contract2x2(A)
        A = contract2x2(A)
    elif pars["initial2x2"]:
        A = contract2x2(A)
    elif pars["initial2x2x2"]:
        A = contract2x2x2(A)
    elif pars["initial2z"]:
        orig_dirs = A.dirs
        A = ncon((A, A),
                 ([-11,-21,-31,1,-51,-61], [-12,1,-32,-42,-52,-62]))
        join_dirs = (None if orig_dirs is None else
                     [orig_dirs[0], orig_dirs[2], orig_dirs[4], orig_dirs[5]])
        A = A.join_indices([0,1], [3,4], [6,7], [8,9], dirs=join_dirs)
    return (A, log_fact)


def generate_As(*args, pars=dict()):
    A, log_fact = generate_A(*args, pars=pars)
    res = ((A,)*8, log_fact)
    # DEBUG
    #res = (tuple(rand_As_pure), log_fact)
    # END DEBUG
    return res


def generate_A_impure(*args, pars=dict()):
    legs = [3] if pars["initial2z"] else [5]
    A_impure = initialtensors.get_initial_impurity(pars, legs=legs)
    log_fact = 0
    if pars["initial4x4"] or pars["initial2x2"] or pars["initial2x2x2"]:
        msg = ("initial2x2, initial4x4 and initial2x2x2 unimplemented for"
               "initial impurities.")
        raise NotImplementedError(msg)
    elif pars["initial2z"]:
        A_pure = initialtensors.get_initial_tensor(pars)
        A_impure = ncon((A_impure, A_pure),
                        ([-11,-21,-31,1,-51,-61], [-12,1,-32,-42,-52,-62]))
        orig_dirs = A_pure.dirs
        join_dirs = (None if orig_dirs is None else
                     [orig_dirs[0], orig_dirs[2], orig_dirs[4], orig_dirs[5]])
        A_impure = A_impure.join_indices([0,1], [3,4], [6,7], [8,9],
                                         dirs=join_dirs)
    return (A_impure, log_fact)


def generate_As_impure(*args, pars=dict()):
    A_impure, log_fact_impure = generate_A_impure(*args, pars=pars)
    A, log_fact_pure = generate_A(*args, pars=pars)
    res = ((A, A_impure, A, A, A, A, A, A), log_fact_impure + log_fact_pure)
    return res


def generate_As_impure111(*args, pars=dict()):
    A_impure, log_fact_impure = generate_A_impure(*args, pars=pars)
    return A_impure, log_fact_impure


def generate_As_impure333(*args, pars=dict()):
    A_impure, log_fact_impure = generate_A_impure(*args, pars=pars)
    A, log_fact_pure = generate_A(*args, pars=pars)
    res = ((A_impure, A, A, A, A, A, A, A),
           (log_fact_impure, log_fact_pure, log_fact_pure, log_fact_pure,
            log_fact_pure, log_fact_pure, log_fact_pure, log_fact_pure))
    # DEBUG
    #res = (tuple(rand_As_impure), (0,)*8)
    #res = (tuple(rand_As_pure), (0,)*8)
    # END DEBUG
    return res



# DEBUG random tensor test
#from tensors import TensorZ2
#T = TensorZ2
#rand_As_pure = [None]*8
#rand_As_pure[0] = T.random(shape=[[1,0], [1,1], [1,2], [1,3], [1,4], [1,5]],
#                           dirs=[1,1,-1,-1,1,-1])
#rand_As_pure[2] = T.random(shape=[[3,0], [3,1], [3,2], [3,3], [3,4], [3,5]],
#                           dirs=[1,1,-1,-1,1,-1])
#rand_As_pure[5] = T.random(shape=[[2,0], [2,1], [2,2], [2,3], [2,4], [2,5]],
#                           dirs=[1,1,-1,-1,1,-1])
#rand_As_pure[7] = T.random(shape=[[4,0], [4,1], [4,2], [4,3], [4,4], [4,5]],
#                           dirs=[1,1,-1,-1,1,-1])
#
#rand_As_pure[1] = T.random(shape=[rand_As_pure[5].shape[2], rand_As_pure[2].shape[3], rand_As_pure[5].shape[0],
#                                  rand_As_pure[2].shape[1], rand_As_pure[0].shape[5], rand_As_pure[0].shape[4]],
#                           dirs=[1,1,-1,-1,1,-1])
#rand_As_pure[3] = T.random(shape=[rand_As_pure[7].shape[2], rand_As_pure[0].shape[3], rand_As_pure[7].shape[0],
#                                  rand_As_pure[0].shape[1], rand_As_pure[2].shape[5], rand_As_pure[2].shape[4]],
#                           dirs=[1,1,-1,-1,1,-1])
#rand_As_pure[4] = T.random(shape=[rand_As_pure[0].shape[2], rand_As_pure[7].shape[3], rand_As_pure[0].shape[0],
#                                  rand_As_pure[7].shape[1], rand_As_pure[5].shape[5], rand_As_pure[5].shape[4]],
#                           dirs=[1,1,-1,-1,1,-1])
#rand_As_pure[6] = T.random(shape=[rand_As_pure[2].shape[2], rand_As_pure[5].shape[3], rand_As_pure[2].shape[0],
#                                  rand_As_pure[5].shape[1], rand_As_pure[7].shape[5], rand_As_pure[7].shape[4]],
#                           dirs=[1,1,-1,-1,1,-1])

# For 3x3x3
#rand_As_impure = [None]*8
#rand_As_impure[0] = T.random(shape=[rand_As_pure[0].shape[0], [5,0], [5,1],
#                                    rand_As_pure[0].shape[3], rand_As_pure[0].shape[4], [5,2]],
#                             dirs=[1,1,-1,-1,1,-1])
#rand_As_impure[2] = T.random(shape=[rand_As_pure[2].shape[0], rand_As_pure[2].shape[1], [5,3],
#                                    [5,4], [5,5], rand_As_pure[2].shape[5]],
#                             dirs=[1,1,-1,-1,1,-1])
#rand_As_impure[5] = T.random(shape=[[6,0], [6,1], rand_As_pure[5].shape[2],
#                                    rand_As_pure[5].shape[3], [6,2], rand_As_pure[5].shape[5]],
#                             dirs=[1,1,-1,-1,1,-1])
#rand_As_impure[7] = T.random(shape=[[6,3], rand_As_pure[7].shape[1], rand_As_pure[7].shape[2],
#                                    [6,4], rand_As_pure[7].shape[4], [6,5]],
#                             dirs=[1,1,-1,-1,1,-1])
#
#rand_As_impure[1] = T.random(shape=[rand_As_pure[1].shape[0], rand_As_impure[2].shape[3], rand_As_impure[5].shape[0],
#                                    rand_As_pure[1].shape[3], rand_As_impure[0].shape[5], rand_As_pure[1].shape[5]],
#                             dirs=[1,1,-1,-1,1,-1])
#rand_As_impure[3] = T.random(shape=[rand_As_pure[3].shape[0], rand_As_pure[3].shape[1], rand_As_impure[7].shape[0],
#                                    rand_As_impure[0].shape[1], rand_As_pure[3].shape[4], rand_As_impure[2].shape[4]],
#                             dirs=[1,1,-1,-1,1,-1])
#rand_As_impure[4] = T.random(shape=[rand_As_impure[0].shape[2], rand_As_impure[7].shape[3], rand_As_pure[4].shape[2],
#                                    rand_As_pure[4].shape[3], rand_As_pure[4].shape[4], rand_As_impure[5].shape[4]],
#                             dirs=[1,1,-1,-1,1,-1])
#rand_As_impure[6] = T.random(shape=[rand_As_impure[2].shape[2], rand_As_pure[6].shape[1], rand_As_pure[6].shape[2],
#                                    rand_As_impure[5].shape[1], rand_As_impure[7].shape[5], rand_As_pure[6].shape[5]],
#                             dirs=[1,1,-1,-1,1,-1])

# For 4x4x4
#rand_As_impure = [None]*8
#rand_As_impure[0] = T.random(shape=[[5,0], rand_As_pure[0].shape[1], rand_As_pure[0].shape[2],
#                                    [5,1], [5,2], rand_As_pure[0].shape[5]],
#                             dirs=[1,1,-1,-1,1,-1])
#rand_As_impure[2] = T.random(shape=[[5,3], [5,4], rand_As_pure[2].shape[2],
#                                    rand_As_pure[2].shape[3], rand_As_pure[2].shape[4], [5,5]],
#                             dirs=[1,1,-1,-1,1,-1])
#rand_As_impure[5] = T.random(shape=[rand_As_pure[5].shape[0], rand_As_pure[5].shape[1], [6,0],
#                                    [6,1], rand_As_pure[5].shape[4], [6,2]],
#                             dirs=[1,1,-1,-1,1,-1])
#rand_As_impure[7] = T.random(shape=[rand_As_pure[7].shape[0], [6,3], [6,4],
#                                    rand_As_pure[7].shape[3], [6,5], rand_As_pure[7].shape[5]],
#                             dirs=[1,1,-1,-1,1,-1])
#
#rand_As_impure[1] = T.random(shape=[rand_As_impure[5].shape[2], rand_As_pure[1].shape[1], rand_As_pure[1].shape[2],
#                                    rand_As_impure[2].shape[1], rand_As_pure[1].shape[4], rand_As_impure[0].shape[4]],
#                             dirs=[1,1,-1,-1,1,-1])
#rand_As_impure[3] = T.random(shape=[rand_As_impure[7].shape[2], rand_As_impure[0].shape[3], rand_As_pure[3].shape[2],
#                                    rand_As_pure[3].shape[3], rand_As_impure[2].shape[5], rand_As_pure[3].shape[5]],
#                             dirs=[1,1,-1,-1,1,-1])
#rand_As_impure[4] = T.random(shape=[rand_As_pure[4].shape[0], rand_As_pure[4].shape[1], rand_As_impure[0].shape[0],
#                                    rand_As_impure[7].shape[1], rand_As_impure[5].shape[5], rand_As_pure[4].shape[5]],
#                             dirs=[1,1,-1,-1,1,-1])
#rand_As_impure[6] = T.random(shape=[rand_As_pure[6].shape[0], rand_As_impure[5].shape[3], rand_As_impure[2].shape[0],
#                                    rand_As_pure[6].shape[3], rand_As_pure[6].shape[4], rand_As_impure[7].shape[4]],
#                             dirs=[1,1,-1,-1,1,-1])

# END DEBUG

