import numpy as np
import initialtensors
import toolbox

version = 1.0

twodee_models = {"ising", "potts3", "sixvertex"}
threedee_models = {"ising3d"}

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
    "symmetry_tensors": {
        "default": False,
        "idfunc":  lambda dataname, pars: True
    },

    # Model dependent parameters.
    # Ising and 3-state Potts
    "beta": {
        "default": 1.,
        "idfunc":  lambda dataname, pars: (
            pars["model"] in {"ising", "potts3", "ising3d"}
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
    if dataname in {"A", "As", "As_impure", "A_impure"}:
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
    else:
        raise ValueError("Unknown dataname: {}".format(dataname))
    return res


def generate_A(*args, pars=dict()):
    A = initialtensors.get_initial_tensor(pars)
    log_fact = 0
    if pars["initial4x4"]:
        A = toolbox.contract2x2(A)
        A = toolbox.contract2x2(A)
    elif pars["initial2x2"]:
        A = toolbox.contract2x2(A)
    elif pars["initial2x2x2"]:
        A = toolbox.contract2x2x2(A)
    return (A, log_fact)


def generate_As(*args, pars=dict()):
    A, log_fact = generate_A(*args, pars=pars)
    res = ((A,)*8, log_fact)
    # DEBUG
    #res = (tuple(rand_As_pure), log_fact)
    # END DEBUG
    return res


def generate_A_impure(*args, pars=dict()):
    A_impure = initialtensors.get_initial_impurity(pars)
    log_fact = 0
    if pars["initial4x4"] or pars["initial2x2"] or pars["initial2x2x2"]:
        msg = ("initial2x2, initial4x4 and initial2x2x2 unimplemented for"
               "initial impurities.")
        raise NotImplementedError(msg)
    return (A_impure, log_fact)


def generate_As_impure(*args, pars=dict()):
    A_impure, log_fact_impure = generate_A_impure(*args, pars=pars)
    A, log_fact_pure = generate_A(*args, pars=pars)
    res = ((A, A, A_impure, A, A, A, A, A), log_fact_impure + log_fact_pure)
    # DEBUG
    #res = (tuple(rand_As_impure), log_fact_impure + log_fact_pure)
    #res = (tuple(rand_As_pure), log_fact_impure + log_fact_pure)
    # END DEBUG
    return res


# DEBUG random tensor test
#from tensors.symmetrytensors import TensorZ2
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
#
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

