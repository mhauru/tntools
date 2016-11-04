import numpy as np
import initialtensors
import toolbox

twodee_models = {"ising", "potts3", "sixvertex"}
threedee_models = {"ising3d"}

parinfo = {
    # Generic parameters
    "model": {
        "type":    "string",
        "default": "",
        "idfunc":  lambda pars: True
    },
    "dtype": {
        "type":    "dtype",
        "default": np.float_,
        "idfunc":  lambda pars: True
    },
    "iter_count": {
        "type":    "int",
        "default": 0,
        "idfunc":  lambda pars: True
    },
    "initial2x2": {
        "type":    "bool",
        "default": False,
        "idfunc":  lambda pars: pars["model"] in twodee_models
    },
    "initial4x4": {
        "type":    "bool",
        "default": False,
        "idfunc":  lambda pars: pars["model"] in twodee_models
    },
    "initial2x2x2": {
        "type":    "bool",
        "default": False,
        "idfunc":  lambda pars: pars["model"] in threedee_models
    },
    "symmetry_tensors": {
        "type":    "bool",
        "default": False,
        "idfunc":  lambda pars: True
    },

    # Model dependent parameters.
    # Ising and 3-state Potts
    "beta": {
        "type":    "float",
        "default": 1.,
        "idfunc":  lambda pars: pars["model"] in {"ising", "potts3"}
    },

    "J": {
        "type":    "float",
        "default": 1.,
        "idfunc":  lambda pars: pars["model"] in {"ising", "potts3"}
    },
    "H": {
        "type":    "float",
        "default": 0.,
        "idfunc":  lambda pars: pars["model"] == "ising"
    },

    # Sixvertex model
    "sixvertex_a": {
        "type":    "float",
        "default": 1.,
        "idfunc":  lambda pars: pars["model"] == "sixvertex"
    },
    "sixvertex_b": {
        "type":    "float",
        "default": 1.,
        "idfunc":  lambda pars: pars["model"] == "sixvertex"
    },
    "sixvertex_c": {
        "type":    "float",
        "default": np.sqrt(2),
        "idfunc":  lambda pars: pars["model"] == "sixvertex"
    },
}


def prereq_pairs(dataname, pars):
    if dataname in {"A", "As"}:
        res = []
    else:
        raise ValueError("Unknown dataname: {}".format(dataname))
    return res


def generate(dataname, *args, pars=dict()):
    if dataname in {"A", "As"}:
        A = initialtensors.get_initial_tensor(pars)
        log_fact = 0
        if pars["initial4x4"]:
            A = toolbox.contract2x2(A)
            A = toolbox.contract2x2(A)
        elif pars["initial2x2"]:
            A = toolbox.contract2x2(A)
    else:
        raise ValueError("Unknown dataname: {}".format(dataname))
    res = (A, log_fact) if dataname == "A" else ((A,)*8, log_fact)
    return res

