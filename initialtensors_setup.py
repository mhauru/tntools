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
        "idfunc":  lambda dataname, pars: pars["model"] in {"ising", "potts3"}
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
    return ((A,)*8, log_fact)


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
    res = ((A_impure, A, A, A, A, A, A, A), log_fact_impure + log_fact_pure)
    return res


