#!/usr/bin/python3

import importlib
from pact import Pact

# A dictionary that maps each possible dataname to a function that takes
# in pars, and gives out the name of the setup module appropriate for
# this dataname and pars. Note that often the return value doesn't even
# depend on pars.
setupmodule_dict = {
    "A": lambda pars: pars["algorithm"] + "_setup",
    "As": lambda pars: pars["algorithm"] + "_setup",
    "T3D_spectrum": lambda pars: "T3D_spectrum" "+_setup"
}


parinfo = {
    "store_data": {
        "type":    "bool",
        "default": False,
    },
}

def get_data(db, dataname, pars, **kwargs):
    pars = copy_update(pars, **kwargs)
    idpars = get_idpars(dataname, pars)
    p = Pact(db)
    try:
        data = p.fetch(dataname, idpars)
    except FileNotFoundError:
        # TODO Storing text output of generation.
        data = generate_data(dataname, pars, db=db)
        if pars["store_data"]:
            p.store(data, dataname, idpars)
    return data


def copy_update(pars, **kwargs):
    pars = pars.copy()
    pars.update(kwargs)
    return pars


def get_idpars(dataname, pars):
    setupmod = get_setupmod(dataname, pars)
    idpars = dict()

    # Get the idpars for the prereqs and update them in.
    prereq_pairs = setupmod.prereq_pairs(dataname, pars)
    for prereq_name, prereq_pars in prereq_pairs:
        prereq_idpars = get_idpars(prereq_name, prereq_pars)
        idpars.update(prereq_pars)

    # Get the idpars for this setupmod, and update those in.
    parinfo = setupmod.parinfo
    idpars = dict()
    for k, v in parinfo.items():
        if v["idfunc"](pars):
            idpars[k] = pars[k]
    if hasattr(setupmod, "idpars_finalize"):
        idpars = setupmod.idpars_finalize(pars, idpars)
    return idpars


def generate_data(dataname, pars, db=None):
    setupmod = get_setupmod(dataname, pars)
    prereq_pairs = setupmod.prereq_pairs(dataname, pars)
    prereqs = []
    for prereq_name, prereq_pars in prereq_pairs:
        if db is not None:
            prereq = get_data(db, prereq_name, prereq_pars)
        else:
            prereq = generate_data(prereq_name, prereq_pars)
        prereq = list(prereq)
        prereqs += prereq
    # TODO Capture output somehow, and store it too.
    data = setupmod.generate(dataname, *prereqs, pars=pars)
    return data


def get_setupmod(dataname, pars):
    modulename = setupmodule_dict[dataname](pars)
    setupmod = importlib.import_module(modulename)
    return setupmod
