import importlib
import logging
import configparser
import numpy as np
import os
from pact import Pact

np.set_printoptions(precision=7)
np.set_printoptions(linewidth=100)

# A dictionary that maps each possible dataname to a function that takes
# in pars, and gives out the name of the setup module appropriate for
# this dataname and pars. Note that often the return value doesn't even
# depend on pars.
setupmodule_dict = {
    "A": lambda pars: pars["algorithm"] + "_setup",
    "As": lambda pars: pars["algorithm"] + "_setup",
    "T3D_spectrum": lambda pars: "T3D_spectrum" + "_setup"
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
        idpars.update(prereq_idpars)

    # Get the idpars for this setupmod, and update those in.
    parinfo = setupmod.parinfo
    for k, v in parinfo.items():
        if v["idfunc"](pars):
            idpars[k] = pars[k]
    if hasattr(setupmod, "version"):
        modulename = setupmodule_dict[dataname](pars)
        idpars[modulename + "_version"] = setupmod.version
    if hasattr(setupmod, "idpars_finalize"):
        idpars = setupmod.idpars_finalize(pars, idpars)
    return idpars


def generate_data(dataname, pars, db=None):
    havedb = True if db is not None else False
    storedata = pars["store_data"] and havedb
    if havedb:
        p = Pact(db)
    if storedata:
        idpars = get_idpars(dataname, pars)
    setupmod = get_setupmod(dataname, pars)
    prereq_pairs = setupmod.prereq_pairs(dataname, pars)
    prereqs = []
    for prereq_name, prereq_pars in prereq_pairs:
        if havedb:
            prereq = get_data(db, prereq_name, prereq_pars)
        else:
            prereq = generate_data(prereq_name, prereq_pars)
        prereq = list(prereq)
        prereqs += prereq

    if storedata:
        handler, filelogger = set_logging_handlers(p, dataname, idpars)
    else:
        filelogger = None
    data = setupmod.generate(dataname, *prereqs, pars=pars,
                             filelogger=filelogger)
    if storedata:
        remove_logging_handlers(logging.getLogger(), handler)
        remove_logging_handlers(filelogger, handler)

    if storedata:
        p.store(data, dataname, idpars)
    return data


def get_setupmod(dataname, pars):
    modulename = setupmodule_dict[dataname](pars)
    setupmod = importlib.import_module(modulename)
    return setupmod


def set_logging_handlers(p, dataname, idpars):
    rootlogger = logging.getLogger()
    filelogger = logging.getLogger("datadispenser_file")
    filelogger.propagate = False

    logfilename = p.generate_path(dataname, idpars, extension=".log")
    os.makedirs(os.path.dirname(logfilename), exist_ok=True)
    filehandler = logging.FileHandler(logfilename, mode='w')
    # TODO DEBUG?
    filehandler.setLevel(logging.INFO)

    parser = configparser.ConfigParser(interpolation=None)
    parser.read('../tools/logging_default.conf')
    fmt = parser.get('formatter_default', 'format')
    datefmt = parser.get('formatter_default', 'datefmt')
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    filehandler.setFormatter(formatter)
    rootlogger.addHandler(filehandler)
    filelogger.addHandler(filehandler)
    return filehandler, filelogger


def remove_logging_handlers(logger, *args):
    for l in args:
        logger.removeHandler(l)


