import importlib
import logging
import configparser
import numpy as np
import os
import multilineformatter
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
    "As_impure": lambda pars: pars["algorithm"] + "_setup",
    "T3D_spectrum": lambda pars: "T3D_spectrum" + "_setup"
}


parinfo = {
    "store_data": {
        "default": True,
    },
}


def apply_parinfo_defaults(pars, parinfo):
    for k, v in parinfo.items():
        if k not in pars:
            pars[k] = v["default"]
    return


def get_data(db, dataname, pars, return_pars=False, **kwargs):
    pars = copy_update(pars, **kwargs)
    apply_parinfo_defaults(pars, parinfo)
    update_default_pars(dataname, pars)
    idpars = get_idpars(dataname, pars)
    p = Pact(db)
    if p.exists(dataname, idpars):
        data = p.fetch(dataname, idpars)
    else:
        data = generate_data(dataname, pars, db=db)
    retval = (data,)
    if return_pars:
        retval += (pars,)
    if len(retval) == 1:
        retval = retval[0]
    return retval


def copy_update(pars, **kwargs):
    pars = pars.copy()
    pars.update(kwargs)
    return pars


def update_default_pars(dataname, pars):
    setupmod = get_setupmod(dataname, pars)

    # Get the pars for the prereqs and update them in.
    prereq_pairs = setupmod.prereq_pairs(dataname, pars)
    prereq_pars_all = dict()
    for prereq_name, prereq_pars in prereq_pairs:
        update_default_pars(prereq_name, prereq_pars)
        prereq_pars_all.update(prereq_pars)

    for k, v in prereq_pars_all.items():
        if k not in pars:
            pars[k] = v

    # Get the default for this setupmod, and update those in.
    parinfo = setupmod.parinfo
    apply_parinfo_defaults(pars, parinfo)
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
        if v["idfunc"](dataname, pars):
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
        prereqs.append(prereq)

    if storedata:
        handler, filelogger = set_logging_handlers(p, dataname, pars)
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


def set_logging_handlers(p, dataname, pars):
    rootlogger = logging.getLogger()
    filelogger = logging.getLogger("datadispenser_file")
    filelogger.propagate = False

    idpars = get_idpars(dataname, pars)
    logfilename = p.generate_path(dataname, idpars, extension=".log")
    os.makedirs(os.path.dirname(logfilename), exist_ok=True)
    filehandler = logging.FileHandler(logfilename, mode='w')
    if "debug" in pars and pars["debug"]:
        filehandler.setLevel(logging.DEBUG)
    else:
        filehandler.setLevel(logging.INFO)

    parser = configparser.ConfigParser(interpolation=None)
    tools_path = os.path.dirname(multilineformatter.__file__)
    parser.read(tools_path + '/logging_default.conf')
    fmt = parser.get('formatter_default', 'format')
    datefmt = parser.get('formatter_default', 'datefmt')
    formatter = multilineformatter.MultilineFormatter(fmt=fmt, datefmt=datefmt)

    filehandler.setFormatter(formatter)
    rootlogger.addHandler(filehandler)
    filelogger.addHandler(filehandler)
    return filehandler, filelogger


def remove_logging_handlers(logger, *args):
    for l in args:
        logger.removeHandler(l)


