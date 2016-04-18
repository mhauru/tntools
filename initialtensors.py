import numpy as np
from scon import scon
from tensors.tensor import Tensor
from tensors.symmetrytensors import TensorZ2, TensorZ3, TensorU1

""" Module for getting the initial tensors for different models. """

def ising_hamiltonian(pars):
    ham = (- pars["J"]*np.array([[ 1,-1],
                                 [-1, 1]],
                                dtype=pars["dtype"])
           + pars["H"]*np.array([[-1, 0],
                                 [ 0, 1]],
                                dtype=pars["dtype"]))
    return ham

def potts3_hamiltonian(pars):
    ham = -pars["J"]*np.eye(3, dtype=pars["dtype"])
    return ham

hamiltonians = {}
hamiltonians["ising"] = ising_hamiltonian
hamiltonians["potts3"] = potts3_hamiltonian

symmetry_classes_dims_qims = {}
symmetry_classes_dims_qims["ising"] = (TensorZ2, [1,1], [0,1])
symmetry_classes_dims_qims["potts3"] = (TensorZ3, [1,1,1], [0,1,2])

# Transformation matrices to the bases where the symmetry is explicit.
symmetry_bases = {}
symmetry_bases["ising"] = np.array([[1, 1],
                                    [1,-1]]) / np.sqrt(2)
phase = np.exp(2j*np.pi/3)
symmetry_bases["potts3"] = np.array([[1,       1,         1],
                                     [1,    phase, phase**2],
                                     [1, phase**2,    phase]],
                                    dtype=np.complex_) / np.sqrt(3)

def get_initial_tensor(pars, **kwargs):
    if kwargs:
        pars = pars.copy()
        pars.update(kwargs)
    if pars["model"].strip().lower() == "sixvertex":
        # This is a special case because its easier to build the
        # Boltzmann weights straight without a Hamiltonian
        return get_initial_sixvertex_tensor(pars)
    model_name = pars["model"].strip().lower()
    ham = hamiltonians[model_name](pars)
    boltz = np.exp(-pars["beta"]*ham)
    T_0 = np.einsum('ab,bc,cd,da->abcd', boltz, boltz, boltz, boltz)
    if pars["symmetry_tensors"]:
        u = symmetry_bases[model_name]
        u_dg = u.T.conjugate()
        T_0 = scon((T_0, u, u, u_dg, u_dg),
                   ([1,2,3,4], [-1,1], [-2,2], [3,-3], [4,-4]))
        cls, dim, qim = symmetry_classes_dims_qims[model_name]
        T_0 = cls.from_ndarray(T_0, shape=[dim]*4, qhape=[qim]*4,
                               dirs=[1,1,-1,-1])
    else:
        T_0 = Tensor.from_ndarray(T_0)
    return T_0


def get_initial_sixvertex_tensor(pars):
    try:
        a = pars["sixvertex_a"]
        b = pars["sixvertex_b"]
        c = pars["sixvertex_c"]
    except KeyError:
        u = pars["sixvertex_u"]
        lmbd = pars["sixvertex_lambda"]
        rho = pars["sixvertex_rho"]
        a = rho*np.sin(lmbd - u)
        b = rho*np.sin(u)
        c = rho*np.sin(lmbd)
    T_0 = np.zeros((2,2,2,2), dtype=pars["dtype"])
    T_0[1,0,0,1] = a
    T_0[0,1,1,0] = a
    T_0[0,0,1,1] = b
    T_0[1,1,0,0] = b
    T_0[0,1,0,1] = c
    T_0[1,0,1,0] = c
    if pars["symmetry_tensors"]:
        dim = [1,1]
        qim = [-1,1]
        T_0 = TensorU1.from_ndarray(T_0, shape=[dim]*4, qhape=[qim]*4,
                                    dirs=[1,1,1,1])
        T_0 = T_0.flip_dir(2)
        T_0 = T_0.flip_dir(3)
    else:
        T_0 = Tensor.from_ndarray(T_0)
    return T_0

