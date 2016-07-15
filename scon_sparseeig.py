import numpy as np
import operator as opr
import itertools as itt
import functools as fct
import scipy.sparse.linalg as spsla
from scon import scon
from tensors.abeliantensor import AbelianTensor


def get_commons(tensor_list):
    errmsg = "tensor_list in scon_sparseeig has inhomogenous "

    types = set(map(type, tensor_list))
    if len(types) > 1:
        raise ValueError(errmsg + "types.")
    commontype = types.pop()

    dtypes = set(t.dtype for t in tensor_list)
    commondtype = np.find_common_type(dtypes, [])

    qoduli = set(t.qodulus if hasattr(t, "qodulus") else None
                    for t in tensor_list)
    if len(qoduli) > 1:
        raise ValueError(errmsg + "qoduli.")
    commonqodulus = qoduli.pop()

    return commontype, commondtype, commonqodulus


def get_free_indexdata(tensor_list, index_list):
    """ Figure out the numbers, dims, qims and dirs of all the free
    indices of the network that we want the eigenvalues of.
    """
    inds = []
    dims = []
    qims = []
    dirs = []
    for t, l in zip(tensor_list, index_list):
        for j, k in enumerate(l):
            if k < 0:
                inds.append(k)
                dims.append(t.shape[j])
                if t.qhape is not None:
                    dirs.append(t.dirs[j])
                    qims.append(t.qhape[j])
                else:
                    dirs.append(None)
                    qims.append(None)
    inds, dims, qims, dirs = zip(*sorted(zip(inds, dims, qims, dirs),
                                         reverse=True))
    return inds, dims, qims, dirs


def get_in_indexdata(commontype, free_dims, free_qims, free_dirs, in_inds):
    in_dims = list(map(free_dims.__getitem__, in_inds))
    in_qims = list(map(free_qims.__getitem__, in_inds))
    if in_qims[0] is None:
        in_qims = None
    in_dirs = map(free_dirs.__getitem__, in_inds)
    try:
        in_dirs = list(map(opr.neg, in_dirs))
    except TypeError:
        in_dirs = None
    in_flatdims = list(map(commontype.flatten_dim, in_dims))
    return in_dims, in_qims, in_dirs, in_flatdims


def update_index_list(index_list, free_inds, in_inds):
    """ Flip the signs of the contraction indices for the vector. """
    c_inds = tuple(map(free_inds.__getitem__, in_inds))
    c_inds_set = set(c_inds)
    # Change the signs of the corresponding indices in index_list.
    index_list = [[-i if i in c_inds_set else i
                   for i in l]
                  for l in index_list]
    c_inds = list(map(opr.neg, c_inds))
    index_list.append(c_inds)
    return index_list


def get_qnums(in_qims, qodulus, qnums_do):
    all_qnums = map(sum, itt.product(*in_qims))
    if commonqodulus is not None:
        all_qnums = set(q % commonqodulus for q in all_qnums)
    else:
        all_qnums = set(all_qnums)
    if qnums_do:
        qnums = sorted(all_qnums & set(qnums_do))
    else:
        qnums = sorted(all_qnums)
    return qnums


def get_eigblocks(scon_op, charge, hermitian, return_eigenvectors,
                  matrix_flatdim, commondtype, **kwargs):
    scon_op_lo = spsla.LinearOperator(
        (matrix_flatdim, matrix_flatdim), fct.partial(scon_op, charge=q),
        dtype=commondtype
    )
    if hermitian:
        res_block = spsla.eigsh(
            scon_op_lo, return_eigenvectors=return_eigenvectors,
            **kwargs
        )
    else:
        res_block = spsla.eigs(
            scon_op_lo, return_eigenvectors=return_eigenvectors,
            **kwargs
        )
    if return_eigenvectors:
        S_block, U_block = res_block
    else:
        S_block = res_block

    order = np.argsort(-np.abs(S_block))
    S_block = S_block[order]
    if return_eigenvectors:
        U_block = U_block[:,order]
        U_block = np.reshape(U_block, in_flatdims+[n_eigs])
        U_block = commontype.from_ndarray(U_block, shape=in_dims+[[n_eigs]],
                                          qhape=in_qims+[[q]],
                                          dirs=in_dirs+[-1])
    retval = (S_block,)
    if return_eigenvectors:
        retval += (U_block,)
    return retval


def get_eig(scon_op, hermitian, n_eigs, return_eigenvectors, in_dims,
            matrix_flatdim, commontype, commondtype, **kwargs):
    scon_op_lo = spsla.LinearOperator((matrix_flatdim, matrix_flatdim),
                                      scon_op, dtype=commondtype)
    if hermitian:
        res = spsla.eigsh(scon_op_lo,
                          return_eigenvectors=return_eigenvectors,
                          **kwargs)
    else:
        res = spsla.eigs(scon_op_lo,
                         return_eigenvectors=return_eigenvectors,
                         **kwargs)
    if return_eigenvectors:
        S, U = res
        U = commontype.from_ndarray(U)
        U = U.reshape(in_dims+[n_eigs])
    else:
        S = res
    order = np.argsort(-np.abs(S))
    S = S[order]
    S = commontype.from_ndarray(S)
    if return_eigenvectors:
        U = U[...,order]
        U = commontype.from_ndarray(U)
    retval = (S,)
    if return_eigenvectors:
        retval += (U,)
    return retval


def preprocess(tensor_list, index_list, in_inds, out_inds,
               print_progress=False, scon_func=None, kwargs={}):
    tensor_list = list(tensor_list)
    index_list = list(index_list)
    in_inds = tuple(in_inds)
    out_inds = tuple(out_inds)
    n_eigs = kwargs.setdefault("k", 6)

    commontype, commondtype, commonqodulus = get_commons(tensor_list)

    free_inds, free_dims, free_qims, free_dirs = get_free_indexdata(
        tensor_list, index_list
    )

    in_dims, in_qims, in_dirs, in_flatdims = get_in_indexdata(
        commontype, free_dims, free_qims, free_dirs, in_inds
    )
    matrix_flatdim = fct.reduce(opr.mul, in_flatdims)

    # Flip the signs of the contraction indices for the vector.
    index_list = update_index_list(index_list, free_inds, in_inds)

    # The permutation on the final legs.
    perm = list(np.argsort(out_inds))

    if scon_func is None:
        def scon_func(v):
            scon_list = tensor_list + [v]
            Av = scon(scon_list, index_list)
            return Av

    # TODO could we initialize an initial guess of commontype and avoid
    # all the to/from ndarray?
    def scon_op(v, charge=0):
        v = np.reshape(v, in_flatdims)
        v = commontype.from_ndarray(v, shape=in_dims, qhape=in_qims,
                                    charge=charge, dirs=in_dirs)
        Av = scon_func(v)
        Av = Av.to_ndarray()
        Av = np.transpose(Av, perm)
        Av = np.reshape(Av, (matrix_flatdim,))
        if print_progress:
            print(".", end='', flush=True)
        return Av

    if print_progress:
        print("Diagonalizing...", end="")

    return (scon_op, matrix_flatdim, in_qims, in_dims,
            commontype, commondtype, commonqodulus, n_eigs)


def scon_sparseeig(tensor_list, index_list, in_inds, out_inds,
                   hermitian=False, print_progress=False, qnums_do=(),
                   return_eigenvectors=True, scon_func=None, **kwargs):
    
    (scon_op, matrix_flatdim, in_qims, in_dims,
     commontype, commondtype, commonqodulus, n_eigs) = preprocess(
        tensor_list, index_list, in_inds, out_inds,
        print_progress=print_progress, kwargs=kwargs
    )

    if issubclass(commontype, AbelianTensor):
        # For AbelianTensors.
        # Figure out the list of charges for eigenvectors.
        qnums = get_qnums(in_qims, commonqodulus, qnums_do)

        # Initialize S and U.
        S_dtype = np.float_ if hermitian else np.complex_
        S = commontype.empty(shape=[[n_eigs]*len(qnums)],
                             qhape=[qnums], invar=False,
                             dirs=[1], dtype=S_dtype)
        if return_eigenvectors:
            U_dtype = commondtype if hermitian else np.complex_
            U = commontype.empty(shape=in_dims+[[n_eigs]*len(qnums)],
                                 qhape=in_qims+[qnums],
                                 dirs=in_dirs+[-1], dtype=U_dtype)

        # Find the eigenvectors in all the charge sectors one by one.
        for q in qnums:
            blocks = get_eigblocks(scon_op, charge, hermitian,
                                   return_eigenvectors, matrix_flatdim,
                                   commondtype, **kwargs)
            S[(q,)] = blocks[0]
            if return_eigenvectors:
                U_block = blocks[1]
                for k, v in U_block.sects.items():
                    U[k] = v

    else:
        # For regular tensors.
        res = get_eig(scon_op, hermitian, n_eigs, return_eigenvectors, in_dims, 
                      matrix_flatdim, commontype, commondtype, **kwargs)
        S = res[0]
        if return_eigenvectors:
            U = res[1]

    if print_progress:
        print()

    retval = (S,)
    if return_eigenvectors:
        retval += (U,)
    if len(retval) == 1:
        retval = retval[0]
    return retval

