import numpy as np
import collections
from tensors import tensor, symmetrytensors, tensorcommon
from functools import reduce
from scon import scon


def contract2x2(T_list, vert_flip=False):
    """ Takes an iterable of rank 4 tensors and contracts a square made
    of them to a single rank 4 tensor. If only a single tensor is given
    4 copies of the same tensor are used. If vert_flip=True the lower
    two are vertically flipped and complex conjugated.
    """
    if isinstance(T_list, (np.ndarray, tensorcommon.TensorCommon)):
        T = T_list
        T_list = [T]*4
    else:
        T_list = list(T_list)
    if type(T_list[0]) is np.ndarray:
        return contract2x2_ndarray(T_list, vert_flip=vert_flip)
    else:
        return contract2x2_Tensor(T_list, vert_flip=vert_flip)

def contract2x2_Tensor(T_list, vert_flip=False):
    if vert_flip:
        def flip(T):
            T.transpose((0,3,2,1))
        flip(T_list[2])
        flip(T_list[3])
    T4 = scon((T_list[0], T_list[1], T_list[2], T_list[3]),
              ([-2,-3,1,3], [1,-4,-6,4], [-1,3,2,-7], [2,4,-5,-8]))
    T4 = T4.join_indices((0,1), (2,3), (4,5), (6,7), dirs=[1,1,-1,-1])
    return T4

def contract2x2_ndarray(T_list, vert_flip=False):
    if vert_flip:
        def flip(T):
            return np.transpose(T.conjugate(), (0,3,2,1))
        T_list[2] = flip(T_list[2])
        T_list[3] = flip(T_list[3])
    T4 = scon((T_list[0], T_list[1], T_list[2], T_list[3]),
              ([-2,-3,1,3], [1,-4,-6,4], [-1,3,2,-7], [2,4,-5,-8]))
    sh = T4.shape
    S = np.reshape(T4, (sh[0]*sh[1], sh[2]*sh[3], sh[4]*sh[5], sh[6]*sh[7]))
    return S


def contract2x2x2(T_list):
    """ Takes an iterable of rank 6 tensors and contracts a cube made
    of them to a single rank 6 tensor. If only a single tensor is given
    8 copies of the same tensor are used.
    """
    if isinstance(T_list, (np.ndarray, tensorcommon.TensorCommon)):
        T = T_list
        T_list = [T]*8
    else:
        T_list = list(T_list)
    if type(T_list[0]) is np.ndarray:
        return contract2x2x2_ndarray(T_list)
    else:
        return contract2x2x2_Tensor(T_list)


def contract2x2x2_Tensor_flipped(T):
    T2 = scon((T, T.conjugate()),
              ([11,-21,-31,-41,-51,-61],
               [11,-22,-32,-42,-52,-62]))
    T4 = scon((T2, T2.conjugate()),
              ([-11,-21,-31,-41,-51,-61,-71,-81,91,101],
               [-12,-22,-32,-42,-52,-62,-72,-82,91,101]))
    T8 = scon((T4, T4.conjugate()),
              ([11,21,31,41,-51,-61,-71,-81,-91,-101,-111,-121,-131,-141,-151,-161],
               [11,21,31,41,-52,-62,-72,-82,-92,-102,-112,-122,-132,-142,-152,-162]))
    T8 = T8.transpose((6,7,4,5,
                       11,15,9,13,
                       2,3,0,1,
                       10,14,8,12,
                       17,21,16,20,
                       19,23,18,22))
    T8 = T8.join_indices((0,1,2,3), (4,5,6,7), (8,9,10,11),
                         (12,13,14,15), (16,17,18,19), (20,21,22,23),
                         dirs=[1,1,-1,-1,1,-1])
    return T8


def contract2x2x2_Tensor(T_list):
    Tcube = scon((T_list[0], T_list[1], T_list[2], T_list[3], T_list[4],
                  T_list[5], T_list[6], T_list[7]),
                 ([-2,-5,7,3,-18,1], [7,-8,-10,9,-19,6],
                  [12,9,-9,-14,20,10], [-1,3,12,-13,-17,4],
                  [-3,-6,5,2,1,-22], [5,-7,-11,8,6,-23],
                  [11,8,-12,-15,10,-24], [-4,2,11,-16,4,-21]))
    S = Tcube.join_indices((0,1,2,3), (4,5,6,7), (8,9,10,11),
                           (12,13,14,15), (16,17,18,19), (20,21,22,23),
                           dirs=[1,1,-1,-1,1,-1])
    return Tcube


def contract2x2x2_ndarray(T_list):
    Tcube = scon((T_list[0], T_list[1], T_list[2], T_list[3], T_list[4],
                  T_list[5], T_list[6], T_list[7]),
                 ([-2,-5,7,3,-18,1], [7,-8,-10,9,-19,6],
                  [12,9,-9,-14,20,10], [-1,3,12,-13,-17,4],
                  [-3,-6,5,2,1,-22], [5,-7,-11,8,6,-23],
                  [11,8,-12,-15,10,-24], [-4,2,11,-16,4,-21]))
    sh = Tcube.shape
    S = np.reshape(Tcube,
                   (sh[0]*sh[1]*sh[2]*sh[3], sh[4]*sh[5]*sh[6]*sh[7],
                    sh[8]*sh[9]*sh[10]*sh[11],
                    sh[12]*sh[13]*sh[14]*sh[15],
                    sh[16]*sh[17]*sh[18]*sh[19],
                    sh[20]*sh[21]*sh[22]*sh[23]))
    return S


def contract3x3(T):
    """ Takes a rank 4 tensor T and contracts a square made of 9 copies of it
    to a single rank 4 tensor.
    """
    T_2 = np.tensordot(T, T, (2,0))
    T_row = np.tensordot(T_2, T, (4,0))
    T_row = np.transpose(T_row, (0,1,3,5,6,2,4,7))
    shp = T_row.shape
    T_row = np.reshape(T_row, (
        shp[0], shp[1]*shp[2]*shp[3], shp[4], shp[5]*shp[6]*shp[7]))
    T_row2 = np.tensordot(T_row, T_row, (3,1))
    T_sqr = np.tensordot(T_row2, T_row, (5,1))
    T_sqr = np.transpose(T_sqr, (5,3,0,1,6,4,2,7))
    shp = T_sqr.shape
    T_sqr = np.reshape(T_sqr,
            (shp[0]*shp[1]*shp[2], shp[3], shp[4]*shp[5]*shp[6], shp[7]))
    return T_sqr


def orthonormalize_in_place(M):
    """ Orthonormalizes the columns of matrix M in place using stabilized
    Gram-Schmidt.
    """
    for i, col1 in enumerate(M.T):
        M[:,i] = col1/np.linalg.norm(col1)
        for j, col2 in enumerate(M[:,i+1:].T):
            M[:,j+i+1] = col2 - col1*np.dot(col1,col2)
    return M


def tensor_frob_norm_sq(A):
    l = len(A.shape)
    indices = list(range(l))
    return np.abs(np.tensordot(A, A.conjugate(), (indices,indices)))


def trivial_projector(d1,d2):
    """ Produces a matrix that projects from R^d2 to R^d1 by keeping the
    first d1 rows. If d1>d2, just return the identity.
    """
    if d1>=d2:
        return np.eye(d2)
    else:
        i = np.eye(d1)
        z = np.zeros((d1,d2-d1))
        return np.append(i,z,axis=1)


def unitarize(M):
    """ Returns the unitary matrix that is closest to the matrix M in
    terms of the Frobenius norm.
    """
    try:
        U,S,V = M.svd()
        result = U.dot(V)
    except (TypeError, AttributeError):
        U,S,V = np.linalg.svd(M)
        result = np.dot(U,V)
    return result


def direct_sum(M,N):
    """ Direct sum of two matrices. """
    # TODO check that M and N are matrices
    Ms = M.shape
    Ns = N.shape
    shape = np.array(Ms) + np.array(Ns)
    try:
        P = type(M).zeros(shape, dtype=M.dtype)
    except AttributeError:
        P = np.zeros(shape, dtype=M.dtype)
    P[:Ms[0], :Ms[1]] = M
    P[-Ns[0]:, -Ns[1]:] = N
    return P


def get_degeneracies(S, tol=1e-12):
    """ Given a list of eigenvalues or singular values, return a list of
    integers that are the degeneracies. S is assumed to be ordered. tol
    sets the tolerance for when values are considered to be the same
    (i.e. degenerate).
    """
    if len(S)<1:
        degs = [0]
    else:
        degs = [1]
        for i,e in enumerate(S[1:]):
            # Note that i starts from 0 even though e starts from S[1]
            if abs((e-S[i])/e) < tol:
                degs[-1] += 1
            else:
                degs += [1]
    return degs


def getCorrespondingEigs(M, N, tolerance=1e-14):
    """ Assuming that M and N can be diagonalized at the same time, return
    two arrays with the eigenvalues of M and N so that eigs_M[i] and
    eigs_N[i] correspond to the same eigenvector and eigs_M is sorted in
    ascending order. The parameter tolerance is for the neglecting numerical
    errors so that abs(x) < tolerance implies x==0.
    """
    eigs_M, V_M = np.linalg.eig(M)
    sorting_indices = np.argsort(eigs_M)
    eigs_M = eigs_M[sorting_indices]
    V_M = V_M[:,sorting_indices]
    # For degenerate eigenvalues, the eigenvectors may not be orthonormal.
    # Fix this by orthonormalizing the eigenbasis using QR decomposition.
    # Because M is assumed to be diagonalizable the resulting vectors
    # should still be eigenvectors of M.
    V_M, R = np.linalg.qr(V_M, mode='complete')
    # Transforming to eigenbasis of M makes N block diagonal, assuming N and M
    # can be diagonalized at the same time.
    N_blockdiag = np.dot(V_M.T.conjugate(), np.dot(N, V_M))
    # Account for numerical errors so that blocks of the block diagonal matrix
    # can be correctly recognized.
    # TODO This still does not work as well as it should.
    for i,row in enumerate(N_blockdiag):
        m = np.max(abs(row))
        for j,e in enumerate(row):
            if(abs(e/m)) < tolerance:
                N_blockdiag[i,j] = 0
    # Get the blocks and diagonalize them individually
    blocks = np.array(getDiagBlocks(N_blockdiag))
    # Calculate the degeneracies of eigs_M
    degs = [1]
    for i,e in enumerate(eigs_M[1:]):
        # Note that i starts from 0 even though e starts from eigs_M[1]
        if abs((e-eigs_M[i])/e) < 1e-12:
            degs[-1] += 1
        else:
            degs += [1]
    # Calculate the dimensions of the blocks
    block_dims = list(map(len, blocks))
    if(degs != block_dims):
        # TODO could also raise an exception or run a loop that changes
        #      tolerance dynamically untill a match is found.
        print("In getCorrespondingEigs, degs and block_dims don't match")
        print("block_dims: " + str(block_dims))
        print("degs: " + str(degs))
    blocks_eigs = [np.linalg.eigvals(M) for M in blocks]
    blocks_eigs = [np.sort(eigs) for eigs in blocks_eigs]
    eigs_N = np.concatenate(blocks_eigs)
    return eigs_M, eigs_N


def tensor_SVD(T, a, b, chis=None, eps=0, hermit=False, print_errors=0,
               return_error=False):
    """ Reshapes the tensor T have indices a on one side and indices b
    on the other, SVDs it as a matrix and reshapes the parts back to the
    original form. a and b should be iterables of integers that number
    the indices of T. 

    The optional argument chis is a list of bond dimensions. The SVD is
    truncated to one of these dimensions chi, meaning that only chi
    largest singular values are kept. If chis is a single integer
    (either within a singleton list or just as a bare integer) this
    dimension is used. If no eps is given, the largest value in chis is
    used. Otherwise the smallest chi in chis is used, such that the
    relative error made in the truncation is smaller than eps.

    If print_errors > 0 truncation errors are printed, with more
    information as print_errors increases through 1,2 and 3.

    By default the function returns the tuple (U,s,V), where in matrix
    terms U.diag(s).V = T, where the equality is appromixate if there is
    truncation.  If return_error = True a fourth value is returned,
    which is the ratio sum_of_discarded_singular_values /
    sum_of_all_singular_values.
    """

    # We want to deal with lists, not tuples or bare integers
    if isinstance(a, collections.Iterable):
        a = list(a)
    else:
        a = [a]
    if isinstance(b, collections.Iterable):
        b = list(b)
    else:
        b = [b]
    assert(len(a) + len(b) == len(T.shape))

    # Permute the indices of T to the right order
    perm = tuple(a+b)
    T_matrix = np.transpose(T, perm)
    # The lists shp_a and shp_b list the dimensions of the bonds in a and b
    shp = T_matrix.shape
    shp_a = shp[:len(a)]
    shp_b = shp[-len(b):]

    # Compute the dimensions of the the matrix that will be formed when
    # indices of a and b are joined together.
    dim_a = 1
    for s in shp_a:
        dim_a = dim_a * s
    dim_b = 1
    for s in shp_b:
        dim_b = dim_b * s
    # Create the matrix and SVD it.
    T_matrix = np.reshape(T_matrix, (dim_a, dim_b))
    U, s, V = np.linalg.svd(T_matrix, full_matrices=False)

    # Format the truncation parameters to canonical form.
    if chis is None:
        if eps > 0:
            # Try all possible chis.
            max_dim = min(dim_a, dim_b)
            chis = list(range(max_dim+1))
    else:
        if isinstance(chis, collections.Iterable):
            chis = list(chis)
        else:
            chis = [chis]
        if eps == 0:
            chis = [max(chis)]
        else:
            chis = sorted(chis)
    
    # Truncate, if truncation dimensions are given.
    if chis is None:
        # No truncation, no error.
        rel_err = 0
    else:
        sum_all_sq = sum(s**2)
        # Find the smallest chi for which the error is small enough.
        # If none is found, use the largest chi.
        for chi in chis:
            sum_disc_sq = sum((s**2)[chi:])
            rel_err_sq = sum_disc_sq/sum_all_sq
            if rel_err_sq <= eps**2:
                break
        # Truncate
        s = s[:chi]
        U = U[:,:chi]
        V = V[:chi,:]
        if print_errors > 0:
            print('----- Tensor SVD error -----')
            print('- Relative truncation error (Frobenius norm) from the '
                  'singular values: %.3e' % np.sqrt(rel_err_sq))
            if print_errors > 3:
                print('- sum(|s^2_discarded|): %.3e' % sum_disc_sq)
                print('- sum(|s^2_all|): %.3e' % sum_all_sq)
            if print_errors > 2:
                # Reconstruct the original T_matrix with the truncated
                # U, s, V and print the error in the reconstruction. The
                # error is computed as the Frobenius norm so it should
                # match the ratio of the discarded eigenvalues printed
                # earlier.
                reco = np.einsum('ij,j->ij', U, s)
                reco = np.dot(reco, V)
                reco_err = np.linalg.norm(T_matrix-reco)
                normalization = np.linalg.norm(T_matrix)
                reco_rel_err = reco_err/normalization
                print('- Relative truncation error (Frobenius norm) from a '
                      'reconstruction: %.3e' % reco_rel_err)
            print('----------------------------')
    # Reshape U and V to tensors with shapes matching the shape of T and
    # return.
    U_tens = np.reshape(U, shp_a + (-1,))
    V_tens = np.reshape(V, (-1,) + shp_b)
    ret_val = U_tens, s, V_tens
    if return_error:
        ret_val = ret_val + (rel_err,)
    return ret_val


def potts3T_0(beta, J, dtype=np.complex_, Z3=True, ndarray=False):
    """ Returns the non-coarse-grained tensor for the 3-state Potts
    model.
    """
    ham = -J*np.eye(3, dtype=dtype)
    boltz = np.exp(-beta*ham)
    T_0 = np.einsum('ab,bc,cd,da->abcd', boltz, boltz, boltz, boltz)
    if Z3:
        phase = np.exp(2j*np.pi/3)
        u = np.array([[1, 1, 1], [1, phase, phase**2], [1, phase**2, phase]],
                     dtype=dtype) / np.sqrt(3)
        u_dg = u.T.conjugate()
        T_0 = scon((T_0, u, u, u_dg, u_dg),
                   ([1,2,3,4], [1,-1], [2,-2], [3,-3], [4,-4]))
        T_0 = symmetrytensors.TensorZ3.from_ndarray(
                T_0, shape=[[1,1,1],[1,1,1],[1,1,1],[1,1,1]], dirs=[1,1,-1,-1])
    elif not ndarray:
        T_0 = tensor.Tensor.from_ndarray(T_0)
    return T_0


def isingT_0(beta, J, H=0, dtype=np.complex_, Z2=True, ndarray=False):
    """ Returns the non-coarse-grained tensor for the Ising model."""
    ham = dtype(J*np.array([[-1,1],[1,-1]]) + H*np.array([[-1,0],[0,1]]))
    boltz = np.exp(-beta*ham)
    T_0 = np.einsum('ab,bc,cd,da->abcd', boltz, boltz, boltz, boltz)
    if Z2:
        u = np.array([[1,1],[1,-1]], dtype=dtype)/np.sqrt(2)
        u_dg = u.T.conjugate()
        T_0 = scon((T_0, u, u, u_dg, u_dg),
                   ([1,2,3,4], [-1,1], [-2,2], [-3,3], [-4,4]))
        T_0 = symmetrytensors.TensorZ2.from_ndarray(T_0,\
                shape=[[1,1],[1,1],[1,1],[1,1]], dirs=[1,1,-1,-1])
    elif not ndarray:
        T_0 = tensor.Tensor.from_ndarray(T_0)
    return T_0


def ising3_initialtensor(beta, J, H=0, dtype=np.complex_, Z2=True):
    """ Returns the non-coarse-grained tensor for the Ising model."""
    ham = dtype(J*np.array([[-1,1],[1,-1]]) + H*np.array([[-1,0],[0,1]]))
    boltz = np.exp(-beta*ham)
    T_0 = np.einsum('ab,bc,cd,da,ef,fg,gh,he,ae,bf,cg,dh->abcdefgh',
                    *(boltz,)*12)
    if Z2:
        u = np.array([[1,1],[1,-1]], dtype=dtype)/np.sqrt(2)
        u_dg = u.T.conjugate()
        T_0 = scon((T_0,
                    u, u, u, u,
                    u_dg, u_dg, u_dg, u_dg),
                   ([1,2,3,4,5,6,7,8],
                    [-1,1], [-2,2], [-3,3], [-4,4],
                    [-5,5], [-6,6], [-7,7], [-8,8]))
        T_0 = symmetrytensors.TensorZ2.from_ndarray(T_0,\
                shape=[[1,1]]*8, dirs=[1,1,1,1,-1,-1,-1,-1])
    else:
        T_0 = tensor.Tensor.from_ndarray(T_0)
    return T_0



def CDL(chi, dtype=np.complex_):
    """ Returns the corner double line with constant bond dimension chi**2. """
    i = np.eye(chi)
    T = np.einsum('ab,cd,ef,gh->habcedgf', i,i,i,i)
    T = np.reshape(T, tuple([chi**2]*4))
    return T


def dimerT_0(dtype=np.complex_):
    T_0 = np.zeros((2,2,2,2), dtype=dtype)
    for i in range(4):
        indices = [0]*4
        indices[i] = 1
        T_0[tuple(indices)] = dtype(1)
    return T_0


def getDiagBlocks(M):
    """ Extract blocks of a block diagonal matrix. No check is made to
    ensure M is a square numpy.ndarray (which it should be for this to work).
    Returns a list of the blocks in order from top left to bottom right.
    """
    # Find the largest squares (i times i) that can be drawn at the
    # non-diagonal corners of M, such that the squares contain nothing
    # but zeros.
    i = 1
    # TODO this could be made more efficient by checking only 1D subarrays
    while (not np.any(M[:i,-i:])) and (not np.any(M[-i:,:i])) and i <= len(M):
        i += 1
    i -= 1
    # If i==0 the matrix is not block diagonal, i.e. has only one block.
    # If i==len(M) the matrix is full of zeros.
    if i==0 or i==len(M):
        return [M]
    # Divide the matrix into four rectangles
    upLeft = M[:i,:i]
    upRight = M[i:,:i]
    downLeft = M[:i,i:]
    downRight = M[i:,i:]
    # If the rectangles in the corners are not empty, try expanding them in the
    # other direction.
    if(np.any(upRight) or np.any(downLeft)):
        upLeft = M[:-i,:-i]
        upRight = M[-i:,:-i]
        downLeft = M[:-i,-i:]
        downRight = M[-i:,-i:]
        # If they still are not empty, the matrix is not block diagonal.
        if(np.any(upRight) or np.any(downLeft)):
            return [M]
    # If get up to this point, the matrix is block diagonal, and upLeft and
    # downRight are squares that form these blocks, although they may still
    # consist of smaller blocks.
    return getDiagBlocks(upLeft) + getDiagBlocks(downRight)

