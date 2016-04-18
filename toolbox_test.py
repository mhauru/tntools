import numpy as np
from toolbox import *

np.set_printoptions(precision=4)
np.set_printoptions(linewidth=120)

def testSumBlocks():
    """ Constructs a list of random number of square matrices of random sizes
    (up to 10), uses directSum to make them into a block diagonal matrix,
    dismantles that matrix with getDiagBlocks and checks that the result
    matches with the original matrices.
    """
    num_of_matrices = np.random.random_integers(1,10)
    dims = np.random.random_integers(1,10,size=num_of_matrices)
    mats = np.array([np.random.rand(d,d) for d in dims])
    sum_mat = mats[0]
    for M in mats[1:]:
        sum_mat = directSum(sum_mat, M)
    blocks = np.array(getDiagBlocks(sum_mat))
    success = all([np.all(a==b) for a,b in zip(mats,blocks)])
    # The check for matching lengths is necessary because zip truncates
    # to the length of the shorter.
    if success and len(mats) == len(blocks):
        print('testSumBlocks OK.')
    else:
        print('Mismatch in testSumBlocks.')
        print('blocks:')
        print(blocks)
        print('mats:')
        print(mats)


def testGetCorrespondingEigs():
    """ Generate random eigenvalues for two matrices and a set of random
    eigenvectors that the matrices share. Write the matrices in the diagonal
    form and then change to a non-eigenvector basis. Test whether
    getCorrespondingEigs manages to simultaneously diagonalize the matrices.
    The first of the matrices may have degenerate eigenvalues, the second one
    may not.
    """
    num_of_eigs_A = np.random.random_integers(2,10)
    # Degeneracies for A. The distribution is chosen on a whim.
    degs = np.ceil(np.random.poisson(0.5, size=num_of_eigs_A)+0.1)
    dim = np.sum(degs)
    eigs_A = np.random.rand(num_of_eigs_A)*1000
    # Repeat some of the eigenvalues according to the degeneracies in degs
    # Please excuse the non-idiomatic one-liner.
    eigs_A = np.array(sum(map(lambda t: [t[0]]*t[1], zip(eigs_A, degs)),[]))
    # No degeneracies for B
    eigs_B = np.random.rand(dim)*1000
    # Generate a random unitary matrix by SVDing a random matrix.
    # I don't know what the distribution is when you do it like this.
    M = np.random.rand(dim,dim)
    U,s,V = np.linalg.svd(M)
    U = np.dot(U,V)
    A = np.dot(U.T.conjugate(), np.dot(np.diag(eigs_A), U))
    B = np.dot(U.T.conjugate(), np.dot(np.diag(eigs_B), U))
    eigs2_B, eigs2_A = getCorrespondingEigs(B,A, 1e-8)
    eigs = sorted(list(zip(eigs_A, eigs_B)))
    eigs2 = sorted(list(zip(eigs2_A, eigs2_B)))
    print(np.array(list(zip(eigs, eigs2))))


def isingT_0_test():
    print(' -isingT_0_test- ')
    dtype = np.complex_
    J = dtype(1)
    beta = 0.44
    T = isingT_0(beta, J, dtype=dtype)
    H = np.array([[1,-1],[-1,1]])
    boltz = np.eye(2) + H*(np.exp(-2*beta*J) - 1)/2
    vert_sym_dif = T - np.transpose(T.conjugate(), (0,3,2,1))
    print('max(abs(vert_sym_dif)):' + str(np.max(np.abs(vert_sym_dif))))
    horz_sym_dif = T - np.transpose(T.conjugate(), (2,1,0,3))
    print('max(abs(horz_sym_dif)):' + str(np.max(np.abs(horz_sym_dif))))
    trace = np.einsum('abab->', T)
    print('tensor_trace' + str(trace))
    print('boltz_00^2 + boltz_11^2: ' + str(boltz[0,0]**2 + boltz[1,1]**2))


isingT_0_test()

