import numpy as np
import networkx as nx

def compute_dist(X):
    # computes all (squared) pairwise Euclidean distances between each data point in X
    # D_ij = <x_i - x_j, x_i - x_j>
    G = np.matmul(X, X.T)
    D = np.diag(G) + np.reshape(np.diag(G), (-1, 1)) - 2 * G
    return D

def compute_kernel(D, eps, d):
    # computes kernel for approximating GL
    # D is matrix of pairwise distances
    K = np.exp(-D/eps) * np.power(eps, -d/2)
    return K

def compute_laplacian(A, q):
    n = A.shape[0]
    theta = 2 * np.pi * q * (A - A.T)
    A_s = 1/2 * (A + A.T)
    degrees = np.sum(A_s, axis=1)
    D = np.diag(np.power(np.sum(A_s, axis=1), -0.5))
    D_mod = D.copy()
    D_mod[np.isinf(D)] = 0
    I_mat = np.identity(n)
    I_mat[np.isinf(D)] = 0
    L  = I_mat - np.matmul(np.matmul(D_mod, A_s), D_mod) * np.exp(1j * theta) 
    return L

def compute_eigen(A, q):
    L = compute_laplacian(A, q)
    S, U = np.linalg.eigh(L)
    S = np.reshape(S.real, (1, -1))
    S[0,0] = 0 # manually enforce this
    U = np.divide(U, np.linalg.norm(U, axis=0, keepdims=True))
    return S, U    

def est_prob_density(dists, eps):
    N = (dists < eps).sum(axis=1, keepdims=True)
    return N

def manifold_normalize(U, d, density, eps):
    coef = math.pow(eps, d) * math.pow(math.pi, (d-1)/2)/math.gamma((d-1)/2 + 1) / d
    norms = np.sqrt(coef * np.divide(np.power(U, 2), density).sum(axis=0, keepdims=True))
    U_normalized = np.divide(U, norms)
    return U_normalized

def compute_wavelet_filter(eigenvec, eigenval, j):
    H = np.einsum('ik,jk->ij', eigenvec * h(eigenval, j), eigenvec)
    return H

def g(lam):
    return np.exp(-lam)

def h(lam,j):
    return g(lam)**(2**(j-1)) - g(lam)**(2**j)

def calculate_wavelet(eigenval,eigenvec,scales):
    dilation = scales.tolist()
    J = dilation[-1]
    wavelet = []
    N = eigenvec.shape[0]
    for dil in dilation:
        if dil == 0:
            wavelet.append(np.identity(N) - np.einsum('ik,jk->ij', eigenvec * g(eigenval), np.conjugate(eigenvec)))
        else:
            wavelet.append(compute_wavelet_filter(eigenvec, eigenval, dil))
    return np.dstack(wavelet), np.einsum('ik,jk->ij', eigenvec * g(eigenval), np.conjugate(eigenvec))

def weighted_wavelet_transform(wavelet, f, N):
    return np.einsum('ijk,j...->i...k', wavelet, f)

def zero_order_feature(Aj, f, N, agg_choice):
    if (agg_choice == "none" or agg_choice == "lowpass"):
        F0 = np.matmul(Aj, f)
        F0 = np.hstack((np.real(F0), np.imag(F0)))
    else:
        norm_list = agg_choice
        this_F0 = np.abs(np.matmul(Aj, f))
        F0 = np.sum(np.power(this_F0, norm_list[0]), axis=0).reshape(-1, 1)
        for i in range(2, len(norm_list)):
            F0 = np.vstack((F0, np.sum(np.power(this_F0, norm_list[i]), axis=0).reshape(-1, 1)))
    return F0

def first_order_feature(psi, Wf, Aj, N, agg_choice):
    F1 = np.abs(np.einsum('ij,j...k->i...k', Aj, Wf))
    if agg_choice == "none":
        F1 = Wf
        F1 = np.reshape(F1, (N, -1), order='F')
        F1 = np.hstack((np.real(F1), np.imag(F1)))
    elif agg_choice == "lowpass":
        F1 = np.einsum('ij,j...k->i...k', Aj, np.abs(Wf))
        F1 = np.reshape(F1, (N, -1), order='F')
        F1 = np.hstack((np.real(F1), np.imag(F1)))
    else:
        norm_list = agg_choice
        this_F1 = F1
        F1 = np.sum(np.power(this_F1, norm_list[0]), axis=0).reshape(-1, 1)
        for i in range(2, len(norm_list)):
            F1 = np.vstack((F1, np.sum(np.power(this_F1, norm_list[i]), axis=0).reshape(-1, 1)))    
    return F1
    
def selected_second_order_feature(psi,Wf,Aj, N, agg_choice):
    temp = np.abs(Wf[...,0])
    F2 = np.einsum('ij,j...->i...', psi[:, :, 1], temp)
    if len(Wf.shape) > 2:
        F2 = np.expand_dims(F2, axis=2)
    for i in range(2,psi.shape[2]):
        temp = np.abs(Wf[...,0:i])
        F2 = np.concatenate((F2, 1/N * np.einsum('i...j,j...k->i...k',psi[..., i], temp)), axis=-1)
    F2 = np.abs(F2)
    if agg_choice == "none":
        F2 = np.reshape(F2, (N, -1), order='F')
        F2 = np.hstack((np.real(F2), np.imag(F2)))
    elif agg_choice == "lowpass":
        F2 = np.einsum('ij,j...k->i...k', Aj, F2)
        F2 = np.reshape(F2, (N, -1), order = 'F')
        F2 = np.hstack((np.real(F2), np.imag(F2)))
    else:
        norm_list = agg_choice
        this_F2 = F2
        F2 = np.sum(np.power(this_F2, norm_list[0]), axis=0).reshape(-1, 1)
        for i in range(2, len(norm_list)):
            F2 = np.vstack((F2, np.sum(np.power(this_F2, norm_list[i]), axis=0).reshape(-1, 1)))
    return F2

def generate_feature(psi,Wf,Aj,f, N, agg_choice="none"):
    F0 = zero_order_feature(Aj, f, N, agg_choice)
    F1 = first_order_feature(psi,Wf,Aj, N, agg_choice)
    F2 = selected_second_order_feature(psi,Wf,Aj, N, agg_choice)
    F = np.concatenate((F0, F1), axis=1)
    F = np.concatenate((F,F2),axis=1)
    return F, [F0, F1, F2]

def compute_all_features(eigenval, eigenvec, signal, N, agg_choice, scales):
    psi,Aj = calculate_wavelet(eigenval,eigenvec,scales)
    Wf = weighted_wavelet_transform(psi,signal, N)
    all_features = generate_feature(psi, Wf, Aj, signal, N, agg_choice)
    return all_features