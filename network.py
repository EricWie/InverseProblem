import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt
import pandas as pd
from functionality import *
from main import *

def inverse_problem_network(A,ls_fourier, eps, w,num_coeff,eps_order=8,unknow_excite=False):
    """solves inverser problem in a network"""
    # number of nodes
    N = len(ls_fourier)
    ls_coeffs = []
    ls_eps = []
    # give excitation the right length
    eps.expand_to_order(ls_fourier[0].order)
    for i in range(N):
        eps_i = np.zeros_like(ls_fourier[0].convert_list())
        for k in range(N):
            # add up excitation from other nodes
            eps_i += A[i,k]*ls_fourier[k].convert_list()
        if i == 0:
            # add exciation to 0st node
            eps_i += eps.convert_list()
        
        perturb_i = fourierseries(eps_i)

        if not unknow_excite:
            # solve inverse problem for each node
            coeff_i = inverse_problem(ls_fourier[i],num_coeff,perturb_i,w,no_linear_coeff=True)
        else: 
            # solve inverse problem for each node + for excitation
            coeff_i, eps_i = inverse_problem_unknown_excite(ls_fourier[i],w,num_coeff,eps_order=eps_order, no_linear_coeff=True)
            ls_eps.append(eps_i)
        ls_coeffs.append(coeff_i)
    if not unknow_excite:
        return ls_coeffs
    else: 
        return ls_coeffs, ls_eps

def inverse_problem_identical(ls_fourier:list[fourierseries],w,num_coeff,eps_order=1, no_linear_coeff=False):
    """every node has the same unknown dynamics only the timeseries is know"""
    # number of nodes
    N = len(ls_fourier)
    order = ls_fourier[0].order
    size = 2 * order +1
    lhs = ls_fourier[0].derivativ(w)
    new_conv_mat = ls_fourier[0].d_mat(num_coeff)

    for i in range(N-1):
        D = ls_fourier[i+1].d_mat(num_coeff)
        new_conv_mat = np.vstack((new_conv_mat,D))
        lhs = np.hstack((lhs,ls_fourier[i+1].derivativ(w)))
    
    if no_linear_coeff:
        new_conv_mat = np.delete(new_conv_mat,1,axis=1)

    for j in range(N):
        eps_mat  = np.zeros((N*size,2*eps_order))
        for i in range(eps_order):
            eps_mat[j*size+i+1,i] = 1
        for i in range(eps_order):
            eps_mat[j*size+order+i+1,i+eps_order] = 1
        new_conv_mat = np.hstack((new_conv_mat,eps_mat))

    inv_mat = np.linalg.pinv(new_conv_mat)
    coeff = inv_mat @ lhs
    if no_linear_coeff:
        #insert 0  so that the size still fits
        coeff = np.insert(coeff,1,0)
    dyn_coeff = coeff[:num_coeff]
    eps_coeff = coeff[num_coeff:]

    pr(dyn_coeff)
    pr(eps_coeff)
def main_network(unknow_excite=False):
    # network matrix
    A = np.array([[-2,1,1],[10,-12,2],[5,0,-10]])
    # coefficants matix
    poly_coeff = ([1,0,-4])
    # exciation at node 0
    eps = fourierseries([0,10,0])
    #periode
    w = 2*np.pi
    # order of fouriercoefficants used
    order = 30
    # order of returned coefficants of intrinsic dynamics
    num_coeff = 3
    ls_fourier = get_info_network(A,poly_coeff,eps,w,order)
    
    if unknow_excite:
        ls_res_coeff, ls_eps = inverse_problem_network(A,ls_fourier, eps, w,num_coeff, unknow_excite=unknow_excite)
        pr(ls_res_coeff)
        for e in ls_eps:
            print(e)
    else:
        ls_res_coeff = inverse_problem_network(A,ls_fourier, eps, w,num_coeff)
        pr(ls_res_coeff)

def main_identical_network():
    # network matrix
    A = np.array([[-2,1,1],[10,-12,2],[5,0,-10]])
    # coefficants matix
    poly_coeff = ([1,0,-4])
    # exciation at node 0
    eps = fourierseries([0,17,0])
    #periode
    w = 2*np.pi
    # order of fouriercoefficants used
    order = 12
    # order of returned coefficants of intrinsic dynamics
    num_coeff = 3
    eps_order = 2
    ls_fourier = get_info_network(A,poly_coeff,eps,w,order)

    inverse_problem_identical(ls_fourier,w,num_coeff,eps_order=eps_order,no_linear_coeff=True)

if __name__ == '__main__':
    main_identical_network()
    #main_network(True)