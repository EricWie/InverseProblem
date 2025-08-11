import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt
import pandas as pd
from functionality import *

def pr(x):
    """nicer looking prints of arrays"""
    df = pd.DataFrame(x)
    print(df)

def inverse_problem(fourier, num_coeff, eps, w, no_linear_coeff=False):
    """normal invers problem with arbitray excitation"""
    # so that eps and fourier are compatible
    eps.expand_to_order(fourier.order)
    #lefthand side of the equation
    lhs = fourier.derivativ(w)
    lhs -= eps.convert_list()

    # solve for coefficant vector with penroseinverse
    new_conv_mat = fourier.d_mat(num_coeff)

    if no_linear_coeff:
        # delet linear row so that it has no impact
        new_conv_mat = np.delete(new_conv_mat,1, axis=1)

    #solve inverse problem
    inv_mat = np.linalg.pinv(new_conv_mat)
    coeff = inv_mat @ lhs   

    if no_linear_coeff:
        #insert 0  so that the size still fits
        coeff = np.insert(coeff,1,0)
    return coeff

def main_inverse_problem(): 
    """solves example inverse problem for 0 node with given exitation"""
    w = 2 * np.pi
    eps = fourierseries([0,10,0])

    func = 'poly'
    if func =='poly':
        #largest coefficant must be positiv else solver diverges
        poly_coeff = np.array([1,0,1,0,-6])
        #how many orders to calculate
        num_coeff = len(poly_coeff)
        add_order = 10
        order = int(num_coeff//2 + add_order)

        # get timeseries data polynomial
        fourier = get_important_info(poly_coeff,start=None,eps=eps, w=w, order=order) # type: ignore

        #real *= (1+0.05*(np.random.rand(len(real))-0.5))
        pr(inverse_problem(fourier,num_coeff,eps,w))

    elif func == 'cos':
        #get timeseries cosine
        num_coeff = 10
        add_order = 8
        order = num_coeff//2 + add_order
        intri = lambda x: 6*np.cos(x)
        fourier = get_important_info(intri,start=np.pi/2,eps=eps, w=w, order=order)
        pr(inverse_problem(fourier,num_coeff,eps,w))

def inverse_problem_network(A,ls_fourier, eps, w,num_coeff):
    """solves inverser problem in a network"""
    # number of nodes
    N = len(ls_fourier)
    ls_coeffs = []
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

        # solve inverse problem for each node
        coeff_i = inverse_problem(ls_fourier[i],num_coeff,perturb_i,w,no_linear_coeff=True)

        ls_coeffs.append(coeff_i)
    return ls_coeffs

def main_network():
    # network matrix
    A = np.array([[-2,1,1],[10,-12,2],[5,0,-10]])
    # coefficants matix
    poly_coeff = ([1,0,-4],[1,0,-6],[1,0,-8])
    # exciation at node 0
    eps = fourierseries([0,10,0])
    #periode
    w = 2*np.pi
    # order of fouriercoefficants used
    order = 22
    # order of returned coefficants of intrinsic dynamics
    num_coeff = 5
    ls_fourier = get_info_network(A,poly_coeff,eps,w,order)
    
    ls_res_coeff = inverse_problem_network(A,ls_fourier, eps, w,num_coeff)
    pr(ls_res_coeff)


def inverse_problem_unknown_excite(fourier,w,num_coeff,eps_order=1):
    """find both taylor expansion of intrinsic dynamics and excitation at the same time"""
    #addition to find the exciation
    N = len(fourier.convert_list())
    eps_mat  = np.zeros((N,2*eps_order))
    for i in range(eps_order):
        eps_mat[i+1,i] = 1
    for i in range(eps_order):
        eps_mat[fourier.order+i+1,i+eps_order] = 1
    #lefthand side of the equation
    lhs = fourier.derivativ(w)

    new_conv_mat = fourier.d_mat(num_coeff)

    new_conv_mat = np.hstack((new_conv_mat,eps_mat))

    inv_mat = np.linalg.pinv(new_conv_mat)
    coeff = inv_mat @ lhs
    return coeff[:-2*eps_order],fourierseries(np.concatenate(([0],coeff[-2*eps_order:])))

def main_unknow_excite():
    """example for determining the excitation in a 0-D system"""
    w = 2 * np.pi
    poly_coeff = [1,0,-4]
    eps = fourierseries([0,10,2,1,3,])
    order = 12
    fourier = get_important_info(poly_coeff,start=None,eps=eps, w=w, order=order) # type: ignore 

    intrinsic, eps_calc = inverse_problem_unknown_excite(fourier,w,num_coeff=4,eps_order=3)
    pr(intrinsic)
    print(eps_calc)


if __name__ == '__main__':
    main_network()