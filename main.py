import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt
import pandas as pd
from functionality import *

def pr(x):
    """nicer looking prints of arrays"""
    df = pd.DataFrame(x)
    print(df)

def inverse_problem(real, D_mat, num_coeff, eps, w):
    """normal invers proble with excitation eps*cos"""
    # number of fourier coefficants used
    diff_coff = dervative_fourier_coefficients(real,w)

    # landside of the equation
    lhs = np.copy(diff_coff)
    lhs[1] += (-eps)
    
    # solve for coefficant vector with penroseinverse
    new_conv_mat = D_mat[:,:num_coeff]
    inv_mat = np.linalg.pinv(new_conv_mat)
    coeff = inv_mat @ lhs
    return coeff

def main_inverse_problem():
    w = 2 * np.pi
    eps = 10
    
    func = 'poly'
    if func =='poly':
        #largest coefficant must be positiv else solver diverges
        poly_coeff = np.array([1, 0, 1,0,-6])
        #how many orders to calculate
        num_coeff = len(poly_coeff)
        add_order = 10
        order = int(num_coeff//2 + add_order)

        # get timeseries data polynomial
        four, real, D_mat = get_important_info(poly_coeff,start=None,eps=eps, w=w, order=order)

        #real *= (1+0.05*(np.random.rand(len(real))-0.5))
        pr(inverse_problem(real, D_mat,num_coeff,eps,w))

    elif func == 'cos':
        #get timeseries cosine
        num_coeff = 10
        add_order = 8
        order = num_coeff//2 + add_order
        intri = lambda x: 6*np.cos(x)
        four, real, D_mat = get_important_info(intri,start=np.pi/2,eps=eps, w=w, order=order)
        pr(inverse_problem(real, D_mat,num_coeff,eps,w))

def inverse_problem_unknow_excite(real,D_mat, w, num_coef, ep_order=1):
    """ finds coeeficants and pertubation from timeseries"""
    # matrix for the fourier coefficants  
    eps_mat  = np.zeros((len(real),2*ep_order))
    for i in range(2*ep_order):
        eps_mat[i+1,i] = 1

    diff_coff = dervative_fourier_coefficients(real,w)

    lhs = diff_coff

    new_conv_mat = D_mat[:,:num_coef]

    new_conv_mat = np.hstack((new_conv_mat,eps_mat))

    inv_mat = np.linalg.pinv(new_conv_mat)
    coeff = inv_mat @ lhs
    return coeff[:-2*ep_order],coeff[-2*ep_order:]

def main_unknow_excite():
    w = 2 * np.pi
    eps = 10
    #largest coefficant must be positiv else solver diverges
    poly_coeff = np.array([1, 0, 1,0,-6])  
    N = len(poly_coeff)
    add_order = 10           
    
    four, real, D_mat = get_important_info(poly_coeff,None, eps, w,  N//2+add_order)

    coeff_pol,coeff_eps = inverse_problem_unknow_excite(real, D_mat, w,N,ep_order=1)
    pr(coeff_pol)
    pr(coeff_eps)


def inverse_problem_network(ls_real,ls_D_mat,w,num_coeff,order_eps):
    N = len(ls_real)

    ls_coeff = []
    ls_eps = []
    # looking at each node seperatly
    for i in range(N):
        real = ls_real[i]
        D_mat = ls_D_mat[i]
        #solving the inv. Problem
        coeff_poly, coeff_eps = inverse_problem_unknow_excite(real,D_mat, w, num_coeff, ep_order=order_eps)

        ls_coeff.append(coeff_poly)
        ls_eps.append(coeff_eps)
    return ls_coeff, ls_eps

def main_network():
    """network reconstruction form time series data"""
    # Parameter definition
    A = np.array([[0,0],[1,0]])
    poly_coeff = [[1,0,-4],[1,0,-4]]
    eps = 10
    w = 2*np.pi
    order = len(poly_coeff[0])
    add_order = 3
    order_eps = 1

    # get timeseriers data from the network
    ls_four,ls_real,ls_D_mat = get_info_network(A,poly_coeff,eps,w,order+add_order)

    #solve Invers Problem in the Network case
    ls_coeff, ls_eps = inverse_problem_network(ls_real,ls_D_mat,w,order,order_eps)
    pr(ls_coeff)
    pr(ls_eps)




if __name__ == '__main__':
    main_inverse_problem()