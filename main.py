import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt
import pandas as pd
from functionality import *

def pr(x):
    """nicer looking prints of arrays"""
    df = pd.DataFrame(x)
    print(df)

def inverse_problem(poly_coeff, eps, w, add_order=0):
    """normal invers proble with excitation eps*cos"""
    # number of fourier coefficants used
    order = len(poly_coeff)//2+add_order

    # get timeseries data
    four, real, D_mat = get_important_info(poly_coeff, eps, w,  order)

    diff_coff = dervative_fourier_coefficients(real,w)

    # landside of the equation
    lhs = np.copy(diff_coff)
    lhs[1] += (-eps)

    if not add_order:
        # solve for coefficant vector
        inv_D = np.linalg.inv(D_mat)
        coeff = inv_D @ lhs
        return coeff
    
    else:
        # solve for coefficant vector with penroseinverse
        new_conv_mat = D_mat[:,:len(poly_coeff)]
        inv_mat = np.linalg.pinv(new_conv_mat)
        coeff = inv_mat @ lhs
        return coeff

def inverse_problem_unknow_excite(poly_coeff, eps,  w,ep_order=1, add_order=0):
    """ finds coeeficants and pertubation from timeseries"""
    order = len(poly_coeff)//2+add_order

    # matrix for the fourier coefficants  
    eps_mat  = np.zeros((2*order+1,2*ep_order))
    for i in range(2*ep_order):
        eps_mat[i+1,i] = 1

    # get timeseries data
    four, real, D_mat = get_important_info(poly_coeff, eps, w,  order)

    diff_coff = dervative_fourier_coefficients(real,w)

    lhs = diff_coff

    new_conv_mat = D_mat[:,:len(poly_coeff)]

    new_conv_mat = np.hstack((new_conv_mat,eps_mat))

    inv_mat = np.linalg.pinv(new_conv_mat)
    coeff = inv_mat @ lhs
    return coeff

def inverse_problem_network(ls_four,ls_D_mat):
    N = len(ls_four)

    for i in range(N):
        yield



def main():
    w = 2 * np.pi
    eps = 10
    #largest coefficant must be positiv else solver diverges
    poly_coeff = np.array([-6, 0, 1,0,1])  

    add_order = 5           
    
    coeff = inverse_problem_unknow_excite(poly_coeff[::-1], eps, w,add_order=add_order)
    pr(coeff)

    coeff = inverse_problem(poly_coeff[::-1], eps, w,add_order=add_order)
    pr(coeff)


def main_network():
    """network reconstruction form time series data"""

    A = np.array([[0,1],[1,0]])
    poly_coeff = [[1,0,-4],[2,0,-10]]
    eps = 1
    w = 2*np.pi
    order = 4

    ls_four,ls_real,ls_D_mat = get_info_network(A,poly_coeff,eps,w,order)

    inverse_problem_network(ls_real,ls_D_mat)





if __name__=='__main__':
    main_network()