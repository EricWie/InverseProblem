import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt
import pandas as pd
from functionality import get_important_info, real_fourier, dervative_fourier_coefficients


def pr(x):
    df = pd.DataFrame(x)
    print(df)

def inverse_problem(poly_coeff, eps, w, add_order=0):

    order = len(poly_coeff)//2+add_order

    four, real, D_mat = get_important_info(poly_coeff, eps, w,  order)

    diff_coff = dervative_fourier_coefficients(real,w)

    lhs = np.copy(diff_coff)
    lhs[1] += (-eps)

    if not add_order:
        inv_D = np.linalg.inv(D_mat)
        coeff = inv_D @ lhs
        return coeff
    
    else:
        new_conv_mat = D_mat[:,:len(poly_coeff)]
        inv_mat = np.linalg.pinv(new_conv_mat)
        coeff = inv_mat @ lhs
        return coeff

def inverse_problem_unknow_excite(poly_coeff, eps,  w,ep_order=1, add_order=0):
    order = len(poly_coeff)//2+add_order

    eps_mat  = np.zeros((2*order+1,2*ep_order))
    for i in range(2*ep_order):
        eps_mat[i+1,i] = 1
    pr(eps_mat)

    four, real, D_mat = get_important_info(poly_coeff, eps, w,  order)

    diff_coff = dervative_fourier_coefficients(real,w)

    lhs = diff_coff

    new_conv_mat = D_mat[:,:len(poly_coeff)]

    new_conv_mat = np.hstack((new_conv_mat,eps_mat))

    inv_mat = np.linalg.pinv(new_conv_mat)
    coeff = inv_mat @ lhs
    return coeff



def main():
    w = 2 * np.pi
    eps = 10
    poly_coeff = np.array([-6, 0, 1,0,1])  

    add_order = 5           
    
    coeff = inverse_problem_unknow_excite(poly_coeff[::-1], eps, w,add_order=add_order)
    pr(coeff)

    coeff = inverse_problem(poly_coeff[::-1], eps, w,add_order=add_order)
    pr(coeff)

if __name__=='__main__':
    main()