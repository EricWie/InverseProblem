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
        pr(new_conv_mat)
        inv_mat = np.linalg.pinv(new_conv_mat)
        coeff = inv_mat @ lhs
        return coeff



def main():
    w = 2 * np.pi
    eps = 10
    poly_coeff = np.array([-4, 0, 1,0,1])  
    add_order = 4           
    
    coeff = inverse_problem(poly_coeff[::-1], eps, w,add_order=add_order)
    pr(coeff)

if __name__=='__main__':
    main()