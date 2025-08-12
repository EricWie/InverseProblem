import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt
import pandas as pd
from functionality import *
from main import *

def inverse_problem_seconde_order(fourier:fourierseries, num_coeff, eps:fourierseries, w):
    """solves seconde oderer inverse problem: d^2x/(dt)^2+a*dx/dt=f(x)+eps(t)"""
    eps.expand_to_order(fourier.order)
    # calculate the derivativs
    deriv_1 = fourierseries(fourier.derivativ(w))
    deriv_2 = fourierseries(deriv_1.derivativ(w))
    
    # lhs of the problem
    lhs = deriv_1.convert_list()-eps.convert_list()

    # solve for coefficant vector with penroseinverse
    new_conv_mat = fourier.d_mat(num_coeff)
    new_conv_mat = np.hstack((new_conv_mat,np.array([deriv_2.convert_list()]).T))
    
    inv_mat = np.linalg.pinv(new_conv_mat)
    coeff = inv_mat @ lhs 
    pr(coeff)
    return coeff

def main_second_order():
    """example for second order inverse problem"""
    w = 2 * np.pi
    a = 1 
    intrinsic = [1,0,-4]
    eps = fourierseries([0,10,0])
    order = 15
    fourier = get_info_seconde_order(a,intrinsic,start=None,eps=eps,w=w,order=order)

    inverse_problem_seconde_order(fourier,len(intrinsic),eps,w)


if __name__ ==  '__main__':
    main_second_order()