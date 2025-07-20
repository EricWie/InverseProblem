import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt
import pandas as pd



def solve_equation(coeff, eps, w,t_start =0, num_periods=100):
    """Solve the equation for the given coefficients and excitation."""

    perturbation = lambda t: eps*np.cos(w*t)

    p = np.poly1d(coeff)
    roots = p.r
    roots = roots[np.isreal(roots)]

    if len(roots)>2:
        raise ValueError
    
    func = lambda t,x: np.poly1d(coeff)(x) + perturbation(t)
    sol = solve_ivp(func, (t_start, 2*np.pi/w*num_periods),(roots[0],),dense_output=True)

    return sol.sol

def solve_network(A,poly_coeff,eps,w,t_start=0, num_periods=100):
    """Solves the equation for the given connection, coefficients and excitation."""
    N = np.shape(A)[0]

    # list for functions (intrinsic dynamics) and roots (starting values)
    f = []
    r = []
    for i in range(N):
        #add function
        p = np.poly1d(poly_coeff[i])
        print(p)
        f.append(p)
        #find root corresponding to stable fixpoint
        roots = p.r
        print(min(roots))
        roots = roots[np.isreal(roots)]

        if len(roots)>2:
            raise ValueError
        r.append(min(roots))
    print(f,r)
    def dxdt(t,x):
        dx = np.zeros_like(x)
        # equation at each node
        for i in range(len(x)):
            dx[i] += f[i](x[i]) + (A@x)[i]
        dx[0] += eps*np.cos(w*t)
        return dx
    
    sol = solve_ivp(dxdt,(t_start, 2*np.pi/w*num_periods),r,dense_output=True)
    return sol.sol



def gen_fourier_coefficantes(t_start,t_end,x,w,N):
    """input x as a function via interpolation of the solution"""
    coeff = np.zeros(2*N+1, dtype=complex)

    for k in range(2*N+1):
        func = lambda t: np.exp(1.0j*(k-N)*w*t)*x(t)
        coeff[k] = quad(func,t_start,t_end,complex_func=True)[0]/(t_end-t_start)
    return np.round(coeff,12)

def real_fourier(coeff,order=None):
    """return coefficant infront of sine , cosine and constant"""
    N = len(coeff)
    res = coeff[N//2+1:]
    const = np.array([np.real(coeff[N//2])])
    cos = 2*np.real(res)
    sin = 2*np.imag(res)
    if order:
        res = np.concatenate((const,cos[:order],sin[:order]),)
    else:
        res = np.concatenate((const,cos,sin),)
    return res

def convol_k_times(coeff, k):
    """Convolve the coefficients k times."""
    if k == 0:
        return coeff
    elif k == 1:
        return np.convolve(coeff, coeff, mode='full',)
    else:
        res = np.convolve(coeff,convol_k_times(coeff, k-1),mode='full')
        return res

def gen_convolution_matrix(coeff):
    """Generate the convolution matrix for the Fourier coefficients."""
    N = len(coeff)
    D = np.zeros((N, N))
    for k in range(N-1):
            res = convol_k_times(coeff, k)
            res = real_fourier(res,order=N//2)
            for i in range(N):
                D[i,k+1] = res[i]
    D[0,0] = 1
    return D

def dervative_fourier_coefficients(coeff,w):
    """Calculate the Fourier coefficients of the derivative of a function."""
    N = len(coeff)
    
    mult = np.arange(1,N//2+1,1)
    const = np.array([0])
    cos = np.array(coeff[1:N//2+1])
    sin = np.array(coeff[N//2+1:])

    res = np.concatenate([const, w*mult*sin, -mult*w*cos])
    return res


def get_important_info(poly_coeff,eps,w,order):
    """ 
    four are the fourier coefficants in the complex world starting with
    e^-i order w t at the zerost entry
    real is of form [const,cos(wt)... cos(order w t), sin ...,sin]
    D_mat are the fourier coefficants of the powers of the time series
    writen as a matrix where the collums are one power of x(t)
    """
    T = 2*np.pi/w

    sol = solve_equation(poly_coeff,eps,w)

    four = gen_fourier_coefficantes(20*T,22*T,sol,w,order)

    real = real_fourier(four)

    D_mat = gen_convolution_matrix(four)

    return four, real, D_mat

def get_info_network(A,polycoff,eps,w, order):
    """give information about the solution of a network in a usefull formate"""
    T = 2*np.pi/w
    N = np.shape(A)[0]
    # solve system
    sol = solve_network(A,polycoff,eps,w)

    four = []
    real = []
    D_mat = []
    for i in range(N):
        sol_node_i = lambda t: sol(t)[i]

        four_i = gen_fourier_coefficantes(20*T,22*T,sol_node_i,w,order)
        real_i = real_fourier(four_i)
        D_mat_i = gen_convolution_matrix(four_i)

        four.append(four_i)
        real.append(real_i)
        D_mat.append(D_mat_i)
    
    return four,real,D_mat




def test_slover_net(test):
    """testet ob die lösung sinvoll aussieht"""
    A = np.array([[0,1],[1,0]])
    poly_coeff = [[1,0,-4],[2,0,-10]]
    eps = 1
    w = 2*np.pi
    order = 4
    if test == 'plot':
        sol = solve_network(A,poly_coeff,eps,w)
        ls_t = np.linspace(0,4,1000)
        ls_x = sol(ls_t).T
        plt.plot(ls_t,ls_x)
        plt.show()

    elif test == 'info':
        get_info_network(A,poly_coeff,eps,w,order)



if __name__ == '__main__':
    test_slover_net('info')
    test_slover_net('plot')