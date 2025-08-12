import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt
import pandas as pd
import numbers


class fourierseries:
    """save fourier coefficantes in a cleaner way"""
    def __init__(self,fouriercoeff):
         self.order = len(fouriercoeff)//2
         self.const = fouriercoeff[0]
         self.cos = fouriercoeff[1:self.order+1]
         self.sin = fouriercoeff[self.order+1:]
    
    def convolution(self):
        return gen_convolution_matrix(self.complex_rep()) #input comlex representation

    def __str__(self) -> str:
        return f"Constante: {self.const} \n Cos: {self.cos} \n Sin: {self.sin}"

    def eval(self,t,w):
        res = self.const
        for j in range(self.order):
            res += self.cos[j]*np.cos((j+1)*w*t)
            res += self.sin[j]*np.sin((j+1)*w*t)
        return res
    
    def expand_to_order(self,order):
        if self.order<order:
            add_oder = order-self.order
            zeros = np.zeros(add_oder)
            self.cos = np.concatenate((self.cos,zeros))
            self.sin = np.concatenate((self.sin,zeros))

    def convert_list(self):
        return np.concatenate(([self.const],self.cos,self.sin))
    
    def complex_rep(self):
        comp_rep = np.zeros(self.order*2+1, dtype=complex)
        for k in range(self.order):
            comp_rep[k]+= self.cos[-k-1]/2-1.0j*self.sin[-k-1]/2
            comp_rep[-k-1]+= self.cos[-k-1]/2+1.0j*self.sin[-k-1]/2
        comp_rep[self.order]=self.const
        return comp_rep
    
    def d_mat(self,degree):
        return self.convolution()[:,:degree]
    
    def derivativ(self,w):
        mult = np.arange(1,self.order+1,1)
        res = np.concatenate(([0],w*mult*self.sin,-w*mult*self.cos))
        return res

        

def solve_equation(intrinsic,start, eps:fourierseries, w,t_start =0, num_periods=100):
    """Solve the equation for the given coefficients and excitation."""
    if not isinstance(eps,fourierseries):
        print("eps should be fourierseries")
    perturbation = lambda t: eps.eval(t,w)
    
    func = lambda t,x: intrinsic(x) + perturbation(t)
    sol = solve_ivp(func, (t_start, 2*np.pi/w*num_periods),(start,),rtol=1e-8,atol=1e-9,method='LSODA',dense_output=True)
    return sol.sol

def solve_network(A,poly_coeff,eps,w,t_start=0, num_periods=100):
    """Solves the equation for the given connection, coefficients and excitation."""
    N = np.shape(A)[0]

    if isinstance(poly_coeff[0],list):
        print("list")
        # list for functions (intrinsic dynamics) and roots (starting values)
        f = []
        r = []
        for i in range(N):
            #add function
            p = np.poly1d(poly_coeff[i])
            f.append(p)
            #find root corresponding to stable fixpoint
            roots = p.r
            roots = roots[np.isreal(roots)]

            if len(roots)>2:
                raise ValueError
            r.append(min(roots))
        def dxdt(t,x):
            dx = np.zeros_like(x)
            # equation at each node
            for i in range(len(x)):
                dx[i] += f[i](x[i]) + (A@x)[i]
            dx[0] += eps.eval(t,w)
            return dx
        
        sol = solve_ivp(dxdt,(t_start, 2*np.pi/w*num_periods),r,dense_output=True)
   
    
    elif isinstance(poly_coeff[0],numbers.Real):
        print("identical dynamics")
        p = np.poly1d(poly_coeff)
        #find root corresponding to stable fixpoint
        roots = p.r
        roots = roots[np.isreal(roots)]

        if len(roots)>2:
            raise ValueError
        root = min(roots)

        def dxdt(t,x):
            dx = np.zeros_like(x)
            # equation at each node
            for i in range(len(x)):
                dx[i] += p(x[i]) + (A@x)[i]
            dx[0] += eps.eval(t,w)
            return dx        

        
        sol = solve_ivp(dxdt,(t_start, 2*np.pi/w*num_periods),np.full(N,root),dense_output=True)
    return sol.sol

def solve_seconde_order(a,intrinsic,start, eps:fourierseries, w , t_start =0, num_periods=100):
    """solves equation of type d^2x/(dt)^2+a*dx/dt=f(x)+eps(t)"""
    if not isinstance(eps,fourierseries):
        print("eps should be fourierseries")
    pertubation = lambda t: eps.eval(t,w)
    # res[0] is dxdt res[1] is ddx/dt^2
    def dydt(t,x):
        return np.array([x[1],(intrinsic(x[0])+pertubation(t)-a*x[1])])
    sol = solve_ivp(dydt, (t_start, 2*np.pi/w*num_periods),(start,0),rtol=1e-8,atol=1e-9,method='LSODA',dense_output=True)
    return sol.sol

def gen_fourier_coefficantes(t_start,t_end,x,w,N):
    """input x as a function via interpolation of the solution"""
    coeff = np.zeros(2*N+1, dtype=complex)

    for k in range(2*N+1):
        func = lambda t: np.exp(1.0j*(k-N)*w*t)*x(t)
        coeff[k] = quad(func,t_start,t_end,complex_func=True,epsrel=1e-9,epsabs=1e-10,limit=1000)[0]/(t_end-t_start)
    return coeff

def check_half_fourier_coefficants(t_start,t_end,x,w):
    """returns fractional orders to check periodicity"""
    func = lambda t: np.exp(1.0j/2*w*t)*x(t)
    coeff = quad(func,t_start,t_end,complex_func=True,epsrel=1e-12,epsabs=1e-12)[0]/(t_end-t_start)
    print(f'the first fractional periode has cos={np.real(coeff)*2} and sin=-{np.imag(coeff)*2}')
    return coeff

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
    #uses complex represenation as input
    N = len(coeff)
    D = np.zeros((N, N))
    for k in range(N-1):
            res = convol_k_times(coeff, k)
            res = real_fourier(res,order=N//2)
            for i in range(N):
                D[i,k+1] = res[i]
    D[0,0] = 1
    return D

def get_important_info(intrinsic,start:float,eps:fourierseries,w:float,order:int)-> fourierseries: 
    """ 
    four are the fourier coefficants in the complex world starting with
    e^-i order w t at the zerost entry
    real is of form [const,cos(wt)... cos(order w t), sin ...,sin]
    D_mat are the fourier coefficants of the powers of the time series
    writen as a matrix where the collums are one power of x(t)
    """
    T = 2*np.pi/w

    if not callable(intrinsic):
        
        p = np.poly1d(intrinsic)
        roots = p.r
        roots = roots[np.isreal(roots)]
        if len(roots)>2:
            raise ValueError
        intrinsic = p
        start = np.real(roots[0])


    sol = solve_equation(intrinsic,start,eps,w)

    four = gen_fourier_coefficantes(20*T,22*T,sol,w,order)
    check_half_fourier_coefficants(20*T,22*T,sol,w)


    real = real_fourier(four)
    if real[0]>10:
        print("waring: might be unstable")

    return fourierseries(real)

def get_info_network(A,polycoff,eps:fourierseries,w:float, order:int):
    """give information about the solution of a network in a usefull formate"""
    T = 2*np.pi/w
    N = np.shape(A)[0]
    # solve system
    sol = solve_network(A,polycoff,eps,w)

    ls_four = []
    for i in range(N):
        sol_node_i = lambda t: sol(t)[i]

        four_i = gen_fourier_coefficantes(20*T,22*T,sol_node_i,w,order//2)
        real_i = real_fourier(four_i)
        ls_four.append(fourierseries(real_i))
    
    return ls_four

def get_info_seconde_order(a,intrinsic,start,eps:fourierseries,w:float,order:int):
    """returns fourier coefficants of the second order 1 node system"""
    T = 2*np.pi/w
    if not callable(intrinsic):
        
        p = np.poly1d(intrinsic)
        roots = p.r
        roots = roots[np.isreal(roots)]
        if len(roots)>2:
            raise ValueError
        intrinsic = p
        start = np.real(roots[0])

    sol = solve_seconde_order(a,intrinsic,start,eps,w)
    position = lambda t: sol(t)[0]

    four = gen_fourier_coefficantes(20*T,22*T,position,w,order)
    check_half_fourier_coefficants(20*T,22*T,position,w)

    real = real_fourier(four)

    if real[0]>10:
        print("waring: might be unstable")

    return fourierseries(real)

def test_slover_net(test):
    """testet ob die lösung sinvoll aussieht"""
    A = np.array([[-10,10],[1,-1]])
    poly_coeff = [1,0,-4]
    eps = fourierseries([0,6,0])
    w = 2*np.pi
    order = 4
    if test == 'plot':
        sol = solve_network(A,poly_coeff,eps,w)
        ls_t = np.linspace(4,10,1000)
        ls_x = sol(ls_t).T
        plt.plot(ls_t,ls_x)
        plt.show()

    elif test == 'info':
        get_info_network(A,poly_coeff,eps,w,order)

def test_2nd_order():
    """test for solver of 2nd order diffrential"""
    w = 2 * np.pi
    a = 0.1
    intrinsic = lambda x: x**2-4
    start = -2
    eps = fourierseries([0,10,0])

    sol = solve_seconde_order(a,intrinsic,start,eps,w)
    ls_t = np.linspace(4,10,1000)
    ls_x = sol(ls_t).T
    plt.plot(ls_t,ls_x)
    plt.show()


if __name__ == '__main__':
    test_2nd_order()