import numpy as np


def conjugate_gradient(A,b):
    """This function determines the solution of the linear system Ax=b by the conjugate gradient
    algorithm."""             
    n = A.shape[0]
    x0,epsilon,kmax = np.zeros((n,1)),1e-16,2000
    r0 = A.dot(x0) - b
    w0 = -r0
    for k in range(kmax):
        alphak = -w0.T.dot(r0) / w0.T.dot(A.dot(w0))
        xk = x0 + alphak*w0
        rk = r0 + alphak*A.dot(w0)
        gammak = rk.T.dot(rk) / r0.T.dot(r0)
        wk = -rk + gammak*w0
        if np.sqrt(rk.T.dot(rk)) < epsilon:
            return xk
        else: 
            r0,x0,w0 = rk,xk,wk
    return xk

def polynomial_smoothing(xi,yi,degre):
    """This function solves the least squares problem by returning the coefficients of the polynomial 
     minimizing."""
    n = len(xi)
    A = np.zeros((n,degre))
    A[:,0] = np.ones((n,))
    for j in range(1,degre):
        A[:,j] = [xi[i]**j for i in range(n)]
    A_T_A = A.T.dot(A)
    b = np.array([yi[i] for i in range(n)])
    b = np.reshape(b,(n,1))
    B = A.T.dot(b)
    w = conjugate_gradient(A_T_A,B)
    return w

def polynomial_fitting(xi,yi,degre):
    """This function creates and returns the polynomial minimizing the least squares problem."""
    w = polynomial_smoothing(xi,yi,degre)
    w = np.reshape(w,(degre,))
    w = w[::-1]
    p = np.poly1d(w)
    return p

def horner(p,x):
    """This function evaluates the value of the polynomial p at x using the Horner method."""
    p = list(p)
    n = len(p)-1
    val = p[0]
    for i in range(n):
        val = val*x+p[i+1]
    return val

def polyval(p,x):
    """This function evaluates the value of the polynomial minimizing p at x."""
    yp = horner(p,x)
    return yp

def polynomial_regression(xi,yi,degre,t):
    """This function uses the least squares method to find and return the value of 
     polynomial minimizing in t."""
    p = polynomial_fitting(xi,yi,degre)
    yp = polyval(p,t)
    return yp