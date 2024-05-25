import library
import numpy as np
import matplotlib.pyplot as plt

#Dots cloud construction
n,degre = 200,15
ti = np.linspace(-5,5,n-1)
epsilon = 0.1
e = 2*epsilon*np.random.rand(1,n)-epsilon
xi = np.linspace(-5,5,n)
yi = 1./(1+(xi*xi))+e
yi = yi[0]
yp = np.zeros_like(xi)
nx = len(xi)

for i in range(nx):
    yp[i] = library.polynomial_regression(xi,yi,degre,xi[i])
    
plt.scatter(xi,yi,label='Points')
plt.plot(xi,yp,'-r')


