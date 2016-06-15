# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:03:14 2016

@author: yoshidakosuke
"""
import numpy as np
import matplotlib.pylab as plt
import GraphicalLasso as GL

np.random.seed(1)
    
t=np.arange(100.)
x1=np.sin(t/10)
    
X=np.random.normal(size=[100,10])
X[:,0]=x1
X[:,1]=x1
X[:,2]=-x1

gl=GL.GraphLasso(rho=0.01)
gl.fit(X).A
    
plt.figure()
plt.plot(gl.history)
plt.title('history',fontsize=20)
plt.show()

mesh=range(10)
X,Y=np.meshgrid(mesh,mesh)
plt.figure()
plt.pcolormesh(X,Y,gl.A)
plt.colorbar()  