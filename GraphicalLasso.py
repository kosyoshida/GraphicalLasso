# -*- coding: utf-8 -*-
"""
============================================================
Graphical Lasso
============================================================

"""""

print(__doc__)
from sklearn import preprocessing
from sklearn import linear_model
import numpy as np


class GraphLasso:
    # X is data (n_samples*n_features)
    # A is precision matrix (n_features*n_features)
    # S is covariance matrix (n_features*n_features)
    # rho is regularizer
    
    # initialization
    def __init__(self,X=None,A=None,S=None,rho=0.1,
                 maxItr=1e+3,tol=1e-2):
        self.X=X
        self.A=A
        self.S=S
        self.rho=rho
        self.maxItr=int(maxItr)
        self.tol=tol
        self.scaler=None
        self.history=[]
        
    # graphical lasso    
    def fit(self,X):
        n_samples,n_features=X.shape[0],X.shape[1]
        
        self.scaler=preprocessing.StandardScaler().fit(X)
        self.X=self.scaler.transform(X)
        
        S=self.X.T.dot(self.X)/n_samples
        A=np.linalg.pinv(S)
        A_old=A
        invA=S
        
        clf=linear_model.Lasso(alpha=self.rho)
        # block cordinate descent
        # we wanna find l and lmbd 
        for i in range(self.maxItr):
            for j in range(n_features):
                R,s,sii=self.get(S)
                W=self.get(invA)[0]
                L=self.get(A)[0]
                
                # find sigma
                sigma=sii+self.rho            
                U,D,V=np.linalg.svd(W)
                W_half=U.dot(np.diag(np.sqrt(D)).dot(U.T))
                
                b=np.linalg.pinv(W_half).dot(s)
                
                # performs lasso  
                beta=-clf.fit(W_half,b).coef_
            
                # find w
                w=W.dot(beta)
                
                l=-beta/(sigma-beta.T.dot(W).dot(beta))
                lmbd=1/(sigma-beta.T.dot(W).dot(beta))
            
                A=self.put(L,l,lmbd)
                invA=self.put(W,w,sigma)
                S=self.put(R,s,sii)
                self.history.append(np.linalg.norm(A-A_old,ord=2))
            
            if np.linalg.norm(A-A_old,ord=2)<self.tol:
                break
            else:
                A_old=A
        
        self.S=S
        self.A=A
        return self
         
            
    # delete pth row and column form ndarray X
    def get(self,S):
        end=S.shape[0]-1
        R=S[:-1,:-1]
        s=S[end,:-1]
        sii=S[end][end]
            
        return [R,s,sii]
    
    def put(self,R,s,sii):
        n=R.shape[0]+1
        X=np.empty([n,n])
        X[1:,1:]=R
        X[1:,0]=s
        X[0,1:]=s
        X[0][0]=sii
        
        return X