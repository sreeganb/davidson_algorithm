#!/Users/StrangeQuark/anaconda3/bin/python

#-------------------------------------------------------------------------------
# Attempt at Davidson algorithm 
# Sree Ganesh
#-------------------------------------------------------------------------------

import numpy as np
import math
import time 

# Build a fake sparse symmetric matrix 
n = 100
sparsity = 0.0001
A = np.zeros((n,n))
for i in range(0,n) : 
    A[i,i] = i + 1
A = A + sparsity*np.random.randn(n,n)
A = (A.T + A)/2

tol = 1e-8             # Convergence tolerance
mmax = 50              # Maximum number of iterations

# Setup the subspace trial vectors
k = 8
eig = 3
t = np.eye(n,k) # initial trial vectors
v = np.zeros((n,n)) # holder for trial vectors as iterations progress
I = np.eye(n) # n*n identity matrix
ritz = np.zeros((n,n))
f = np.zeros((n,n))
#-------------------------------------------------------------------------------
# Begin iterations  
#-------------------------------------------------------------------------------
#start_davidson = time.time()
iter = 0
for m in range(k,mmax,k):
    iter = iter + 1
    print ("Iteration no:", iter)
    if iter==1:  # for first iteration add normalized guess vectors to matrix v
        for l in range(m):
            print (l)
            v[:,l] = t[:,l]/(np.linalg.norm(t[:,l]))
    # Matrix-vector products, form the projected Hamiltonian in the subspace
    T = np.linalg.multi_dot([v[:,:m].T,A,v[:,:m]]) # selects fastest evaluation order
    w, vects = np.linalg.eig(T) # Diagonalize the subspace Hamiltonian
    jj = 0
    #***************************************************************************
    # For each eigenvector of T build a Ritz vector, precondition it and check
    # if the norm is greater than a set threshold.
    #***************************************************************************
    for ii in range(m): #for each new eigenvector of T
        for j in range(n): # diagonal preconditioner build for each eigenvalue
            f[j,j] = A[j,j] - w[ii] 
        ritz[:,ii] = np.dot(np.linalg.inv(f),np.linalg.multi_dot([A,v[:,:m],vects[:,ii]]))
        if np.linalg.norm(ritz[:,ii]) > 1e-4 :
            ritz[:,ii] = ritz[:,ii]/(np.linalg.norm(ritz[:,ii]))
            v[:,m+jj] = ritz[:,ii]
            jj = jj + 1
    q, r = np.linalg.qr(v[:,:m+jj-1])
    print (len(q))
#    for kk in range(m+jj):
#        v[:,kk] = q[:,kk]
#    print (w)
