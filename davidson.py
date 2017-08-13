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

tol = 1e-8              # Convergence tolerance
mmax = 50              # Maximum number of iterations

# Setup the subspace trial vectors
k = 8
eig = 3
t = np.eye(n,k) # trial vectors
v = np.zeros((n,n))
I = np.eye(n)
res = np.zeros((n,k))
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
        for l in range(k):
            v[:,l] = t[:,l]/(np.linalg.norm(t[:,l]))
    # Matrix-vector products, form the projected Hamiltonian in the subspace
    T = np.linalg.multi_dot([v[:,:m].T,A,v[:,:m]]) # apparently selects fastest evaluation order
    w, vects = np.linalg.eig(T) # Diagonalize the subspace Hamiltonian
    # Build residual vectors
    for j in range(n):
        f[j,j] = A[j,j] - w[1] 
    for j in range(k):
        res[:,j] = np.linalg.multi_dot([A,v[:,:m],vects[:,1]]) 
#    print (res)
    print (w)
