#!/Users/StrangeQuark/anaconda3/bin/python

#-------------------------------------------------------------------------------
# Attempt at Davidson algorithm 
# Sree Ganesh
#-------------------------------------------------------------------------------

import numpy as np
import math
import time 
from numpy.linalg import multi_dot

# Build a fake sparse symmetric matrix 
n = 100
sparsity = 0.001
A = np.zeros((n,n))
for i in range(0,n) : 
    A[i,i] = i + 1
A = A + sparsity*np.random.randn(n,n)
A = (A.T + A)/2

tol = 1e-8              # Convergence tolerance
mmax = n/2              # Maximum number of iterations

# Setup the subspace trial vectors
k = 8
eig = 3
t = np.eye(n,k) # trial vectors
v = np.zeros((n,n))
I = np.eye(n)
#start_davidson = time.time()
#-------------------------------------------------------------------------------
# Begin iterations  
#-------------------------------------------------------------------------------
iter = 0
for m in xrange(k,mmax,k):
    iter = iter + 1
    print "Iteration no:" % iter
    # Matrix-vector products
    T = multidot([t.T,A,t])
    


#for m in xrange(k,mmax,k):
#    if m <= k:
#        for j in xrange(0,k):
#            v[:,j] = t[:,j]/np.linalg.norm(t[:,j]) # normalize the vectors
#print((t))
#print ((v))
