#!/modfac/home/sreeganb/anaconda2/bin/python

#-------------------------------------------------------------------------------
# Power iteration
#-------------------------------------------------------------------------------

import numpy as np

n=100
sparsity = 0.0005  
A = np.zeros((n,n))  
for i in range(0,n):  
    A[i,i] = i^2 + 1  
    A = A + sparsity*np.random.randn(n,n)  
    A = (A.T + A)/2  

# Build a random trial vector
B=np.random.rand(100)
j=0
norm_mat=np.zeros(2)
for i in range(1,2000):
    C = np.dot(A,B)
    nor = np.linalg.norm(C)
    B = C/(nor)
    j=j+1
    if j==1: 
        print ('just the first iteration, give me a break')
        norm_mat[0]=nor
    else: 
        norm_mat[1] = norm_mat[0]
        norm_mat[0] = nor
        diff = abs(norm_mat[1] - norm_mat[0])
        if diff < 10e-10:
            print ('power iteration converged at iteration number:', j)
            break
        else:
            continue

approx = np.dot(B.T,np.dot(A,B))/np.linalg.norm(B)
print ('power iteration dominant eigenvalue=', approx)

w, v = np.linalg.eig(A)
diff = w[0] - approx
print ('exact dominant eigenvalue=',w[0])
print ('difference:', diff)
#print 'exact eigenvalues'
#print w
#print v

#-------------------------------------------------------------------------------
# Rayleigh quotient correction 
#-------------------------------------------------------------------------------
B = np.random.rand(100)
for i in range(1,1000):
    rcoeff = np.dot(B.T,np.dot(A,B))/np.linalg.norm(B)
    rmat = rcoeff*np.eye(100)
    C = np.dot(np.linalg.inv(A-rcoeff),B)
    nor = np.linalg.norm(C)
    B = C/(nor)

approx = np.dot(B.T,np.dot(A,B))/np.linalg.norm(B)
print ('with Rayleight correction step')
print (approx)

