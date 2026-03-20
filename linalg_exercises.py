import numpy as np
from scipy import linalg

# a. Define matrix A
A = np.array([[1,-2,3],
             [4,5,6],
              [7,1,9]])

#b. Define a vector b
b = np.array([1, 2,3])

#c. solve the linear system of equations Ax = b
x = linalg.solve(A,b)
print('c. solution to linear system: ', x)

#d. Check that your solution is correct by plugging it into the equation 
value_check = np.dot(A,x)

print("d. A @ x =", value_check)
print("d. equal b", np.allclose(value_check, b))

#e.Repeat with random 3x3 matrix B instead of vector b

B = np.random.randint(1,10, (3,3))
print("matrix B", B)
X =linalg.solve(A,B)
print("e. solution X", X)

value_check_B = np.dot(A,X)
print("e.A @ X", value_check_B)
print("e. equals B?", np.allclose(value_check_B, B))


#f. Solve the eigenvalue problem for the matrix A and print the eigenvalues and eigenvectors

eigenvalues, eigenvectors = linalg.eig(A)
print("\nf. Eigenvalues:", eigenvalues)
print("f. Eigenvectors:\n", eigenvectors)


#g. Calculate the inverse, determinant of A
A_inv = linalg.inv(A)

print('g. inverse of A', A_inv)

A_det = linalg.det(A)

print('g. determinant of A', A_det)

#h. Calculate the norm of A with different orders

A_norm_first = linalg.norm(A,1)

print("h. norm 1 of A", A_norm_first)

A_norm_second=linalg.norm(A, 2)
print("h. norm 2 of A", A_norm_second)



