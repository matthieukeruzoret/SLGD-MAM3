import numpy as np
import matplotlib.pyplot as plt

def generate_linear_system(n):
  """
  Generates a linear system with a diagonally dominant matrix A and vector b.

  Args:
    n: Dimension of the system (size of the matrix A).

  Returns:
    A: Coefficient matrix (numpy array).
    b: Right-hand side vector (numpy array).
  """

  A = np.zeros((n, n))
  for i in range(n):
    for j in range(n):
      if j != i :
        A[i,j] = -1
      else :
        A[i,j]  = 5*(i+1)
  b = np.random.rand(n)

  return A, b

def jacobi_method(A, b, x0, tol=1e-5, max_iter=1000):
  """
  Jacobi method for dense matrices.

  Args:
    A: Dense coefficient matrix (numpy array).
    b: Right-hand side vector (numpy array).
    x0: Initial guess for the solution vector (numpy array).
    tol: Tolerance for convergence.
    max_iter: Maximum number of iterations.

  Returns:
    x: Approximate solution vector.
    iterations: Number of iterations performed.
    time_taken: Time taken for the iterations.
  """
  n= A.shape[0]
  x = x0.copy()
  errors = []
  for k in range(max_iter):
    x_new = np.zeros_like(x)
    for i in range(n):
      somme = 0
      for j in range(n):
        if i!=j:
          somme += A[i,j]*x[j]
      x_new[i] = (b[i] - somme)/A[i,i]
    error = np.linalg.norm(x_new - x)
    errors.append(error)
    if error < tol:
        break
    x = x_new
  return x_new, k+1, errors


def jacobi_method2(A, b, x0, tol=1e-5, max_iter=1000):
    """
    Implements the Jacobi method for solving the linear system Ax = b.

    Args:
        A: Coefficient matrix (numpy array).
        b: Right-hand side vector (numpy array).
        x0: Initial guess for the solution vector (numpy array).
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        x: Approximate solution vector.
        iterations: Number of iterations performed.
        errors: List of errors between successive approximations.
    """
    n = A.shape[0]
    x = x0.copy()
    errors = []

    for k in range(max_iter):
        x_new = np.zeros_like(x)  # Initialize a new solution vector

        for i in range(n):
            sum_ = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - sum_) / A[i, i]

        # Calculate the error (norm of the difference between successive approximations)
        error = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(error)

        if error < tol:
            break

        x = x_new.copy()  # Update x for the next iteration

    return x, k+1, errors  # Return the last approximation if max_iter is reached

def rayon_spectral(A):
  """
  Args : A: Coefficient matrix (numpy array).
  Return : p : Spectral radius of the matrix "T"
  """
  n = A.shape[0]
  D = np.zeros_like(A)
  for i in range(n):
    D[i,i] = A[i,i]
  T = np.dot(np.linalg.inv(D),(A-D))
  p = max(abs(np.linalg.eigvals(T)))
  print("Le rayon spectral est égale à", p)

def diagonale_dominante(A):
  """
  Cette fonction détermine la nature diagonale dominante d'une matrice dense.
  """
  n = A.shape[0]
  for i in range(n):
      # Calcul de la somme des éléments non-diagonaux
      somme_hors_diag = sum(abs(A[i, j]) for j in range(n) if j != i)
      if (abs(A[i, i]) < somme_hors_diag):
          print("La matrice n'est pas diagonale dominante.")
          return
  print("La matrice est diagonale dominante.")
  return

def plot_error(errors, iterations):
    plt.figure(figsize=(10, 10))
    plt.plot(range(iterations), errors, marker='o', linestyle='-')
    plt.semilogy(range(iterations), errors, marker='o', linestyle='-')  # Use semilogy for log-scale on y-axis
    plt.xlabel("Iterations")
    plt.ylabel("Error estimate")
    plt.title("Error vs Iterations (Jacobi Method)")
    plt.grid(True)
    plt.show()


# Example usage:
n = 111

A, b = generate_linear_system(n)  # Generate a linear system

x0 = np.zeros(np.size(b))

# Solve using Jacobi method
x_jacobi, iterations, errors = jacobi_method(A, b, x0)

# Calculate exact solution
x_exact = np.linalg.solve(A, b)

# Calculate spectral radius to see if the Jacobi's method converges
rayon_spectral(A)

diagonale_dominante(A)

# Print results
print(f"Iterations: {iterations}")
print(f"Solution Jacobi: {x_jacobi}")
print(f"Exact solution: {x_exact}")

# Plot the error
plot_error(errors, iterations)


