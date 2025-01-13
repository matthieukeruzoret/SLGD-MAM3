import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import time
import matplotlib.pyplot as plt

def generate_sparse_tridiagonal_matrix(n, diagonal_value=5, off_diagonal_value=-1):
    """
    Generates a sparse tridiagonal matrix.

    Args:
        n: Dimension of the system (size of the matrix A).
        diagonal_value: Value for the diagonal elements.
        off_diagonal_value: Value for the off-diagonal elements.

    Returns:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
    """
    # Main diagonal and lower/upper diagonals
    main_diag = np.full(n, diagonal_value)
    upper_diag = np.full(n - 1, off_diagonal_value)
    lower_diag = np.full(n - 1, off_diagonal_value)
    # Construct sparse matrix
    data = np.concatenate((main_diag, upper_diag, lower_diag))
    rows = np.concatenate((np.arange(n), np.arange(n - 1), np.arange(1, n)))
    cols = np.concatenate((np.arange(n), np.arange(1, n), np.arange(n - 1)))
    A = csr_matrix((data, (rows, cols)), shape=(n, n))

    b = np.random.rand(n)
    return A, b


def jacobi_dense(A, b, x0, tol=1e-5, max_iter=1000):
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
    n = A.shape[0]
    x = x0.copy()
    errors = []
    start_time = time.time()
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
    end_time=time.time()
    time_taken = end_time - start_time
    return x, k+1, time_taken,errors  # Return the last approximation if max_iter is reached


def jacobi_sparse(A, b, x0, tol=1e-5, max_iter=10000):
    """
    Jacobi method for sparse matrices.

    Args:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
        x0: Initial guess for the solution vector (numpy array).
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        x: Approximate solution vector.
        iterations: Number of iterations performed.
        time_taken: Time taken for the iterations.
    """
    start_time = time.time()
    x = x0.copy()
    D_inv = 1.0 / A.diagonal()
    R = A - sparse.diags(A.diagonal())
    errors = []
    for k in range(max_iter):
        x_new = D_inv * (b - R @ x)
        error = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(error)
        if error < tol:
            break
        x = x_new
    end_time = time.time()
    time_taken = end_time - start_time
    return x, k+1, time_taken, errors

def plot_error_jac(errors_dense,errors_sparse):
    plt.figure(figsize=(10, 6))
    plt.plot(errors_dense, label="Jacobi Method dense", marker='o', linestyle='-', markersize=4)
    plt.plot(errors_sparse, label="Jacobi Methode sparse", marker='s', linestyle='--', markersize=4)
    plt.yscale("log")  # Log scale for the error
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

# Comparison of performance for varying matrix dimensions
n = 111
results = []

A_sparse, b = generate_sparse_tridiagonal_matrix(n)
x0 = np.zeros(np.size(b))

# Construct dense matrix for reference
A_dense = A_sparse.toarray()

# Jacobi Dense
_, iter_dense, time_dense, errors_dense = jacobi_dense(A_dense, b, x0)
print(f"Dense ({n}x{n}): {iter_dense} itérations, {time_dense} secondes")

# Jacobi Sparse
_, iter_sparse, time_sparse, errors_sparse= jacobi_sparse(A_sparse, b, x0)
print(f"Sparse ({n}x{n}): {iter_sparse} itérations, {time_sparse} secondes")

# Affichage des résultats
results.append((n, time_dense, time_sparse))
print("\nn\tDense Time (s)\tSparse Time (s)")
for res in results:
    print(f"{res[0]} \t{res[1]}\t \t{res[2]}")

plot_error_jac(errors_dense,errors_sparse)
