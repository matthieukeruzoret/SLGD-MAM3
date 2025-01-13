import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import time
import math

def generate_simple_sparse_tridiagonal_matrix(n, diagonal_value=5, off_diagonal_value=-1):
    """
    Generates a sparse tridiagonal matrix, ensuring no overlaps.

    Args:
        n: Dimension of the system (size of the matrix A).
        diagonal_value: Value for the diagonal elements.
        off_diagonal_value: Value for the off-diagonal elements.

    Returns:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        A_dense: equivalent Dense matrix (numpy array)
        b: Right-hand side vector (numpy array).
    """
    data = []
    for i in range(n-1):
        data.append(diagonal_value)
        data.append(off_diagonal_value)
        data.append(off_diagonal_value)
    data.append(diagonal_value)
    data = np.array(data)

    ind_ligne = np.array([0,0])
    for i in range(1,n-1):
        temp = np.full(3,i)
        ind_ligne = np.concatenate((ind_ligne, temp), axis=0)
    ind_ligne = np.concatenate((ind_ligne, np.array([n-1,n-1])), axis=0)

    ind_cols = np.array([0,1])
    for i in range(n-2):
        temp = np.arange(i, i+3)
        ind_cols = np.concatenate((ind_cols, temp), axis=0)
    ind_cols = np.concatenate((ind_cols, np.array([n-2,n-1])), axis=0)

    # Conversion en format CSR
    A_sparse = csr_matrix((data, (ind_ligne, ind_cols)), shape=(n, n))

    # Construction de la version dense pour référence
    A_dense = A_sparse.toarray()

    # Génération du vecteur b
    b = np.random.rand(n)

    return A_sparse, A_dense, b

def generate_sparse_tridiagonal_matrix(n):
    """
    Generates a sparse tridiagonal matrix which represents the 2nd order finite difference Laplacian
    operator in dimension 1 (n+1 subintervals of the [0,1] interval with boundary conditions 0 at 0 and 1)

    Args:
        n: Dimension of the system (size of the matrix A).

    Returns:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
    """
    h = 1 / (n + 1) #définition du pas
    data_diag = 2 / h**2
    data_off_diag = -1 / h**2
    A_sparse, A_dense, b = generate_simple_sparse_tridiagonal_matrix(n, data_diag, data_off_diag)
    #condition aux bords
    b[0] = 0
    b[n-1] = 0
    return A_sparse, A_dense, b

def generate_sparse_tridiagonal_matrix_pendule(n):
    """
    Generates a sparse tridiagonal matrix with the specific values.

    Args:
        n: Dimension of the system (size of the matrix A).

    Returns:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
    """
    g=9.81
    l=0.22
    w0 = math.sqrt(g/l)
    for i in range(n):
        h = 1 / (n + 1)
        data_diag = 2 / h**2 + w0**2
        data_off_diag = -1/h**2
    A_sparse, A_dense, b = generate_simple_sparse_tridiagonal_matrix(n, data_diag, data_off_diag)
    return A_sparse, A_dense, b

def generate_sparse_tridiagonal_matrix_nd(n):
    """
    Generates a sparse tridiagonal matrix which represents the 2nd order finite difference Laplacian
    operator in dimension 1 (n+1 subintervals of the [0,1] interval with boundary conditions 0 at 0 and 1)

    Args:
        n: Dimension of the system (size of the matrix A).

    Returns:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
    """
    data_diag = 3
    data_off_diag = -2
    A_sparse, A_dense, b = generate_simple_sparse_tridiagonal_matrix(n, data_diag, data_off_diag)
    return A_sparse, A_dense, b

def jacobi_sparse_with_error(A, b, x0, x_exact, tol=1e-6, max_iter=10000):
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
        errors: List of errors between the exact solution and approximations.
    """
    start_time = time.time()
    D_inv = 1.0 / A.diagonal()  # Inverse of the diagonal elements
    L_plus_U = A - sparse.diags(A.diagonal())

    x = x0.copy()
    errors = []
    for _ in range(max_iter):
        x_new = D_inv * (b - L_plus_U @ x)
        error = np.linalg.norm(x_new - x_exact, np.inf)
        errors.append(error)
        x = x_new
        if error < tol:
            break
    end_time = time.time()
    time_taken = end_time - start_time
    return x, len(errors), errors, time_taken

def gs_sparse_with_error(A, b, x0, x_exact, tol=1e-6, max_iter=10000):
    """
    Gauss-Seidel method for sparse matrices.

    Args:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
        x0: Initial guess for the solution vector (numpy array).
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        x: Approximate solution vector.
        iterations: Number of iterations performed.
        errors: List of errors between the exact solution and approximations.
    """
    n = A.shape[0]
    x = x0.copy()
    errors = []
    start_time= time.time()

    for iter_count in range(max_iter):
        x_new = x.copy()

        for i in range(n):
            sum_L = A[i, :i].dot(x_new[:i]).item()
            sum_U = A[i, i+1:].dot(x[i+1:]).item()
            x_new[i] = (b[i] - sum_L - sum_U) / A[i, i]

        error = np.linalg.norm(x_new - x_exact, np.inf)
        errors.append(error)

        if error < tol:
            break
        x = x_new
    end_time = time.time()
    time_taken = end_time - start_time
    return x, iter_count + 1, errors, time_taken

def SOR_sparse_with_error(A, b, x0, x_exact,omega, tol=1e-6, max_iter=10000):
    """
    Gauss-Seidel method for sparse matrices.

    Args:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
        x0: Initial guess for the solution vector (numpy array).
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        x: Approximate solution vector.
        iterations: Number of iterations performed.
        errors: List of errors between the exact solution and approximations.
    """
    n = A.shape[0]
    x = x0.copy()
    errors = []
    start_time = time.time()

    for iter_count in range(max_iter):
        x_new = x.copy()

        for i in range(n):
            sum_L = A[i, :i].dot(x_new[:i]).item()
            sum_U = A[i, i+1:].dot(x[i+1:]).item()
            s_i = (b[i] - sum_L - sum_U) / A[i, i]
            x_new[i] = omega * s_i + (1-omega)*x_new[i]

        error = np.linalg.norm(x_new - x_exact, np.inf)
        errors.append(error)

        if error < tol:
            break
        x = x_new
    end_time = time.time()
    time_taken = end_time - start_time
    return x, iter_count + 1, errors, time_taken

def plot_error_jac_gs_sor(errors_jac,errors_gs,errors_sor):
    plt.figure(figsize=(10, 6))
    plt.plot(errors_jac, label="Jacobi Method", marker='o', linestyle='-', markersize=4)
    plt.plot(errors_gs, label="Gauss-Seidel Method", marker='s', linestyle='--', markersize=4)
    plt.plot(errors_sor, label=f"SOR Method: omega = {omega}", marker='P', linestyle='--', markersize=4)
    plt.yscale("log")  # Log scale for the error
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

def plot_error_sor(errors):
    plt.figure(figsize=(10, 10))
    i=0
    for element in errors:
        plt.plot(element,label=f"omega = {w_p[i]}", marker='o', linestyle='-')
        i+=1
    plt.yscale("log")  # Log scale for the error
    plt.xlabel("Iterations")
    plt.ylabel("Error estimate")
    plt.title("Error vs Iterations (sor Method)")
    plt.legend()
    plt.grid(True)
    plt.show()

def rayon_spectral_jacobi(A):
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
    return p

def rayon_spectral_gs(A):
    """
    Args : A: Coefficient matrix (numpy array).
    Return : p : Spectral radius of the matrix "T"
    """
    n = A.shape[0]
    U = np.zeros_like(A)
    for i in range(n):
        for j in range(n):
            if j>i:
                U[i,j] = A[i,j]
    L_plus_D = A - U
    T = np.dot(np.linalg.inv(L_plus_D),U)
    p = max(abs(np.linalg.eigvals(T)))
    return p

def rayon_spectral_SOR(A,omega):
   #Calcul du rayon spectral pour SOR2
   D = np.diag(np.diag(A))
   #print("MATRICE D : ",D)
   # Préconditionneur de SOR2
   L = np.tril(A,-1)
   U = np.triu(A,1)
   M = D - L*omega
   N = D*(1  - omega) + U
   M_inv = np.linalg.inv(M)
   print("MATRICE INV : ",C_inv)
   T = M_inv @ N * omega
   #print("MATRICE T : ",T)
   val_propre = np.linalg.eigvals(T)
   #print(val_propre)
   ray_spectral = max(abs(val_propre))
   print("le rayon spectral est : ",ray_spectral)

#exemples d'utilisation des différentes fonctions
results = []
n = 25# Taille de la matrice
x0 = np.zeros(n)  # Solution initiale
A_sparse, A_dense, b = generate_sparse_tridiagonal_matrix_pendule(n)
x_exact = np.dot(np.linalg.inv(A_dense), b)

#calcul du rayon spectral
rho = rayon_spectral_jacobi(A_dense)
if rho <1:
    omega = 2/(1+ math.sqrt(1 - abs(rho)**2)) #on calcul le omega optimal si le rayon spectral < 1
else : omega =1.6 #sinon on assigne une valeur chosie

x_jc, iter_jac, errors_jac, time_jac= jacobi_sparse_with_error(A_sparse, b, x0, x_exact)
x_gs, iter_gs, errors_gs, time_gs = gs_sparse_with_error(A_sparse, b, x0, x_exact)
x_sor, iter_sor, errors_sor, time_sor = SOR_sparse_with_error(A_sparse, b, x0, x_exact,omega)

# Affichage des temps pour chaque méthode
results.append((n, time_jac, time_gs, time_sor))
print("\nn\tjac Time (s) \tgs Time (s) \tsor Time (s)")
for res in results:
    print(f"{res[0]} \t{res[1]}\t \t{res[2]} \t{res[3]}")

plot_error_jac_gs_sor(errors_jac,errors_gs,errors_sor) #affichage des résultats

#affichage des résultats selon différents oméga (méthode SOR)
errors_sor_w = []
w_p = []
for i in range(1,10):
    w = 1 + i/10
    _, _, error_sor,_= SOR_sparse_with_error(A_sparse, b, x0, x_exact,w)
    errors_sor_w.append(error_sor)
    w_p.append(w)
plot_error_sor(errors_sor_w)
