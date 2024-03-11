import numpy as np

def jacobi(A, b, x0, tol=1e-6, max_iter=1000):
    n = len(b)
    x = np.copy(x0)
    x_new = np.zeros_like(x)
    for k in range(max_iter):
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x_new[i] = (b[i] - sigma) / A[i][i]
        if np.linalg.norm(x_new - x) < tol:
            return x_new, k + 1
        x[:] = x_new
    return x, max_iter

def gauss_seidel(A, b, x0, tol=1e-6, max_iter=1000):
    n = len(b)
    x = np.copy(x0)
    x_new = np.zeros_like(x)
    for k in range(max_iter):
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x_new[j]
            x_new[i] = (b[i] - sigma) / A[i][i]
        if np.linalg.norm(x_new - x) < tol:
            return x_new, k + 1
        x[:] = x_new
    return x, max_iter

def gaussian_elimination(A, b):
    n = len(b)
    for i in range(n):
        pivot_row = i
        for j in range(i + 1, n):
            if abs(A[j][i]) > abs(A[pivot_row][i]):
                pivot_row = j
        A[[i, pivot_row]] = A[[pivot_row, i]]
        b[[i, pivot_row]] = b[[pivot_row, i]]
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            b[j] -= factor * b[i]
            for k in range(i, n):  # Aqui estava o erro
                A[j][k] -= factor * A[i][k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i] / A[i][i]
        for j in range(i - 1, -1, -1):
            b[j] -= A[j][i] * x[i]
    return x

def select_method():
    method = input("Escolha o método (Jacobi, Gauss-Seidel, Eliminação de Gauss): ")
    return method.lower()

def input_matrix():
    n = int(input("Digite o tamanho da matriz quadrada (n x n): "))
    print("Digite os elementos da matriz linha por linha:")
    A = []
    for i in range(n):
        row = list(map(float, input().split()))
        A.append(row)
    return np.array(A)

def input_vector(n):
    print("Digite os elementos do vetor separados por espaço:")
    b = list(map(float, input().split()))
    return np.array(b)

# Exemplo de uso
method = select_method()

if method == "jacobi":
    A = input_matrix()
    b = input_vector(len(A))
    x0 = input_vector(len(A))  # Vetor inicial x^0
    tol = float(input("Digite o critério de parada: "))
    x, iterations = jacobi(A, b, x0, tol)
    print("Solução pelo método de Jacobi:", x)
    print("Número de iterações:", iterations)
elif method == "gauss-seidel":
    A = input_matrix()
    b = input_vector(len(A))
    x0 = np.zeros(len(A))
    tol = float(input("Digite o critério de parada: "))
    x, iterations = gauss_seidel(A, b, x0, tol)
    print("Solução pelo método de Gauss-Seidel:", x)
    print("Número de iterações:", iterations)
elif method == "eliminação de gauss":
    A = input_matrix()
    b = input_vector(len(A))
    x = gaussian_elimination(A, b)
    print("Solução pelo método de Eliminação de Gauss:", x)
else:
    print("Método não reconhecido.")
