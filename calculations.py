# calculations.py
# Modul pro matematické výpočty interpolačních metod.

import numpy as np
import math

# --- Funkce pro Newtonovu interpolaci ---

def calculate_newton_coeffs(x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
    """Vypočítá koeficienty c_i pro Newtonův interpolační polynom."""
    n = len(x_nodes)
    coeffs = np.copy(y_nodes).astype(float)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coeffs[i] = (coeffs[i] - coeffs[i - 1]) / (x_nodes[i] - x_nodes[i - j])
    return coeffs

def evaluate_newton_poly(coeffs: np.ndarray, x_nodes: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vyhodnotí Newtonův polynom pro dané body x."""
    n = len(coeffs)
    y = np.full_like(x, coeffs[n - 1], dtype=float)
    for i in range(n - 2, -1, -1):
        y = coeffs[i] + (x - x_nodes[i]) * y
    return y

# --- Funkce pro B-interpolaci ---

def calculate_B_basis_func(k: int, x: np.ndarray) -> np.ndarray:
    """
    Vypočítá hodnotu k-té bázové funkce B_k(x).
    B_k(x) = (1/k!) * Σ [(-1)^(k-j) * C(k,j) * (j+1)^x]
    """
    if k < 0: return np.zeros_like(x)
    total_sum = np.zeros_like(x, dtype=float)
    for j in range(k + 1):
        comb = math.comb(k, j)
        term = ((-1)**(k - j)) * comb * np.power(j + 1, x, dtype=float)
        total_sum += term
    return total_sum / float(math.factorial(k))

def calculate_d_coeffs(y_nodes: np.ndarray) -> np.ndarray:
    """Vypočítá koeficienty d_k pro F(x) = Σ d_k * B_k(x)."""
    n = len(y_nodes) - 1
    d_coeffs = np.zeros(n + 1)
    for i in range(n + 1):
        # Vypočítá sumu d_k * B_k(i) pro k < i
        sum_part = sum(d_coeffs[k] * calculate_B_basis_func(k, np.array([i]))[0] for k in range(i))
        # d_i = y_i - Σ_{k=0}^{i-1} d_k * B_k(i)
        d_coeffs[i] = y_nodes[i] - sum_part
    return d_coeffs

def evaluate_B_interpolation(d_coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vyhodnotí finální interpolační funkci F(x) = Σ d_k * B_k(x)."""
    n = len(d_coeffs) - 1
    total_y = np.zeros_like(x, dtype=float)
    for k in range(n + 1):
        if not np.isclose(d_coeffs[k], 0):
            total_y += d_coeffs[k] * calculate_B_basis_func(k, x)
    return total_y