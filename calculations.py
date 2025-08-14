# calculations.py
# Modul pro matematické výpočty interpolačních metod.
# VERZE S VYSOKOU PŘESNOSTÍ pro B-interpolaci pomocí modulu 'decimal'.

import numpy as np
import math
from decimal import Decimal, getcontext

# --- Globální nastavení přesnosti pro modul 'decimal' ---
# Nastavíme pracovní přesnost na 50 desetinných míst.
getcontext().prec = 50


# --- Funkce pro Newtonovu interpolaci (zůstává beze změny, používá rychlé floaty) ---

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


# --- Funkce pro B-interpolaci (přepsané pro vysokou přesnost) ---

def calculate_B_basis_func(k: int, x: np.ndarray) -> np.ndarray:
    """
    Vypočítá hodnotu k-té bázové funkce B_k(x) s vysokou přesností.
    """
    if k < 0:
        return np.zeros_like(x, dtype=float)

    x_list = x.tolist()
    results = []

    factorial_k = Decimal(math.factorial(k))
    one = Decimal(1)

    for x_val in x_list:
        x_dec = Decimal(x_val)
        total_sum = Decimal(0)

        for j in range(k + 1):
            comb = Decimal(math.comb(k, j))
            sign = one if (k - j) % 2 == 0 else -one
            base = Decimal(j + 1)

            # --- ZDE JE KLÍČOVÁ OPRAVA ---
            # Metoda .power() funguje jen pro celočíselné exponenty.
            # Pro reálné exponenty použijeme identitu a^x = exp(x * ln(a)).
            # Knihovna 'decimal' má vysoce přesné verze .ln() a .exp().
            power_result = (x_dec * base.ln()).exp()
            term = sign * comb * power_result
            total_sum += term

        results.append(float(total_sum / factorial_k))

    return np.array(results)

def calculate_d_coeffs(y_nodes: np.ndarray) -> np.ndarray:
    """Vypočítá koeficienty d_k s vysokou přesností."""
    n = len(y_nodes) - 1
    d_coeffs = [Decimal(0)] * (n + 1)
    y_nodes_dec = [Decimal(y) for y in y_nodes]

    for i in range(n + 1):
        sum_part = Decimal(0)
        i_dec_array = np.array([float(i)])

        for k in range(i):
            basis_val_array = calculate_B_basis_func(k, i_dec_array)
            basis_val = Decimal(basis_val_array[0])
            sum_part += d_coeffs[k] * basis_val

        d_coeffs[i] = y_nodes_dec[i] - sum_part

    return np.array([float(d) for d in d_coeffs])

def evaluate_B_interpolation(d_coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vyhodnotí finální interpolační funkci F(x) = Σ d_k * B_k(x) s vysokou přesností."""
    n = len(d_coeffs) - 1
    total_y = np.zeros_like(x, dtype=float)

    for k in range(n + 1):
        if np.isclose(d_coeffs[k], 0):
            continue

        basis_values = calculate_B_basis_func(k, x)
        total_y += d_coeffs[k] * basis_values

    return total_y