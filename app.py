# app_newton_simple.py
# Streamlit aplikace: Newtonova polynomiální interpolace (zjednodušené ovládání)
# Spusť: streamlit run app_newton_simple.py

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple
from fractions import Fraction
import re

# --- Konfigurace ---
PLOT_RESOLUTION = 1000  # Pevný počet bodů pro vykreslení křivky (dostatečně hladké)

# --- Pomocné a výpočtové funkce ---

def float_to_fraction_str(num: float, limit_denominator: int = 1000) -> str:
    """Převede float na řetězec reprezentující zlomek."""
    if np.isclose(num, 0):
        return "0"

    fraction = Fraction(num).limit_denominator(limit_denominator)
    if fraction.denominator == 1:
        return str(fraction.numerator)
    else:
        return f"{fraction.numerator}/{fraction.denominator}"

def calculate_newton_coeffs(x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
    """Vypočítá koeficienty c_i pro Newtonův interpolační polynom."""
    n = len(x_nodes)
    coeffs = np.copy(y_nodes).astype(float)

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coeffs[i] = (coeffs[i] - coeffs[i-1]) / (x_nodes[i] - x_nodes[i-j])

    return coeffs

def evaluate_newton_poly(coeffs: np.ndarray, x_nodes: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vyhodnotí Newtonův polynom pro dané body x pomocí Hornerova schématu."""
    n = len(coeffs)
    y = np.full_like(x, coeffs[n-1])

    for i in range(n - 2, -1, -1):
        y = coeffs[i] + (x - x_nodes[i]) * y

    return y

# --- Funkce pro UI a formátování ---

def render_controls() -> Tuple[str, float, float]:
    """Vykreslí ovládací prvky a vrátí jejich hodnoty."""
    st.header("Nastavení interpolace")

    st.subheader("Interpolační body")
    default_points = "0, 1, 4, 9, 16, 25"
    points_str = st.text_input(
        "Zadejte Y-ové hodnoty bodů (oddělené čárkou)",
        value=default_points
    )
    st.caption("Program automaticky přiřadí X-ové hodnoty: 0, 1, 2, 3, ...")

    st.markdown("---")

    st.subheader("Rozsah grafu")
    x_min = st.number_input("x min", value=-2.0, step=1.0, format="%.2f")
    x_max = st.number_input("x max", value=7.0, step=1.0, format="%.2f")

    return points_str, x_min, x_max

def format_polynomial_string(coeffs: np.ndarray, x_nodes: np.ndarray) -> str:
    """Vytvoří čitelný LaTeX řetězec reprezentující Newtonův polynom."""
    poly_parts = []

    first_coeff_str = float_to_fraction_str(coeffs[0])
    if first_coeff_str != "0":
        poly_parts.append(first_coeff_str)

    for i in range(1, len(coeffs)):
        coeff_val = coeffs[i]
        if np.isclose(coeff_val, 0):
            continue

        sign = "+" if coeff_val > 0 else "-"
        coeff_abs_str = float_to_fraction_str(abs(coeff_val))

        if poly_parts:
            poly_parts.append(f" {sign} ")
        elif sign == "-":
             poly_parts.append(f"{sign}")

        if coeff_abs_str != "1":
            poly_parts.append(f"{coeff_abs_str} ")

        for j in range(i):
            if np.isclose(x_nodes[j], 0):
                poly_parts.append("x")
            else:
                term_sign = "-" if x_nodes[j] > 0 else "+"
                term_val = abs(x_nodes[j])
                poly_parts.append(f"(x {term_sign} {term_val:.4g})")

    if not poly_parts:
        return "P(x) = 0"

    latex_str = "".join(poly_parts)
    latex_str = latex_str.replace(') (', r') \cdot (')
    latex_str = latex_str.replace(' x(', r' \cdot x(')
    latex_str = re.sub(r'(\d+)/(\d+)', r'\\frac{\1}{\2}', latex_str)

    return "P(x) = " + latex_str

def create_plot(x_plot: np.ndarray, y_plot: np.ndarray, x_nodes: np.ndarray, y_nodes: np.ndarray) -> go.Figure:
    """Vytvoří Plotly graf s interpolační křivkou a zadanými body."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines', name='Interpolovaný polynom P(x)'))
    fig.add_trace(go.Scatter(x=x_nodes, y=y_nodes, mode='markers', name='Zadané body', marker=dict(size=10, color='red')))
    fig.update_layout(xaxis_title="x", yaxis_title="P(x)", hovermode="x unified", template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(showgrid=True, zeroline=True)
    fig.update_yaxes(showgrid=True, zeroline=True)
    return fig

# --- Hlavní funkce aplikace ---
def main():
    """Hlavní funkce, která spouští aplikaci."""
    st.set_page_config(page_title="Newtonova interpolace", layout="wide")
    st.title("Newtonova polynomiální interpolace")
    st.markdown("Zadejte hodnoty a sledujte, jak se z nich sestaví polynom s koeficienty ve tvaru zlomků.")

    col_plot, col_controls = st.columns([3, 2], gap="large")

    with col_controls:
        points_str, x_min, x_max = render_controls()

    try:
        y_nodes = np.array([float(val.strip()) for val in points_str.split(',') if val.strip()])
        if len(y_nodes) < 1:
            st.warning("Zadejte alespoň jednu Y-ovou hodnotu.")
            st.stop()

        x_nodes = np.arange(len(y_nodes))
        coeffs = calculate_newton_coeffs(x_nodes, y_nodes)

        # Použijeme pevnou hodnotu pro vykreslení
        x_plot = np.linspace(x_min, x_max, PLOT_RESOLUTION)
        y_plot = evaluate_newton_poly(coeffs, x_nodes, x_plot)

    except (ValueError, IndexError):
        st.error("Chyba ve vstupu. Ujistěte se, že zadáváte pouze čísla oddělená čárkou.")
        st.stop()

    with col_plot:
        st.header("Výsledky")

        st.subheader("Odvozené koeficienty (cᵢ)")
        st.write("Koeficienty `c_i` Newtonova polynomu, zobrazené jako zlomky:")

        coeff_display_list = [f"c{i} = {float_to_fraction_str(c)}" for i, c in enumerate(coeffs)]
        st.code('\n'.join(coeff_display_list), language='text')

        st.subheader("Interpolovaný polynom P(x)")
        polynomial_str = format_polynomial_string(coeffs, x_nodes)
        st.latex(polynomial_str)

        st.header("Graf")
        fig = create_plot(x_plot, y_plot, x_nodes, y_nodes)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()