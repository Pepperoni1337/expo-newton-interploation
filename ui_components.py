# ui_components.py
# Modul pro komponenty uživatelského rozhraní (Streamlit a Plotly).

import streamlit as st
import plotly.graph_objects as go
from fractions import Fraction
import re
import numpy as np

# --- Formátovací funkce (beze změny) ---
def float_to_fraction_str(num: float, limit_denominator: int = 1000) -> str:
    if np.isclose(num, 0): return "0"
    fraction = Fraction(num).limit_denominator(limit_denominator)
    return str(fraction.numerator) if fraction.denominator == 1 else f"{fraction.numerator}/{fraction.denominator}"

def format_newton_string(coeffs: list, x_nodes: list) -> str:
    poly_parts = []
    first_coeff_str = float_to_fraction_str(coeffs[0])
    if first_coeff_str != "0" or len(coeffs) == 1: poly_parts.append(first_coeff_str)
    for i in range(1, len(coeffs)):
        coeff_val = coeffs[i]
        if np.isclose(coeff_val, 0): continue
        sign = "+" if coeff_val > 0 else "-"
        coeff_abs_str = float_to_fraction_str(abs(coeff_val))
        if poly_parts: poly_parts.append(f" {sign} ")
        elif sign == "-": poly_parts.append(f"{sign}")
        if coeff_abs_str != "1": poly_parts.append(f"{coeff_abs_str} ")
        for j in range(i):
            if np.isclose(x_nodes[j], 0): poly_parts.append("x")
            else: poly_parts.append(f"(x - {x_nodes[j]:.4g})")
    if not poly_parts: return "P_n(x) = 0"
    latex_str = "".join(poly_parts).replace(')(', r') \cdot (').replace(')x', r') \cdot x')
    latex_str = re.sub(r'(-?\d+)/(\d+)', r'\\frac{\1}{\2}', latex_str)
    return "P_n(x) = " + latex_str

def format_B_interpolation_string(d_coeffs: list) -> str:
    parts = []
    for k, d in enumerate(d_coeffs):
        if np.isclose(d, 0): continue
        sign_char = "-" if d < 0 else "+"
        d_abs = abs(d)
        coeff_str = float_to_fraction_str(d_abs)
        term = f"B_{k}(x)"
        if coeff_str != "1":
            frac = Fraction(d_abs).limit_denominator(1000)
            term = f"\\frac{{{frac.numerator}}}{{{frac.denominator}}} \\cdot {term}"
        if not parts:
            parts.append(f"- {term}" if d < 0 else term)
        else:
            parts.append(f" {sign_char} {term}")
    if not parts: return "F_n(x) = 0"
    latex_str = "".join(parts).strip()
    if latex_str.startswith("+ "): latex_str = latex_str[2:]
    return f"F_n(x) = {latex_str}"

# --- UI prvky ---

def render_controls():
    """Vykreslí ovládací prvky s volbou režimu rozsahu."""
    st.header("Nastavení")
    default_points = "0, 1, 8, 27, 64"
    points_str = st.text_input(
        "Zadejte Y-ové hodnoty bodů (oddělené čárkou)", value=default_points)

    st.subheader("Rozsah grafu")
    mode = st.radio("Režim rozsahu", ["Automatický", "Manuální"], key="range_mode", horizontal=True)

    x_min, x_max = None, None
    if mode == "Manuální":
        c1, c2 = st.columns(2)
        with c1:
            x_min = st.number_input("X min", value=-5.0, step=1.0, format="%.1f")
        with c2:
            x_max = st.number_input("X max", value=10.0, step=1.0, format="%.1f")

    return points_str, mode, x_min, x_max

# --- ZDE JE ZMĚNA ---
def create_plot(x_plot, y_newton, y_b_interp, x_nodes, y_nodes, x_range, y_range) -> go.Figure:
    """Vytvoří Plotly graf se stejným měřítkem os."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_plot, y=y_newton, mode='lines', name='Newtonův polynom P_n(x)'))
    fig.add_trace(go.Scatter(x=x_plot, y=y_b_interp, mode='lines', name='B-interpolace F_n(x)', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=x_nodes, y=y_nodes, mode='markers', name='Zadané body', marker=dict(size=10, color='red')))

    fig.update_xaxes(range=x_range)

    # Nastavení stejného měřítka os
    fig.update_yaxes(
        range=y_range,
        scaleanchor="x",    # Ukotví měřítko osy Y na osu X
        scaleratio=1,       # Nastaví poměr měřítka na 1:1
    )

    fig.update_layout(
        xaxis_title="x", yaxis_title="y", hovermode="x unified", template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig