# ui_components.py
# Modul pro komponenty uživatelského rozhraní (Streamlit a Plotly).

import streamlit as st
import plotly.graph_objects as go
from fractions import Fraction
import re
import numpy as np  # <-- TENTO ŘÁDEK BYL PŘIDÁN

# --- Formátovací funkce ---

def float_to_fraction_str(num: float, limit_denominator: int = 1000) -> str:
    """Převede float na řetězec reprezentující zlomek."""
    if np.isclose(num, 0): return "0"
    fraction = Fraction(num).limit_denominator(limit_denominator)
    return str(fraction.numerator) if fraction.denominator == 1 else f"{fraction.numerator}/{fraction.denominator}"

def format_newton_string(coeffs: list, x_nodes: list) -> str:
    """Vytvoří čitelný LaTeX řetězec pro Newtonův polynom."""
    poly_parts = []
    first_coeff_str = float_to_fraction_str(coeffs[0])
    if first_coeff_str != "0" or len(coeffs) == 1:
        poly_parts.append(first_coeff_str)

    for i in range(1, len(coeffs)):
        coeff_val = coeffs[i]
        if np.isclose(coeff_val, 0): continue  # Upraveno pro konzistenci
        sign = "+" if coeff_val > 0 else "-"
        coeff_abs_str = float_to_fraction_str(abs(coeff_val))
        if poly_parts: poly_parts.append(f" {sign} ")
        elif sign == "-": poly_parts.append(f"{sign}")
        if coeff_abs_str != "1": poly_parts.append(f"{coeff_abs_str} ")
        for j in range(i):
            if np.isclose(x_nodes[j], 0): poly_parts.append("x") # Upraveno
            else: poly_parts.append(f"(x - {x_nodes[j]:.4g})")

    if not poly_parts: return "P_n(x) = 0"
    latex_str = "".join(poly_parts).replace(')(', r') \cdot (').replace(')x', r') \cdot x')
    latex_str = re.sub(r'(-?\d+)/(\d+)', r'\\frac{\1}{\2}', latex_str)
    return "P_n(x) = " + latex_str

def format_B_interpolation_string(d_coeffs: list) -> str:
    """Vytvoří čitelný LaTeX řetězec pro funkci F(x)."""
    parts = []
    for k, d in enumerate(d_coeffs):
        if np.isclose(d, 0): continue  # Upraveno pro konzistenci

        sign_char = "-" if d < 0 else "+"
        d_abs = abs(d)
        coeff_str = float_to_fraction_str(d_abs)
        term = f"B_{k}(x)"

        # Zjednodušení pro koeficient 1
        if coeff_str != "1":
            # Použití \frac pro hezčí zlomky
            frac = Fraction(d_abs).limit_denominator(1000)
            term = f"\\frac{{{frac.numerator}}}{{{frac.denominator}}} \\cdot {term}"

        if not parts:
            parts.append(f"- {term}" if d < 0 else term)
        else:
            parts.append(f" {sign_char} {term}")

    if not parts: return "F_n(x) = 0"

    latex_str = "".join(parts).strip()
    # Odstranění úvodního plus, pokud tam je
    if latex_str.startswith("+ "):
        latex_str = latex_str[2:]

    return f"F_n(x) = {latex_str}"


# --- UI prvky ---

def render_controls() -> str:
    """Vykreslí ovládací prvky."""
    st.header("Nastavení")
    default_points = "0, 1, 8, 27, 64"  # y = x^3
    points_str = st.text_input(
        "Zadejte Y-ové hodnoty bodů (oddělené čárkou)", value=default_points)
    st.caption("Program automaticky přiřadí X-ové hodnoty: 0, 1, 2, ... a přizpůsobí graf.")
    return points_str

def create_plot(x_plot, y_newton, y_b_interp, x_nodes, y_nodes, x_range, y_range) -> go.Figure:
    """Vytvoří Plotly graf s oběma křivkami a pevně nastaveným rozsahem."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_plot, y=y_newton, mode='lines', name='Newtonův polynom P_n(x)'))
    fig.add_trace(go.Scatter(x=x_plot, y=y_b_interp, mode='lines', name='B-interpolace F_n(x)', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=x_nodes, y=y_nodes, mode='markers', name='Zadané body', marker=dict(size=10, color='red')))

    fig.update_xaxes(range=x_range, autorange=False, zeroline=True, showgrid=True)
    fig.update_yaxes(range=y_range, autorange=False, zeroline=True, showgrid=True)

    fig.update_layout(
        xaxis_title="x", yaxis_title="y", hovermode="x unified", template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig