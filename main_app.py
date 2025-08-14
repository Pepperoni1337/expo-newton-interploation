# main_app.py
# Hlavní soubor Streamlit aplikace pro porovnání interpolačních metod.
# Spusťte tento soubor: streamlit run main_app.py

import streamlit as st
import numpy as np
import math
import calculations as calc
import ui_components as ui

# --- Konfigurace ---
POINTS_PER_UNIT = 100
PADDING_FACTOR = 0.1

def main():
    """Hlavní funkce, která spouští a řídí aplikaci."""
    st.set_page_config(page_title="Porovnání interpolací", layout="wide")
    st.title("Porovnání interpolačních metod")

    col_results, col_controls = st.columns([3, 1], gap="large")

    with col_controls:
        points_str, mode, manual_x_min, manual_x_max = ui.render_controls()

    try:
        y_nodes = np.array([float(val.strip()) for val in points_str.split(',') if val.strip()])
        if len(y_nodes) < 1:
            st.warning("Zadejte alespoň jednu Y-ovou hodnotu."); st.stop()

        n = len(y_nodes) - 1
        x_nodes = np.arange(n + 1)

        # --- Výpočet rozsahu grafu podle zvoleného režimu ---
        if mode == "Automatický":
            x_min, x_max = np.min(x_nodes), np.max(x_nodes)
            x_span = x_max - x_min if x_max > x_min else 1.0
            x_padding = x_span * PADDING_FACTOR
            x_axis_range = [x_min - x_padding, x_max + x_padding]
        else: # Manuální režim
            if manual_x_min >= manual_x_max:
                st.sidebar.error("Hodnota 'X min' musí být menší než 'X max'.")
                st.stop()
            x_axis_range = [manual_x_min, manual_x_max]

        # Y rozsah vypočítáme vždy automaticky pro nejlepší zobrazení
        y_min, y_max = np.min(y_nodes), np.max(y_nodes)
        y_span = y_max - y_min if y_max > y_min else 1.0
        y_padding = y_span * PADDING_FACTOR
        y_axis_range = [y_min - y_padding, y_max + y_padding]

        # --- Chytré generování bodů pro vykreslení ---
        num_smooth_points = int((x_axis_range[1] - x_axis_range[0]) * POINTS_PER_UNIT)
        smooth_points = np.linspace(x_axis_range[0], x_axis_range[1], max(2, num_smooth_points))
        start_int = math.ceil(x_axis_range[0])
        end_int = math.floor(x_axis_range[1])
        integer_points = np.arange(start_int, end_int + 1, dtype=float)
        x_plot = np.union1d(smooth_points, integer_points)

        # --- Výpočty interpolací ---
        newton_coeffs = calc.calculate_newton_coeffs(x_nodes, y_nodes)
        d_coeffs = calc.calculate_d_coeffs(y_nodes)
        y_plot_newton = calc.evaluate_newton_poly(newton_coeffs, x_nodes, x_plot)
        y_plot_b = calc.evaluate_B_interpolation(d_coeffs, x_plot)

    except (ValueError, IndexError):
        st.error("Chyba ve vstupu. Zkontrolujte formát zadaných bodů."); st.stop()

    with col_results:
        st.header("Grafické porovnání")
        fig = ui.create_plot(x_plot, y_plot_newton, y_plot_b, x_nodes, y_nodes, x_axis_range, y_axis_range)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Newtonova interpolace")
            st.code('\n'.join([f"c{i} = {ui.float_to_fraction_str(c)}" for i, c in enumerate(newton_coeffs)]), 'text')
            st.latex(ui.format_newton_string(newton_coeffs, x_nodes))
        with col2:
            st.subheader("2. B-interpolace")
            st.code('\n'.join([f"d{k} = {ui.float_to_fraction_str(d)}" for k, d in enumerate(d_coeffs)]), 'text')
            st.latex(ui.format_B_interpolation_string(d_coeffs))
            st.caption(f"kde $B_k(x) = \\frac{{1}}{{k!}} \\sum_{{j=0}}^{{k}} (-1)^{{k-j}} \\binom{{k}}{{j}} (j+1)^x$")

if __name__ == "__main__":
    main()