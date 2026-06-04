"""
Interactive double-slit simulation app.
Author contribution: Paul Völker implemented this project part.
"""

import streamlit as st
import matplotlib.pyplot as plt

from modes import MODES
from physics.double_slit import de_broglie_wavelength


def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


st.set_page_config(page_title="Double-Slit Simulation", page_icon="⚛", layout="wide")

st.title("Double-Slit Simulation for Massive Particles")

if "playing" not in st.session_state:
    st.session_state.playing = False

if "progress" not in st.session_state:
    st.session_state.progress = 0.0

with st.sidebar:
    st.header("Controls")

    mode_name = st.selectbox("Simulation mode", list(MODES.keys()))

    st.subheader("Particle")
    mass = st.number_input("Mass [kg]", value=9.11e-31, format="%.3e")
    velocity = st.number_input("Velocity [m/s]", value=2.00e6, format="%.3e")

    st.subheader("Geometry")
    slit_distance = st.number_input("Slit center distance [m]", value=8.00e-7, format="%.3e")
    slit_width = st.number_input("Single slit width [m]", value=1.50e-7, format="%.3e")
    source_distance = st.number_input("Source → slit distance [m]", value=0.20, format="%.3e")
    screen_distance = st.number_input("Slit → screen distance [m]", value=0.50, format="%.3e")
    screen_width = st.number_input("Screen half-width [m]", value=0.006, format="%.3e")

    which_path = st.checkbox("Which-path measurement", value=False)

selected_mode = MODES[mode_name]
wavelength = de_broglie_wavelength(mass, velocity)

st.subheader("Animation")

col_toggle, col_reset, col_slider = st.columns([1.4, 1.0, 6.0])

with col_toggle:
    toggle_label = "⏸ Pause" if st.session_state.playing else "▶ Play"

    if st.button(toggle_label, use_container_width=True):
        st.session_state.playing = not st.session_state.playing

with col_reset:
    if st.button("↺ Reset", use_container_width=True):
        st.session_state.playing = False
        st.session_state.progress = 0.0
        rerun_app()

with col_slider:
    if st.session_state.playing:
        st.progress(st.session_state.progress)
    else:
        st.session_state.progress = st.slider(
            "Progress",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.progress),
            step=0.005,
            label_visibility="collapsed",
        )

params = {
    "mass": mass,
    "velocity": velocity,
    "slit_distance": slit_distance,
    "slit_width": slit_width,
    "source_distance": source_distance,
    "screen_distance": screen_distance,
    "screen_width": screen_width,
    "which_path": which_path,
    "progress": st.session_state.progress,
}

fig_top, fig_screen = selected_mode.render(params)

col1, col2 = st.columns([1.6, 1.0])

with col1:
    st.subheader("Top View: Source → Slits → Screen")
    st.pyplot(fig_top, clear_figure=True)

with col2:
    st.subheader("Screen Pattern")
    st.pyplot(fig_screen, clear_figure=True)

plt.close(fig_top)
plt.close(fig_screen)

st.subheader("Explanation")
st.info(selected_mode.EXPLANATION)

st.subheader("Current values")
st.write(f"de Broglie wavelength: `{wavelength:.3e} m`")
st.write(f"mass: `{mass:.3e} kg`")
st.write(f"velocity: `{velocity:.3e} m/s`")
st.write(f"source → slit: `{source_distance:.3e} m`")
st.write(f"slit → screen: `{screen_distance:.3e} m`")
st.write(f"slit center distance: `{slit_distance:.3e} m`")
st.write(f"single slit width: `{slit_width:.3e} m`")
st.write(f"which-path measurement: `{which_path}`")

if st.session_state.playing:
    st.session_state.progress += 0.035

    if st.session_state.progress >= 1.0:
        st.session_state.progress = 1.0
        st.session_state.playing = False

    rerun_app()