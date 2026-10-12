import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from matplotlib.patches import Patch

from georef import GridGeoref
from ndvi import get_landsat_ndvi_and_temp_matrix
from sim_class import simModel
from simulation import create_polygon, plot_overlay
from unet import (
    _fit_grid,
    _window_from_reports,
    build_inputs,
    load_unet,
)

# Ship exactly these four checkpoints in the repo, all with window_size=6.
# Keys are the forecast horizon k each model was trained for.
MODELS = {
    1: "models/unet_h1_w6.pth",
    2: "models/unet_h2_w6.pth",
    3: "models/unet_h3_w6.pth",
    4: "models/unet_h4_w6.pth",
}
WINDOW_SIZE = 6  # shared across all four

# simModel is created in preallocated mode, which requires a fixed maximum
# number of days up front. Users can still step the sim in chunks; if they try
# to exceed this, sim.step raises IndexError and the app shows an error.
MAX_SIM_DAYS = 365


@st.cache_resource
def get_unet(horizon):
    """Load one model per horizon and cache it across sessions."""
    path = MODELS.get(horizon)
    if path is None:
        return None, None
    p = Path(path)
    if not p.exists():
        return None, None
    model, ckpt = load_unet(str(p), device="cpu")
    model.eval()
    return model, ckpt


# Show the popup on first load
if "show_popup" not in st.session_state:
    st.session_state.show_popup = True


@st.dialog("What is this?", dismissible=False)
def quick_test_popup():
    st.markdown("""
    This page lets you watch a crop disease spread across a real agricultural
    field and see an AI model try to predict where it will go next.
    """)

    st.markdown("**What's real and what's simulated**")
    st.markdown("""
    - The vegetation map comes from real satellite imagery at the location
      you choose.
    - The disease outbreak is simulated. You control how fast it spreads,
     how wind carries it, and detection rates.
    - The AI forecast is a neural network trained on hundreds of simulated
      outbreaks. It only sees the detection reports, not the true infection,
      and predicts where the outbreak will be in the next 1 to 4 days.
    """)

    st.markdown(
        "[Check out this video](https://youtu.be/Y1aokfLRu8k) "
        "for a step-by-step walkthrough."
    )

    st.warning(
        "This is a research demo, not a real outbreak warning system.\n\n"
        "All disease spread shown on this page is simulated.\n\n"
        "NDVI images will be cropped to 80x80 grids for model inputs"
    )
    agreed = st.checkbox("I understand that this website is a demo")

    if st.button("Close", disabled=not agreed):
        st.session_state.show_popup = False
        st.rerun()


# Open the dialog if the flag is True
if st.session_state.show_popup:
    quick_test_popup()


# -----------------------------------------------------------------------------
# UNet prediction / comparison
# -----------------------------------------------------------------------------
def run_unet_prediction(horizon, threshold=0.5):
    """
    Run one model (chosen by horizon k) on the current simulation state
    and display its k-day forecast.
    """
    model, ckpt = get_unet(horizon)
    if model is None:
        st.error(
            f"Model for horizon {horizon} not found at {MODELS[horizon]}. "
            f"Add the .pth file to the models/ folder."
        )
        return

    sim = st.session_state.get("sim")
    if sim is None:
        st.warning(
            "No simulation yet. Click 'Initialize Spreading Capabilities' and "
            "run some timesteps first."
        )
        return

    reports = sim.report_grids
    if reports.shape[0] < WINDOW_SIZE:
        st.warning(
            f"Need at least {WINDOW_SIZE} simulated days to build the "
            f"model input; have {reports.shape[0]}. "
            f"Run the simulation longer."
        )
        return

    info = ckpt["training_info"]
    cfg = ckpt["model_config"]
    k = cfg.get("n_horizons", horizon)
    target_size = tuple(info.get("target_size", (80, 80)))
    subtract_one = info.get("subtract_one", True)
    log1p_window = info.get("log1p_window", True)

    ndvi = sim.ndvi

    window = _window_from_reports(reports, WINDOW_SIZE).astype(np.int32)
    th, tw = target_size
    ndvi_fit = _fit_grid(ndvi, th, tw, pad_value=0.0)
    window_fit = _fit_grid(window, th, tw, pad_value=0)

    if subtract_one:
        m = window_fit > 0
        window_fit = window_fit.copy()
        window_fit[m] -= 1

    x_seq = build_inputs(ndvi_fit, window_fit, log1p_window=log1p_window)

    with torch.no_grad():
        logits_list = model(x_seq)
    probs = torch.sigmoid(torch.stack(logits_list)).cpu().numpy()  # (D, k, H, W)

    t = probs.shape[0] - 1
    current_window = window_fit[t]
    current_I = _fit_grid(sim.I, th, tw, 0.0)

    # --- figure ---
    fig, axes = plt.subplots(2, k + 2, figsize=(3 * (k + 2), 6.5))

    ax = axes[0, 0]
    im = ax.imshow(current_window, cmap="Greys", origin="upper")
    ax.set_title(f"window input (day {reports.shape[0]})", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[0, 1]
    im = ax.imshow(current_I, cmap="hot", vmin=0, vmax=1, origin="upper")
    ax.set_title("current infected (sim)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([])
    ax.set_yticks([])

    for c in range(k):
        ax = axes[0, c + 2]
        im = ax.imshow(probs[t, c], cmap="hot", vmin=0, vmax=1, origin="upper")
        ax.set_title(f"pred +{c + 1}  (m={probs[t, c].mean():.3f})", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes[1, 0]
    ax.imshow(
        (current_window > 0).astype(float), cmap="Greys", vmin=0, vmax=1, origin="upper"
    )
    ax.set_title("reported cells", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1, 1]
    ax.imshow(current_I, cmap="hot", vmin=0, vmax=1, origin="upper")
    ax.set_title("infected truth", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    for c in range(k):
        ax = axes[1, c + 2]
        mask = (probs[t, c] > threshold).astype(float)
        ax.imshow(mask, cmap="hot", vmin=0, vmax=1, origin="upper")
        ax.set_title(f"mask +{c + 1}  ({mask.mean() * 100:.1f}%)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"UNet forecast (horizon k={k}, window={WINDOW_SIZE}, threshold={threshold})",
        fontsize=12,
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Predicted infection coverage by horizon**")
    cols = st.columns(k)
    for c in range(k):
        mask = probs[t, c] > threshold
        cols[c].metric(
            f"+{c + 1} day",
            f"{mask.mean() * 100:.1f}%",
            help=f"mean prob = {probs[t, c].mean():.4f}",
        )


def run_unet_comparison(threshold=0.5):
    """
    Run every shipped model on the current simulation state and show the
    predicted infection mask for each model's own horizon side by side.
    Because horizon and window size are fixed across models, the input
    tensor is identical for all four; only the head differs.
    """
    sim = st.session_state.get("sim")
    if sim is None:
        st.warning("No simulation yet. Initialize and run some timesteps first.")
        return

    reports = sim.report_grids
    if reports.shape[0] < WINDOW_SIZE:
        st.warning(
            f"Need at least {WINDOW_SIZE} simulated days; have {reports.shape[0]}."
        )
        return

    ndvi = sim.ndvi
    window = _window_from_reports(reports, WINDOW_SIZE).astype(np.int32)

    # all models share target_size / subtract_one, so use the first one
    first_model, first_ckpt = get_unet(min(MODELS.keys()))
    if first_model is None:
        st.error("No models available.")
        return
    info = first_ckpt["training_info"]
    target_size = tuple(info.get("target_size", (80, 80)))
    subtract_one = info.get("subtract_one", True)
    log1p_window = info.get("log1p_window", True)

    th, tw = target_size
    ndvi_fit = _fit_grid(ndvi, th, tw, 0.0)
    window_fit = _fit_grid(window, th, tw, 0)
    if subtract_one:
        m = window_fit > 0
        window_fit = window_fit.copy()
        window_fit[m] -= 1

    x_seq = build_inputs(ndvi_fit, window_fit, log1p_window=log1p_window)

    current_I = _fit_grid(sim.I, th, tw, 0.0)

    rows = []
    for horizon in sorted(MODELS.keys()):
        model, _ = get_unet(horizon)
        if model is None:
            continue
        with torch.no_grad():
            logits_list = model(x_seq)
        probs = torch.sigmoid(torch.stack(logits_list)).cpu().numpy()
        t = probs.shape[0] - 1
        # take the final horizon this model predicts
        pred = probs[t, -1]
        rows.append((horizon, pred, current_I))

    if not rows:
        st.warning("No models loaded.")
        return

    fig, axes = plt.subplots(1, len(rows) + 2, figsize=(3 * (len(rows) + 2), 3.5))

    ax = axes[0]
    ax.imshow(current_I, cmap="hot", vmin=0, vmax=1, origin="upper")
    ax.set_title("sim truth", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    ax.imshow(
        (window_fit[-1] > 0).astype(float), cmap="Greys", vmin=0, vmax=1, origin="upper"
    )
    ax.set_title("reported", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    for i, (horizon, pred, _) in enumerate(rows):
        ax = axes[i + 2]
        ax.imshow(pred, cmap="hot", vmin=0, vmax=1, origin="upper")
        cov = (pred > threshold).mean() * 100
        ax.set_title(f"k={horizon} pred  ({cov:.1f}%)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Forecast comparison across horizons", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


st.set_page_config(layout="wide")


# -----------------------------------------------------------------------------
# Satellite data
# -----------------------------------------------------------------------------
def get_NDVI(lati, longi, dist):
    status_placeholder = st.empty()

    status_placeholder.info("Attempting to get data from API...")

    time.sleep(1)
    try:
        ndvi, temp = get_landsat_ndvi_and_temp_matrix(
            lati, longi, half_side_meters=dist
        )
        st.session_state["ndvi"] = ndvi
        st.session_state["temps"] = temp
        st.write("Got Ndvi")
        st.session_state["lati"] = lati
        st.session_state["long"] = longi
        st.session_state["dist"] = dist
        # New NDVI invalidates any running sim.
        st.session_state.pop("sim", None)
        status_placeholder.success("Data Acquired")
    except Exception as e:
        print(f"An error occurred: {e}")
        status_placeholder.error(f"Failed to fetch data. Error {e}")


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------
def init_matrix(spreadability, variance, wind_x, wind_y, magni_mult):
    """
    Build a fresh simModel: dispersal kernel, initial infection, and
    preallocated history/report buffers. Replaces the old manual
    S/I/R + conv_matrix + reports_list bookkeeping.
    """
    status_placeholder = st.empty()
    ndvi = st.session_state["ndvi"]
    status_placeholder.info("Calculating Kernel")

    sim = simModel(
        ndvi=ndvi,
        center_coords=(st.session_state["lati"], st.session_state["long"]),
        dist_per_grid=15,
        fp=0.01,
        fn=0.01,
        beta=st.session_state["beta"],
        gamma=st.session_state["gamma"],
        total_steps=MAX_SIM_DAYS,
    )

    # simModel.createMatrix does exactly what generate_wind_kernel used to.
    sim.createMatrix(
        radius=int(spreadability),
        sigma=variance,
        wind_x=wind_x * magni_mult,
        wind_y=-wind_y * magni_mult,
    )

    # Place one infected cell in the middle third of the grid (same behaviour
    # as start_SIR_random_middle) and build the lat/lon lookup matrix.
    sim.initSim(mode="middle")

    st.session_state["sim"] = sim
    st.session_state["var"] = variance
    st.session_state["spread"] = spreadability
    st.session_state["wind_x"] = wind_x
    st.session_state["wind_y"] = wind_y
    st.session_state["magni_mult"] = magni_mult

    status_placeholder.success("Finished")
    ndvi = st.session_state["ndvi"]
    print(ndvi.min(), (ndvi < 0).sum(), ndvi.shape)


def simulate_steps(timestep):
    """
    Advance the current simModel by `timestep` days. Report grids and history
    are written internally by sim.step -> sim._single_step, so this function
    only needs to render the SIR composite figure.
    """
    status_placeholder = st.empty()
    sim = st.session_state["sim"]
    status_placeholder.info("Starting")

    try:
        sim.step(steps=int(timestep))
    except IndexError as e:
        # simModel preallocates history/report buffers; stepping past
        # total_steps raises IndexError.
        status_placeholder.error(
            f"Reached the preallocated simulation length "
            f"({MAX_SIM_DAYS} days): {e}"
        )
        return

    status_placeholder.success("Finished")

    rgb_grid = np.stack([sim.I, sim.R, sim.S], axis=-1)

    fig2, ax2 = plt.subplots(figsize=(8, 8))

    ax2.imshow(rgb_grid, origin="upper")

    ax2.set_title("Current simulation", fontsize=14, pad=15)
    ax2.axis("off")

    legend_elements = [
        Patch(facecolor="blue", label="Susceptible (S)"),
        Patch(facecolor="red", label="Infectious (I)"),
        Patch(facecolor="green", label="Recovered (R)"),
    ]

    ax2.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))

    fig2.tight_layout()
    st.session_state["Simulation_graph"] = fig2
    st.session_state["Simulation_ax"] = ax2


# Hides the declaration block of the website
def toggle_layout():
    st.session_state.show_all = not st.session_state.show_all


st.title("NDVI Based Crop SIR Simulation")

with st.expander("About this app", expanded=True):
    st.write("""
    NDVI is Normalized Difference Vegetation Index. NDVI is a measuremnet comparing the 
    difference between near infrared and red light divided by their sum. This is a helpful \n\n
    approximation for how dense an area's vegetation is, since healthy plants absorb most 
    visible red light and reflect a large amount of near-infrared light.
    See more about NDVI [here](https://www.earthdata.nasa.gov/topics/land-surface/normalized-difference-vegetation-index-ndvi).
    """)


# code block below is to make sure the variables are saved when the declaration code is toggled
if "ndvi" not in st.session_state:
    st.session_state["ndvi"] = np.load("Default_ndvi_data.npy")
if "wind_x" not in st.session_state:
    st.session_state["wind_x"] = 0.0
if "wind_y" not in st.session_state:
    st.session_state["wind_y"] = 0.0
if "magni_mult" not in st.session_state:
    st.session_state["magni_mult"] = 1.0
if "beta" not in st.session_state:
    st.session_state["beta"] = 0.3
if "gamma" not in st.session_state:
    st.session_state["gamma"] = 0.0
if "show_all" not in st.session_state:
    st.session_state.show_all = True
if "lati" not in st.session_state:
    st.session_state["lati"] = 37.498056
if "long" not in st.session_state:
    st.session_state["long"] = -120.812583
if "dist" not in st.session_state:
    st.session_state["dist"] = 3000
if "spread" not in st.session_state:
    st.session_state["spread"] = 5
if "var" not in st.session_state:
    st.session_state["var"] = 3


if st.session_state.show_all:
    st.header("Declare Variables")
    with st.expander("About the Variables", expanded=True):
        st.write("""
        Range is the size of the ndvi to request from the planetary computer api, please note, the current minimum to be used with the U-Net is 80x80 with all NDVI grid calculations being cropped to that size. \n\n
        Approximately 15 square meters is one grid value. If you do not wish to see the predictions, there is no restriction on size. \n\n
        Maximum Spread Range is the maximum distance one infected plant can potentially infect another.\n\n
        Spread Concentration how likely the disease is to spread to its maximum range, issues occur when the value provided is higher than the spread range and values between 0 and 1. \n\n
        Rate of transmissions is an additional scalar to adjust how likely a plant is to get infected, value must be between 0 and 1.
        """)

    # satellite data init
    sat_init_1, sat_init_2, sat_init_3 = st.columns(3)

    with sat_init_1:
        lat = st.number_input("Latitude", value=st.session_state["lati"])
    with sat_init_2:
        long = st.number_input("Longitude", value=st.session_state["long"])
    with sat_init_3:
        dist = st.number_input("Range", value=st.session_state["dist"])

    # for creating the conv matrix
    mat1, mat2, mat3 = st.columns(3)
    with mat1:
        spreadability = st.number_input(
            "Maximum Spread Range", value=st.session_state["spread"]
        )
    with mat2:
        variance = st.number_input(
            "Spread Concentration", value=st.session_state["var"]
        )
    with mat3:
        beta = st.number_input(
            "Rate of Transmission",
            value=st.session_state["beta"],
            min_value=0.0,
            max_value=1.0,
        )

    # slider input test
    wind1, wind2, wind3, wind4 = st.columns(4)
    with wind1:
        wind_x = st.slider(
            "Wind Horizontal Component",
            min_value=-1.0,
            max_value=1.0,
            value=st.session_state["wind_x"],
            step=0.01,
        )
    with wind2:
        wind_y = st.slider(
            "Wind Vertical Component",
            min_value=-1.0,
            max_value=1.0,
            value=st.session_state["wind_y"],
            step=0.01,
        )
    with wind3:
        magni_mult = st.slider(
            "Magnitude (useless without wind vector)",
            min_value=0.1,
            max_value=5.0,
            value=st.session_state["magni_mult"],
            step=0.1,
        )

    # normalizing
    magnitude = np.sqrt(wind_x**2 + wind_y**2)

    if magnitude > 0:
        st.session_state["norm_x"] = wind_x / magnitude
        st.session_state["norm_y"] = wind_y / magnitude
    else:
        st.session_state["norm_x"], st.session_state["norm_y"] = 0.0, 0.0

    # creating unit fig diagram
    unitfig, unitax = plt.subplots(figsize=(5, 5))

    # drawing unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    unitax.plot(
        np.cos(theta),
        np.sin(theta),
        color="lightgray",
        linestyle="--",
        label="Unit Circle",
    )

    # highlighting axes
    unitax.axhline(0, color="black", linewidth=0.8)
    unitax.axvline(0, color="black", linewidth=0.8)

    # drawing the vector

    unitax.quiver(
        0,
        0,
        st.session_state["norm_x"],
        st.session_state["norm_y"],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="royalblue",
        width=0.015,
    )

    # small dot at the tip
    unitax.plot(
        st.session_state["norm_x"],
        st.session_state["norm_y"],
        marker="o",
        color="royalblue",
        markersize=6,
    )

    # formatting the plot
    unitax.set_xlim([-1.2, 1.2])
    unitax.set_ylim([-1.2, 1.2])
    unitax.set_aspect(
        "equal"
    )  # Forces the plot to be a perfect square, preventing oval distortion
    unitax.grid(True, which="both", linestyle=":", alpha=0.5)
    unitax.set_title("Resultant Unit Vector", fontsize=12)

    #  displaying info sideby side
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.metric("Normalized X", f"{st.session_state['norm_x']:.5f}")
        st.metric("Normalized Y", f"{st.session_state['norm_y']:.5f}")

        show_graph = st.toggle("Toggle Vector Graph", value=True)

    with col2:
        # Use st.pyplot to render the Matplotlib figure directly
        if show_graph:
            st.pyplot(unitfig)

    simcol1, simcol2, simcol3 = st.columns([0.25, 0.25, 0.5])

    with simcol1:
        if st.button(
            f"Get NDVI data at {st.session_state['lati']}, {st.session_state['long']}"
        ):
            get_NDVI(lat, long, dist)
    with simcol2:
        if st.button("Initialize Spreading Capabilities"):
            init_matrix(
                spreadability,
                variance,
                st.session_state["norm_x"],
                st.session_state["norm_y"],
                magni_mult,
            )

tab1, tab2, tab3 = st.tabs(
    ["Satellite NDVI View", "SIR Simulation View", "Temperature display"]
)
fig, ax = plt.subplots()

with tab1:
    st.subheader("NDVI Display")
    st.button("Show/Hide Variable Declaration", on_click=toggle_layout)

    graph_col, padding_col = st.columns([0.5, 0.5])

    with graph_col:
        ax.imshow(st.session_state["ndvi"], vmin=0.0, origin="upper")
        st.session_state["ndvi_fig"] = fig
        st.session_state["ndvi_ax"] = ax
        st.pyplot(fig)

with tab2:
    st.subheader("SIR Simulation")
    select_sim_col, sim_col, filer_col = st.columns([0.25, 0.25, 0.5])

    with select_sim_col:
        timestep = st.number_input("Number of time steps to calculate", value=10)
        threshhold = st.number_input("threshhold", value=0.01)
    with sim_col:
        if st.button(f"Simulate {timestep} Timesteps"):
            # Auto-init a sim if the user hasn't clicked Initialize yet.
            if "sim" not in st.session_state:
                init_matrix(
                    st.session_state["spread"],
                    st.session_state["var"],
                    st.session_state["norm_x"],
                    st.session_state["norm_y"],
                    st.session_state["magni_mult"],
                )

            simulate_steps(timestep)
            sim = st.session_state["sim"]

            # simModel.generate_latlon_matrix (called during initSim) already
            # populates sim.coordinate_matrix, so we don't need the local
            # helper any more.
            georef = GridGeoref.from_locs(sim.coordinate_matrix, dist_per_cell=15)
            polygon = create_polygon(
                prob_grid=sim.I,
                georef=georef,
                threshold=threshhold,
                method="contour",
            )
            st.session_state["Simulation_graph"] = plot_overlay(
                grid=sim.I, polygon=polygon, georef=georef
            )

    graph_col, padding_col = st.columns([0.5, 0.5])
    st.markdown("---")
    st.subheader("UNet forecast")

    c1, c2, c3 = st.columns([0.4, 0.3, 0.3])
    with c1:
        unet_threshold = st.slider("prediction threshold", 0.1, 0.9, 0.5, 0.05)
    with c2:
        unet_horizon = st.selectbox(
            "model horizon (k)", sorted(MODELS.keys()), index=len(MODELS) - 1
        )
    with c3:
        st.write("")  # spacer
        st.write("")
        show_single = st.button("Show single forecast")
        show_all = st.button("Compare all horizons")

    if show_single:
        run_unet_prediction(unet_horizon, threshold=unet_threshold)

    if show_all:
        run_unet_comparison(threshold=unet_threshold)
    with graph_col:
        if "Simulation_graph" in st.session_state:
            fig2 = st.session_state["Simulation_graph"]
            ax2 = st.session_state["Simulation_ax"]
            st.pyplot(fig2)


with tab3:
    st.subheader("Temperature Display")

    graph_col2, padding_col2 = st.columns([0.5, 0.5])

    with graph_col2:
        if "temps" in st.session_state:
            fig3, ax3 = plt.subplots()

            im = ax3.imshow(st.session_state["temps"], origin="upper")

            fig3.colorbar(im, ax=ax3)

            st.pyplot(fig3)