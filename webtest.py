import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
import torch
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from shapely.geometry import MultiPoint, Polygon
from skimage import measure

from georef import GridGeoref
from ndvi import get_landsat_ndvi_and_temp_matrix, get_landsat_ndvi_matrix
from simulation import (
    create_polygon,
    generate_wind_kernel,
    plot_overlay,
    sir_spatial_step,
    start_SIR_random,
    start_SIR_random_middle,
)
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
WINDOW_SIZE = 6   # shared across all four


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
        "[Tutorial unfinished, will provide link soon](https://google.com) "
        "for a step-by-step walkthrough."
    )

    st.warning(
        "This is a research demo, not a real outbreak warning system.\n\n"
        "All disease spread shown on this page is simulated.\n\n"
        "NDVI images will be cropped to 80x80 grids for model inputs"
    )

    if st.button("Close"):
        st.session_state.show_popup = False
        st.rerun()

# Open the dialog if the flag is True
if st.session_state.show_popup:
    quick_test_popup()

def generate_report_grid(I, S, ndvi, fp=0.01, fn=0.01, seed=None):
    """
    Mimic simModel.createReportGrid on the current simulation state.
    Produces a (H, W) uint8 grid of report counts per cell for one day.
    """
    rng = np.random.default_rng(seed)
    tp = rng.poisson(I * ndvi*10)
    if fn > 0:
        tp = rng.binomial(tp, 1.0 - fn)
    fp_arr = rng.poisson(S * ndvi * fp)
    return np.clip(tp + fp_arr, 0, 255).astype(np.uint8)


def run_unet_prediction(horizon, threshold=0.5, fp=0.01, fn=0.01):
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

    reports_list = st.session_state.get("reports_list", [])
    if len(reports_list) < WINDOW_SIZE:
        st.warning(
            f"Need at least {WINDOW_SIZE} simulated days to build the "
            f"model input; have {len(reports_list)}. "
            f"Run the simulation longer."
        )
        return

    info = ckpt["training_info"]
    cfg = ckpt["model_config"]
    k = cfg.get("n_horizons", horizon)
    target_size = tuple(info.get("target_size", (80, 80)))
    subtract_one = info.get("subtract_one", True)
    log1p_window = info.get("log1p_window", True)

    reports = np.stack(reports_list, axis=0)   # (D, H, W)
    ndvi = st.session_state["ndvi"]

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
    current_I = _fit_grid(st.session_state["I"], th, tw, 0.0)

    # --- figure ---
    fig, axes = plt.subplots(2, k + 2, figsize=(3 * (k + 2), 6.5))

    ax = axes[0, 0]
    im = ax.imshow(current_window, cmap="Greys", origin="upper")
    ax.set_title(f"window input (day {len(reports_list)})", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[0, 1]
    im = ax.imshow(current_I, cmap="hot", vmin=0, vmax=1, origin="upper")
    ax.set_title("current infected (sim)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([]); ax.set_yticks([])

    for c in range(k):
        ax = axes[0, c + 2]
        im = ax.imshow(probs[t, c], cmap="hot", vmin=0, vmax=1, origin="upper")
        ax.set_title(f"pred +{c+1}  (m={probs[t, c].mean():.3f})", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 0]
    ax.imshow((current_window > 0).astype(float), cmap="Greys",
              vmin=0, vmax=1, origin="upper")
    ax.set_title("reported cells", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 1]
    ax.imshow(current_I, cmap="hot", vmin=0, vmax=1, origin="upper")
    ax.set_title("infected truth", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    for c in range(k):
        ax = axes[1, c + 2]
        mask = (probs[t, c] > threshold).astype(float)
        ax.imshow(mask, cmap="hot", vmin=0, vmax=1, origin="upper")
        ax.set_title(f"mask +{c+1}  ({mask.mean()*100:.1f}%)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"UNet forecast (horizon k={k}, window={WINDOW_SIZE}, "
        f"threshold={threshold})",
        fontsize=12,
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Predicted infection coverage by horizon**")
    cols = st.columns(k)
    for c in range(k):
        mask = (probs[t, c] > threshold)
        cols[c].metric(f"+{c+1} day", f"{mask.mean()*100:.1f}%",
                       help=f"mean prob = {probs[t, c].mean():.4f}")


def run_unet_comparison(threshold=0.5):
    """
    Run every shipped model on the current simulation state and show the
    predicted infection mask for each model's own horizon side by side.
    Because horizon and window size are fixed across models, the input
    tensor is identical for all four; only the head differs.
    """
    reports_list = st.session_state.get("reports_list", [])
    if len(reports_list) < WINDOW_SIZE:
        st.warning(
            f"Need at least {WINDOW_SIZE} simulated days; have "
            f"{len(reports_list)}."
        )
        return

    reports = np.stack(reports_list, axis=0)
    ndvi = st.session_state["ndvi"]

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

    current_I = _fit_grid(st.session_state["I"], th, tw, 0.0)

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
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    ax.imshow((window_fit[-1] > 0).astype(float), cmap="Greys",
              vmin=0, vmax=1, origin="upper")
    ax.set_title("reported", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    for i, (horizon, pred, _) in enumerate(rows):
        ax = axes[i + 2]
        ax.imshow(pred, cmap="hot", vmin=0, vmax=1, origin="upper")
        cov = (pred > threshold).mean() * 100
        ax.set_title(f"k={horizon} pred  ({cov:.1f}%)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Forecast comparison across horizons", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
st.set_page_config(layout="wide")


# functions
def get_NDVI(lati, longi, dist):

    status_placeholder = st.empty()

    status_placeholder.info("Attempting to get data from API...")

    time.sleep(1)
    try:
        ndvi, temp = get_landsat_ndvi_and_temp_matrix(lati, longi, half_side_meters=dist)
        st.session_state["ndvi"] = ndvi
        st.session_state["temps"] = temp
        st.write("Got Ndvi")
        st.session_state["lati"] = lati
        st.session_state["long"] = long
        st.session_state["dist"] = dist
        if "conv_matrix" in st.session_state:
            del st.session_state["conv_matrix"]

        status_placeholder.success("Data Acquired")
    except Exception as e:
        print(f"An error occurred: {e}")
        status_placeholder.error(f"Failed to fetch data. Error {e}")
def generate_latlon_matrix(input_matrix, center_coords, dist_per_cell=15):
    """
    creates a matrix of shape input_matrix with each value relating of the estimated long/lat

    INPUTS
    ---------
    input_matrix = 2d numpy array
    center_idx = tuple (row,column) indicating the indicie of the center (might be removable)
    center_coords = tuple (lat, long) cooresponding to the coord of the center idx
    dist_per_cell = distance for each cell, in meters

    RETURNS
    ---------
    numpy array with shape of input and one additional layer, representing each spaces relative coordinate
    """
    rows, cols = input_matrix.shape
    center_row, center_col = input_matrix.shape
    center_lat, center_lon = center_coords

    # constants
    R = 6378137.0  # radius of the earth
    RAD_PER_DEG = np.pi / 180.0
    DEG_PER_RAD = 180.0 / np.pi

    cos_lat = np.cos(center_lat * RAD_PER_DEG)
    lat_scale = (dist_per_cell / R) * DEG_PER_RAD
    lon_scale = (dist_per_cell / (R * cos_lat)) * DEG_PER_RAD

    # creating meshgrid
    y_indices = -(np.arange(rows) - center_row)
    x_indices = np.arange(cols) - center_col

    x_offsets, y_offsets = np.meshgrid(x_indices, y_indices)

    # calculating
    target_lats = center_lat + (y_offsets * lat_scale)
    target_lons = center_lon + (x_offsets * lon_scale)

    # combining
    coordinate_matrix = np.dstack((target_lats, target_lons))

    return coordinate_matrix

def init_matrix(spreadability, variance, wind_x, wind_y, magni_mult):

    status_placeholder = st.empty()
    ndvi = st.session_state["ndvi"]
    status_placeholder.info("Calculating Kernel")
    conv_matrix = generate_wind_kernel(
        radius=spreadability,
        sigma=variance,
        wind_x=wind_x * magni_mult,
        wind_y=-wind_y * magni_mult,
    )
    S, I, R = start_SIR_random_middle(ndvi)
    
    st.session_state["var"] = variance
    st.session_state["spread"] = spreadability
    st.session_state["S"] = S
    st.session_state["I"] = I
    st.session_state["R"] = R
    st.session_state["conv_matrix"] = conv_matrix
    st.session_state["reports_list"] = []
    st.session_state["timetotal"] = 0
    try:  # to differentiate between function being run with or without the code block being displayed
        st.session_state["beta"] = beta
        st.session_state["gamma"] = gamma
        st.session_state["wind_x"] = wind_x
        st.session_state["wind_y"] = wind_y
        st.session_state["magni_mult"] = magni_mult
    except Exception as e:
        print(f"An error occurred: {e}")
        status_placeholder.error(f"Failed to init Matrix. Error {e}")

    status_placeholder.success("Finished")



def simulate_steps(timestep):

    status_placeholder = st.empty()
    ndvi = st.session_state["ndvi"]
    status_placeholder.info("Starting")
    timetotal = st.session_state["timetotal"]
    conv_matrix = st.session_state["conv_matrix"]
    S = st.session_state["S"]
    I = st.session_state["I"]
    R = st.session_state["R"]
    for t in range(timestep):
        timetotal += 1
        S, I, R = sir_spatial_step(
            S, I, R, ndvi, conv_matrix,
            st.session_state["beta"], st.session_state["gamma"],
        )
        report = generate_report_grid(I, S, ndvi)
        st.session_state.setdefault("reports_list", []).append(report)
    status_placeholder.success("Finished")
    st.session_state["S"] = S
    st.session_state["I"] = I
    st.session_state["R"] = R

    rgb_grid = np.stack([I, R, S], axis=-1)

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

    # Apply tight layout formatting to the overall figure
    fig2.tight_layout()
    st.session_state["Simulation_graph"] = fig2
    st.session_state["Simulation_ax"] = ax2

    # st.pyplot(fig2)


# Hides the declaration block of the website
def toggle_layout():
    st.session_state.show_all = not st.session_state.show_all


st.title("NDVI based crop SIR Simulation")
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
            "maximum spread range", value=st.session_state["spread"]
        )
    with mat2:
        variance = st.number_input(
            "spread concentration", value=st.session_state["var"]
        )
    with mat3:
        beta = st.number_input(
            "Rate of transmission",
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
            "Wind Horizontal component",
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
    # with simcol3:
    #     if st.button(f"Simulate {timestep} timesteps"):
    #         simulate_steps(timestep)

tab1, tab2, tab3 = st.tabs(["Satellite NDVI View", "SIR Simulation View", "Temperature display"])
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
        threshhold = st.number_input("threshhold", value=.01)
    with sim_col:
        if st.button(f"Simulate {timestep} Timesteps"):
            if "conv_matrix" not in st.session_state:
                init_matrix(
                    st.session_state["spread"],
                    st.session_state["var"],
                    st.session_state["norm_x"],
                    st.session_state["norm_y"],
                    st.session_state["magni_mult"],
                )
            
            simulate_steps(timestep)
            locs = generate_latlon_matrix(input_matrix=st.session_state["I"],center_coords=(st.session_state["lati"],st.session_state["long"]))
            georef = GridGeoref.from_locs(locs, dist_per_cell=15)
            polygon = create_polygon(prob_grid= st.session_state["I"], georef = georef, threshold=threshhold, method="contour")
            st.session_state["Simulation_graph"] = plot_overlay(grid= st.session_state["I"], polygon = polygon, georef = georef)

    graph_col, padding_col = st.columns([0.5, 0.5])
    st.markdown("---")
    st.subheader("UNet forecast")

    c1, c2, c3 = st.columns([0.4, 0.3, 0.3])
    with c1:
        unet_threshold = st.slider(
            "prediction threshold", 0.1, 0.9, 0.5, 0.05
        )
    with c2:
        unet_horizon = st.selectbox(
            "model horizon (k)", sorted(MODELS.keys()), index=len(MODELS) - 1
        )
    with c3:
        st.write("")   # spacer
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

    
