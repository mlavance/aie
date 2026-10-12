"""
Spatial SIR-style plant disease simulation.

The model stores susceptible (S), infected (I), and recovered/removed (R)
proportions for each grid cell. Infection spreads between cells via a 2D
Gaussian kernel, and observations/reports can include false positives and
false negatives.
"""

import math
import random
import uuid
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import convolve2d


class simModel:
    """
    Spatial SIR-like simulator over an NDVI grid.

    Parameters
    ----------
    ndvi : array-like
        Per-cell NDVI values. Higher values generally increase infection pressure.
    center_coords : tuple
        (latitude, longitude) of the grid center.
    dist_per_grid : float
        Grid-cell size in meters.
    fp : float
        False-positive probability for reports.
    fn : float
        False-negative probability for reports.
    beta : float or None
        Infection/transmission rate. Must be supplied together with gamma.
    gamma : float or None
        Recovery/removal rate. Must be supplied together with beta.
    """

    def __init__(
        self,
        ndvi,
        center_coords,
        dist_per_grid=15,
        fp=0.01,
        fn=0.01,
        beta=None,
        gamma=None,
        total_steps=None,
        recovery=False,
    ):
        """
        ndvi : ndvi grid from landsat via planetary computer api
        center coords : cordinates of the center of the ndvi reading
        dist_per_grid : square meters each grid cell covers from the api, defaults to 15
        fp/fn : false rates for positive and negative detections from image recognition
        beta : spread rate
        gamma : removal rate #NOTE defaults to false because image detection model might 
                be able to detect dead plants and consider those as infected for spread calculations
        total_steps : how many steps to preallocate #NOTE can be used by end user, but primarily for generating training data
        recovery : toggle for above gamma to be used in the randomBG function
        """
        #checking inputs
        if fn > 1 or fn < 0: 
            raise ValueError(f"False Negative must be between 0 and 1, was passed {fn}")

        if fp > 1 or fp < 0:
            raise ValueError(f"False Positive must be between 0 and 1, was passed {fp}")

        if not ((beta is None) ^ (gamma is None)):
            self.beta = beta
            self.gamma = gamma
        else:
            raise ValueError(
                "Only one beta or gamma was provided, requires neither or both"
            )

        if beta is not None and (beta > 1 or beta < 0):
            raise ValueError(f"Beta must be between 0 and 1, was passed {beta}")

        if gamma is not None and (gamma > 1 or gamma < 0):
            raise ValueError(f"Gamma must be between 0 and 1, was passed {gamma}")

        self.ndvi = ndvi
        self.coords = center_coords
        self.dist = dist_per_grid
        self.fp = fp
        self.fn = fn
        self.scaleReports = None
        self.recovery = recovery
        # Storage mode. If total_steps is given we preallocate fixed-size
        # history buffers (fast, memory-efficient for training data generation).
        # Otherwise we fall back to the simple list+stack path for small sims.
        if total_steps is not None:
            self.total_steps = int(total_steps)
            self._prealloc = True
            self._S_buf = None  # allocated in randomMiddle once shape is known
            self._I_buf = None
            self._reports_buf = None  # array in prealloc mode, list in non-prealloc
            self._hist_count = 0  # number of frames written so far
        else:
            self.total_steps = None
            self._prealloc = False
            self._S_hist_arr = None  # plain stacked array, non-prealloc path
            self._I_hist_arr = None
            self._reports_buf = None

    def createMatrix(self, radius=2, sigma=1.0, wind_x=0.0, wind_y=0.0):
        """
        Create and store a normalized 2D Gaussian dispersal kernel.

        radius : int
            Kernel radius. The full kernel size is (2 * radius + 1) by
            (2 * radius + 1). For example, radius=2 gives a 5x5 kernel.
        sigma : float
            Standard deviation of the Gaussian; larger values spread infection
            farther in one step.
        wind_x, wind_y : float
            Offsets that shift the Gaussian peak away from the center. This can
            represent wind-biased spread.
        """
        # radius of 2 => 5x5 convolution matrix
        size = 2 * radius + 1

        # Create a coordinate grid centered at (0, 0).
        # For size=5, ax = [-2, -1, 0, 1, 2].
        ax = np.linspace(-(size // 2), size // 2, size)
        xx, yy = np.meshgrid(ax, ax)

        # Evaluate an unnormalized Gaussian. Subtracting wind_x/wind_y shifts
        # the peak of the kernel in the x/y directions.
        kernel = np.exp(-((xx - wind_x) ** 2 + (yy - wind_y) ** 2) / (2 * sigma**2))

        # Normalize so the kernel sums to 1. This keeps convolution output on
        # a comparable scale to the input infected density.
        self.kernel = kernel / np.sum(kernel)

    def randomBG(self, choice=None):
        """
        Randomly set beta and gamma according to a disease archetype.

        Archetypes:
          - "unviable": spreads quickly but dies out quickly.
          - "slow": low transmission and low recovery; persistent but slow.
          - "normal": moderate transmission and low recovery; sustained spread.
          - "fast": high transmission and moderate recovery; rapid outbreak.

        If `choice` is invalid or None, the current implementation randomly
        chooses an archetype and then immediately overwrites it with "fast".
        """
        if choice in ["unviable", "slow", "normal", "fast"]:
            disease_type = choice
        else:
            disease_type = random.choice(["slow", "normal", "fast"])

        while True:
            if disease_type == "unviable":
                # Disease spreads quickly but dies out too fast to cause outbreak.
                # High beta for fast spread, but very high gamma for quick recovery.
                beta = random.uniform(0.30, 0.60)
                gamma = random.uniform(0.30, 0.60)
                # Beta/gamma ratio ~ 0.5-2.0 = disease can't sustain itself.

            elif disease_type == "slow":
                # Slow spread with slow death/recovery.
                # Low beta and low gamma = persistent but slow-moving disease.
                beta = random.uniform(0.05, 0.15)
                gamma = random.uniform(0.01, 0.04)
                # Beta/gamma ratio ~ 2-10 = slow but sustainable spread.

            elif disease_type == "normal":
                # Normal spread with slow death/recovery.
                # Moderate beta with low gamma = steady, sustained outbreak.
                beta = random.uniform(0.15, 0.35)
                gamma = random.uniform(0.02, 0.08)
                # Beta/gamma ratio ~ 3-15 = good sustained spread.

            else:  # fast
                # Fast spread with normal death/recovery.
                # High beta with moderate gamma = rapid, visible outbreak.
                beta = random.uniform(0.35, 0.70)
                gamma = random.uniform(0.08, 0.20)
                # Beta/gamma ratio ~ 2-8 = fast but not instantly saturating.

            # Ensure the sampled parameters match the intended disease type.
            ratio = beta / gamma

            if disease_type == "unviable":
                # Must be unviable (ratio < 2.0).
                if ratio < 2.0:
                    break
            elif disease_type == "slow":
                # Slow spread with slow death (ratio 2-10).
                if 2.0 < ratio < 10.0:
                    break
            elif disease_type == "normal":
                # Normal spread with slow death (ratio 3-15).
                if 3.0 < ratio < 15.0:
                    break
            else:  # fast
                # Fast spread with normal death (ratio 2-8).
                if 2.0 < ratio < 8.0:
                    break

        self.beta = beta
        self.gamma = gamma
        if not self.recovery:
            self.gamma = 0

    def setBG(self, beta, gamma):
        """Manually set the transmission and recovery rates."""
        self.beta = beta
        self.gamma = gamma

    def randomMatrix(self):
        """Create a randomized Gaussian dispersal kernel and store it."""
        radius = random.randint(3, 7)
        sigma = random.uniform(1, 4)
        wind_x = random.uniform(-4, 4)
        wind_y = random.uniform(-4, 4)

        self.createMatrix(radius=radius, sigma=sigma, wind_x=wind_x, wind_y=wind_y)

    def initSim(self, mode="middle"):
        """
        Initialize the SIR arrays and set the simulation day to 0.

        mode="middle" places the initial infection near the middle of the grid.
        mode="random" is intended to place it randomly, but the `randomStart`
        method is not defined in this file.
        """
        if mode not in ("middle", "random"):
            raise ValueError("Mode not recognized, must be either middle or random")

        if mode == "middle":
            self.randomMiddle()
        else:
            # NOTE: randomStart() is referenced but not defined in this file.
            self.randomStart()

        self.day = 0

    def randomMiddle(self):
        """
        Initialize one infected cell in the middle third of the grid.

        Prefers cells with NDVI > 0.5. Tries up to 10 random draws within the
        middle third; if none meet the threshold, uses the last draw anyway
        (with a warning) so the simulation can still proceed.

        Sets:
        self.S, self.I, self.R : current SIR arrays
        self.S_hist, self.I_hist : history arrays for S and I
        self.coordinate_matrix : per-cell latitude/longitude lookup
        """
        input_np = np.asarray(self.ndvi)
        I = np.zeros_like(input_np)
        rando = I.shape

        row_lo = math.floor(rando[0] / 3)
        row_hi = 2 * row_lo - 1
        col_lo = math.floor(rando[1] / 3)
        col_hi = 2 * col_lo - 1

        # Guard against degenerate small grids
        if row_hi <= row_lo:
            row_hi = row_lo + 1
        if col_hi <= col_lo:
            col_hi = col_lo + 1

        row = col = 0
        found = False
        for _ in range(30):
            row = np.random.randint(row_lo, row_hi)
            col = np.random.randint(col_lo, col_hi)
            if self.ndvi[row, col] > 0.3:
                found = True
                break

        if not found:
            print(
                f"Warning: no cell with NDVI > 0.3 found in 30 tries; "
                f"using ({row}, {col}) with NDVI={self.ndvi[row, col]:.3f}"
            )

        I[row, col] = .2

        # 2. Create susceptible and recovered arrays.
        S = np.zeros_like(input_np) + 1 - I
        R = np.zeros_like(input_np)

        # Save the initial timestep in the object.
        self.S = S
        self.I = I
        self.R = R

        # Build the latitude/longitude lookup for every grid cell.
        self.generate_latlon_matrix()

        # Save initial history.
        n_rows, n_cols = I.shape
        if self._prealloc:
            self._S_buf = np.empty(
                (self.total_steps + 1, n_rows, n_cols), dtype=np.float32
            )
            self._I_buf = np.empty(
                (self.total_steps + 1, n_rows, n_cols), dtype=np.float32
            )
            self._reports_buf = np.empty(
                (self.total_steps, n_rows, n_cols), dtype=np.uint16
            )
            self._S_buf[0] = S
            self._I_buf[0] = I
            self._hist_count = 1
        else:
            self._reports_buf = []
            self.S_hist = S
            self.I_hist = I

    @property
    def S_hist(self):
        """
        Return S history as (T, n_rows, n_cols).

        Preallocated mode: zero-copy slice of the internal buffer covering only
        the frames written so far.
        Non-preallocated mode: the stacked array held in self._S_hist_arr,
        updated by np.vstack in randomMiddle/step.
        """
        if self._prealloc:
            if self._S_buf is None:
                return None
            return self._S_buf[: self._hist_count]
        return self._S_hist_arr

    @S_hist.setter
    def S_hist(self, value):
        if self._prealloc:
            raise RuntimeError(
                "Cannot assign to S_hist in preallocated mode; "
                "write to self._S_buf instead"
            )
        self._S_hist_arr = value

    @property
    def I_hist(self):
        """Same contract as S_hist, for the infected compartment."""
        if self._prealloc:
            if self._I_buf is None:
                return None
            return self._I_buf[: self._hist_count]
        return self._I_hist_arr

    @I_hist.setter
    def I_hist(self, value):
        if self._prealloc:
            raise RuntimeError(
                "Cannot assign to I_hist in preallocated mode; "
                "write to self._I_buf instead"
            )
        self._I_hist_arr = value

    @property
    def report_grids(self):
        """
        Per-day report grids accumulated so far, shape (D, n_rows, n_cols).

        Preallocated mode: zero-copy slice of the internal uint16 buffer,
        covering days 1..self.day.
        Non-preallocated mode: the list is stacked on demand.
        Returns an empty (0, H, W) array if the sim has not stepped yet.
        """
        shape = (0,) + self.I.shape
        if self._reports_buf is None:
            return np.empty(shape, dtype=np.uint16)
        if self._prealloc:
            return self._reports_buf[: self.day]
        if len(self._reports_buf) == 0:
            return np.empty(shape, dtype=np.uint16)
        return np.stack(self._reports_buf, axis=0)

    def _single_step(self):
        """
        Advance the simulation by exactly one day.

        Internal helper used by step() and complete_sim(). Does the SIR update,
        clips to [0, 1], appends to history (preallocated buffer or vstack),
        and generates reports for the new day.
        """
        # Infection pressure at each cell: weighted sum of nearby infected cells.
        infected_pressure = convolve2d(
            self.I, self.kernel, mode="same", boundary="fill"
        )

        # New infections depend on transmission rate, susceptible density,
        # NDVI (plant density/vulnerability proxy), and infection pressure.
        new_infections = self.beta * self.S * self.ndvi * infected_pressure

        # Recovery/removal is proportional to the infected density.
        new_recoveries = self.gamma * self.I

        # Do not infect or recover more than the available amount in a cell.
        new_infections = np.minimum(new_infections, self.S)
        new_recoveries = np.minimum(new_recoveries, self.I)

        # Compute the next SIR state.
        S_next = self.S - new_infections
        I_next = self.I + new_infections - new_recoveries
        R_next = self.R + new_recoveries

        # Keep each compartment within [0, 1].
        self.S = np.clip(S_next, 0.0, 1.0)
        self.I = np.clip(I_next, 0.0, 1.0)
        self.R = np.clip(R_next, 0.0, 1.0)

        # Append the new state to history.
        if self._prealloc:
            if self._hist_count > self.total_steps:
                raise IndexError(
                    f"step would exceed preallocated total_steps={self.total_steps}"
                )
            self._S_buf[self._hist_count] = self.S
            self._I_buf[self._hist_count] = self.I
            self._hist_count += 1
        else:
            self.S_hist = np.vstack((self.S_hist, self.S.copy()))
            self.I_hist = np.vstack((self.I_hist, self.I.copy()))

        # Update reports. On the first call, self.reports does not exist yet.
        self.day = getattr(self, "day", 0) + 1
        day_reports = self.createReports()
        prev = getattr(self, "reports", None)
        if prev is None:
            self.reports = day_reports
        else:
            self.reports = np.vstack((prev, day_reports))

        if self._prealloc:
            self._reports_buf[self.day - 1] = self.createReportGrid()
        else:
            self._reports_buf.append(self.createReportGrid())

    def step(self, steps=1):
        """
        Advance the simulation by `steps` days.

        Works in both storage modes. In preallocated mode, the cumulative number
        of steps across all calls must not exceed self.total_steps.

        inputs
        --------
        steps = number of days to advance (must be >= 1)

        returns
        -------
        none, mutates self.S / self.I / self.R / history / reports
        """
        if steps < 1:
            raise AttributeError(
                f"Step was called with {steps} steps, must call with 1 or more"
            )
        for _ in range(steps):
            self._single_step()

    def complete_sim(self):
        """
        Fill the preallocated history buffer by running all remaining steps.

        Convenience wrapper for the training-data workflow: call initSim(), then
        complete_sim() once, and the sim is fully run with no further loop
        management needed.

        Only valid in preallocated mode (total_steps passed to __init__). If the
        sim has already been advanced via step(), only the remaining days are run.

        returns
        -------
        none, mutates the sim to its final state
        """
        if not self._prealloc:
            raise RuntimeError(
                "complete_sim requires total_steps to be set at init. "
                "Use step(n) for non-preallocated sims."
            )

        remaining = self.total_steps + 1 - self._hist_count
        if remaining <= 0:
            return  # already complete, no-op
        self.step(steps=remaining)

    def getBG(self):
        """Print and return the current beta and gamma values."""
        print(f"Beta is: {self.beta}")
        print(f"Gamma is: {self.gamma}")
        return self.beta, self.gamma

    def createReports(self, seed=None):
        """
        Generate observation reports for the current day.

        Returns
        -------
        reports : ndarray, shape (M, 3)
            Each row is [latitude, longitude, day]. Returns an empty (0, 3)
            array if no reports are generated.

        Current assumptions:
          - The entire area is scanned by the disease-detection model.
          - True positives are drawn as Poisson counts from I * NDVI.
          - False negatives are applied with probability self.fn.
          - False positives are drawn as Poisson counts from S * NDVI * self.fp.
        """
        rng = np.random.default_rng(seed)

        # NDVI can be negative (water, cloud, shadow, bare soil). Poisson
        # requires lam >= 0, so clip before sampling.
        plantidx = np.clip(np.asarray(self.ndvi, dtype=float), 0.0, None)

        # True positives: Poisson from infected * NDVI.
        tp_lam = np.clip(self.I * plantidx, 0.0, None)
        tp_counts = rng.poisson(tp_lam)

        # False positives: Poisson from susceptible * NDVI * fp.
        fp_lam = np.clip(self.S * plantidx * self.fp, 0.0, None)
        fp_counts = rng.poisson(fp_lam)

        # False negatives: each true detection is missed with prob self.fn.
        # Binomial thinning preserves the Poisson process exactly.
        if self.fn > 0:
            tp_counts = rng.binomial(tp_counts, 1.0 - self.fn)

        total = tp_counts + fp_counts
        if not np.any(total):
            return np.empty((0, 3), dtype=float)

        # Expand each cell's count into repeated row/col indices.
        rows, cols = np.nonzero(total)
        n = total[rows, cols]
        rows_rep = np.repeat(rows, n)
        cols_rep = np.repeat(cols, n)

        # Shuffle so TP and FP reports are interleaved rather than
        # false positives always trailing at the end.
        perm = rng.permutation(len(rows_rep))
        rows_rep = rows_rep[perm]
        cols_rep = cols_rep[perm]

        # self.coordinate_matrix is (rows, cols, 2), last axis [lat, lon].
        lat = self.coordinate_matrix[rows_rep, cols_rep, 0].astype(float)
        lon = self.coordinate_matrix[rows_rep, cols_rep, 1].astype(float)

        # Jitter each report to a random point inside its grid cell rather
        # than always reporting the exact cell-center coordinate.
        lat = lat + rng.uniform(-self.lat_scale / 2, self.lat_scale / 2, size=lat.shape)
        lon = lon + rng.uniform(-self.lon_scale / 2, self.lon_scale / 2, size=lon.shape)

        day_col = np.full(lat.shape, self.day, dtype=float)
        return np.column_stack([lat, lon, day_col])

    def outputReports(self):
        # small function to convert the output from genReports() to a format readable by the st-gnn
        # can be implemented into genReports later
        return [{"lat": r[0], "lon": r[1], "t": r[2]} for r in self.reports]

    def rasterize_reports(self, reports):
        """
        Invert generate_latlon_matrix to bin (lat, lon) reports back onto
        the same grid, accumulating counts per cell.
        """
        rows, cols = self.I.shape
        grid = np.zeros((rows, cols), dtype=np.float32)
        if reports.shape[0] == 0:
            return grid

        lat, lon = reports[:, 0], reports[:, 1]
        row = np.round(
            self.center_row - (lat - self.center_lat) / self.lat_scale
        ).astype(int)
        col = np.round(
            self.center_col + (lon - self.center_lon) / self.lon_scale
        ).astype(int)

        valid = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
        row, col = row[valid], col[valid]

        np.add.at(grid, (row, col), 1)
        return grid

    def generate_latlon_matrix(self):
        """
        Build a per-cell latitude/longitude lookup matrix.

        The grid is centered on self.coords. Rows move north/south and columns
        move east/west. Distances are converted from meters to degrees using
        an equirectangular approximation around the center latitude.
        """
        rows, cols = self.I.shape
        center_row, center_col = rows // 2, cols // 2
        center_lat, center_lon = self.coords

        # Constants.
        R = 6378137.0  # Earth radius in meters.
        RAD_PER_DEG = np.pi / 180.0
        DEG_PER_RAD = 180.0 / np.pi

        # Longitude degrees are smaller near the poles, so scale by cos(latitude).
        cos_lat = np.cos(center_lat * RAD_PER_DEG)
        lat_scale = (self.dist / R) * DEG_PER_RAD
        lon_scale = (self.dist / (R * cos_lat)) * DEG_PER_RAD

        # Create index offsets relative to the grid center.
        # y_indices increases upward (north), so row 0 is north of the center.
        y_indices = -(np.arange(rows) - center_row)
        x_indices = np.arange(cols) - center_col

        x_offsets, y_offsets = np.meshgrid(x_indices, y_indices)

        # Convert offsets to latitude/longitude offsets.
        target_lats = center_lat + (y_offsets * lat_scale)
        target_lons = center_lon + (x_offsets * lon_scale)

        # Stack into an (rows, cols, 2) array: last axis is [lat, lon].
        coordinate_matrix = np.dstack((target_lats, target_lons))

        self.coordinate_matrix = coordinate_matrix
        self.lat_scale = lat_scale
        self.lon_scale = lon_scale
        self.center_lat, self.center_lon = center_lat, center_lon
        self.center_row, self.center_col = center_row, center_col

    def save(
        self,
        output_dir="simdata",
        ext=".npz",
        infected_threshold=0.1,
        include_S_hist=False,
        include_I_hist=False,
        include_coordinate_matrix=False,
        include_kernel=False,
        ndvi_as_uint16=False,
    ):
        """
        Save a training episode. By default only fields the model actually
        consumes are written; everything else is optional metadata.

        Default contents (essentials only)
        ----------------------------------
        ndvi              : (n_rows, n_cols) float32 (or uint16 if requested)
        reports           : (N, 3) float16 [lat, lon, day]
        ground_truth      : (T, n_rows, n_cols) uint8  (packed? no — see below)
        center_lat/lon    : float64
        dist_per_grid     : float32
        lat_scale/lon_scale : float32 (needed for jitter / georef reconstruction)

        Optional (off by default, for reproducibility / debugging)
        ----------------------------------------------------------
        include_S_hist               : (T, n_rows, n_cols) float32 ~8 MB
        include_I_hist               : (T, n_rows, n_cols) float32 ~8 MB
        include_coordinate_matrix    : (n_rows, n_cols, 2) float32 ~320 KB
        include_kernel               : small, but only useful for re-running sim
        ndvi_as_uint16               : store NDVI as uint16 (lossless if source
                                    was 16-bit PNG; use load_sim(scale=True) to
                                    recover)

        inputs
        --------
        output_dir = folder to write into
        ext = file extension (default ".npz")
        infected_threshold = threshold used to derive ground_truth from I_hist
        include_* = optional fields to add (see above)
        ndvi_as_uint16 = compress NDVI by storing the raw 16-bit integer range

        returns
        -------
        filepath = Path to the written file
        """
        filepath = Path(output_dir) / f"{self.coords[0]}_{self.coords[1]}{ext}"
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # --- reports: float32 by default (precision-safe at cell scale) ---
        if getattr(self, "reports", None) is None or len(self.reports) == 0:
            reports_arr = np.empty((0, 3), dtype=np.float32)
        else:
            reports_arr = np.asarray(self.reports, dtype=np.float32)

        # --- labels: uint8, derived from I_hist at the fixed threshold ---
        i_hist = np.asarray(self.I_hist, dtype=np.float32)
        ground_truth = (i_hist > infected_threshold).astype(np.uint8)

        # --- ndvi: optionally store as uint16 ---
        ndvi_arr = np.asarray(self.ndvi)
        if ndvi_as_uint16:
            # round-trip via the raw 16-bit integer range
            ndvi_arr = np.clip(np.round(ndvi_arr * 65535.0), 0, 65535).astype(np.uint16)
        else:
            ndvi_arr = ndvi_arr.astype(np.float32)

        payload = {
            # --- essentials ---
            "ndvi": ndvi_arr,
            "reports": reports_arr,
            "ground_truth": ground_truth,
            "center_lat": np.float64(self.coords[0]),
            "center_lon": np.float64(self.coords[1]),
            "dist_per_grid": np.float32(self.dist),
            "lat_scale": np.float32(getattr(self, "lat_scale", 0.0)),
            "lon_scale": np.float32(getattr(self, "lon_scale", 0.0)),
            "infected_threshold": np.float32(infected_threshold),
            "ndvi_uint16_encoded": np.uint8(ndvi_as_uint16),
        }

        # --- optional fields ---
        if include_S_hist:
            payload["S_hist"] = np.asarray(self.S_hist, dtype=np.float32)
        if include_I_hist:
            payload["I_hist"] = i_hist
        if include_coordinate_matrix:
            payload["coordinate_matrix"] = np.asarray(
                self.coordinate_matrix, dtype=np.float32
            )
        if include_kernel:
            if hasattr(self, "kernel"):
                payload["kernel"] = np.asarray(self.kernel, dtype=np.float32)
            else:
                payload["kernel"] = np.zeros((1, 1), dtype=np.float32)

        # --- provenance: always cheap, always useful for filtering ---
        payload["beta"] = np.float32(self.beta if self.beta is not None else np.nan)
        payload["gamma"] = np.float32(self.gamma if self.gamma is not None else np.nan)
        payload["fp"] = np.float32(self.fp)
        payload["fn"] = np.float32(self.fn)

        np.savez_compressed(filepath, **payload)
        return filepath

    @classmethod
    def load(cls, path, scale_ndvi=True):
        """
        Reconstruct a simModel from a saved .npz file.

        inputs
        --------
        path = path to the .npz file
        scale_ndvi = if the file stored NDVI as uint16, divide by 65535 to
                     recover the [0, 1] float values

        returns
        -------
        sim = simModel in non-preallocated mode with history and reports loaded
        """
        data = np.load(path, allow_pickle=False)

        ndvi = data["ndvi"]
        if bool(data["ndvi_uint16_encoded"]) and scale_ndvi:
            ndvi = ndvi.astype(np.float32) / 65535.0

        sim = cls(
            ndvi=ndvi,
            center_coords=(float(data["center_lat"]), float(data["center_lon"])),
            dist_per_grid=float(data["dist_per_grid"]),
            fp=float(data["fp"]),
            fn=float(data["fn"]),
            beta=None if np.isnan(data["beta"]) else float(data["beta"]),
            gamma=None if np.isnan(data["gamma"]) else float(data["gamma"]),
        )
        sim.lat_scale = float(data["lat_scale"])
        sim.lon_scale = float(data["lon_scale"])
        sim.reports = data["reports"]

        if "I_hist" in data:
            sim.I_hist = data["I_hist"]
        else:
            sim.I_hist = data["ground_truth"].astype(np.float32)

        if "S_hist" in data:
            sim.S_hist = data["S_hist"]
        if "kernel" in data:
            sim.kernel = data["kernel"]
        if "coordinate_matrix" in data:
            sim.coordinate_matrix = data["coordinate_matrix"]

        return sim

    def save_h5(self, output_dir="simdata", ext=".h5"):
        """
        Save a completed simulation run to HDF5 for training.

        Stores continuous S/I history (the regression targets), the full raw
        report stream, NDVI, and all grid-georeferencing metadata needed to
        re-rasterize reports later — either during training-data assembly or
        against real drone reports in production.

        Requires the sim to have been stepped at least once (self.reports is
        only created inside _single_step, on the first call to step()).

        Returns
        -------
        filepath : Path
            Path to the written .h5 file.
        """
        if getattr(self, "reports", None) is None:
            raise RuntimeError(
                "No reports found — call step()/complete_sim() before saving."
            )

        run_id = uuid.uuid4()
        filepath = Path(output_dir) / f"{run_id}{ext}"
        filepath.parent.mkdir(parents=True, exist_ok=True)

        i_hist = np.asarray(self.I_hist, dtype=np.float32)
        s_hist = np.asarray(self.S_hist, dtype=np.float32)
        ndvi_arr = np.asarray(self.ndvi, dtype=np.float32)
        reports_arr = (
            np.asarray(self.reports, dtype=np.float64)
            if len(self.reports)
            else np.empty((0, 3), dtype=np.float64)
        )

        with h5py.File(filepath, "w") as f:
            # Time-series grids
            f.create_dataset("i_hist", data=i_hist, compression="gzip", chunks=True)
            f.create_dataset("s_hist", data=s_hist, compression="gzip", chunks=True)

            # Raw report stream: (N, 3) = [lat, lon, day]
            f.create_dataset("reports", data=reports_arr, compression="gzip")

            # Static covariate
            f.create_dataset("ndvi", data=ndvi_arr, compression="gzip", chunks=True)

            # Grid geometry, needed to re-rasterize reports identically later
            f.attrs["center_lat"] = float(self.center_lat)
            f.attrs["center_lon"] = float(self.center_lon)
            f.attrs["center_row"] = int(self.center_row)
            f.attrs["center_col"] = int(self.center_col)
            f.attrs["lat_scale"] = float(self.lat_scale)
            f.attrs["lon_scale"] = float(self.lon_scale)
            f.attrs["dist_per_grid"] = float(self.dist)
            f.attrs["n_days"] = int(i_hist.shape[0])

            # Provenance / disease parameters, for filtering runs by archetype later
            f.attrs["run_id"] = str(run_id)
            f.attrs["beta"] = float(self.beta) if self.beta is not None else np.nan
            f.attrs["gamma"] = float(self.gamma) if self.gamma is not None else np.nan
            f.attrs["fp"] = float(self.fp)
            f.attrs["fn"] = float(self.fn)

        return filepath

    @classmethod
    def load_h5(cls, path):
        """
        Reconstruct a simModel from an HDF5 file written by save_h5.

        Returns an instance in non-preallocated mode with S_hist/I_hist and
        reports restored directly (no re-simulation). Ready to feed straight
        into build_input_sequence-style assembly, or to resume stepping.

        Caveats
        -------
        - R history isn't saved (save_h5 only stores S_hist/I_hist), so R is
        reconstructed as zeros. That's correct for the current SI-only setup
        (gamma forced to 0 in randomBG), but would be wrong if you ever run
        a true SIR simulation with recovery — you'd need to add R_hist to
        save_h5/load_h5 first.
        - The dispersal kernel isn't saved either. If you want to resume
        stepping this sim (not just read its history), call createMatrix()
        or randomMatrix() again before calling step() — otherwise self.kernel
        won't exist and _single_step will fail.

        Parameters
        ----------
        path : str or Path
            Path to the .h5 file.

        Returns
        -------
        sim : simModel
        """
        path = Path(path)
        with h5py.File(path, "r") as f:
            i_hist = f["i_hist"][:]
            s_hist = f["s_hist"][:]
            reports_arr = f["reports"][:]
            ndvi_arr = f["ndvi"][:]

            center_lat = float(f.attrs["center_lat"])
            center_lon = float(f.attrs["center_lon"])
            center_row = int(f.attrs["center_row"])
            center_col = int(f.attrs["center_col"])
            lat_scale = float(f.attrs["lat_scale"])
            lon_scale = float(f.attrs["lon_scale"])
            dist_per_grid = float(f.attrs["dist_per_grid"])
            n_days = int(f.attrs["n_days"])

            beta_raw = f.attrs["beta"]
            gamma_raw = f.attrs["gamma"]
            fp = float(f.attrs["fp"])
            fn = float(f.attrs["fn"])
            run_id = f.attrs.get("run_id", None)

        beta = None if np.isnan(beta_raw) else float(beta_raw)
        gamma = None if np.isnan(gamma_raw) else float(gamma_raw)

        sim = cls(
            ndvi=ndvi_arr,
            center_coords=(center_lat, center_lon),
            dist_per_grid=dist_per_grid,
            fp=fp,
            fn=fn,
            beta=beta,
            gamma=gamma,
            total_steps=None,  # non-prealloc mode, so S_hist/I_hist setters work
        )

        # Restore history + current state directly, bypassing randomMiddle.
        sim.S_hist = s_hist
        sim.I_hist = i_hist
        sim.S = s_hist[-1]
        sim.I = i_hist[-1]
        sim.R = np.zeros_like(ndvi_arr, dtype=np.float32)  # see docstring caveat

        sim.reports = reports_arr
        sim.day = n_days - 1  # matches how day increments in _single_step

        # Restore grid geometry from saved attrs rather than recomputing, to
        # avoid any float drift across a save/load round trip.
        sim.center_lat, sim.center_lon = center_lat, center_lon
        sim.center_row, sim.center_col = center_row, center_col
        sim.lat_scale, sim.lon_scale = lat_scale, lon_scale

        rows, cols = ndvi_arr.shape
        y_indices = -(np.arange(rows) - center_row)
        x_indices = np.arange(cols) - center_col
        x_offsets, y_offsets = np.meshgrid(x_indices, y_indices)
        sim.coordinate_matrix = np.dstack(
            (
                center_lat + y_offsets * lat_scale,
                center_lon + x_offsets * lon_scale,
            )
        )

        if run_id is not None:
            sim.run_id = str(run_id)

        return sim

    def crop_ndvi(self, size=100):
        M, N = self.ndvi.shape

        # Compute starting indices for the crop
        start_row = (M - size) // 2
        start_col = (N - size) // 2

        self.ndvi = self.ndvi[
            start_row : start_row + size, start_col : start_col + size
        ]

        return 1

    def createReportGrid(self, seed=None):
        """
        Generate a per-cell count of observation reports for the current day.

        Returns
        -------
        report_grid : ndarray, shape (n_rows, n_cols), dtype uint16
            Number of reports originating from each cell.
        """
        rng = np.random.default_rng(seed)

        # Clip NDVI to non-negative: Poisson requires lam >= 0.
        plantidx = np.clip(np.asarray(self.ndvi, dtype=float), 0.0, None)
        if self.scaleReports is not None:
            plantidx = plantidx * self.scaleReports

        tp_lam = np.clip(self.I * plantidx, 0.0, None)
        tp_counts = rng.poisson(tp_lam)

        if self.fn > 0:
            tp_counts = rng.binomial(tp_counts, 1.0 - self.fn)

        fp_lam = np.clip(self.S * plantidx * self.fp, 0.0, None)
        fp_counts = rng.poisson(fp_lam)

        return (tp_counts + fp_counts).astype(np.uint16)

    def savewindow(self, size):
        """
        Sliding window sum of the accumulated report grids.

        For each day t, result[t] is the sum of report grids from day
        max(0, t - size + 1) through day t. The window always ends at the
        current day and looks back at most `size` days.

        inputs
        --------
        size : int, number of days in the sliding window (>= 1)

        returns
        -------
        windowed : (day, n_rows, n_cols) uint16 array
            windowed[t] is the count-per-cell sum over the last `size` days
            ending at day t+1.
        """
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")

        reports = self.report_grids
        if reports.size == 0:
            return np.empty((0,) + reports.shape[1:], dtype=np.uint16)

        # cast to int32 so cumsum cannot overflow uint16
        cumsum = np.cumsum(reports.astype(np.int32), axis=0)
        D = cumsum.shape[0]

        if size >= D:
            return cumsum.astype(np.uint16)

        result = np.empty_like(cumsum)
        result[:size] = cumsum[:size]
        result[size:] = cumsum[size:] - cumsum[:-size]
        return result.astype(np.uint16)

    def save_window(
        self, output_dir="simdata_windows", ext=".npz", infected_threshold=0.1
    ):
        """
        Save a training file for unet_report.py containing only what the model
        and loss need:

        ndvi         : (H, W) float32
        reports      : (D, H, W) uint8 raw per-day report grids
        ground_truth : (D, H, W) uint8 binary infection mask
        center_lat   : scalar float64
        center_lon   : scalar float64

        Window size is a training-time choice; the loader computes the sliding
        window from `reports` on the fly. Files are named by UUID so two sims
        with the same grid coordinates cannot collide.

        inputs
        --------
        output_dir : folder to write into
        ext : file extension
        infected_threshold : threshold to derive ground_truth from I_hist

        returns
        -------
        filepath : Path to the written file
        """
        if getattr(self, "day", 0) == 0:
            raise RuntimeError(
                "no reports accumulated yet; call step() or complete_sim() first"
            )

        reports = self.report_grids

        i_hist = self.I_hist
        if i_hist.shape[0] == self.day + 1:
            i_hist = i_hist[1:]
        elif i_hist.shape[0] != self.day:
            raise RuntimeError(
                f"I_hist has {i_hist.shape[0]} frames, expected {self.day} or {self.day + 1}"
            )

        if reports.shape != i_hist.shape:
            raise RuntimeError(
                f"reports {reports.shape} and ground truth {i_hist.shape} mismatch"
            )

        ground_truth = (i_hist > infected_threshold).astype(np.uint8)

        filepath = Path(output_dir) / f"{uuid.uuid4().hex}{ext}"
        filepath.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            filepath,
            ndvi=np.asarray(self.ndvi, dtype=np.float32),
            reports=np.asarray(reports, dtype=np.uint8),
            ground_truth=ground_truth,
            center_lat=np.float64(self.coords[0]),
            center_lon=np.float64(self.coords[1]),
        )
        return filepath
