# %%
# IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
import rioxarray

# %% [markdown]
# Function to get data from LANDSAT.<br>
# Uses Band 4 (red light) and Band 5 (NIR) to calculate NDVI values over a grid

# %%
def get_landsat_ndvi_matrix(lat, lon, half_side_meters=500):
  """
  connects to microsoft planetary computer stac api to acquire landsat band 4 and 5
  for ndvi calculation

  inputs
  lat - latitude of the center point
  lon - longitude of the center point
  half_side_meters - distance from the center to the edge of the square
      NOTE final result is not always a square, but usually within 1 (NxN+-1)
      to approximately convert from array shape NxN to half_side_meters multiply N by 15

  returns
  numpy array of ndvi values based on inputs
  """
  # initialize the stac client and authenticate
  print("connecting to API")
  catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
  )

  # creating bounding box in degrees from the meter inputs
  lat_buffer = half_side_meters / 111000.0
  lon_buffer = half_side_meters / (111000.0 * np.cos(np.radians(lat)))

  bbox = [
    lon - lon_buffer,  # min longitude
    lat - lat_buffer,  # min latitude
    lon + lon_buffer,  # max longitude
    lat + lat_buffer   # max latitude
  ]

  # searching for most recent data
  print("searching for recent landsat items")
  search = catalog.search(
    collections=["landsat-c2-l2"],
    bbox=bbox,
    max_items=5,
    query={"eo:cloud_cover": {"lt": 10}}
  )

  items = list(search.item_collection())
  if not items:
    raise ValueError("No matching Landsat images found for the given criteria.")

  latest_item = sorted(items, key=lambda x: x.properties['datetime'], reverse=True)[0]
  print(f"Found best match: {latest_item.id} (Date: {latest_item.properties['datetime']})")
  # can be commented out for bulk testing

  # checking metadata
  red_key = "red" if "red" in latest_item.assets else "b4"
  nir_key = "nir08" if "nir08" in latest_item.assets else "b5"

  # cropping result to only the specified area
  print("streaming and cropping Band 4 (Red)") # can be commented out
  with rioxarray.open_rasterio(latest_item.assets[red_key].href) as red_src:
    red_cropped = red_src.rio.clip_box(
    minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], crs="EPSG:4326"
    ).squeeze()

  # cropping result to only the specified area
  print("streaming and cropping Band 5 (NIR)") # can be commented out
  with rioxarray.open_rasterio(latest_item.assets[nir_key].href) as nir_src:
    nir_cropped = nir_src.rio.clip_box(
    minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], crs="EPSG:4326"
    ).squeeze()

  # ensure both arrays have the same shape
  if red_cropped.shape != nir_cropped.shape:
    print("Slight array shape mismatch detected. Re-aligning pixels...")
    nir_cropped = nir_cropped.rio.reproject_match(red_cropped)

  # sanity check, ensures division later is properly executed and stored
  red = red_cropped.values.astype(np.float64)
  nir = nir_cropped.values.astype(np.float64)

  # ndvi calculation
  print("calculating ndvi array")
  # Using np.errstate prevents warnings if both NIR and Red are 0 in a pixel (e.g. data gaps)
  with np.errstate(divide='ignore', invalid='ignore'):
    ndvi = (nir - red) / (nir + red)
    # replace NaN and infinities with 0
    ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
    # NDVI is signed (-1..1); negative pixels are non-vegetated.
    # Clip to [0, 1] so downstream Poisson draws are valid.
    ndvi = np.clip(ndvi, 0.0, 1.0)

  print(f"NDVI processing complete. Result shape: {ndvi.shape}")
  return ndvi

def get_landsat_ndvi_and_temp_matrix(lat, lon, half_side_meters=500):
  """
  Connects to Microsoft Planetary Computer STAC API to acquire Landsat bands
  needed for NDVI (red, NIR) and surface temperature (thermal IR).

  Inputs
  ------
  lat, lon : float  center point
  half_side_meters : float  half-width of requested square in meters

  Returns
  -------
  (ndvi, temperature_celsius) : tuple of 2D numpy arrays
      ndvi                - NDVI (unitless, -1..1)
      temperature_celsius - Landsat C2 L2 Surface Temperature in °C
  """
  # ---- connect to STAC ----
  print("connecting to API")
  catalog = pystac_client.Client.open(
      "https://planetarycomputer.microsoft.com/api/stac/v1",
      modifier=planetary_computer.sign_inplace,
  )

  # ---- bounding box ----
  lat_buffer = half_side_meters / 111000.0
  lon_buffer = half_side_meters / (111000.0 * np.cos(np.radians(lat)))
  bbox = [
      lon - lon_buffer,
      lat - lat_buffer,
      lon + lon_buffer,
      lat + lat_buffer,
  ]

  # ---- search ----
  print("searching for recent landsat items")
  search = catalog.search(
      collections=["landsat-c2-l2"],
      bbox=bbox,
      max_items=5,
      query={"eo:cloud_cover": {"lt": 10}},
  )
  items = list(search.item_collection())
  if not items:
      raise ValueError("No matching Landsat images found for the given criteria.")

  latest_item = sorted(
      items, key=lambda x: x.properties["datetime"], reverse=True
  )[0]
  print(f"Found best match: {latest_item.id} "
        f"(Date: {latest_item.properties['datetime']})")

  # ---- asset keys ----
  red_key     = "red"   if "red"   in latest_item.assets else "b4"
  nir_key     = "nir08" if "nir08" in latest_item.assets else "b5"
  thermal_key = "lwir11" if "lwir11" in latest_item.assets else "b10"

  # ---- Red band ----
  print("streaming and cropping Band 4 (Red)")
  with rioxarray.open_rasterio(latest_item.assets[red_key].href) as red_src:
      red_cropped = red_src.rio.clip_box(
          minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3],
          crs="EPSG:4326",
      ).squeeze()

  # ---- NIR band ----
  print("streaming and cropping Band 5 (NIR)")
  with rioxarray.open_rasterio(latest_item.assets[nir_key].href) as nir_src:
      nir_cropped = nir_src.rio.clip_box(
          minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3],
          crs="EPSG:4326",
      ).squeeze()

  # ---- Thermal band (Landsat band 10 / ST_B10) ----
  print("streaming and cropping Band 10 (Thermal IR / Surface Temp)")
  with rioxarray.open_rasterio(latest_item.assets[thermal_key].href) as th_src:
      thermal_cropped = th_src.rio.clip_box(
          minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3],
          crs="EPSG:4326",
      ).squeeze()

  # ---- shape alignment ----
  if red_cropped.shape != nir_cropped.shape:
      print("Slight array shape mismatch detected. Re-aligning NIR...")
      nir_cropped = nir_cropped.rio.reproject_match(red_cropped)

  if thermal_cropped.shape != red_cropped.shape:
      print("Slight array shape mismatch detected. Re-aligning thermal...")
      thermal_cropped = thermal_cropped.rio.reproject_match(red_cropped)

  # ---- to float arrays ----
  red     = red_cropped.values.astype(np.float64)
  nir     = nir_cropped.values.astype(np.float64)
  thermal = thermal_cropped.values.astype(np.float64)

  # ---- NDVI ----
  print("calculating ndvi array")
  with np.errstate(divide="ignore", invalid="ignore"):
      ndvi = (nir - red) / (nir + red)
      ndvi = np.nan_to_num(ndvi, nan=0.0)

  # ---- Surface Temperature (Celsius) ----
  # USGS Landsat Collection 2 Level-2 ST_B10 coefficients:
  #   ST_K = DN * 0.00341802 + 149.0
  #   ST_C = ST_K - 273.15
  print("calculating surface temperature array")
  temperature_kelvin  = thermal * 0.00341802 + 149.0
  temperature_celsius = temperature_kelvin - 273.15

  # Fill nodata / invalid (0 DN produces ~149 K). Anything below ~150 K is bogus.
  temperature_celsius = np.where(
      (temperature_celsius < -100.0) | (temperature_celsius > 100.0),
      np.nan,
      temperature_celsius,
  )
  temperature_celsius = np.nan_to_num(temperature_celsius, nan=0.0)

  print(f"Processing complete. NDVI shape: {ndvi.shape}, "
        f"Temp shape: {temperature_celsius.shape}")

  return ndvi, temperature_celsius

# %%
# target_lat = 37.498056
# target_lon = -120.812583

# try:
#   ndvi_matrix = get_landsat_ndvi_matrix(target_lat, target_lon, half_side_meters=1500)
#   print("Found Data")

#   print("\nSample NDVI 5x5 grid slice:")
#   print(np.round(ndvi_matrix[:5, :5], 3))

# except Exception as e:
#   print(f"An error occurred: {e}")

# # %%
# print(ndvi_matrix.shape)

# # %%
# plt.imshow(ndvi_matrix, vmin=0.0, vmax=1.0, origin='upper')


