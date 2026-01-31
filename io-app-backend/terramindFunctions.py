
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import torchvision.transforms as T
import terratorch
import base64
from io import BytesIO
from PIL import Image
import gc

# =========================================
# CONFIGURATION DEFAULTS
# =========================================

DEFAULT_BUFFER_KM = 5
DEFAULT_MAX_CLOUD_COVER = 10
DEFAULT_DAYS_BACK = 180

TARGET_SIZE = 224
TIMESTEPS = 50
BRIGHTNESS_BOOST = 2.5
NORMALIZATION_MODE = "offset"

# Spectral index thresholds
NDWI_THRESHOLD = 0.1
MNDWI_THRESHOLD = 0.1
NDVI_THRESHOLD = 0.3
NDBI_THRESHOLD = 0.0
BSI_THRESHOLD = 0.1

USE_WATER_CORRECTION = True
USE_VEGETATION_CORRECTION = True
USE_BUILDING_CORRECTION = True
USE_BARE_SOIL_CORRECTION = True

# =========================================
# ESA WORLDCOVER CLASSES
# =========================================

ESA_CLASSES = {
    0: "No data",
    10: "Trees / Forest",
    20: "Shrubs",
    30: "Grass / Meadows",
    40: "Cultivated crops",
    50: "Buildings",
    60: "Bare ground",
    70: "Snow and ice",
    80: "Water",
    90: "Wetlands",
    95: "Mangroves",
    100: "Lichens and moss"
}

ESA_COLORS = {
    0: [0, 0, 0],
    10: [0, 100, 0],
    20: [255, 187, 34],
    30: [255, 255, 76],
    40: [240, 150, 255],
    50: [250, 0, 0],
    60: [180, 180, 180],
    70: [240, 240, 240],
    80: [0, 100, 200],
    90: [0, 150, 160],
    95: [0, 207, 117],
    100: [250, 230, 160]
}

INDEX_TO_ESA = {
    0: 0, 1: 10, 2: 20, 3: 30, 4: 40, 5: 50,
    6: 60, 7: 70, 8: 80, 9: 90, 10: 95, 11: 100
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[DEVICE] Computing device: {device}")

# =========================================
# GLOBAL MODEL CACHE
# =========================================

_CURRENT_MODEL = None
_CURRENT_MODEL_NAME = None

def get_model(model_name):
    """
    Loads and caches TerraMind model for inference.
    Implements global cache to avoid loading same model multiple times.
    Clears memory when switching models.

    Args:
        model_name: str - model identifier from TerraTorch registry

    Returns:
        model object ready for inference

    Raises:
        Exception if model loading fails
    """
    global _CURRENT_MODEL, _CURRENT_MODEL_NAME
    if _CURRENT_MODEL is not None and _CURRENT_MODEL_NAME == model_name:
        return _CURRENT_MODEL
    if _CURRENT_MODEL is not None:
        del _CURRENT_MODEL
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("[CLEANUP] Memory cleared after previous model.")

    from terratorch import FULL_MODEL_REGISTRY
    print("[LOADING] Loading model (first time only)...")

    try:
        model = FULL_MODEL_REGISTRY.build(
            model_name,  # Use the name passed from parameter
            modalities=["S2L2A"],
            output_modalities=["LULC"],
            pretrained=True,
            standardize=True,
        ).to(device)

        model.eval()

        # Update global cache
        _CURRENT_MODEL = model
        _CURRENT_MODEL_NAME = model_name

        print(f"[SUCCESS] Model {model_name} ready for use.")
        return _CURRENT_MODEL

    except Exception as e:
        print(f"[ERROR] Error loading model {model_name}: {e}")
        # If it fails, try loading default or raise error
        raise e

# =========================================
# HELPER FUNCTIONS
# =========================================

def get_coordinates_from_name(place_name):
    """
    Geocodes place name to geographic coordinates using Nominatim.

    Args:
        place_name: str - name of the location

    Returns:
        tuple of (latitude, longitude) or None if not found
    """
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="terramind_fast")
        location = geolocator.geocode(place_name)
        if location:
            print(f"[LOCATION] {location.address}")
            return location.latitude, location.longitude
        return None
    except:
        return None


def download_sentinel2(lat, lon, buffer_km, max_cloud_cover, days_back):
    """
    Downloads Sentinel-2 L2A satellite data for specified location.
    Uses Planetary Computer STAC API to find and load cloud-optimized GeoTIFF data.

    Args:
        lat: latitude coordinate
        lon: longitude coordinate
        buffer_km: search radius in kilometers
        max_cloud_cover: maximum allowed cloud cover percentage
        days_back: days to search back in time

    Returns:
        tuple of (stacked_array, date, scene_id) or None if no data found
    """
    import pystac_client
    import odc.stac
    import planetary_computer
    import math
    import numpy as np
    from datetime import datetime, timedelta
    from pyproj import Transformer

    print(f"[DOWNLOAD] Downloading data for: {lat:.4f}, {lon:.4f} (Radius: {buffer_km}km)")

    # 1. Calculate Mercator scale factor for given latitude
    # This corrects map distortion (in Poland 1 meter real ≈ 1.6 meters in EPSG:3857)
    scale_factor = 1.0 / math.cos(math.radians(lat))

    # 2. Prepare transformers
    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    # 3. Calculate center in Mercator meters
    center_x, center_y = to_3857.transform(lon, lat)

    # 4. Calculate extent in Mercator meters accounting for scale
    # half_side = (buffer_km * 1000) / 2  <-- If buffer_km is side length
    # half_side = (buffer_km * 1000)      <-- If buffer_km is radius (distance from center)
    # We assume buffer_km as radius (consistent with "radius" logic):
    half_side_mercator = (buffer_km * 1000) * scale_factor

    min_x, min_y = center_x - half_side_mercator, center_y - half_side_mercator
    max_x, max_y = center_x + half_side_mercator, center_y + half_side_mercator

    # 5. Convert back to degrees for STAC (required by API)
    west, south = to_4326.transform(min_x, min_y)
    east, north = to_4326.transform(max_x, max_y)
    bbox_geo = [west, south, east, north]

    # --- Data Search ---
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    date_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox_geo,
        datetime=date_range,
        query={"eo:cloud_cover": {"lte": max_cloud_cover}}
    )

    items = list(search.items())
    if not items:
        print("[ERROR] No data matching criteria found")
        return None

    best_item = sorted(items, key=lambda x: x.properties.get('eo:cloud_cover', 100))[0]

    # 6. Load data
    # bbox_geo (degrees) defines the area, crs="EPSG:3857" enforces web map format
    data = odc.stac.load(
        [best_item],
        bands=["B01", "B02", "B03", "B04", "B05", "B06",
               "B07", "B08", "B8A", "B09", "B11", "B12"],
        bbox=bbox_geo,
        crs="EPSG:3857",
        resolution=10
    )

    # Map variables to numpy array
    stacked = np.stack([data[b].values[0] for b in data.data_vars], axis=0)

    print(f"[SUCCESS] Downloaded image size: {stacked.shape}")
    return stacked, best_item.datetime.strftime('%Y-%m-%d'), best_item.id

# Input data format: (12, H, W) numpy array (12 Sentinel-2 bands)
def prepare_input(data_12ch):
    """
    Preprocesses raw satellite data tensor for model inference.
    Normalizes, resizes, and converts to PyTorch tensor.

    Args:
        data_12ch: numpy array of shape (12, H, W) with 12 Sentinel-2 bands

    Returns:
        torch.Tensor of shape (1, 12, 224, 224) ready for model inference
    """
    tensor = torch.from_numpy(data_12ch.astype(np.float32))
    tensor = torch.nan_to_num(tensor, nan=0.0)

    if NORMALIZATION_MODE == "offset":
        tensor = tensor - 1000.0

    tensor = torch.clamp(tensor, min=0)

    h, w = tensor.shape[1], tensor.shape[2]
    min_dim = min(h, w)

    transform = T.Compose([
        T.CenterCrop(min_dim),
        T.Resize((TARGET_SIZE, TARGET_SIZE), antialias=True)
    ])

    return transform(tensor).unsqueeze(0)

def run_inference(model, input_tensor):
    """
    Executes AI model inference on preprocessed satellite data.

    Args:
        model: TerraMind neural network model
        input_tensor: torch.Tensor of shape (1, 12, 224, 224)

    Returns:
        torch.Tensor containing LULC (Land Use Land Cover) predictions
    """
    print(f"[RUNNING] Running AI model...")
    with torch.no_grad():
        output = model(
            {"S2L2A": input_tensor.to(device)},
            verbose=False,
            timesteps=TIMESTEPS
        )
    return output["LULC"].detach()

def decode_output(lulc_tensor):
    """
    Converts model output tensor to class map with ESA WorldCover labels.
    Maps class indices to standardized ESA class codes.

    Args:
        lulc_tensor: Model output tensor (4D or 3D depending on inference mode)

    Returns:
        numpy.ndarray with ESA class codes (0-100)
    """
    if lulc_tensor.ndim == 4 and lulc_tensor.shape[1] > 1:
        class_indices = lulc_tensor.argmax(dim=1)[0].cpu().numpy()
        if class_indices.max() <= 11:
            class_map = np.vectorize(lambda x: INDEX_TO_ESA.get(x, 0))(class_indices)
        else:
            class_map = class_indices
    else:
        class_map = lulc_tensor[0, 0].cpu().numpy().astype(int)
    return class_map

def calculate_spectral_indices(input_tensor):
    """
    Calculates spectral indices from Sentinel-2 multispectral bands.
    Includes: NDWI, MNDWI, AWEI, NDVI, EVI, NDBI, BSI.

    Args:
        input_tensor: torch.Tensor containing 12 Sentinel-2 bands

    Returns:
        dict with spectral indices as numpy arrays
    """
    blue = input_tensor[0, 1].cpu().numpy() / 10000.0
    green = input_tensor[0, 2].cpu().numpy() / 10000.0
    red = input_tensor[0, 3].cpu().numpy() / 10000.0
    nir = input_tensor[0, 7].cpu().numpy() / 10000.0
    swir1 = input_tensor[0, 10].cpu().numpy() / 10000.0
    swir2 = input_tensor[0, 11].cpu().numpy() / 10000.0

    eps = 1e-8
    indices = {}
    indices['ndwi'] = (green - nir) / (green + nir + eps)
    indices['mndwi'] = (green - swir1) / (green + swir1 + eps)
    indices['awei'] = 4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)
    indices['ndvi'] = (nir - red) / (nir + red + eps)
    indices['evi'] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)
    indices['ndbi'] = (swir1 - nir) / (swir1 + nir + eps)
    indices['bsi'] = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + eps)

    return indices

def generate_index_masks(indices):
    """
    Generates binary masks for each spectral vegetation and water index.
    Uses predefined thresholds to create masks for water, vegetation, buildings, bare soil.

    Args:
        indices: dict with spectral indices (ndwi, mndwi, awei, ndvi, evi, ndbi, bsi)

    Returns:
        dict with 7 binary masks as numpy boolean arrays
    """
    masks = {}

    # Water mask (NDWI)
    masks['water_ndwi'] = indices['ndwi'] > NDWI_THRESHOLD

    # Water mask (MNDWI)
    masks['water_mndwi'] = indices['mndwi'] > MNDWI_THRESHOLD

    # Water mask (AWEI)
    masks['water_awei'] = indices['awei'] > 0

    # Vegetation mask (NDVI)
    masks['vegetation_ndvi'] = indices['ndvi'] > NDVI_THRESHOLD

    # Vegetation mask (EVI)
    masks['vegetation_evi'] = indices['evi'] > 0.3

    # Buildings mask (NDBI)
    masks['buildings_ndbi'] = (indices['ndbi'] > NDBI_THRESHOLD) & (indices['ndvi'] < 0.2)

    # Bare soil mask (BSI)
    masks['baresoil_bsi'] = (indices['bsi'] > BSI_THRESHOLD) & (indices['ndvi'] < 0.1)

    return masks

def apply_hybrid_corrections(class_map, indices):
    """
    Applies hybrid corrections to raw model output using spectral indices.
    Corrects water, vegetation, buildings, and bare soil classifications.
    Creates correction layers showing where corrections were applied.

    Args:
        class_map: numpy array with ESA class codes from raw model
        indices: dict with calculated spectral indices

    Returns:
        tuple of (corrected_class_map, correction_layers_dict)
    """
    hybrid_map = class_map.copy()
    correction_layers = {}

    if USE_WATER_CORRECTION:
        water_mask = ((indices['ndwi'] > NDWI_THRESHOLD) | (indices['mndwi'] > MNDWI_THRESHOLD) | (indices['awei'] > 0))
        already_water = (class_map == 80) | (class_map == 90)
        correction_layers['water'] = water_mask & ~already_water
        hybrid_map[correction_layers['water']] = 80

    if USE_VEGETATION_CORRECTION:
        strong_vegetation = (indices['ndvi'] > 0.5) & (indices['evi'] > 0.3)
        not_water = (hybrid_map != 80) & (hybrid_map != 90)
        correction_layers['vegetation'] = strong_vegetation & not_water & (hybrid_map != 10)
        hybrid_map[correction_layers['vegetation']] = 10

    if USE_BUILDING_CORRECTION:
        building_mask = ((indices['ndbi'] > NDBI_THRESHOLD) & (indices['ndvi'] < 0.2) & (indices['ndwi'] < 0))
        can_be_building = (hybrid_map != 80) & (hybrid_map != 10)
        correction_layers['buildings'] = building_mask & can_be_building & (hybrid_map != 50)
        hybrid_map[correction_layers['buildings']] = 50

    if USE_BARE_SOIL_CORRECTION:
        bare_mask = ((indices['bsi'] > BSI_THRESHOLD) & (indices['ndvi'] < 0.1) & (indices['ndwi'] < 0) & (indices['ndbi'] < 0.1))
        can_be_bare = (hybrid_map != 80) & (hybrid_map != 10) & (hybrid_map != 50)
        correction_layers['baresoil'] = bare_mask & can_be_bare & (hybrid_map != 60)
        hybrid_map[correction_layers['baresoil']] = 60

    return hybrid_map, correction_layers

# =========================================
# NEW: MASK VISUALIZATION FUNCTIONS
# =========================================

def create_rgb_image(input_tensor, brightness=BRIGHTNESS_BOOST):
    """
    Creates natural color RGB image from Sentinel-2 multispectral tensor.
    Uses red, green, blue bands with brightness scaling and normalization.

    Args:
        input_tensor: torch.Tensor with 12 Sentinel-2 bands
        brightness: float - brightness multiplier (default 2.5)

    Returns:
        numpy.ndarray (H, W, 3) uint8 RGB image
    """
    red = input_tensor[0, 3].cpu().numpy()
    green = input_tensor[0, 2].cpu().numpy()
    blue = input_tensor[0, 1].cpu().numpy()

    rgb = np.stack([red, green, blue], axis=-1)

    if NORMALIZATION_MODE == "offset":
        rgb = rgb + 1000.0

    rgb = rgb / 10000.0 * brightness
    rgb = np.clip(rgb, 0, 1)
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    return rgb_uint8

def create_segmentation_image(class_map):
    """
    Converts class map to colored segmentation visualization.
    Maps ESA class codes to predefined RGB colors for display.

    Args:
        class_map: numpy array with ESA class codes (0-100)

    Returns:
        numpy.ndarray (H, W, 3) uint8 RGB image with class colors
    """
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in ESA_COLORS.items():
        mask = class_map == class_id
        rgb[mask] = color

    return rgb

def create_mask_visualization(mask, color=[255, 0, 0]):
    """
    Creates colored visualization of binary mask.
    Renders mask pixels in specified color with gray background for contrast.

    Args:
        mask: numpy boolean array (True/False)
        color: list [R, G, B] for True pixels (default red [255, 0, 0])

    Returns:
        numpy.ndarray (H, W, 3) uint8 RGB image with mask visualization
    """
    h, w = mask.shape
    rgb = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background

    rgb[mask] = color  # Fill mask with color
    rgb[~mask] = [240, 240, 240]  # Gray background for better visibility

    return rgb

def calculate_class_percentages(class_map):
    """
    Calculates pixel count and percentage coverage for each ESA class.
    Includes class names and statistics for result reporting.

    Args:
        class_map: numpy array with ESA class codes

    Returns:
        dict with class statistics {class_id: {name, count, percentage}}
    """
    total_pixels = class_map.size
    percentages = {}

    for class_id, class_name in ESA_CLASSES.items():
        count = np.sum(class_map == class_id)
        if count > 0:
            percentages[class_id] = {
                'name': class_name,
                'count': int(count),
                'percentage': round(count / total_pixels * 100, 2)
            }

    return percentages

def image_to_base64(image_array):
    """
    Encodes numpy image array to base64-encoded PNG string for transmission.

    Args:
        image_array: numpy.ndarray containing image data

    Returns:
        str - base64-encoded PNG image suitable for HTML/JSON transmission
    """
    img = Image.fromarray(image_array)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# =========================================
# MAIN ANALYTICAL FUNCTION
# =========================================

def analyze(location_data, buffer_km=DEFAULT_BUFFER_KM, max_cloud_cover=DEFAULT_MAX_CLOUD_COVER,
            days_back=DEFAULT_DAYS_BACK, show_visualization=False, save_files=False, model_name="terramind_v1_large_generate"):
    """
    Main analysis pipeline for land cover classification and spectral analysis.
    Downloads Sentinel-2 data, runs AI model, applies corrections, generates visualizations.

    Args:
        location_data: list/tuple [lat, lon] or str place name
        buffer_km: int - search radius in kilometers (default 5)
        max_cloud_cover: int - maximum cloud percentage (default 10)
        days_back: int - days to search back (default 180)
        show_visualization: bool - display plots (not used in API)
        save_files: bool - save results to disk (not used in API)
        model_name: str - TerraMind model name

    Returns:
        dict with success status, images (base64), masks, statistics, metadata

    Example:
        result = analyze([50.0540, 19.9352], buffer_km=5, max_cloud_cover=20)
    """

    lat, lon = None, None
    title = "Unknown"

    # 1. Recognize data type
    if isinstance(location_data, list) or isinstance(location_data, tuple):
        lat, lon = location_data
        title = f"{lat:.4f}N, {lon:.4f}E"
    elif isinstance(location_data, str):
        coords = get_coordinates_from_name(location_data)
        if not coords:
            print("[ERROR] Coordinates not found for given name.")
            return None
        lat, lon = coords
        title = location_data
    else:
        print("[ERROR] Invalid location format")
        return None

    print(f"\n{'='*60}")
    print(f"[START] ANALYSIS START: {title}")
    print(f"{'='*60}")
    print(f"[LOCATION] Coordinates: {lat:.6f}, {lon:.6f}")
    print(f"[RADIUS] Radius: {buffer_km} km")
    print(f"[CLOUDS] Max cloud cover: {max_cloud_cover}%")
    print(f"[HISTORY] Days back: {days_back}")
    print(f"{'='*60}\n")

    # 2. Download data
    result = download_sentinel2(lat, lon, buffer_km, max_cloud_cover, days_back)
    if result is None:
        return None
    data, date, scene_id = result

    # Save original size before scaling
    original_height, original_width = data.shape[1], data.shape[2]

    # 3. Process and run AI
    input_tensor = prepare_input(data)
    print(f"[TENSOR] Input tensor size: {input_tensor.shape}")

    model = get_model(model_name)
    lulc_output = run_inference(model, input_tensor)

    # 4. Decode (RAW model - without corrections)
    class_map_raw = decode_output(lulc_output)

    # 5. Calculate spectral indices
    indices = calculate_spectral_indices(input_tensor)

    # 6. Generate masks for indices
    index_masks = generate_index_masks(indices)

    # 7. Apply corrections (final map)
    class_map_final, correction_layers = apply_hybrid_corrections(class_map_raw, indices)

    # =========================================
    # GENERATING ALL VISUALIZATIONS
    # =========================================

    print("[VISUALIZATION] Generating visualizations...")

    # 1. RGB (satellite)
    rgb_image = create_rgb_image(input_tensor)

    # 2. Raw TerraMind (without corrections)
    raw_segmentation = create_segmentation_image(class_map_raw)

    # 3. Final segmentation (with corrections)
    final_segmentation = create_segmentation_image(class_map_final)

    # 4. Index masks
    mask_images = {}
    mask_colors = {
        'water_ndwi': [0, 150, 255],      # Blue
        'water_mndwi': [0, 100, 200],     # Dark blue
        'water_awei': [100, 200, 255],    # Light blue
        'vegetation_ndvi': [0, 150, 0],   # Green
        'vegetation_evi': [50, 200, 50],  # Light green
        'buildings_ndbi': [255, 0, 0],    # Red
        'baresoil_bsi': [180, 140, 100],  # Brown
    }

    for mask_name, mask in index_masks.items():
        color = mask_colors.get(mask_name, [128, 128, 128])
        mask_images[mask_name] = create_mask_visualization(mask, color)

    # 5. Statistics
    statistics = calculate_class_percentages(class_map_final)

    # =========================================
    # PREPARING RESULT
    # =========================================

    frontend_result = {
        'success': True,
        'lat': lat,
        'lon': lon,
        'title': title,
        'date': date,
        'scene_id': scene_id,
        'statistics': statistics,

        # MAIN IMAGES
        'rgb_base64': image_to_base64(rgb_image),
        'raw_segmentation_base64': image_to_base64(raw_segmentation),
        'segmentation_base64': image_to_base64(final_segmentation),

        # INDEX MASKS
        'masks': {
            mask_name: image_to_base64(mask_img)
            for mask_name, mask_img in mask_images.items()
        },

        # For frontend compatibility
        'class_map': class_map_final.tolist(),

        # Original image dimensions
        'image_width': original_width,
        'image_height': original_height
    }

    print("[SUCCESS] Analysis completed successfully!")

    return frontend_result

if __name__ == "__main__":
    print("\n" + "="*70)
    print("[TEST] TEST MODE - terramindFunctions.py")
    print("="*70)

    result = analyze([50.0540, 19.9352], buffer_km=3, max_cloud_cover=10, days_back=60)

    if result:
        print("\n" + "="*70)
        print("[RESULT] GENERATED IMAGES:")
        print("="*70)
        print(f"  [OK] RGB: {len(result['rgb_base64'])} chars")
        print(f"  [OK] Raw TerraMind: {len(result['raw_segmentation_base64'])} chars")
        print(f"  [OK] Final segmentation: {len(result['segmentation_base64'])} chars")
        print(f"  [OK] Index masks: {len(result['masks'])} pieces")
        for mask_name in result['masks'].keys():
            print(f"      - {mask_name}")
        print("="*70)
