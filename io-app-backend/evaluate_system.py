import torch
import terramindFunctions as tm
from terratorch import FULL_MODEL_REGISTRY
from metrics import calculate_precision,calculate_recall,calculate_accuracy, calculate_miou, calculate_fw_iou, calculate_dice_score

# [CONFIGURATION] Device setup
DEVICE = tm.device

def load_model(model_name):
    """Loads selected model from TerraTorch registry. Falls back to Large version if loading fails."""
    print(f"[LOADING] Loading model: {model_name}...")
    try:
        model = FULL_MODEL_REGISTRY.build(
            model_name,
            modalities=["S2L2A"],
            output_modalities=["LULC"],
            pretrained=True,
            standardize=True,
        ).to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"[WARNING] Model loading error {model_name}: {e}")
        print(f"[FALLBACK] Attempting terramind_v1_large_generate...")
        try:
            model = FULL_MODEL_REGISTRY.build(
                "terramind_v1_large_generate",
                modalities=["S2L2A"],
                output_modalities=["LULC"],
                pretrained=True,
                standardize=True,
            ).to(DEVICE)
            model.eval()
            return model
        except Exception as e2:
            print(f"[ERROR] Fallback model loading error: {e2}")
            return None

def run_evaluation_with_models(lat, lon, buffer_km=5, max_cloud_cover=20, days_back=120, model_a_name=None, model_b_name=None):
    """
    Runs comparison of two selected models on satellite imagery.
    Downloads data once and processes both models with spectral corrections.
    Computes metrics for both raw and corrected outputs.

    Args:
        lat: latitude coordinate
        lon: longitude coordinate
        buffer_km: search radius in kilometers
        model_a_name: name of first model (default: terramind_v1_small_generate)
        model_b_name: name of second model (default: terramind_v1_large_generate)

    Returns:
        dict with comparison metrics, class maps, and imagery data
    """
    if model_a_name is None:
        model_a_name = 'terramind_v1_small_generate'
    if model_b_name is None:
        model_b_name = 'terramind_v1_large_generate'

    print(f"[COMPARE] Model comparison for: {lat}, {lon}")
    print(f"   Model A: {model_a_name}")
    print(f"   Model B: {model_b_name}")

    # [DOWNLOAD] Download data once for both models (time efficient)
    dl_result = tm.download_sentinel2(lat, lon, buffer_km, max_cloud_cover, days_back)

    if dl_result is None:
        return {"error": "Satellite data unavailable for given analysis parameters."}

    raw_data, date, scene_id = dl_result

    # [DIMENSIONS] Save original image dimensions before scaling to 224x224
    original_height, original_width = raw_data.shape[1], raw_data.shape[2]
    print(f"[DIMENSIONS] Original image size: {original_width}x{original_height}")

    # [PREPARE] Prepare common data for both models
    input_tensor = tm.prepare_input(raw_data)
    # [INDICES] Calculate spectral indices once and use for both models (consistency)
    indices = tm.calculate_spectral_indices(input_tensor)

    # ==========================================
    # [MODEL_A] Model A processing
    # ==========================================
    print(f"[PROCESSING] Processing: {model_a_name}...")
    model_a = load_model(model_a_name)
    if model_a is None:
        return {"error": f"Error loading model {model_a_name}"}

    raw_output_a = tm.run_inference(model_a, input_tensor)
    map_a_raw = tm.decode_output(raw_output_a)
    # [CORRECTIONS] Apply spectral corrections
    map_a, _ = tm.apply_hybrid_corrections(map_a_raw, indices)
    del model_a

    # ==========================================
    # [MODEL_B] Model B processing
    # ==========================================
    print(f"[PROCESSING] Processing: {model_b_name}...")
    model_b = load_model(model_b_name)
    if model_b is None:
        return {"error": f"Error loading model {model_b_name}"}

    raw_output_b = tm.run_inference(model_b, input_tensor)
    map_b_raw = tm.decode_output(raw_output_b)
    # [CORRECTIONS] Apply spectral corrections
    map_b, _ = tm.apply_hybrid_corrections(map_b_raw, indices)
    del model_b

    # [CLEANUP] Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ==========================================
    # [METRICS] Calculate evaluation metrics
    # ==========================================
    print("[METRICS] Computing metrics...")

    # [RAW_METRICS] Metrics for RAW segmentation (without spectral indices)
    print("  [RAW] Computing metrics for raw segmentation (without spectral indices)...")
    acc_raw = calculate_accuracy(map_a_raw, map_b_raw)
    miou_raw, iou_details_raw = calculate_miou(map_a_raw, map_b_raw)
    fw_iou_raw = calculate_fw_iou(map_a_raw, map_b_raw)
    dice_raw = calculate_dice_score(map_a_raw, map_b_raw)
    mean_precision_raw, precision_details_raw = calculate_precision(map_a_raw, map_b_raw)
    mean_recall_raw, recall_details_raw = calculate_recall(map_a_raw, map_b_raw)

    # [COMBINE_RAW] Combine details for RAW
    combined_details_raw = {}
    for class_name in iou_details_raw.keys():
        combined_details_raw[class_name] = {
            "iou": iou_details_raw.get(class_name, 0.0),
            "precision": precision_details_raw.get(class_name, 0.0),
            "recall": recall_details_raw.get(class_name, 0.0)
        }

    # [CORRECTED_METRICS] Metrics for CORRECTED segmentation (with spectral corrections)
    print("  [CORRECTED] Computing metrics for corrected segmentation (with spectral indices)...")
    acc = calculate_accuracy(map_a, map_b)
    miou, iou_details = calculate_miou(map_a, map_b)
    fw_iou = calculate_fw_iou(map_a, map_b)
    dice = calculate_dice_score(map_a, map_b)

    # [NEW_METRICS] Invoke precision and recall functions
    mean_precision, precision_details = calculate_precision(map_a, map_b)
    mean_recall, recall_details = calculate_recall(map_a, map_b)

    # [COMBINE_CORRECTED] Combine details for CORRECTED
    combined_details = {}

    # [KEYS] Extract class names from IoU details (always computed)
    for class_name in iou_details.keys():
        combined_details[class_name] = {
            "iou": iou_details.get(class_name, 0.0),
            "precision": precision_details.get(class_name, 0.0),
            "recall": recall_details.get(class_name, 0.0)
        }

    return {
        "status": "success",
        "metrics": {
            "raw": {
                "accuracy": acc_raw,
                "miou": miou_raw,
                "fw_iou": fw_iou_raw,
                "dice": dice_raw,
                "mean_precision": mean_precision_raw,
                "mean_recall": mean_recall_raw,
                "class_details": combined_details_raw
            },
            "corrected": {
                "accuracy": acc,
                "miou": miou,
                "fw_iou": fw_iou,
                "dice": dice,
                "mean_precision": mean_precision,
                "mean_recall": mean_recall,
                "class_details": combined_details
            }
        },
        "maps": {
            "modelA": map_a,
            "modelB": map_b,
            "modelA_raw": map_a_raw,
            "modelB_raw": map_b_raw
        },
        "raw_data": raw_data,
        "input_tensor": input_tensor,
        "indices": indices,
        "date": date,
        "image_width": original_width,
        "image_height": original_height
    }
