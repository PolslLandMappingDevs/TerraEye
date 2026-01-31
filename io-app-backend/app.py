from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import terramindFunctions as tm
import os
import traceback
from terramindFunctions import analyze
from evaluate_system import run_evaluation_with_models

app = Flask(__name__, static_folder='static', static_url_path='')

CORS(app)

# =========================================
# API ENDPOINTS
# =========================================
@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({
        'mapboxAccessToken': os.environ.get('MAPBOX_ACCESS_TOKEN', ''),
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_endpoint():
    try:
        data = request.json
        print(f"[REQUEST] Received request: {data}")

        location = data.get('location', {})
        lat = location.get('lat')
        lng = location.get('lng')

        params = data.get('params', {})
        buffer_km = params.get('bufferKm', 5)
        max_cloud_cover = params.get('maxCloudCover', 20)
        days_back = params.get('daysBack', 120)
        model_name = params.get('model', 'terramind_v1_base_generate')

        if lat is None or lng is None:
            return jsonify({
                'success': False,
                'error': 'Missing coordinates (lat/lng)'
            }), 400

        print(f"[RUNNING] Starting analysis for: [{lat}, {lng}]")
        print(f"[PARAMETERS] Buffer={buffer_km}km, MaxCloudCover<{max_cloud_cover}%, HistoryDays={days_back}")

        result = analyze(
            location_data=[lat, lng],
            buffer_km=buffer_km,
            max_cloud_cover=max_cloud_cover,
            days_back=days_back,
            model_name=model_name,
            show_visualization=False,
            save_files=False
        )

        if result is None:
            return jsonify({
                'success': False,
                'error': 'Satellite data unavailable for given analysis parameters.'
            }), 404

        response = {
            'success': True,
            'lat': result.get('lat'),
            'lon': result.get('lon'),
            'radius_km': buffer_km,
            'title': result.get('title'),
            'date': result.get('date'),
            'scene_id': result.get('scene_id'),
            'statistics': result.get('statistics'),
            'image': f"data:image/png;base64,{result.get('segmentation_base64')}",
            'rgb': f"data:image/png;base64,{result.get('rgb_base64')}",
            'raw_segmentation': f"data:image/png;base64,{result.get('raw_segmentation_base64')}",
            'masks': {
                mask_name: f"data:image/png;base64,{mask_base64}"
                for mask_name, mask_base64 in result.get('masks', {}).items()
            }
        }

        print(f"[SUCCESS] Sending result to frontend")
        return jsonify(response)

    except Exception as e:
        print(f"[ERROR] Error during analysis: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/advanced-analyze', methods=['POST'])
def advanced_analyze_endpoint():
    """
    Endpoint for advanced comparative analysis with metrics.
    Compares two selected models and returns:
    - Images from both models (RGB, raw, final)
    - Spectral masks
    - Comparative metrics
    """
    try:
        data = request.json
        print(f"[REQUEST] Received advanced analysis request: {data}")

        location = data.get('location', {})
        lat = location.get('lat')
        lng = location.get('lng')

        params = data.get('params', {})
        buffer_km = params.get('bufferKm', 5)
        max_cloud_cover = params.get('maxCloudCover', 20)
        days_back = params.get('daysBack', 120)
        model_a = params.get('modelA', 'terramind_v1_small_generate')
        model_b = params.get('modelB', 'terramind_v1_large_generate')

        if lat is None or lng is None:
            return jsonify({
                'success': False,
                'error': 'Missing coordinates (lat/lng)'
            }), 400

        print(f"[RUNNING] Starting advanced analysis for: [{lat}, {lng}]")
        print(f"[COMPARISON] {model_a} vs {model_b}")
        print(f"[PARAMETERS] Buffer={buffer_km}km, MaxCloudCover<{max_cloud_cover}%, HistoryDays={days_back}")

        # Run evaluate_system.run_evaluation_with_models()
        eval_result = run_evaluation_with_models(lat, lng, buffer_km, max_cloud_cover, days_back, model_a, model_b)

        if 'error' in eval_result:
            return jsonify({
                'success': False,
                'error': eval_result.get('error')
            }), 404

        # Convert maps to images (using terramindFunctions)
        print("[CONVERTING] Converting maps to images...")
        import base64
        import io
        from PIL import Image
        import numpy as np

        # Maps with corrections and without corrections
        model_a_map = eval_result['maps']['modelA']  # With corrections
        model_a_map_raw = eval_result['maps']['modelA_raw']  # Without corrections
        model_b_map = eval_result['maps']['modelB']  # With corrections
        model_b_map_raw = eval_result['maps']['modelB_raw']  # Without corrections
        raw_data = eval_result['raw_data']
        input_tensor = eval_result['input_tensor']

        # Function to convert numpy RGB array to base64
        def rgb_to_base64(rgb_array):
            """Converts RGB array to PNG base64"""
            img = Image.fromarray(rgb_array.astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')

        # RGB from raw_data (using create_rgb_image from terramindFunctions)
        rgb_image = tm.create_rgb_image(input_tensor)
        rgb_base64 = rgb_to_base64(rgb_image)

        # Segmentation maps - raw (without corrections) and final (with corrections)
        model_a_raw_segmentation = tm.create_segmentation_image(model_a_map_raw)
        model_a_segmentation = tm.create_segmentation_image(model_a_map)
        model_b_raw_segmentation = tm.create_segmentation_image(model_b_map_raw)
        model_b_segmentation = tm.create_segmentation_image(model_b_map)

        model_a_raw_seg_base64 = rgb_to_base64(model_a_raw_segmentation)
        model_a_seg_base64 = rgb_to_base64(model_a_segmentation)
        model_b_raw_seg_base64 = rgb_to_base64(model_b_raw_segmentation)
        model_b_seg_base64 = rgb_to_base64(model_b_segmentation)

        # Spectral masks (from indices)
        indices = eval_result['indices']
        masks_dict = {}

        # Generate masks like in analyze()
        index_masks = tm.generate_index_masks(indices)

        if isinstance(index_masks, dict):
            for mask_name, mask_array in index_masks.items():
                try:
                    # Convert binary masks to 0-255
                    mask_binary = mask_array.astype(np.uint8) * 255
                    img_mask = Image.fromarray(mask_binary, mode='L')
                    buf_mask = io.BytesIO()
                    img_mask.save(buf_mask, format='PNG')
                    buf_mask.seek(0)
                    masks_dict[mask_name] = base64.b64encode(buf_mask.read()).decode('utf-8')
                except Exception as e:
                    print(f"[WARNING] Error converting mask {mask_name}: {e}")

        # Format the response
        response = {
            'success': True,
            'date': eval_result.get('date'),
            'lat': lat,
            'lon': lng,
            'radius_km': buffer_km,
            'image_width': eval_result.get('image_width', 512),
            'image_height': eval_result.get('image_height', 512),

            # IMAGES
            'modelA': {
                'name': model_a.split('_')[2].upper(),
                'rgb': f"data:image/png;base64,{rgb_base64}",
                'raw_segmentation': f"data:image/png;base64,{model_a_raw_seg_base64}",
                'image': f"data:image/png;base64,{model_a_seg_base64}",
                'masks': {
                    mask_name: f"data:image/png;base64,{mask_base64}"
                    for mask_name, mask_base64 in masks_dict.items()
                }
            },
            'modelB': {
                'name': model_b.split('_')[2].upper(),
                'rgb': f"data:image/png;base64,{rgb_base64}",
                'raw_segmentation': f"data:image/png;base64,{model_b_raw_seg_base64}",
                'image': f"data:image/png;base64,{model_b_seg_base64}",
                'masks': {
                    mask_name: f"data:image/png;base64,{mask_base64}"
                    for mask_name, mask_base64 in masks_dict.items()
                }
            },

            # METRICS
            'metrics': eval_result.get('metrics', {})
        }

        print(f"[SUCCESS] Sending advanced analysis to frontend")
        print(f"   Image date: {eval_result.get('date')}")
        print(f"   Metrics: {list(eval_result.get('metrics', {}).keys())}")

        return jsonify(response)

    except Exception as e:
        print(f"[ERROR] Error during advanced analysis: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Backend is working correctly'})


# =========================================
# FRONTEND SERVING
# =========================================

@app.route('/')
def serve_index():
    """Serves the main index.html file of the frontend."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    """Serves remaining static files (js, css, images)."""
    return send_from_directory(app.static_folder, path)

@app.errorhandler(404)
def not_found(e):
    """Handles page refresh (React Router) - redirects back to index.html."""
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("[START] TerraMind server ready on Hugging Face!")
    print("="*60)

    # Port 7860 is required by Hugging Face
    port = int(os.environ.get("FLASK_RUN_PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
