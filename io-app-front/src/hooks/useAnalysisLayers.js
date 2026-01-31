import { useEffect } from 'react'
import { calculateBounds, getImageByPath } from '../utils/mapUtils'

export const useAnalysisLayers = (mapInstance, isMapLoaded, analysisResult, layersConfig, localOpacity) => {
  useEffect(() => {
    if (!isMapLoaded || !mapInstance || !analysisResult || !layersConfig) {
      return;
    }

    const coords = calculateBounds(analysisResult.lat, analysisResult.lon, analysisResult.radius_km || 5);

    const updateLayer = (id, layerKey, opacity) => {
      const sourceId = `source-${id}`;
      const layerId = `layer-${id}`;
      const selectedValue = layersConfig[layerKey];

      if (mapInstance.getSource(sourceId)) {
        if (mapInstance.getLayer(layerId)) mapInstance.removeLayer(layerId);
        mapInstance.removeSource(sourceId);
      }

      if (selectedValue === 'none' || !selectedValue) {
        return;
      }

      const imageData = getImageByPath(analysisResult, layersConfig[layerKey]);

      if (imageData) {
        mapInstance.addSource(sourceId, {
          type: 'image',
          url: imageData,
          coordinates: coords
        });

        const firstLabelId = mapInstance.getStyle().layers.find(l => l.type === 'symbol')?.id;

        mapInstance.addLayer({
          id: layerId,
          type: 'raster',
          source: sourceId,
          paint: {
            'raster-opacity': opacity,
            'raster-fade-duration': 800,
          }
        }, firstLabelId);
      }
    };

    updateLayer('first', 'firstLayer', 1);
    updateLayer('second', 'secondLayer', localOpacity);

  }, [isMapLoaded, mapInstance, analysisResult, layersConfig]);

  useEffect(() => {
    if (isMapLoaded && mapInstance?.getLayer('layer-second')) {
      mapInstance.setPaintProperty('layer-second', 'raster-opacity', localOpacity);
    }
  }, [localOpacity, isMapLoaded, mapInstance]);
};
