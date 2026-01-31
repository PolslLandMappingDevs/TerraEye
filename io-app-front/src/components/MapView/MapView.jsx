import { useRef, useState, useEffect } from 'react'
import { useConfig } from '../../hooks/useConfig'
import { useMap } from '../../hooks/useMap'
import { useAnalysisLayers } from '../../hooks/useAnalysisLayers'
import MapLegend from '../MapLegend/MapLegend'
import LayerOpacitySlider from '../LayerOpacitySlider/LayerOpacitySlider'
import { IconPlus, IconMinus, IconMapPin } from '@tabler/icons-react'
import { Tooltip } from '@mantine/core'

import styles from './MapView.module.css';

function MapView({
                   selectedLocation,
                   onLocationSelect,
                   analysisResult,
                   layersConfig
                 }) {
  const mapContainerRef = useRef(null);
  const { mapboxAccessToken } = useConfig();

  const [localOpacity, setLocalOpacity] = useState(0.5);

  const { mapInstance, isMapLoaded, handleZoomIn, handleZoomOut, handleFlyBackToAnalysis } = useMap(
    mapContainerRef,
    mapboxAccessToken,
    selectedLocation,
    analysisResult
  );

  useAnalysisLayers(
    mapInstance,
    isMapLoaded,
    analysisResult,
    layersConfig,
    localOpacity,
  );


  useEffect(() => {
    if (!isMapLoaded || !mapInstance) return;

    const handleMapClick = (e) => {
      const { lng, lat } = e.lngLat;
      onLocationSelect({ lat, lng, isSearch: false });
    };

    mapInstance.on('click', handleMapClick);
    return () => mapInstance.off('click', handleMapClick);
  }, [isMapLoaded, mapInstance, onLocationSelect]);

  return (
    <div className={styles.mapWrapper}>
      <div className={styles.mapContainer} ref={mapContainerRef} />
      {analysisResult && (
        <>
          <MapLegend analysisResult={analysisResult} />
          <LayerOpacitySlider
            opacity={localOpacity}
            onChange={setLocalOpacity}
          />
        </>
      )}
      <div className={styles.mapActionsPanel}>
        <Tooltip label="Zoom In" position="left" withArrow offset={15} openDelay={500} closeDelay={200} classNames={{ tooltip: styles.customTooltip }}>
          <button onClick={handleZoomIn} className={styles.mapActionButton}>
            <IconPlus size={20} stroke={1.5} />
          </button>
        </Tooltip>

        <div className={styles.actionDivider} />

        <Tooltip label="Zoom Out" position="left" withArrow offset={15} openDelay={500} closeDelay={200} classNames={{ tooltip: styles.customTooltip }}>
          <button onClick={handleZoomOut} className={styles.mapActionButton}>
            <IconMinus size={20} stroke={1.5} />
          </button>
        </Tooltip>

        <div className={styles.actionDivider} />

        <Tooltip label="Fly Back to Analysis" position="left" withArrow offset={15} openDelay={500} closeDelay={200} classNames={{ tooltip: styles.customTooltip }}>
          <button onClick={handleFlyBackToAnalysis} className={styles.mapActionButton}>
            <IconMapPin size={20} stroke={1.5} />
          </button>
        </Tooltip>
      </div>
    </div>
  );
}

export default MapView;
