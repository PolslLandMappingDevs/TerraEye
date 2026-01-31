import { useState, useEffect, useRef } from 'react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'

export const useMap = (mapContainerRef, mapboxAccessToken, selectedLocation, analysisResult) => {
  const mapRef = useRef(null);
  const markerRef = useRef(null);
  const [mapInstance, setMapInstance] = useState(null);
  const [isMapLoaded, setIsMapLoaded] = useState(false);

  useEffect(() => {
    if (!mapContainerRef.current || !mapboxAccessToken || mapRef.current) {
      return;
    }

    mapboxgl.accessToken = mapboxAccessToken;

    mapRef.current = new mapboxgl.Map({
      container: mapContainerRef.current,
      style: 'mapbox://styles/mapbox/satellite-streets-v12',
      center: [20.0, 50.0],
      zoom: 4,
      projection: 'globe',
      attributionControl: false,
    });

    mapRef.current.on('load', () => {
      setIsMapLoaded(true);
      setMapInstance(mapRef.current);
    });

    mapRef.current.addControl(
      new mapboxgl.AttributionControl({
        compact: true
      }),
      'bottom-right'
    );

    return () => {
      mapRef.current?.remove();
      markerRef.current?.remove();
      mapRef.current = null;
      setMapInstance(null);
      setIsMapLoaded(false);
    }
  }, [mapContainerRef, mapboxAccessToken]);

  useEffect(() => {
    if (!isMapLoaded || !mapRef.current || !selectedLocation) {
      return;
    }

    const { lat, lng, isSearch } = selectedLocation;

    if (isSearch) {
      mapRef.current.flyTo({
        center: [lng, lat],
        zoom: 12,
        essential: true,
        duration: 2500
      });

      if (markerRef.current) {
        markerRef.current.remove();
        markerRef.current = null;
      }
    }
    else {
      if (markerRef.current) markerRef.current.remove();

      markerRef.current = new mapboxgl.Marker({ color: 'red' })
        .setLngLat([lng, lat])
        .addTo(mapRef.current);
    }

  }, [selectedLocation, isMapLoaded]);

  useEffect(() => {
    if (isMapLoaded && mapRef.current && analysisResult && selectedLocation) {
      const { lat, lng } = selectedLocation;

      mapRef.current.flyTo({
        center: [lng, lat],
        zoom: 12,
        essential: true,
        duration: 2500,
        curve: 1.2
      });
    }
  }, [analysisResult, isMapLoaded]);

  const handleZoomIn = () => {
    mapRef.current?.zoomIn({ duration: 300 });
  }

  const handleZoomOut = () => {
    mapRef.current?.zoomOut({ duration: 300 });
  }

  const handleFlyBackToAnalysis = () => {
    const target = analysisResult || selectedLocation;

    if (mapRef.current && target) {
      mapRef.current.flyTo({
        center: [target.lng || target.lon, target.lat],
        zoom: 12,
        essential: true,
        duration: 2500,
      });
    }
  }

  return { mapInstance, isMapLoaded, handleZoomIn, handleZoomOut, handleFlyBackToAnalysis };
}
