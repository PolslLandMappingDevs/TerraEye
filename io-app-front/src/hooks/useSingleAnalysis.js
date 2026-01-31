import { useState } from 'react'
import axios from 'axios'

export const useSingleAnalysis = (onAnalysisComplete) => {
  const [isLoading, setIsLoading] = useState(false);
  const [isError, setIsError] = useState(null);

  const runAnalysis = async (location, filters) => {
    if (!location || location.isSearch) return;
    setIsLoading(true);
    setIsError(null);

    const payload = {
      location: { lat: location.lat, lng: location.lng },
      params: {
        bufferKm: filters.radius,
        maxCloudCover: filters.cloudCover,
        daysBack: filters.daysRange,
        model: filters.model
      }
    };

    try {
      const response = await axios.post('/api/analyze', payload);
      if (response.data.success && onAnalysisComplete) {
        onAnalysisComplete(response.data);
      } else {
        setIsError(response.data.error || 'Server returned an unsuccessful status');
      }
    } catch (error) {
      console.error('Connection error', error);
      const message = error.response?.data?.error || 'Backend connection error. Please try again later.'
      setIsError(message);
    } finally {
      setIsLoading(false);
    }
  };

  return { runAnalysis, isLoading, isError };
};
