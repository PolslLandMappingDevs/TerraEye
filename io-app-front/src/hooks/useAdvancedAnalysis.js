import { useState } from 'react'
import axios from 'axios'

export const useAdvancedAnalysis = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [comparisonData, setComparisonData] = useState(null);
  const [isError, setIsError] = useState(null);

  const runComparison = async (location, filters, modelA, modelB) => {
    if (!location || location.isSearch) return;
    setIsLoading(true);
    setComparisonData(null);
    setIsError(null);

    const payload = {
      location: { lat: location.lat, lng: location.lng },
      params: {
        bufferKm: filters.radius,
        maxCloudCover: filters.cloudCover,
        daysBack: filters.daysRange,
        modelA,
        modelB
      }
    };

    try {
      const response = await axios.post('/api/advanced-analyze', payload);
      if (response.data.success) {
        setComparisonData({
          modelA: response.data.modelA,
          modelB: response.data.modelB,
          metrics: response.data.metrics
        });
      }
    } catch (error) {
      console.error('Connection error', error);
      const message = error.response?.data?.error || 'Backend connection error. Please try again later.'
      setIsError(message);
    } finally {
      setIsLoading(false);
    }
  };

  return { runComparison, comparisonData, setComparisonData, isLoading, isError };
};
