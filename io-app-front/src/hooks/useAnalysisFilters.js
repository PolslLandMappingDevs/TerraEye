import { useState } from 'react'

export const useAnalysisFilters = () => {
  const [filters, setFilters] = useState({
    cloudCover: 20,
    radius: 5,
    daysRange: 120,
    model: 'terramind_v1_base_generate'
  });

  const updateFilter = (key, value) => {
    setFilters((prev) => ({
      ...prev,
      [key]: value
    }));
  };

  return {
    filters,
    updateFilter,
  };
};
