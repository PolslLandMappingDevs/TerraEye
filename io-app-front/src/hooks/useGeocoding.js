import { useState, useEffect } from 'react'
import { useDebouncedValue } from '@mantine/hooks'
import axios from 'axios'

export const useGeocoding = (query) => {
  const [locations, setLocations] = useState([]);
  const [isQueryLoading, setIsQueryLoading] = useState(false);
  const [debouncedQuery] = useDebouncedValue(query, 400);

  useEffect(() => {
    if (!debouncedQuery || debouncedQuery.length < 3) {
      setIsQueryLoading(false);
      return;
    }

    const source = axios.CancelToken.source();

    const fetchLocations = async () => {
      setIsQueryLoading(true);

      try {
        const response = await axios.get('https://geocoding-api.open-meteo.com/v1/search', {
          params: {
            name: debouncedQuery,
            count: 5,
            language: 'en',
            format: 'json'
          },
          cancelToken: source.token
        });

        const results = response.data.results || [];

        const formattedLocations = results.map(item => ({
          value: [item.name, item.admin1, item.country].filter(Boolean).join(', '),
          lat: item.latitude,
          lng: item.longitude,
        }));

        const uniqueLocations = formattedLocations.filter((v, i, a) =>
          a.findIndex(t => t.value === v.value) === i
        );

        setLocations(uniqueLocations);
      } catch (error) {
        if (!axios.isCancel(error)) {
          console.error('Geocoding error: ', error);
        }
      } finally {
        setIsQueryLoading(false);
      }
    }

    fetchLocations().catch(console.error);

    return () => {
      source.cancel('Operation canceled by the user.');
    };
  }, [debouncedQuery]);

  return { locations, isQueryLoading, setLocations };
}
