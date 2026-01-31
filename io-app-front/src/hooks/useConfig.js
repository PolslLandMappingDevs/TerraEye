import { useState, useEffect } from 'react'
import { api } from '../services/api'

export const useConfig = () => {
  const [mapboxAccessToken, setMapboxAccessToken] = useState(null);
  const [status, setStatus] = useState('loading');

  useEffect(() => {
    api.get('/config')
      .then(response => {
        setMapboxAccessToken(response.data.mapboxAccessToken);
        setStatus('success');
      })
      .catch(error => {
        console.error('Axios error: ', error.message);
        setStatus('error');
      })
  }, []);

  return { mapboxAccessToken, status };
}
