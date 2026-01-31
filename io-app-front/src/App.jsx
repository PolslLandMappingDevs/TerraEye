import { useState } from "react"
import AnalysisPanel from './components/AnalysisPanel/AnalysisPanel'
import MapView from './components/MapView/MapView'
import { MantineProvider, createTheme } from '@mantine/core'
import '@mantine/core/styles.css'

const theme = createTheme({
  fontFamily: 'Google Sans, sans-serif',
  headings: {
    fontFamily: 'Google Sans, sans-serif',
  },
});

function App() {
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [layersConfig, setLayersConfig] = useState({
    firstLayer: 'rgb',
    secondLayer: 'image',
  });

  const handleAnalysisComplete = (data) => {
    setAnalysisResult(data);
  };

  const handleLocationChange = (locationData) => {
    setSelectedLocation(locationData);
  };

  return (
    <MantineProvider forceColorScheme='dark' theme={ theme }>
      <AnalysisPanel
        selectedLocation={selectedLocation}
        onLocationSelect={handleLocationChange}
        onAnalysisComplete={handleAnalysisComplete}
        layersConfig={layersConfig}
        onLayersChange={setLayersConfig}
        analysisResult={analysisResult}
      />
      <MapView
        selectedLocation={selectedLocation}
        onLocationSelect={handleLocationChange}
        analysisResult={analysisResult}
        layersConfig={layersConfig}
      />
    </MantineProvider>
  )
}

export default App
