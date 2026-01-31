import { useState } from 'react'
import { Title, Text, Stack, Button, Group, Divider } from '@mantine/core'
import { IconArrowsLeftRight, IconPlayerPlay, IconMapPin, IconAlertCircle } from '@tabler/icons-react'
import SearchBar from '../SearchBar/SearchBar'
import AnalysisOptions from '../AnalysisOptions/AnalysisOptions'
import { LayerSelector } from '../LayerSelector/LayerSelector'
import { useAnalysisFilters } from '../../hooks/useAnalysisFilters'
import { useSingleAnalysis } from '../../hooks/useSingleAnalysis'
import { useAdvancedAnalysis } from '../../hooks/useAdvancedAnalysis'
import styles from './AnalysisPanel.module.css'
import AdvancedAnalysisModal from "../AdvancedAnalysisModal/AdvancedAnalysisModal";

function AnalysisPanel({
                         selectedLocation,
                         onAnalysisComplete,
                         onLocationSelect,
                         layersConfig,
                         onLayersChange,
                         analysisResult
                       }) {
  const [compareOpened, setCompareOpened] = useState(false);
  const { filters, updateFilter } = useAnalysisFilters();
  const { runAnalysis, isLoading: isSingleLoading, isError: isSingleError } = useSingleAnalysis(onAnalysisComplete);
  const {
    runComparison,
    comparisonData,
    setComparisonData,
    isLoading: isAdvancedLoading,
    isError: isAdvancedError,
  } = useAdvancedAnalysis();
  const isLocationReady = selectedLocation && !selectedLocation.isSearch;

  const handleOpenCompare = () => {
    setComparisonData(null);
    setCompareOpened(true);
  };

  return (
    <div className={styles.sidebar}>
      <div className={styles.scrollArea}>
        <Title order={3} c="white" mb="md">TerraEye</Title>

        <SearchBar onLocationSelect={onLocationSelect} />

        <div className={styles.locationBadge}>
          {isLocationReady ? (
            <Stack gap={2}>
              <Text size="xs" c="dimmed" fw={700} tt="uppercase">Analysis Target:</Text>
              <Group gap="xs">
                <IconMapPin size={14} color="#228be6" />
                <Text size="sm" c="blue.4" fw={500}>
                  {selectedLocation.lat.toFixed(5)}, {selectedLocation.lng.toFixed(5)}
                </Text>
              </Group>
            </Stack>
          ) : (
            <Text size="sm" c="dimmed" fs="italic">
              {selectedLocation?.isSearch
                ? "Now click on the map to place a pin."
                : "Search for a city or click on the map."}
            </Text>
          )}
        </div>

        <Divider my="lg" color="gray.8" />

        <AnalysisOptions values={filters} onChange={updateFilter} />

        <Stack gap="sm" mt="30px">
          {isSingleError && (
            <div className={styles.errorBox}>
              <IconAlertCircle size={16} />
              <Text size="xs" fw={600}>{isSingleError}</Text>
            </div>
          )}
          <Button
            leftSection={<IconPlayerPlay size={18} />}
            onClick={() => runAnalysis(selectedLocation, filters)}
            disabled={!isLocationReady || isSingleLoading}
            loading={isSingleLoading}
            size="md"
            fullWidth
            className={styles.runAnalysisButton}
          >
            Run Analysis
          </Button>

          <Button
            variant="outline"
            leftSection={<IconArrowsLeftRight size={18} />}
            onClick={handleOpenCompare}
            disabled={!isLocationReady || isAdvancedLoading}
            size="md"
            fullWidth
            className={styles.advancedAnalysisButton}
          >
            Advanced Analysis
          </Button>
        </Stack>

        {analysisResult && (
          <LayerSelector
            layersConfig={layersConfig}
            onLayersChange={(key, val) => onLayersChange(prev => ({ ...prev, [key]: val }))}
          />
        )}
      </div>

      <AdvancedAnalysisModal
        opened={compareOpened}
        onClose={() => setCompareOpened(false)}
        onRunCompare={(mA, mB) => runComparison(selectedLocation, filters, mA, mB)}
        isLoading={isAdvancedLoading}
        isError={isAdvancedError}
        results={comparisonData}
      />
    </div>
  );
}

export default AnalysisPanel
