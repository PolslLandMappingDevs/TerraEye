import { Select, Stack, Divider } from '@mantine/core';
import styles from './LayerSelector.module.css';

const layerOptions = [
  {
    group: 'Main Images',
    items: [
      { label: 'None', value: 'none' },
      { label: 'Satellite (RGB)', value: 'rgb' },
      { label: 'Raw Output', value: 'raw_segmentation' },
      { label: 'Output (with Spectral Indices)', value: 'image' },
    ]
  },
  {
    group: 'Spectral Indices (Masks)',
    items: [
      { label: 'Vegetation (NDVI)', value: 'masks.vegetation_ndvi' },
      { label: 'Water (NDWI)', value: 'masks.water_ndwi' },
      { label: 'Buildings (NDBI)', value: 'masks.buildings_ndbi' },
      { label: 'Bare Soil (BSI)', value: 'masks.baresoil_bsi' }
    ]
  }
];

export function LayerSelector({ layersConfig, onLayersChange }) {
  return (
    <div className={styles.layersBox}>
      <Divider
        my="lg"
        color="gray.8"
        label="Map Layers"
        labelPosition="center"
        styles={{ label: { color: '#868e96', fontWeight: 600 } }}
      />
      <Stack gap="md">
        <Select
          label="Base Layer"
          data={layerOptions}
          value={layersConfig.firstLayer}
          onChange={(val) => onLayersChange('firstLayer', val)}
          classNames={{
            input: styles.selectInput,
            dropdown: styles.selectDropdown,
            option: styles.selectOption,
          }}
          comboboxProps={{ withinPortal: true, zIndex: 11000 }}
          allowDeselect={false}
        />

        <Select
          label="Overlay Layer"
          data={layerOptions}
          value={layersConfig.secondLayer}
          onChange={(val) => onLayersChange('secondLayer', val)}
          classNames={{
            input: styles.selectInput,
            dropdown: styles.selectDropdown,
            option: styles.selectOption,
          }}
          comboboxProps={{ withinPortal: true, zIndex: 11000 }}
          allowDeselect={false}
        />
      </Stack>
    </div>
  );
}
