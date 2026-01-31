import { Slider, Select, NumberInput, Text, Stack } from '@mantine/core'
import classes from './AnalysisOptions.module.css'

export default function AnalysisOptions({ values, onChange }) {

  const handleChange = (key, val) => {
    onChange(key, val);
  };

  return (
    <Stack gap="md">
      <div>
        <div className={classes.header}>
          <Text size="sm" fw={600} c="dimmed">Max Cloud Cover</Text>
          <Text size="sm" fw={700} c="blue">{values.cloudCover}%</Text>
        </div>
        <Slider
          value={values.cloudCover}
          onChange={(val) => handleChange('cloudCover', val)}
          min={0}
          max={100}
          step={1}
          label={null}
          size="sm"
          classNames={{ track: classes.sliderTrack, thumb: classes.sliderThumb }}
        />
      </div>

      <div>
        <div className={classes.header}>
          <Text size="sm" fw={600} c="dimmed">Analysis Radius</Text>
          <Text size="sm" fw={700} c="blue">{values.radius} km</Text>
        </div>
        <Slider
          value={values.radius}
          onChange={(val) => handleChange('radius', val)}
          min={1}
          max={10}
          step={1}
          label={null}
          size="sm"
          classNames={{ track: classes.sliderTrack, thumb: classes.sliderThumb }}
        />
      </div>

      <NumberInput
        label="Time Range (days back)"
        value={values.daysRange}
        onChange={(val) => handleChange('daysRange', val)}
        min={1}
        max={365}
        size="sm"
        classNames={{ label: classes.inputLabel, input: classes.inputField }}
      />

      <Select
        label="Model Version"
        allowDeselect={false}
        data={[
          { value: 'terramind_v1_tiny_generate', label: 'Terramind v1 Tiny' },
          { value: 'terramind_v1_small_generate', label: 'Terramind v1 Small' },
          { value: 'terramind_v1_base_generate', label: 'Terramind v1 Base' },
          { value: 'terramind_v1_large_generate', label: 'Terramind v1 Large' },
        ]}
        value={values.model}
        onChange={(val) => handleChange('model', val)}
        size="sm"
        classNames={{
          label: classes.inputLabel,
          input: classes.inputField,
          dropdown: classes.inputField,
          option: classes.inputField
        }}
      />
    </Stack>
  );
}
