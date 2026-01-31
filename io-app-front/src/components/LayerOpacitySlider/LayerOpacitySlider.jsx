import { Stack, Group, Text, Slider } from '@mantine/core';
import styles from './LayerOpacitySlider.module.css';

export default function LayerOpacitySlider({ opacity, onChange }) {
  return (
    <div className={styles.floatingContainer}>
      <div className={styles.glassPanel}>
        <Stack gap={6}>
          <Group justify="space-between" align="center">
            <Text size="xs" fw={700} c="dimmed" tt="uppercase" style={{ letterSpacing: '1px' }}>
              Overlay Opacity
            </Text>
            <Text size="md" c="blue.4" fw={700}>
              {Math.round(opacity * 100)}%
            </Text>
          </Group>

          <Slider
            value={opacity * 100}
            onChange={(val) => onChange(val / 100)}
            min={0}
            max={100}
            label={null}
            color="blue"
            size="xl"
            thumbSize={24}
            className={styles.sliderWidth}
            styles={{
              track: { backgroundColor: 'rgba(255,255,255,0.1)' }
            }}
          />

          <Group justify="space-between">
            <Text size="10px" c="dimmed">Base Layer</Text>
            <Text size="10px" c="dimmed">Overlay Layer</Text>
          </Group>
        </Stack>
      </div>
    </div>
  );
}
