import { Text, Stack, Group } from '@mantine/core';
import { getMetadata } from '../../utils/mapUtils';
import styles from './MapLegend.module.css';

export default function MapLegend({ analysisResult }) {
  if (!analysisResult || !analysisResult.statistics) return null;

  const stats = Object.values(analysisResult.statistics)
    .filter(item => item.percentage > 0)
    .sort((a, b) => b.percentage - a.percentage);

  if (stats.length === 0) return null;

  return (
    <div className={styles.container}>
      <div className={styles.glassPanel}>
        <Text
          size="xs"
          fw={700}
          c="dimmed"
          tt="uppercase"
          mb="sm"
          style={{ letterSpacing: '1px' }}
        >
          Legend
        </Text>

        <Stack gap={8}>
          {stats.map((item) => {
            const { label, color } = getMetadata(item.name);
            return (
              <Group key={item.name} justify="space-between" align="center" wrap="nowrap">
                <div className={styles.itemLabel}>
                  <span
                    className={styles.colorBadge}
                    style={{ background: color }}
                  />
                  <Text size="xs" c="gray.3" style={{ lineHeight: 1 }}>
                    {label}
                  </Text>
                </div>

                <Text size="xs" fw={700} c="white">
                  {item.percentage.toFixed(1)}%
                </Text>
              </Group>
            );
          })}
        </Stack>
      </div>
    </div>
  );
}
