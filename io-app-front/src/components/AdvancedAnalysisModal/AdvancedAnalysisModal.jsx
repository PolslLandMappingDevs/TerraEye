import { Modal, Select, Button, Group, Stack, Text, Center } from '@mantine/core'
import { IconAlertCircle } from '@tabler/icons-react'
import { useState } from 'react'
import classes from './AdvancedAnalysisModal.module.css'

const CLASS_METADATA = {
  'Drzewa / Las': 'Trees / Forest',
  'Zarośla': 'Shrubland',
  'Trawa / Łąki': 'Grassland',
  'Uprawy rolne': 'Crops',
  'Zabudowa': 'Built area',
  'Goły grunt': 'Bare Ground',
  'Śnieg i lód': 'Snow & Ice',
  'Woda': 'Water',
  'Tereny podmokłe': 'Flooded vegetation',
  'Namorzyny': 'Mangroves',
  'Mchy i porosty': 'Moss & Lichen',
  'Brak danych': 'No Data'
};

const MASK_INFO = {
  water_ndwi: { name: "NDWI - Water", description: "Open water bodies", color: "#1971c2", formula: "(G - NIR)/(G + NIR)" },
  water_mndwi: { name: "MNDWI - Urban Water", description: "Water in urban areas", color: "#1864ab", formula: "(G - SWIR)/(G + SWIR)" },
  water_awei: { name: "AWEI - Automated", description: "Shadow suppression", color: "#0b7285", formula: "4(G-SWIR)-(0.25NIR+2.75SWIR2)" },
  vegetation_ndvi: { name: "NDVI - Vegetation", description: "Plant health", color: "#2f9e44", formula: "(NIR - R)/(NIR + R)" },
  vegetation_evi: { name: "EVI - Enhanced Veg", description: "Dense canopy", color: "#5c940d", formula: "2.5(NIR-R)/(NIR+6R-7.5B+1)" },
  buildings_ndbi: { name: "NDBI - Built-up", description: "Urban structures", color: "#c92a2a", formula: "(SWIR - NIR)/(SWIR + NIR)" },
  baresoil_bsi: { name: "BSI - Bare Soil", description: "Soil detection", color: "#d9480f", formula: "((SWIR+R)-(NIR+B))/((SWIR+R)+(NIR+B))" }
};

export default function AdvancedAnalysisModal({ opened, onClose, onRunCompare, isLoading, isError, results }) {
  const [modelA, setModelA] = useState('terramind_v1_small_generate');
  const [modelB, setModelB] = useState('terramind_v1_large_generate');

  const modelOptions = [
    { value: 'terramind_v1_tiny_generate', label: 'Terramind v1 Tiny' },
    { value: 'terramind_v1_small_generate', label: 'Terramind v1 Small' },
    { value: 'terramind_v1_base_generate', label: 'Terramind v1 Base' },
    { value: 'terramind_v1_large_generate', label: 'Terramind v1 Large' },
  ];

  const getModelLabel = (model) => {
    const parts = model.split('_');
    return parts[2] ? parts[2].toUpperCase() + " Model" : model;
  };

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title="ADVANCED ANALYSIS - Quantitative comparison of TerraMind models"
      size="100%"
      centered
      classNames={{ content: classes.modalContent, header: classes.modalHeader, body: classes.modalBody }}
      overlayProps={{ backgroundOpacity: 0.85, blur: 12 }}
    >
      <Stack gap="xl" h="100%">
        {!results && (
          <Center h={500}>
            <Stack align="center" gap="xl" w="100%" maw={600}>
              <Text size="lg" fw={500} c="dimmed" lts={1}>Select models to compare</Text>
              <Group grow w="100%">
                <Select label="Model A" data={modelOptions} allowDeselect={false} value={modelA} onChange={setModelA} disabled={isLoading} />
                <Select label="Model B" data={modelOptions} allowDeselect={false} value={modelB} onChange={setModelB} disabled={isLoading} />
              </Group>
              <Button fullWidth size="lg" color="blue" onClick={() => onRunCompare(modelA, modelB)} loading={isLoading} className={classes.startAnalysisButton}>
                Start Analysis
              </Button>
              {isError && (
                <div className={classes.errorBox}>
                  <IconAlertCircle size={16} />
                  <Text size="xs" fw={600}>{isError}</Text>
                </div>
              )}
            </Stack>
          </Center>
        )}

        {results && !isLoading && (
          <div className={classes.dualLayout}>

            <div className={classes.modelColumn}>
              <div className={classes.columnHeader}>{getModelLabel(modelA)}</div>

              <div className={classes.imagesGrid}>
                <ImageCard
                  src={results.modelA.rgb}
                  title="Satellite RGB (Model Input)"
                  subtitle=""
                  borderColor="#1971c2"
                />
                <ImageCard
                  src={results.modelA.raw_segmentation}
                  title="Raw Model Output"
                  subtitle=""
                  borderColor="#f08c00"
                />
                <ImageCard
                  src={results.modelA.image}
                  title="Refined Output (With Spectral Indices)"
                  subtitle=""
                  borderColor="#2f9e44"
                />
              </div>
            </div>

            <div className={classes.modelColumn}>
              <div className={classes.columnHeader}>{getModelLabel(modelB)}</div>

              <div className={classes.imagesGrid}>
                <ImageCard
                  src={results.modelB.rgb}
                  title="Satellite RGB (Model Input)"
                  subtitle=""
                  borderColor="#1971c2"
                />
                <ImageCard
                  src={results.modelB.raw_segmentation}
                  title="Raw Model Output"
                  subtitle=""
                  borderColor="#f08c00"
                />
                <ImageCard
                  src={results.modelB.image}
                  title="Refined Output (With Spectral Indices)"
                  subtitle=""
                  borderColor="#2f9e44"
                />
              </div>
            </div>

            <div className={classes.masksSection}>
              <div className={classes.masksHeader}>Spectral Indices Masks</div>
              <div className={classes.masksGrid}>
                {Object.entries(results.modelA.masks).map(([key, src]) => {
                  const info = MASK_INFO[key] || { name: key, color: '#444', formula: 'N/A' };
                  return (
                    <div key={key} className={classes.maskCard} style={{ borderColor: info.color }}>
                      <div className={classes.maskTitleBar} style={{ backgroundColor: info.color }}>
                        {info.name}
                      </div>
                      <img src={src} className={classes.maskImg} alt={info.name} />
                      <div className={classes.maskInfo}>
                        <div className={classes.maskDesc}>{info.description}</div>
                        <div className={classes.maskFormula}>{info.formula}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {results.metrics && (
              <div className={classes.metricsSection}>

                {results.metrics.raw && (
                  <div className={classes.metricsSubsection}>
                    <div className={classes.metricsHeader}>Metrics - Raw Segmentation (without Spectral Indices)</div>

                    <div className={classes.mainMetricsGrid}>
                      <MetricCard
                        label="Pixel Accuracy"
                        value={results.metrics.raw.accuracy?.toFixed(2)}
                        unit="%"
                        color="#1971c2"
                      />
                      <MetricCard
                        label="mIoU"
                        value={results.metrics.raw.miou?.toFixed(2)}
                        unit="%"
                        color="#2f9e44"
                      />
                      <MetricCard
                        label="Frequency Weighted IoU"
                        value={results.metrics.raw.fw_iou?.toFixed(2)}
                        unit="%"
                        color="#f08c00"
                      />
                      <MetricCard
                        label="Dice Score"
                        value={results.metrics.raw.dice?.toFixed(2)}
                        unit="%"
                        color="#c92a2a"
                      />
                      <MetricCard
                        label="Mean Precision"
                        value={results.metrics.raw.mean_precision?.toFixed(2)}
                        unit="%"
                        color="#7950f2"
                      />
                      <MetricCard
                        label="Mean Recall"
                        value={results.metrics.raw.mean_recall?.toFixed(2)}
                        unit="%"
                        color="#15aabf"
                      />
                    </div>

                    {results.metrics.raw.class_details && Object.keys(results.metrics.raw.class_details).length > 0 && (
                      <div className={classes.classDetailsSection}>
                        <div className={classes.classDetailsHeader}>Per-Class Metrics</div>
                        <div className={classes.classDetailsGrid}>
                          {Object.entries(results.metrics.raw.class_details).map(([className, metrics]) => (
                            <div key={className} className={classes.classMetricCard}>
                              <div className={classes.classMetricCardTitle}>{CLASS_METADATA[className] || className}</div>
                              <div className={classes.classMetricValues}>
                                <div className={classes.classMetricRow}>
                                  <span>IoU:</span>
                                  <span className={classes.metricValue}>{metrics.iou?.toFixed(2)}%</span>
                                </div>
                                <div className={classes.classMetricRow}>
                                  <span>Precision:</span>
                                  <span className={classes.metricValue}>{metrics.precision?.toFixed(2)}%</span>
                                </div>
                                <div className={classes.classMetricRow}>
                                  <span>Recall:</span>
                                  <span className={classes.metricValue}>{metrics.recall?.toFixed(2)}%</span>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {results.metrics.corrected && (
                  <div className={classes.metricsSubsection}>
                    <div className={classes.metricsHeader}>Metrics - Corrected Segmentation (with Spectral Indices)</div>

                    <div className={classes.mainMetricsGrid}>
                      <MetricCard
                        label="Pixel Accuracy"
                        value={results.metrics.corrected.accuracy?.toFixed(2)}
                        unit="%"
                        color="#1971c2"
                      />
                      <MetricCard
                        label="mIoU"
                        value={results.metrics.corrected.miou?.toFixed(2)}
                        unit="%"
                        color="#2f9e44"
                      />
                      <MetricCard
                        label="Frequency Weighted IoU"
                        value={results.metrics.corrected.fw_iou?.toFixed(2)}
                        unit="%"
                        color="#f08c00"
                      />
                      <MetricCard
                        label="Dice Score"
                        value={results.metrics.corrected.dice?.toFixed(2)}
                        unit="%"
                        color="#c92a2a"
                      />
                      <MetricCard
                        label="Mean Precision"
                        value={results.metrics.corrected.mean_precision?.toFixed(2)}
                        unit="%"
                        color="#7950f2"
                      />
                      <MetricCard
                        label="Mean Recall"
                        value={results.metrics.corrected.mean_recall?.toFixed(2)}
                        unit="%"
                        color="#15aabf"
                      />
                    </div>

                    {results.metrics.corrected.class_details && Object.keys(results.metrics.corrected.class_details).length > 0 && (
                      <div className={classes.classDetailsSection}>
                        <div className={classes.classDetailsHeader}>Per-Class Metrics</div>
                        <div className={classes.classDetailsGrid}>
                          {Object.entries(results.metrics.corrected.class_details).map(([className, metrics]) => (
                            <div key={className} className={classes.classMetricCard}>
                              <div className={classes.classMetricCardTitle}>{CLASS_METADATA[className] || className}</div>
                              <div className={classes.classMetricValues}>
                                <div className={classes.classMetricRow}>
                                  <span>IoU:</span>
                                  <span className={classes.metricValue}>{metrics.iou?.toFixed(2)}%</span>
                                </div>
                                <div className={classes.classMetricRow}>
                                  <span>Precision:</span>
                                  <span className={classes.metricValue}>{metrics.precision?.toFixed(2)}%</span>
                                </div>
                                <div className={classes.classMetricRow}>
                                  <span>Recall:</span>
                                  <span className={classes.metricValue}>{metrics.recall?.toFixed(2)}%</span>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

              </div>
            )}

          </div>
        )}
      </Stack>
    </Modal>
  );
}

function ImageCard({ src, title, subtitle, borderColor }) {
  return (
    <div className={classes.imageCard}>
      <div className={classes.cardTitle} style={{ color: borderColor }}>{title}</div>
      <div className={classes.cardFrame} style={{ borderColor: borderColor }}>
        <img src={src} className={classes.cardImg} alt={title} />
      </div>
      <div className={classes.cardSubtitle}>{subtitle}</div>
    </div>
  );
}

function MetricCard({ label, value, unit, color, highlight = false }) {
  return (
    <div className={classes.metricCard} style={{
      borderColor: color,
      backgroundColor: highlight ? 'rgba(27, 137, 23, 0.1)' : 'rgba(0, 0, 0, 0.2)'
    }}>
      <div className={classes.metricLabel}>{label}</div>
      <div className={classes.metricValueContainer} style={{ color: color }}>
        <span className={classes.metricValueText}>{value}</span>
        <span className={classes.metricUnit}>{unit}</span>
      </div>
    </div>
  );
}
