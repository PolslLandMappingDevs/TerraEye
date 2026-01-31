export const calculateBounds = (centerLat, centerLng, radiusKm) => {
  const R = 6378137;
  const scaleFactor = 1.0 / Math.cos(centerLat * Math.PI / 180);
  const mercatorExtent = (radiusKm * 1000) * scaleFactor;
  const x = centerLng * (Math.PI / 180) * R;
  const latRad = centerLat * (Math.PI / 180);
  const y = Math.log(Math.tan((Math.PI / 4) + (latRad / 2))) * R;

  const minX = x - mercatorExtent;
  const maxX = x + mercatorExtent;
  const minY = y - mercatorExtent;
  const maxY = y + mercatorExtent;

  const toLng = (mercX) => (mercX / R) * (180 / Math.PI);
  const toLat = (mercY) => (2 * Math.atan(Math.exp(mercY / R)) - Math.PI / 2) * (180 / Math.PI);

  return [
    [toLng(minX), toLat(maxY)], [toLng(maxX), toLat(maxY)],
    [toLng(maxX), toLat(minY)], [toLng(minX), toLat(minY)]
  ];
};

export const getImageByPath = (result, path) => {
  if (!result || !path) return null;
  if (path.startsWith('masks.')) {
    const maskKey = path.split('.')[1];
    return result.masks ? result.masks[maskKey] : null;
  }
  return result[path];
};

export const CLASS_METADATA = {
  'Drzewa / Las': { label: 'Trees / Forest', color: 'rgb(0, 100, 0)' },
  'Zarośla': { label: 'Shrubland', color: 'rgb(255, 187, 34)' },
  'Trawa / Łąki': { label: 'Grassland', color: 'rgb(255, 255, 76)' },
  'Uprawy rolne': { label: 'Crops', color: 'rgb(240, 150, 255)' },
  'Zabudowa': { label: 'Built area', color: 'rgb(250, 0, 0)' },
  'Goły grunt': { label: 'Bare Ground', color: 'rgb(180, 180, 180)' },
  'Śnieg i lód': { label: 'Snow & Ice', color: 'rgb(240, 240, 240)' },
  'Woda': { label: 'Water', color: 'rgb(0, 100, 200)' },
  'Tereny podmokłe': { label: 'Flooded vegetation', color: 'rgb(0, 150, 160)' },
  'Namorzyny': { label: 'Mangroves', color: 'rgb(0, 207, 117)' },
  'Mchy i porosty': { label: 'Moss & Lichen', color: 'rgb(250, 230, 160)' },
  'Brak danych': { label: 'No Data', color: 'rgb(100, 100, 100)' },
  // English labels returned by backend
  'No data': { label: 'No data', color: 'rgb(100, 100, 100)' },
  'Trees / Forest': { label: 'Trees / Forest', color: 'rgb(0, 100, 0)' },
  'Shrubs': { label: 'Shrubs', color: 'rgb(255, 187, 34)' },
  'Grass / Meadows': { label: 'Grass / Meadows', color: 'rgb(255, 255, 76)' },
  'Cultivated crops': { label: 'Cultivated crops', color: 'rgb(240, 150, 255)' },
  'Buildings': { label: 'Buildings', color: 'rgb(250, 0, 0)' },
  'Bare ground': { label: 'Bare ground', color: 'rgb(180, 180, 180)' },
  'Snow and ice': { label: 'Snow and ice', color: 'rgb(240, 240, 240)' },
  'Water': { label: 'Water', color: 'rgb(0, 100, 200)' },
  'Wetlands': { label: 'Wetlands', color: 'rgb(0, 150, 160)' },
  'Mangroves': { label: 'Mangroves', color: 'rgb(0, 207, 117)' },
  'Lichens and moss': { label: 'Lichens and moss', color: 'rgb(250, 230, 160)' }
};

export const getMetadata = (className) => {
  return CLASS_METADATA[className] || { label: className, color: '#666' };
};
