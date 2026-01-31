import { useState } from 'react'
import { useGeocoding } from '../../hooks/useGeocoding'
import { Autocomplete, Loader, CloseButton } from '@mantine/core'
import { IconSearch } from '@tabler/icons-react'
import styles from './SearchBar.module.css'

function SearchBar({ onLocationSelect }) {
  const [inputValue, setInputValue] = useState('');
  const { locations, isQueryLoading, setLocations } = useGeocoding(inputValue);

  const changeHandler = (value) => {
    setInputValue(value);

    if (value.length < 3) {
      setLocations([]);
    }
  };

  const selectHandler = (selectedValue) => {
    const selected = locations.find((loc) => loc.value === selectedValue);

    if (selected) {
      setInputValue(selected.value);

      if (onLocationSelect) {
        onLocationSelect({
          lat: selected.lat,
          lng: selected.lng,
          label: selected.value,
          isSearch: true
        });
      }
    }
  };

  const clearHandler = () => {
    setInputValue('');
    setLocations([]);
  }

  return (
    <Autocomplete
      classNames={{
        input: styles.searchInput,
        dropdown: styles.searchDropdown,
        option: styles.searchOption,
      }}
      placeholder='Search for a city...'
      value={ inputValue }
      onChange={ changeHandler }
      onOptionSubmit={ selectHandler }
      data={ locations.map((loc) => loc.value) }
      leftSection={ <IconSearch size={16} /> }
      rightSectionPointerEvents='all'
      rightSection={
        isQueryLoading ? (
          <Loader size='xs'/>
        ) : inputValue ? (
          <CloseButton onClick={ clearHandler } />
        ) : null
      }
      comboboxProps={{ zIndex: 10000, withinPortal: true }}
    />
  )
}

export default SearchBar
