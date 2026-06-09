export type IndexingProvider = 'deepseek' | 'google_studio'
export type IndexedProvider = IndexingProvider | 'legacy' | 'unknown'

export const INDEXING_PROVIDER_STORAGE_KEY = 'indexing-provider'
export const DEFAULT_INDEXING_PROVIDER: IndexingProvider = 'deepseek'

export function isIndexingProvider(value: string): value is IndexingProvider {
  return value === 'deepseek' || value === 'google_studio'
}

export function getStoredIndexingProvider(storage: Storage = window.localStorage): IndexingProvider {
  const stored = storage.getItem(INDEXING_PROVIDER_STORAGE_KEY)
  return stored && isIndexingProvider(stored) ? stored : DEFAULT_INDEXING_PROVIDER
}

export function setStoredIndexingProvider(provider: IndexingProvider, storage: Storage = window.localStorage) {
  storage.setItem(INDEXING_PROVIDER_STORAGE_KEY, provider)
}

export function getIndexingProviderLabel(provider: IndexedProvider): string {
  switch (provider) {
    case 'deepseek':
      return 'DeepSeek'
    case 'google_studio':
      return 'Google Studio'
    case 'legacy':
      return 'Legacy'
    default:
      return 'Unknown'
  }
}
