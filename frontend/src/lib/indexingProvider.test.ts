import { describe, expect, it } from 'vitest'

import {
  DEFAULT_INDEXING_PROVIDER,
  getIndexingProviderLabel,
  getStoredIndexingProvider,
  isIndexingProvider,
  setStoredIndexingProvider,
} from './indexingProvider'

describe('indexingProvider helpers', () => {
  it('falls back to deepseek for missing or invalid storage values', () => {
    const storage = {
      getItem: () => 'not-real',
      setItem: () => undefined,
    }

    expect(getStoredIndexingProvider(storage as Storage)).toBe(DEFAULT_INDEXING_PROVIDER)
  })

  it('persists the selected provider', () => {
    let stored = ''
    const storage = {
      getItem: () => stored,
      setItem: (_key: string, value: string) => {
        stored = value
      },
    }

    setStoredIndexingProvider('google_studio', storage as Storage)

    expect(stored).toBe('google_studio')
    expect(getStoredIndexingProvider(storage as Storage)).toBe('google_studio')
  })

  it('returns friendly labels for supported and legacy providers', () => {
    expect(isIndexingProvider('deepseek')).toBe(true)
    expect(isIndexingProvider('google_studio')).toBe(true)
    expect(isIndexingProvider('legacy')).toBe(false)
    expect(getIndexingProviderLabel('deepseek')).toBe('DeepSeek')
    expect(getIndexingProviderLabel('google_studio')).toBe('Google Studio')
    expect(getIndexingProviderLabel('legacy')).toBe('Legacy')
    expect(getIndexingProviderLabel('unknown')).toBe('Unknown')
  })
})
