import { describe, expect, it } from 'vitest'

import { DEFAULT_API_BASE_URL, getApiBaseUrl } from './apiBaseUrl'

describe('getApiBaseUrl', () => {
  it('prefers VITE_API_BASE_URL', () => {
    expect(
      getApiBaseUrl({
        VITE_API_BASE_URL: 'http://backend.example.com/api',
        VITE_API_URL: 'http://old.example.com/api',
      })
    ).toBe('http://backend.example.com/api')
  })

  it('falls back to VITE_API_URL for compatibility', () => {
    expect(
      getApiBaseUrl({
        VITE_API_URL: 'http://backend.example.com/api',
      })
    ).toBe('http://backend.example.com/api')
  })

  it('falls back to localhost when no env is set', () => {
    expect(getApiBaseUrl({})).toBe(DEFAULT_API_BASE_URL)
  })
})
