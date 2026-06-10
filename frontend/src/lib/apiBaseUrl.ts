type ApiEnv = {
  VITE_API_BASE_URL?: string
  VITE_API_URL?: string
}

export const DEFAULT_API_BASE_URL = 'http://localhost:8000/api'

export function getApiBaseUrl(env: ApiEnv = (import.meta as any).env): string {
  return env.VITE_API_BASE_URL || env.VITE_API_URL || DEFAULT_API_BASE_URL
}
