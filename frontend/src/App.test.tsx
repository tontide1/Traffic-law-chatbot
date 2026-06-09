import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, afterEach } from 'vitest'

import App from './App'
import client from './api/client'

vi.mock('./api/client', () => ({
  default: {
    get: vi.fn().mockResolvedValue({
      data: [
        { id: '1', source: 'a.pdf', status: 'processed', indexed_provider: 'google_studio' },
        { id: '2', source: 'b.pdf', status: 'processed', indexed_provider: 'legacy' },
      ],
    }),
  },
}))

vi.mock('./components/ChatInterface', () => ({
  default: () => <div>Chat Interface</div>,
}))

describe('App', () => {
  afterEach(() => {
    window.localStorage.clear()
    vi.clearAllMocks()
  })

  it('restores the selected provider from localStorage', async () => {
    window.localStorage.setItem('indexing-provider', 'google_studio')

    render(<App />)

    expect(screen.getByRole('button', { name: 'Google Studio' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('renders provider badges from the documents API', async () => {
    render(<App />)

    await waitFor(() => expect(client.get).toHaveBeenCalledWith('/documents'))

    expect(screen.getAllByText('Google Studio')[0]).toBeInTheDocument()
    expect(screen.getByText('Legacy')).toBeInTheDocument()
  })
})
