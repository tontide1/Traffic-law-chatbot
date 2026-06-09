import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, afterEach } from 'vitest'

import FileUpload from './FileUpload'
import client from '../api/client'

vi.mock('../api/client', () => ({
  default: {
    post: vi.fn().mockResolvedValue({ data: { message: 'ok' } }),
  },
}))

describe('FileUpload', () => {
  afterEach(() => {
    vi.clearAllMocks()
  })

  it('sends the selected provider in FormData', async () => {
    render(<FileUpload selectedProvider="google_studio" />)

    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    const file = new File(['hello'], 'law.txt', { type: 'text/plain' })

    fireEvent.change(input, { target: { files: [file] } })
    fireEvent.click(screen.getByRole('button', { name: /begin indexing/i }))

    await waitFor(() => expect(client.post).toHaveBeenCalledTimes(1))

    const [, formData] = vi.mocked(client.post).mock.calls[0]
    expect((formData as FormData).get('provider')).toBe('google_studio')
  })

  it('shows provider-aware helper and loading copy', async () => {
    let resolveRequest: ((value: { data: { message: string } }) => void) | undefined
    vi.mocked(client.post).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveRequest = resolve
        }) as Promise<{ data: { message: string } }>
    )

    render(<FileUpload selectedProvider="deepseek" />)

    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    const file = new File(['hello'], 'law.txt', { type: 'text/plain' })

    fireEvent.change(input, { target: { files: [file] } })

    expect(screen.getByText('Will build with: DeepSeek')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /begin indexing/i }))

    expect(screen.getByText('Building knowledge graph with DeepSeek...')).toBeInTheDocument()

    resolveRequest?.({ data: { message: 'ok' } })
    await waitFor(() => expect(screen.getByText('ok')).toBeInTheDocument())
  })
})
