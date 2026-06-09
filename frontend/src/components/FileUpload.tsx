import { useState } from 'react'
import { Upload, File, X, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react'
import client from '../api/client'
import { IndexingProvider, getIndexingProviderLabel, DEFAULT_INDEXING_PROVIDER } from '../lib/indexingProvider'

interface FileUploadProps {
  onSuccess?: () => void
  onUploadStateChange?: (isUploading: boolean) => void
  selectedProvider?: IndexingProvider
}

export default function FileUpload({
  onSuccess,
  onUploadStateChange,
  selectedProvider = DEFAULT_INDEXING_PROVIDER,
}: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null)
  const [status, setStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle')
  const [message, setMessage] = useState('')

  const providerLabel = getIndexingProviderLabel(selectedProvider)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setStatus('idle')
      setMessage('')
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setStatus('uploading')
    onUploadStateChange?.(true)
    const formData = new FormData()
    formData.append('file', file)
    formData.append('provider', selectedProvider)

    try {
      const response = await client.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      setStatus('success')
      setMessage(response.data.message)
      setFile(null)
      if (onSuccess) onSuccess()
    } catch (error: any) {
      setStatus('error')
      setMessage(error.response?.data?.detail || 'Upload failed')
    } finally {
      onUploadStateChange?.(false)
    }
  }

  return (
    <div className="space-y-4">
      {!file ? (
        <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-xl cursor-pointer hover:bg-muted/50 transition-colors group">
          <div className="flex flex-col items-center justify-center pt-5 pb-6">
            <Upload className="w-6 h-6 text-muted-foreground group-hover:text-primary transition-colors mb-2" />
            <p className="text-xs text-muted-foreground font-medium">Click to upload legal PDFs</p>
          </div>
          <input type="file" className="hidden" accept=".pdf,.txt" onChange={handleFileChange} />
        </label>
      ) : (
        <div className="p-3 border rounded-xl bg-card flex items-center justify-between animate-in zoom-in-95">
          <div className="flex items-center gap-3 overflow-hidden">
            <div className="p-2 bg-primary/10 rounded-lg">
              <File className="w-4 h-4 text-primary" />
            </div>
            <span className="text-xs font-medium truncate">{file.name}</span>
          </div>
          <button onClick={() => setFile(null)} className="p-1 hover:bg-muted rounded-full">
            <X className="w-4 h-4 text-muted-foreground" />
          </button>
        </div>
      )}

      {file && status === 'idle' && (
        <>
          <p className="text-[10px] text-muted-foreground px-1">{`Will build with: ${providerLabel}`}</p>
          <button
            onClick={handleUpload}
            className="w-full py-2 bg-primary text-primary-foreground rounded-lg text-xs font-bold hover:opacity-90 transition-all shadow-lg shadow-primary/20"
          >
            Begin Indexing
          </button>
        </>
      )}

      {status === 'uploading' && (
        <div className="flex items-center justify-center gap-2 text-xs font-medium text-muted-foreground py-2">
          <Loader2 className="w-3 h-3 animate-spin" />
          <span>{`Building knowledge graph with ${providerLabel}...`}</span>
        </div>
      )}

      {status === 'success' && (
        <div className="flex items-center gap-2 text-[10px] font-bold text-emerald-600 bg-emerald-50 p-2 rounded-lg border border-emerald-100">
          <CheckCircle2 className="w-3 h-3" />
          <span>{message}</span>
        </div>
      )}

      {status === 'error' && (
        <div className="flex items-center gap-2 text-[10px] font-bold text-red-600 bg-red-50 p-2 rounded-lg border border-red-100">
          <AlertCircle className="w-3 h-3" />
          <span>{message}</span>
        </div>
      )}
    </div>
  )
}
