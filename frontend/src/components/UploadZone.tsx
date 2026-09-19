import { useRef, useState } from 'react'

interface UploadZoneProps {
  onFileSelected: (file: File) => void
  disabled?: boolean
}

export function UploadZone({ onFileSelected, disabled }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  function handleFiles(files: FileList | null) {
    const file = files?.[0]
    if (file) onFileSelected(file)
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        if (!disabled) setIsDragging(true)
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setIsDragging(false)
        if (!disabled) handleFiles(e.dataTransfer.files)
      }}
      onClick={() => !disabled && inputRef.current?.click()}
      className={`flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-12 text-center transition-colors
        ${disabled ? 'cursor-not-allowed opacity-50 border-neutral-300 dark:border-neutral-700' : 'cursor-pointer'}
        ${isDragging ? 'border-lime-500 bg-lime-500/10' : 'border-neutral-300 dark:border-neutral-700 hover:border-lime-500'}`}
    >
      <div className="text-4xl">🎾</div>
      <p className="text-lg font-medium text-neutral-900 dark:text-neutral-100">
        Drop your groundstroke video here
      </p>
      <p className="text-sm text-neutral-500 dark:text-neutral-400">
        or click to browse — filmed from behind the baseline works best
      </p>
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        className="hidden"
        disabled={disabled}
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  )
}
