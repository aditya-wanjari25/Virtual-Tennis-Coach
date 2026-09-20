import { useRef, useState } from 'react'

interface UploadZoneProps {
  onFileSelected: (file: File) => void
  disabled?: boolean
}

const TIPS = [
  { icon: '📹', label: 'Film from behind', hint: 'Baseline view, full body in frame' },
  { icon: '🎾', label: 'A few swings', hint: 'Three or four groundstrokes is plenty' },
  { icon: '⏱️', label: 'Keep it short', hint: 'Five to fifteen seconds' },
]

export function UploadZone({ onFileSelected, disabled }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  function handleFiles(files: FileList | null) {
    const file = files?.[0]
    if (file) onFileSelected(file)
  }

  return (
    <div className="flex flex-col gap-5">
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
        role="button"
        tabIndex={disabled ? -1 : 0}
        onKeyDown={(e) => {
          if (!disabled && (e.key === 'Enter' || e.key === ' ')) {
            e.preventDefault()
            inputRef.current?.click()
          }
        }}
        className={`group relative overflow-hidden rounded-3xl border-2 border-dashed px-5 py-12 text-center transition-all duration-300 sm:px-8 sm:py-16
          ${disabled ? 'cursor-not-allowed border-ink-700 opacity-50' : 'cursor-pointer'}
          ${
            isDragging
              ? 'scale-[1.01] border-court-400 bg-court-400/10'
              : 'border-ink-600 bg-ink-850/40 hover:border-court-500 hover:bg-ink-850'
          }`}
      >
        {/* Accent glow, only on hover/drag — keeps the resting state calm. */}
        <div
          className={`pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,var(--color-court-500)_0%,transparent_65%)] transition-opacity duration-500
            ${isDragging ? 'opacity-[0.14]' : 'opacity-0 group-hover:opacity-[0.07]'}`}
        />

        <div className="relative flex flex-col items-center gap-4">
          <div
            className={`flex h-16 w-16 items-center justify-center rounded-2xl text-3xl transition-transform duration-300
              ${isDragging ? 'scale-110 bg-court-400/20' : 'bg-ink-800 group-hover:scale-105'}`}
          >
            🎾
          </div>
          <div>
            <p className="text-lg font-semibold tracking-tight text-neutral-100">
              {isDragging ? 'Drop it here' : 'Drop your swing video'}
            </p>
            <p className="mt-1.5 text-sm text-neutral-500">
              or <span className="text-court-400 underline decoration-court-400/40 underline-offset-4">browse files</span>
              <span className="mx-2 text-ink-600">·</span>
              MP4, MOV, WebM
            </p>
          </div>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          className="hidden"
          disabled={disabled}
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        {TIPS.map((t) => (
          <div key={t.label} className="rounded-2xl border border-ink-700/60 bg-ink-850/40 px-4 py-3.5">
            <div className="flex items-center gap-2">
              <span className="text-sm">{t.icon}</span>
              <span className="text-sm font-medium text-neutral-200">{t.label}</span>
            </div>
            <p className="mt-1 text-xs leading-relaxed text-neutral-500">{t.hint}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
