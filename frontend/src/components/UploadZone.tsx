import { useRef, useState } from 'react'
import { LIMITS } from '../api'

interface UploadZoneProps {
  onFileSelected: (file: File) => void
  disabled?: boolean
}

const TIPS = [
  { icon: '📹', label: 'Film from behind', hint: 'Baseline view, full body in frame' },
  { icon: '🎾', label: 'A few swings', hint: 'Three or four groundstrokes is plenty' },
  { icon: '⏱️', label: 'Keep it short', hint: 'Five to fifteen seconds' },
]

// Mirrors the server's ALLOWED_SUFFIXES, so the file picker offers exactly what
// the backend will accept rather than everything the OS calls a video.
const ACCEPT = '.mp4,.mov,.webm,.avi,.mpeg,.mpg,.3gp'

/** Read a video's duration in the browser, or null if we can't.
 *
 *  Fails open on purpose. Browsers can't decode every container the server
 *  accepts -- Chrome frequently can't read HEVC, which is exactly what iPhones
 *  record by default -- and a pre-check that blocks footage the server would
 *  have processed happily is worse than no pre-check at all. null means
 *  "couldn't tell, let the server decide".
 */
function readDuration(file: File): Promise<number | null> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file)
    const video = document.createElement('video')
    let settled = false

    const done = (value: number | null) => {
      if (settled) return
      settled = true
      URL.revokeObjectURL(url)
      resolve(value)
    }

    // Don't let a container the browser half-understands hang the picker.
    const timer = window.setTimeout(() => done(null), 5000)
    const finish = (value: number | null) => {
      window.clearTimeout(timer)
      done(value)
    }

    video.preload = 'metadata'
    video.onloadedmetadata = () => finish(Number.isFinite(video.duration) ? video.duration : null)
    video.onerror = () => finish(null)
    video.src = url
  })
}

export function UploadZone({ onFileSelected, disabled }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [rejection, setRejection] = useState<string | null>(null)
  const [checking, setChecking] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  // Catches the two rejections worth catching before an upload rather than
  // after one. The server re-checks everything (and checks codec, resolution
  // and frame count besides) -- this only saves the user a pointless wait.
  async function handleFiles(files: FileList | null) {
    const file = files?.[0]
    if (!file) return

    setRejection(null)

    const maxBytes = LIMITS.maxUploadMB * 1024 * 1024
    if (file.size > maxBytes) {
      const mb = Math.round(file.size / (1024 * 1024))
      setRejection(
        `That file is ${mb} MB — the limit is ${LIMITS.maxUploadMB} MB. A few seconds of footage is all we need.`,
      )
      return
    }

    setChecking(true)
    const duration = await readDuration(file)
    setChecking(false)

    if (duration !== null && duration > LIMITS.maxDurationS) {
      setRejection(
        `That clip is ${Math.round(duration)}s long — the limit is ${LIMITS.maxDurationS}s. Trim it to the few swings you want looked at.`,
      )
      return
    }

    onFileSelected(file)
  }

  const busy = disabled || checking

  return (
    <div className="flex flex-col gap-5">
      <div
        onDragOver={(e) => {
          e.preventDefault()
          if (!busy) setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setIsDragging(false)
          if (!busy) handleFiles(e.dataTransfer.files)
        }}
        onClick={() => !busy && inputRef.current?.click()}
        role="button"
        tabIndex={busy ? -1 : 0}
        onKeyDown={(e) => {
          if (!busy && (e.key === 'Enter' || e.key === ' ')) {
            e.preventDefault()
            inputRef.current?.click()
          }
        }}
        className={`group relative overflow-hidden rounded-3xl border-2 border-dashed px-5 py-12 text-center transition-all duration-300 sm:px-8 sm:py-16
          ${busy ? 'cursor-not-allowed border-ink-700 opacity-50' : 'cursor-pointer'}
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
              <span className="mx-2 text-ink-600">·</span>
              up to {LIMITS.maxDurationS}s
            </p>
          </div>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="hidden"
          disabled={busy}
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      {rejection && (
        <p role="alert" className="text-center text-sm text-red-400">
          {rejection}
        </p>
      )}

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
