import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { getSwings, videoUrl } from '../api'

const OBS_LABELS: Record<string, string> = {
  lower_body: 'Lower body',
  balance: 'Balance',
  timing_and_preparation: 'Timing & prep',
  spacing_to_ball: 'Spacing to ball',
  contact_point: 'Contact point',
  not_visible: "Couldn't see",
}

// Order matters: lead with what a coach looks at first.
const OBS_ORDER = ['lower_body', 'balance', 'contact_point', 'spacing_to_ball', 'timing_and_preparation', 'not_visible']

export function SwingBreakdown({ jobId }: { jobId: string }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [selected, setSelected] = useState(1)
  const [duration, setDuration] = useState(0)
  const [playhead, setPlayhead] = useState(0)

  const { data: swings, isError } = useQuery({
    queryKey: ['swings', jobId],
    queryFn: () => getSwings(jobId),
  })

  // Jump the video to a swing's contact, backing up slightly so you see the
  // approach rather than landing on the frame after impact.
  function seekTo(time: number, index: number) {
    setSelected(index)
    const v = videoRef.current
    if (!v) return
    v.currentTime = Math.max(0, time - 0.7)
    void v.play().catch(() => {}) // autoplay can be blocked; not worth surfacing
  }

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    const onTime = () => setPlayhead(v.currentTime)
    const onMeta = () => setDuration(v.duration || 0)
    v.addEventListener('timeupdate', onTime)
    v.addEventListener('loadedmetadata', onMeta)
    return () => {
      v.removeEventListener('timeupdate', onTime)
      v.removeEventListener('loadedmetadata', onMeta)
    }
  }, [])

  if (isError || !swings?.length) return null

  const current = swings.find((s) => s.index === selected) ?? swings[0]
  const notes = OBS_ORDER.filter((k) => current.observations[k]).map((k) => [k, current.observations[k]] as const)

  return (
    <div className="animate-rise overflow-hidden rounded-3xl border border-ink-700/60 bg-ink-850/40">
      <div className="border-b border-ink-700/60 px-8 py-5">
        <h2 className="font-semibold tracking-tight text-neutral-100">Swing by swing</h2>
        <p className="mt-1 text-sm text-neutral-500">Pick a swing to jump to it and see what was found.</p>
      </div>

      <div className="p-8">
        <div className="overflow-hidden rounded-2xl bg-black">
          <video
            ref={videoRef}
            src={videoUrl(jobId)}
            controls
            muted
            playsInline
            preload="metadata"
            className="max-h-[420px] w-full object-contain"
          />
        </div>

        {/* Contact markers. Sits under the video rather than overlaying the
            native controls, which would fight with them on mobile. */}
        {duration > 0 && (
          <div className="relative mt-3 h-8">
            <div className="absolute inset-x-0 top-3 h-1 rounded-full bg-ink-700" />
            <div
              className="absolute top-3 h-1 rounded-full bg-court-500/40"
              style={{ width: `${Math.min(100, (playhead / duration) * 100)}%` }}
            />
            {swings.map((s) => {
              const pct = (s.contact_time_s / duration) * 100
              const isActive = s.index === selected
              return (
                <button
                  key={s.index}
                  onClick={() => seekTo(s.contact_time_s, s.index)}
                  title={`Swing ${s.index} — contact at ${s.contact_time_s.toFixed(1)}s`}
                  aria-label={`Jump to swing ${s.index}`}
                  className="absolute top-0 -translate-x-1/2"
                  style={{ left: `${pct}%` }}
                >
                  <span
                    className={`block h-7 w-7 rounded-full border-2 text-xs font-semibold leading-[1.5rem] transition-all
                      ${
                        isActive
                          ? 'scale-110 border-court-400 bg-court-400 text-ink-900'
                          : 'border-ink-600 bg-ink-800 text-neutral-400 hover:border-court-500 hover:text-court-300'
                      }`}
                  >
                    {s.index}
                  </span>
                </button>
              )
            })}
          </div>
        )}

        <div className="mt-8 grid gap-6 md:grid-cols-5">
          <div className="md:col-span-3">
            <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-neutral-500">
              What we saw — swing {current.index}
            </h3>
            <dl className="flex flex-col gap-3">
              {notes.map(([key, text]) => (
                <div key={key} className="border-l-2 border-ink-700 pl-3.5">
                  <dt className="text-xs font-medium text-court-300">{OBS_LABELS[key] ?? key}</dt>
                  <dd className="mt-0.5 text-sm leading-relaxed text-neutral-400">{text}</dd>
                </div>
              ))}
              {notes.length === 0 && (
                <p className="text-sm text-neutral-600">No visual notes recorded for this swing.</p>
              )}
            </dl>
          </div>

          <div className="md:col-span-2">
            <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-neutral-500">
              Vs your other swings
            </h3>
            <div className="flex flex-col gap-3.5">
              {Object.entries(current.relative).map(([label, value]) => (
                <div key={label}>
                  <div className="mb-1.5 flex items-baseline justify-between">
                    <span className="text-xs text-neutral-400">{label}</span>
                    <span className="text-[10px] uppercase tracking-wide text-neutral-600">
                      {value > 0.66 ? 'most' : value < 0.34 ? 'least' : 'mid'}
                    </span>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded-full bg-ink-700">
                    <div
                      className="h-full rounded-full bg-court-500 transition-all duration-500"
                      style={{ width: `${Math.max(4, value * 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
            <p className="mt-4 text-[11px] leading-relaxed text-neutral-600">
              Relative to your own swings in this video — not a score.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
