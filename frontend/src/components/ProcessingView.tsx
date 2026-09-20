import type { Stage } from '../api'

/** Ordered so we can tell which steps are already behind us. The copy says what
 *  each step actually does -- a 30s wait is more tolerable when it's legible. */
const STEPS: { id: Stage; label: string; detail: string }[] = [
  { id: 'tracking', label: 'Tracking your body', detail: 'Finding joint positions in every frame' },
  { id: 'watching', label: 'Watching your swings', detail: 'Looking at balance, footwork and timing' },
  { id: 'coaching', label: 'Writing your feedback', detail: 'Working out what matters most' },
]

export function ProcessingView({ stage }: { stage: Stage | null }) {
  // Before the first stage lands, treat it as step one rather than showing
  // nothing -- the upload has finished, so work really is underway.
  const activeIndex = Math.max(0, STEPS.findIndex((s) => s.id === stage))

  return (
    <div className="animate-rise overflow-hidden rounded-3xl border border-ink-700/60 bg-ink-850/40">
      <div className="relative h-1 overflow-hidden bg-ink-800">
        <div className="animate-sweep absolute inset-y-0 w-1/2 bg-gradient-to-r from-transparent via-court-400 to-transparent" />
      </div>

      <div className="p-8">
        <div className="mb-8">
          <h2 className="text-lg font-semibold tracking-tight text-neutral-100">Analyzing</h2>
        </div>

        <ol className="flex flex-col gap-5">
          {STEPS.map((step, i) => {
            const done = i < activeIndex
            const active = i === activeIndex
            return (
              <li key={step.id} className="flex items-start gap-4">
                <div
                  className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border text-xs transition-colors duration-500
                    ${done ? 'border-court-500 bg-court-500/15 text-court-400' : ''}
                    ${active ? 'animate-pulse-ring border-court-400 bg-court-400 text-ink-900' : ''}
                    ${!done && !active ? 'border-ink-600 text-neutral-600' : ''}`}
                >
                  {done ? '✓' : i + 1}
                </div>
                <div className="min-w-0">
                  <p
                    className={`text-sm font-medium transition-colors duration-500
                      ${active ? 'text-neutral-100' : done ? 'text-neutral-400' : 'text-neutral-600'}`}
                  >
                    {step.label}
                  </p>
                  {active && <p className="mt-0.5 text-xs leading-relaxed text-neutral-500">{step.detail}</p>}
                </div>
              </li>
            )
          })}
        </ol>
      </div>
    </div>
  )
}
