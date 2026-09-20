import Markdown from 'react-markdown'
import type { Job } from '../api'
import { ChatPanel } from './ChatPanel'
import { ProcessingView } from './ProcessingView'
import { SwingBreakdown } from './SwingBreakdown'

export function StatusView({ job, onReset }: { job: Job; onReset: () => void }) {
  if (job.status === 'pending' || job.status === 'processing') {
    return <ProcessingView stage={job.stage} />
  }

  if (job.status === 'error') {
    return (
      <div className="animate-rise flex flex-col gap-4 rounded-3xl border border-red-900/60 bg-red-950/20 p-5 sm:p-8">
        <div className="flex items-center gap-2.5">
          <span className="text-xl">⚠️</span>
          <h2 className="font-semibold text-red-300">Couldn't analyze this one</h2>
        </div>
        <p className="text-sm leading-relaxed text-red-200/70">{job.error}</p>
        <p className="text-xs leading-relaxed text-neutral-500">
          Most often this means no swings were detected — check the whole body is in frame and the
          camera is behind the baseline.
        </p>
        <button
          onClick={onReset}
          className="self-start rounded-xl bg-red-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-red-500"
        >
          Try another video
        </button>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-5">
      <div className="animate-rise overflow-hidden rounded-3xl border border-ink-700/60 bg-ink-850/40">
        <div className="h-1 bg-gradient-to-r from-court-600 via-court-400 to-court-600" />

        <div className="p-5 sm:p-8">
          <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2.5">
              <span className="text-xl">🎾</span>
              <h2 className="text-lg font-semibold tracking-tight text-neutral-100">Your feedback</h2>
            </div>
            {job.swing_count != null && (
              <span className="rounded-full border border-court-500/30 bg-court-500/10 px-3 py-1 text-xs font-medium text-court-300">
                {job.swing_count} {job.swing_count === 1 ? 'swing' : 'swings'} analyzed
              </span>
            )}
          </div>

          {/* Explicit prose overrides: the feedback is free-form markdown from
              the model, so the typography has to hold up whatever shape it takes. */}
          <div
            className="prose prose-invert max-w-none
              prose-p:text-[15px] prose-p:leading-[1.75] prose-p:text-neutral-300
              prose-strong:text-neutral-100 prose-strong:font-semibold
              prose-headings:text-neutral-100 prose-headings:tracking-tight
              prose-li:text-neutral-300 prose-li:leading-relaxed
              prose-em:text-court-300 prose-em:not-italic"
          >
            <Markdown>{job.feedback}</Markdown>
          </div>
        </div>
      </div>

      <SwingBreakdown jobId={job.id} />

      <ChatPanel jobId={job.id} />

      <button
        onClick={onReset}
        className="self-center rounded-xl border border-ink-700 px-5 py-2.5 text-sm font-medium text-neutral-400 transition-colors hover:border-court-500/50 hover:text-court-300"
      >
        Analyze another video
      </button>
    </div>
  )
}
