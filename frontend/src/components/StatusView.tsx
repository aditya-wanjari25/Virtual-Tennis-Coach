import Markdown from 'react-markdown'
import type { Job } from '../api'

const STAGE_LABELS: Record<string, string> = {
  pending: 'Queued…',
  processing: 'Analyzing your swing — extracting pose data and generating feedback…',
}

export function StatusView({ job, onReset }: { job: Job; onReset: () => void }) {
  if (job.status === 'pending' || job.status === 'processing') {
    return (
      <div className="flex flex-col items-center gap-4 rounded-2xl border border-neutral-200 dark:border-neutral-800 p-12 text-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-lime-500 border-t-transparent" />
        <p className="text-neutral-700 dark:text-neutral-300">{STAGE_LABELS[job.status]}</p>
      </div>
    )
  }

  if (job.status === 'error') {
    return (
      <div className="flex flex-col gap-4 rounded-2xl border border-red-300 dark:border-red-900 bg-red-50 dark:bg-red-950/30 p-8">
        <p className="font-medium text-red-700 dark:text-red-400">Something went wrong analyzing this video</p>
        <p className="text-sm text-red-600 dark:text-red-400/80">{job.error}</p>
        <button
          onClick={onReset}
          className="self-start rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700"
        >
          Try another video
        </button>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-6 rounded-2xl border border-neutral-200 dark:border-neutral-800 p-8">
      <div className="flex items-center gap-2">
        <span className="text-2xl">🎾</span>
        <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">Coach's feedback</h2>
      </div>
      <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:leading-relaxed">
        <Markdown>{job.feedback}</Markdown>
      </div>
      <button
        onClick={onReset}
        className="self-start rounded-lg bg-lime-600 px-4 py-2 text-sm font-medium text-white hover:bg-lime-700"
      >
        Analyze another video
      </button>
    </div>
  )
}
