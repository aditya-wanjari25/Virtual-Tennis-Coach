import { useMutation, useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { getJob, uploadVideo } from './api'
import { StatusView } from './components/StatusView'
import { UploadZone } from './components/UploadZone'

function App() {
  const [jobId, setJobId] = useState<string | null>(null)

  const uploadMutation = useMutation({
    mutationFn: uploadVideo,
    onSuccess: (data) => setJobId(data.job_id),
  })

  const jobQuery = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId!),
    enabled: jobId !== null,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      // Poll faster than before: stage changes are the point of the progress
      // view, and a 3s interval made the steps feel laggy.
      return status === 'done' || status === 'error' ? false : 1500
    },
  })

  function reset() {
    setJobId(null)
    uploadMutation.reset()
  }

  return (
    <div className="relative min-h-screen bg-ink-900 text-neutral-200">
      {/* Single soft accent wash behind the header. Fixed and non-interactive
          so it never intercepts clicks or scrolls with content. */}
      <div
        aria-hidden
        className="pointer-events-none fixed inset-x-0 top-0 h-[420px] bg-[radial-gradient(ellipse_70%_100%_at_50%_0%,var(--color-court-500)_0%,transparent_70%)] opacity-[0.06]"
      />

      <div className="relative mx-auto max-w-3xl px-5 py-14 sm:py-20">
        <header className="mb-12 text-center">
          <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-ink-700 bg-ink-850/60 px-3.5 py-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-court-400" />
            <span className="text-xs font-medium tracking-wide text-neutral-400">
              Pose tracking + video analysis
            </span>
          </div>

          <h1 className="text-4xl font-bold tracking-tight text-neutral-50 sm:text-5xl">
            Virtual Tennis <span className="text-court-400">Coach</span>
          </h1>
          <p className="mx-auto mt-4 max-w-md text-[15px] leading-relaxed text-neutral-400">
            Upload a groundstroke video. Get specific, measured feedback — then ask
            follow-up questions about it.
          </p>
        </header>

        {jobId === null ? (
          <div className="animate-rise flex flex-col gap-4">
            <UploadZone
              onFileSelected={(file) => uploadMutation.mutate(file)}
              disabled={uploadMutation.isPending}
            />
            {uploadMutation.isPending && (
              <div className="flex items-center justify-center gap-2.5 text-sm text-neutral-400">
                <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-court-400 border-t-transparent" />
                Uploading…
              </div>
            )}
            {uploadMutation.isError && (
              <p className="text-center text-sm text-red-400">{uploadMutation.error.message}</p>
            )}
          </div>
        ) : jobQuery.data ? (
          <StatusView job={jobQuery.data} onReset={reset} />
        ) : jobQuery.isError ? (
          <div className="animate-rise rounded-3xl border border-red-900/60 bg-red-950/20 p-8 text-center">
            <p className="text-sm text-red-300">{jobQuery.error.message}</p>
            <button
              onClick={reset}
              className="mt-5 rounded-xl bg-red-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-red-500"
            >
              Try again
            </button>
          </div>
        ) : (
          <div className="rounded-3xl border border-ink-700/60 bg-ink-850/40 p-12 text-center text-sm text-neutral-500">
            Loading…
          </div>
        )}

        <footer className="mt-16 text-center text-xs text-neutral-600">
          Measurements from pose tracking · Observations from video · Filmed from behind the baseline
        </footer>
      </div>
    </div>
  )
}

export default App
