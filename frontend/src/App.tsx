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
      return status === 'done' || status === 'error' ? false : 3000
    },
  })

  function reset() {
    setJobId(null)
    uploadMutation.reset()
  }

  return (
    <div className="min-h-screen bg-neutral-50 dark:bg-neutral-950">
      <div className="mx-auto max-w-2xl px-4 py-16">
        <header className="mb-10 text-center">
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">Virtual Tennis Coach</h1>
          <p className="mt-2 text-neutral-500 dark:text-neutral-400">
            Upload a groundstroke video, get grounded, data-driven feedback.
          </p>
        </header>

        {jobId === null ? (
          <div className="flex flex-col gap-3">
            <UploadZone onFileSelected={(file) => uploadMutation.mutate(file)} disabled={uploadMutation.isPending} />
            {uploadMutation.isPending && (
              <p className="text-center text-sm text-neutral-500 dark:text-neutral-400">Uploading…</p>
            )}
            {uploadMutation.isError && (
              <p className="text-center text-sm text-red-600 dark:text-red-400">{uploadMutation.error.message}</p>
            )}
          </div>
        ) : jobQuery.data ? (
          <StatusView job={jobQuery.data} onReset={reset} />
        ) : jobQuery.isError ? (
          <div className="rounded-2xl border border-red-300 dark:border-red-900 bg-red-50 dark:bg-red-950/30 p-8 text-center">
            <p className="text-red-700 dark:text-red-400">{jobQuery.error.message}</p>
            <button onClick={reset} className="mt-4 rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700">
              Try again
            </button>
          </div>
        ) : (
          <div className="rounded-2xl border border-neutral-200 dark:border-neutral-800 p-12 text-center text-neutral-500">
            Loading…
          </div>
        )}
      </div>
    </div>
  )
}

export default App
