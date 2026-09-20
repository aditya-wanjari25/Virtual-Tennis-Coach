import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import { getChatHistory, sendChat, type ChatMessage } from '../api'

const SUGGESTIONS = [
  'Which swing was my best?',
  'Why does that matter?',
  'What should I work on first?',
]

export function ChatPanel({ jobId }: { jobId: string }) {
  const [draft, setDraft] = useState('')
  const queryClient = useQueryClient()
  const endRef = useRef<HTMLDivElement>(null)

  const { data: messages = [] } = useQuery({
    queryKey: ['chat', jobId],
    queryFn: () => getChatHistory(jobId),
  })

  const mutation = useMutation({
    mutationFn: (message: string) => sendChat(jobId, message),
    // Show the user's message straight away rather than after the round trip --
    // the coach can take several seconds when it re-watches the video.
    onMutate: async (message) => {
      await queryClient.cancelQueries({ queryKey: ['chat', jobId] })
      const previous = queryClient.getQueryData<ChatMessage[]>(['chat', jobId]) ?? []
      queryClient.setQueryData<ChatMessage[]>(['chat', jobId], [
        ...previous,
        { role: 'user', content: message },
      ])
      return { previous }
    },
    onError: (_err, _msg, context) => {
      queryClient.setQueryData(['chat', jobId], context?.previous ?? [])
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['chat', jobId] })
    },
  })

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, mutation.isPending])

  function submit(text: string) {
    const trimmed = text.trim()
    if (!trimmed || mutation.isPending) return
    mutation.mutate(trimmed)
    setDraft('')
  }

  return (
    <div className="mt-6 flex flex-col gap-4 rounded-2xl border border-neutral-200 dark:border-neutral-800 p-6">
      <div className="flex items-center gap-2">
        <span className="text-xl">💬</span>
        <h2 className="font-semibold text-neutral-900 dark:text-neutral-100">Ask your coach</h2>
      </div>

      {messages.length === 0 && !mutation.isPending && (
        <div className="flex flex-wrap gap-2">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => submit(s)}
              className="rounded-full border border-neutral-300 dark:border-neutral-700 px-3 py-1.5 text-sm text-neutral-600 dark:text-neutral-400 hover:border-lime-500 hover:text-lime-600"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {messages.length > 0 && (
        <div className="flex max-h-96 flex-col gap-4 overflow-y-auto">
          {messages.map((m, i) => (
            <div
              key={i}
              className={
                m.role === 'user'
                  ? 'self-end max-w-[85%] rounded-2xl bg-lime-600 px-4 py-2 text-white'
                  : 'self-start max-w-[90%] text-neutral-800 dark:text-neutral-200'
              }
            >
              {m.role === 'user' ? (
                m.content
              ) : (
                <div className="prose prose-sm prose-neutral dark:prose-invert max-w-none">
                  <Markdown>{m.content}</Markdown>
                </div>
              )}
            </div>
          ))}
          {mutation.isPending && (
            <div className="flex items-center gap-2 self-start text-sm text-neutral-500">
              <div className="h-3 w-3 animate-spin rounded-full border-2 border-lime-500 border-t-transparent" />
              thinking — may be re-watching your video…
            </div>
          )}
          <div ref={endRef} />
        </div>
      )}

      {mutation.isError && (
        <p className="text-sm text-red-600 dark:text-red-400">{mutation.error.message}</p>
      )}

      <form
        onSubmit={(e) => {
          e.preventDefault()
          submit(draft)
        }}
        className="flex gap-2"
      >
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Was my elbow bent at contact?"
          disabled={mutation.isPending}
          className="flex-1 rounded-lg border border-neutral-300 dark:border-neutral-700 bg-transparent px-3 py-2 text-neutral-900 dark:text-neutral-100 placeholder:text-neutral-400 focus:border-lime-500 focus:outline-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={mutation.isPending || !draft.trim()}
          className="rounded-lg bg-lime-600 px-4 py-2 text-sm font-medium text-white hover:bg-lime-700 disabled:opacity-40"
        >
          Ask
        </button>
      </form>
    </div>
  )
}
