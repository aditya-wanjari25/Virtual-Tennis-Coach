import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import { LIMITS, getChatHistory, sendChat, type ChatMessage } from '../api'

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
    <div className="animate-rise flex flex-col gap-5 rounded-3xl border border-ink-700/60 bg-ink-850/40 p-5 sm:p-8">
      <div className="flex items-center gap-2.5">
        <span className="text-xl">💬</span>
        <h2 className="font-semibold tracking-tight text-neutral-100">Ask your coach</h2>
      </div>

      {messages.length === 0 && !mutation.isPending && (
        <div className="flex flex-col gap-3">
          <p className="text-sm leading-relaxed text-neutral-500">
            Follow up on anything above. The coach can re-watch a specific swing to answer.
          </p>
          <div className="flex flex-wrap gap-2">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => submit(s)}
                className="rounded-full border border-ink-600 px-3.5 py-1.5 text-sm text-neutral-400 transition-colors hover:border-court-500/60 hover:bg-court-500/5 hover:text-court-300"
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      {messages.length > 0 && (
        <div className="flex max-h-[26rem] flex-col gap-5 overflow-y-auto pr-1">
          {messages.map((m, i) => (
            <div
              key={i}
              className={
                m.role === 'user'
                  ? 'max-w-[85%] self-end rounded-2xl rounded-br-md bg-court-500 px-4 py-2.5 text-sm font-medium text-ink-900'
                  : 'max-w-[92%] self-start'
              }
            >
              {m.role === 'user' ? (
                m.content
              ) : (
                <div className="prose prose-sm prose-invert max-w-none prose-p:leading-relaxed prose-p:text-neutral-300 prose-strong:text-neutral-100">
                  <Markdown>{m.content}</Markdown>
                </div>
              )}
            </div>
          ))}
          {mutation.isPending && (
            <div className="flex items-center gap-2.5 self-start text-sm text-neutral-500">
              <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-court-400 border-t-transparent" />
              thinking — may be re-watching your video…
            </div>
          )}
          <div ref={endRef} />
        </div>
      )}

      {mutation.isError && <p className="text-sm text-red-400">{mutation.error.message}</p>}

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
          // Matches the server's cap, so the limit is felt as the field simply
          // stopping rather than as a 422 after pressing Ask.
          maxLength={LIMITS.maxChatChars}
          disabled={mutation.isPending}
          className="flex-1 rounded-xl border border-ink-600 bg-ink-900/60 px-4 py-3 text-base sm:text-sm text-neutral-100 transition-colors placeholder:text-neutral-600 focus:border-court-500 focus:outline-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={mutation.isPending || !draft.trim()}
          className="shrink-0 rounded-xl bg-court-500 px-5 py-3 text-sm font-semibold text-ink-900 transition-colors hover:bg-court-400 disabled:opacity-30"
        >
          Ask
        </button>
      </form>
    </div>
  )
}
