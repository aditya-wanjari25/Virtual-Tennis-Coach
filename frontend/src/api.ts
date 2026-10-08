const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"

/** Upload limits, mirrored from the backend guardrails for pre-checks and
 *  copy. The server is the authority -- these only exist so an oversized file
 *  fails instantly instead of after a long pointless upload. */
export const LIMITS = {
  maxUploadMB: 100,
  maxDurationS: 90,
  maxChatChars: 2000,
} as const

/** Pull FastAPI's error message out of a failed response.
 *
 *  HTTPException sends {detail: "..."}; Pydantic validation failures send
 *  {detail: [{...}, ...]}. The guardrails put a player-readable sentence in
 *  the first form, so showing it beats showing a status code -- "that clip is
 *  120s long, the limit is 90s" is actionable where "Upload failed (422)" is
 *  not. The list form is a schema bug rather than something a user can fix,
 *  so that falls back to the generic message. */
async function errorMessage(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json()
    if (typeof body?.detail === "string" && body.detail.trim()) return body.detail
  } catch {
    // Non-JSON body (a proxy's HTML 413 page, say) -- nothing to extract.
  }
  return `${fallback} (${res.status})`
}

export type JobStatus = "pending" | "processing" | "done" | "error"
export type Stage = "tracking" | "watching" | "coaching"

export interface Job {
  id: string
  status: JobStatus
  stage: Stage | null
  created_at: string
  feedback: string | null
  error: string | null
  swing_count: number | null
}

export async function uploadVideo(file: File): Promise<{ job_id: string }> {
  const formData = new FormData()
  formData.append("file", file)

  const res = await fetch(`${API_BASE_URL}/videos`, { method: "POST", body: formData })
  if (!res.ok) {
    throw new Error(await errorMessage(res, "Upload failed"))
  }
  return res.json()
}

export async function getJob(jobId: string): Promise<Job> {
  const res = await fetch(`${API_BASE_URL}/videos/${jobId}`)
  if (!res.ok) {
    throw new Error(`Could not fetch job status (${res.status})`)
  }
  return res.json()
}

export interface Swing {
  index: number
  contact_time_s: number
  /** Qualitative notes, keyed by aspect (lower_body, balance, ...). */
  observations: Record<string, string>
  /** Each metric as a 0-1 position within THIS video's range. Deliberately not
   *  absolute values — our angles are 2D projections, so only the comparison
   *  across swings is meaningful. */
  relative: Record<string, number>
}

export function videoUrl(jobId: string): string {
  return `${API_BASE_URL}/videos/${jobId}/file`
}

export async function getSwings(jobId: string): Promise<Swing[]> {
  const res = await fetch(`${API_BASE_URL}/videos/${jobId}/swings`)
  if (!res.ok) {
    throw new Error(`Could not load swing breakdown (${res.status})`)
  }
  return res.json()
}

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
}

export async function sendChat(jobId: string, message: string): Promise<ChatMessage> {
  const res = await fetch(`${API_BASE_URL}/videos/${jobId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  })
  if (!res.ok) {
    throw new Error(await errorMessage(res, "Coach could not answer"))
  }
  return res.json()
}

export async function getChatHistory(jobId: string): Promise<ChatMessage[]> {
  const res = await fetch(`${API_BASE_URL}/videos/${jobId}/chat`)
  if (!res.ok) {
    throw new Error(`Could not load conversation (${res.status})`)
  }
  return res.json()
}
