const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"

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
    throw new Error(`Upload failed (${res.status})`)
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
    throw new Error(`Coach could not answer (${res.status})`)
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
