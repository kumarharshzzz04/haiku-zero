export type ModelInfo = {
  id: string
  label: string
  path: string
  epoch: number | null
  val_loss: number | null
  meta_error?: string
}

export type Poem = {
  lines: string[]
  pattern: number[]
  distance: number
  exact: boolean
  strict_ok: boolean | null
  logprob: number
  repeated_rate: number
  unattested_rate: number
  seed_hit: boolean
}

export type GenerateRequest = {
  prompt: string
  model: string
  count: number
  strict: boolean
  seed_first: boolean
  best_of: number
  temperature: number
  seed: number | null
}

export type GenerateResponse = {
  model: string
  model_label: string
  epoch: number | null
  val_loss: number | null
  prompt: string
  strict: boolean
  seed_first: boolean
  best_of: number
  seed: number | null
  poems: Poem[]
  seconds: number
  load_seconds: number | null
}

async function readJson<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    const message = body && typeof body.error === "string" ? body.error : response.statusText
    throw new Error(body?.hint ? `${message} (${body.hint})` : message)
  }
  return body as T
}

export async function fetchModels(): Promise<{ models: ModelInfo[]; default: string | null }> {
  return readJson(await fetch("/api/models"))
}

export async function generateHaiku(request: GenerateRequest): Promise<GenerateResponse> {
  return readJson(
    await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    }),
  )
}
