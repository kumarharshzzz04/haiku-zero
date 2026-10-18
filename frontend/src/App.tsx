import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react"
import { ArrowLeft, ArrowRight, Check, Copy, Menu, Sparkles, X } from "lucide-react"

import { HaikuConsole, type ConsoleSettings } from "@/components/haiku-console"
import MorphOrb from "@/components/ui/ai-thinking-orb-and-input"
import SakuraEditorialPoster, {
  type SakuraEditorialKeyword,
} from "@/components/ui/sakura-editorial-poster"
import {
  fetchModels,
  generateHaiku,
  type GenerateResponse,
  type ModelInfo,
  type Poem,
} from "@/lib/api"
import { SAKURA_SCENES } from "@/lib/sakura-scenes"
import { cn } from "@/lib/utils"

const DEFAULT_SETTINGS: ConsoleSettings = {
  prompt: "old pond",
  model: "checkpoints",
  count: 2,
  strict: true,
  seed_first: true,
  best_of: 4,
  temperature: 0.8,
  seed: "",
  scene: 0,
  view: "poster",
}

/** The reveal is a tween, not a scroll: the cover has no scrollbar to drive it. */
const COVER_DURATION_MS = 1900

/** Long enough to watch the orb think, short enough not to feel stalled. */
const CHAT_MIN_THINK_MS = 2300

function meterTag(poem: Poem): string {
  return poem.exact ? "5 · 7 · 5 exact" : `near ${poem.pattern.join(" · ")}`
}

/** The huge poster word: the seed if there is one, otherwise the poem's longest word. */
function posterTitle(prompt: string, poem: Poem): string {
  const seed = prompt.trim()
  if (seed) return seed
  const words = poem.lines
    .join(" ")
    .split(/\s+/)
    .filter((word) => /^[A-Za-z']+$/.test(word))
  return words.sort((a, b) => b.length - a.length)[0] ?? "haiku"
}

function posterKeywords(result: GenerateResponse): SakuraEditorialKeyword[] {
  return [
    { label: result.strict ? "strict meter" : "free meter" },
    { label: `best of ${result.best_of}` },
    { label: result.model_label.split(" - ")[0] },
  ]
}

function stamp(date: Date): string {
  return `${String(date.getMonth() + 1).padStart(2, "0")}.${String(date.getDate()).padStart(2, "0")} ${date.getFullYear()}`
}

async function copyText(text: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text)
  } catch {
    /* clipboard blocked (http) - nothing useful to do */
  }
}

function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [done, setDone] = useState(false)
  return (
    <button
      type="button"
      onClick={() => {
        void copyText(text)
        setDone(true)
        window.setTimeout(() => setDone(false), 1400)
      }}
      className="inline-flex items-center gap-2 rounded-md border border-white/12 px-3 py-1.5 text-xs text-[#f6eee8]/70 transition hover:border-white/25 hover:text-[#f6eee8]"
    >
      {done ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      {done ? "copied" : label}
    </button>
  )
}

function ResultMeta({ result }: { result: GenerateResponse }) {
  const bits = [
    result.model_label,
    result.epoch != null ? `epoch ${result.epoch}` : null,
    result.val_loss != null ? `val ${result.val_loss.toFixed(4)}` : null,
    result.strict ? "strict 5-7-5" : "free meter",
    result.seed_first ? "seed leads" : "seed free",
    `best of ${result.best_of}`,
    result.seed != null ? `seed ${result.seed}` : "seed random",
    `${result.poems.length} poem${result.poems.length === 1 ? "" : "s"} in ${result.seconds}s`,
  ].filter(Boolean)

  const exact = result.poems.filter((poem) => poem.exact).length
  return (
    <div className="flex w-full flex-wrap items-center gap-x-3 gap-y-2 px-6 pb-5 text-xs font-light text-[#f6eee8]/50">
      {bits.map((bit) => (
        <span key={bit} className="tracking-wide">
          {bit}
        </span>
      ))}
      <span className="text-[#e7a3ae]">
        exact {exact}/{result.poems.length}
      </span>
      <span className="ml-auto">
        <CopyButton
          text={result.poems.map((poem) => poem.lines.join("\n")).join("\n\n")}
          label="Copy all"
        />
      </span>
    </div>
  )
}

function SheetView({ result, generatedAt }: { result: GenerateResponse; generatedAt: Date }) {
  return (
    <section className="w-full px-6 pb-24">
      <div className="divide-y divide-white/8 overflow-hidden rounded-xl border border-white/10 bg-white/[0.03]">
        {result.poems.map((poem, index) => (
          <article key={index} className="grid gap-4 p-5 md:grid-cols-12">
            <div className="md:col-span-7">
              {poem.lines.map((line, lineIndex) => (
                <p
                  key={lineIndex}
                  className="font-editorial text-xl leading-snug text-[#f6eee8]"
                >
                  {line}
                </p>
              ))}
            </div>
            <div className="md:col-span-5">
              <div className="flex flex-wrap gap-2 text-[11px] font-light tracking-wide">
                <span
                  className={cn(
                    "rounded-full px-2.5 py-1",
                    poem.exact
                      ? "bg-[#e7a3ae]/20 text-[#f0b4bd]"
                      : "bg-amber-400/15 text-amber-200",
                  )}
                >
                  {meterTag(poem)}
                </span>
                <span className="rounded-full bg-white/8 px-2.5 py-1 text-[#f6eee8]/65">
                  logprob {poem.logprob.toFixed(3)}
                </span>
                <span className="rounded-full bg-white/8 px-2.5 py-1 text-[#f6eee8]/65">
                  repeats {(poem.repeated_rate * 100).toFixed(0)}%
                </span>
                <span
                  className={cn(
                    "rounded-full px-2.5 py-1",
                    poem.unattested_rate > 0.15
                      ? "bg-amber-400/15 text-amber-200"
                      : "bg-white/8 text-[#f6eee8]/65",
                  )}
                  title="share of words the training corpus never contained (invented words)"
                >
                  invented words {(poem.unattested_rate * 100).toFixed(0)}%
                </span>
                {result.prompt ? (
                  <span
                    className={cn(
                      "rounded-full px-2.5 py-1",
                      poem.seed_hit
                        ? "bg-[#e7a3ae]/20 text-[#f0b4bd]"
                        : "bg-white/8 text-[#f6eee8]/65",
                    )}
                  >
                    {poem.seed_hit ? "uses your word" : "ignores your word"}
                  </span>
                ) : null}
                {poem.strict_ok === false ? (
                  <span className="rounded-full bg-red-400/15 px-2.5 py-1 text-red-200">
                    line closed short
                  </span>
                ) : null}
              </div>
              <p className="mt-3 text-[11px] font-light text-[#f6eee8]/40">
                seed {result.seed != null ? result.seed + index : "random"} · {result.model_label} ·{" "}
                {stamp(generatedAt)}
              </p>
            </div>
          </article>
        ))}
      </div>
    </section>
  )
}

type PanelProps = {
  models: ModelInfo[]
  settings: ConsoleSettings
  loading: boolean
  error: string | null
  patch: (next: Partial<ConsoleSettings>) => void
  onGenerate: () => void
  onRandomize: () => void
  onCover: () => void
}

/** The settings drawer, shared by the desk (sticky column) and the writing stage (overlay). */
function SettingsPanel({
  open,
  onClose,
  overlay,
  ...console
}: PanelProps & { open: boolean; onClose: () => void; overlay: boolean }) {
  return (
    <>
      {open ? (
        <button
          type="button"
          aria-label="Close the panel"
          onClick={onClose}
          className="fixed inset-0 z-30 bg-black/60 backdrop-blur-sm lg:hidden"
        />
      ) : null}

      <aside
        // the writing stage draws the panel off-canvas, so its controls must not be tabbable
        inert={overlay && !open}
        className={cn(
          "fixed inset-y-0 left-0 z-40 w-[20rem] shrink-0 overflow-y-auto border-r border-white/10 bg-[#100d0c] transition-transform duration-300",
          overlay ? "" : "lg:sticky lg:top-0 lg:z-auto lg:h-screen lg:translate-x-0",
          open ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <HaikuConsole
          models={console.models}
          settings={console.settings}
          loading={console.loading}
          error={console.error}
          onPatch={console.patch}
          onGenerate={console.onGenerate}
          onRandomize={console.onRandomize}
          onCover={console.onCover}
        />
      </aside>
    </>
  )
}

const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)"

/** Subscribed, not read during render: the value is external state React may re-check at will. */
function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const query = window.matchMedia(REDUCED_MOTION_QUERY)
      query.addEventListener("change", onChange)
      return () => query.removeEventListener("change", onChange)
    },
    () => window.matchMedia(REDUCED_MOTION_QUERY).matches,
  )
}

/** Cover progress 0 → 1: the wordmark slides in, then the copy and the start button follow. */
function useCoverReveal(active: boolean): number {
  const reduced = usePrefersReducedMotion()
  const [progress, setProgress] = useState(0)
  const raf = useRef(0)

  useEffect(() => {
    if (!active || reduced) return
    const started = performance.now()
    const step = () => {
      const t = Math.min(1, (performance.now() - started) / COVER_DURATION_MS)
      setProgress(1 - Math.pow(1 - t, 3))
      if (t < 1) raf.current = window.requestAnimationFrame(step)
    }
    raf.current = window.requestAnimationFrame(step)
    return () => window.cancelAnimationFrame(raf.current)
  }, [active, reduced])

  return reduced ? 1 : progress      // settled state is derived, never set
}

function Cover({ onEnter, sceneIndex }: { onEnter: () => void; sceneIndex: number }) {
  const progress = useCoverReveal(true)
  const [today] = useState(() => stamp(new Date()))
  const scene = SAKURA_SCENES[sceneIndex] ?? SAKURA_SCENES[0]

  return (
    <div className="relative h-screen w-full overflow-hidden">
      <SakuraEditorialPoster
        preview
        forceProgress={progress}
        title="HAIKU"
        titleTail="ZERO"
        titleMaxFraction={0.42}
        scrim="strong"
        fullBleed
        copySize="compact"
        sceneSrc={scene.src}
        sceneAlt={scene.alt}
        foregroundSrc="/branch.png"
        foregroundAlt="Cherry blossom branch in front of the wordmark"
        keywords={[
          { label: "五七五" },
          { label: "public-domain verse" },
          { label: "手元" },
        ]}
        headline="Five, seven, five | 五・七・五。"
        body="A 15.8M-parameter model trained on public-domain poetry, then fine-tuned on real haiku. It runs on the machine under your desk and answers in under three seconds."
        subheadline="Meter is enforced while it writes. 書かせながら、数える。"
        footerLeft="HAIKU ZERO"
        footerCenter="俳句 · ローカル"
        footerRight={today}
        socialHandle="strict 5-7-5 · seed leads · 0% invented words"
        cta={
          <button
            type="button"
            onClick={onEnter}
            style={{ opacity: progress > 0.85 ? 1 : 0 }}
            className="inline-flex items-center gap-2 rounded-md bg-[#e7a3ae] px-5 py-2.5 text-sm font-medium tracking-wide text-[#2c1a1e] transition hover:bg-[#f0b4bd]"
          >
            <Sparkles className="h-4 w-4" />
            Start writing
          </button>
        }
      />
    </div>
  )
}

export function App() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [settings, setSettings] = useState<ConsoleSettings>(DEFAULT_SETTINGS)
  const [result, setResult] = useState<GenerateResponse | null>(null)
  const [generatedAt, setGeneratedAt] = useState(() => new Date())
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [stage, setStage] = useState<"cover" | "ask" | "desk">("cover")
  const [sidebarOpen, setSidebarOpen] = useState(false)

  useEffect(() => {
    let alive = true
    fetchModels()
      .then((data) => {
        if (!alive) return
        setModels(data.models)
        setSettings((current) => ({
          ...current,
          model: data.models.some((model) => model.id === current.model)
            ? current.model
            : (data.default ?? current.model),
        }))
      })
      .catch((err: Error) => alive && setError(`Could not reach the API: ${err.message}`))
    return () => {
      alive = false
    }
  }, [])

  const patch = useCallback((next: Partial<ConsoleSettings>) => {
    setSettings((current) => ({ ...current, ...next }))
  }, [])

  const run = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await generateHaiku({
        prompt: settings.prompt,
        model: settings.model,
        count: settings.count,
        strict: settings.strict,
        seed_first: settings.seed_first,
        best_of: settings.best_of,
        temperature: settings.temperature,
        seed: settings.seed.trim() === "" ? null : Number(settings.seed),
      })
      setResult(response)
      setGeneratedAt(new Date())
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }, [settings])

  /**
   * The chat's answer. The orb shows the poem it gets back; the same request also fills the desk,
   * so the poster is already waiting when the card resolves.
   */
  const ask = useCallback(
    async (text: string): Promise<string> => {
      setLoading(true)
      setError(null)
      setSettings((current) => ({ ...current, prompt: text }))
      try {
        const response = await generateHaiku({
          prompt: text,
          model: settings.model,
          count: settings.count,
          strict: settings.strict,
          seed_first: settings.seed_first,
          best_of: settings.best_of,
          temperature: settings.temperature,
          seed: settings.seed.trim() === "" ? null : Number(settings.seed),
        })
        setResult(response)
        setGeneratedAt(new Date())
        if (response.poems.length === 0) return "The model returned nothing."
        // one haiku per stanza: the card draws each line on its own row
        return response.poems.map((poem) => poem.lines.join("\n")).join("\n\n")
      } catch (err) {
        const message = (err as Error).message
        setError(message)
        return `The model did not answer — ${message}`
      } finally {
        setLoading(false)
      }
    },
    [settings],
  )

  const scene = SAKURA_SCENES[settings.scene] ?? SAKURA_SCENES[0]
  const panel = {
    models,
    settings,
    loading,
    error,
    patch,
    onGenerate: () => void run(),
    onRandomize: () => patch({ seed: String(Math.floor(Math.random() * 100000)) }),
    onCover: () => {
      setStage("cover")
      window.scrollTo({ top: 0 })
    },
  }

  if (stage === "cover") {
    return (
      <Cover
        sceneIndex={settings.scene}
        onEnter={() => {
          setStage("ask")
          setSidebarOpen(false)
          window.scrollTo({ top: 0 })
        }}
      />
    )
  }

  if (stage === "ask") {
    return (
      <div className="relative min-h-screen bg-[#1a1114]">
        <MorphOrb onSubmit={ask} minThinkMs={CHAT_MIN_THINK_MS} />

        <SettingsPanel {...panel} overlay open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

        <div className="pointer-events-none fixed inset-x-0 top-0 z-20 flex items-start justify-between gap-3 px-5 py-5 sm:px-8">
          <div className="pointer-events-auto flex items-center gap-3">
            <button
              type="button"
              aria-label="Back to the cover"
              onClick={() => setStage("cover")}
              className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-white/12 text-[#f6eee8]/60 transition hover:border-white/30 hover:text-[#f6eee8]"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
            </button>
            <p className="text-[10px] font-light tracking-[0.34em] text-[#f6eee8]/40 uppercase">
              俳句 · write
            </p>
          </div>

          <div className="pointer-events-auto flex items-center gap-2">
            {result ? (
              <button
                type="button"
                onClick={() => setStage("desk")}
                className="inline-flex items-center gap-2 rounded-full bg-[#e7a3ae] px-4 py-2 text-xs font-medium tracking-wide text-[#2c1a1e] transition hover:bg-[#f0b4bd]"
              >
                the poster
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            ) : null}
            <button
              type="button"
              onClick={() => setSidebarOpen((open) => !open)}
              className="inline-flex items-center gap-2 rounded-full border border-white/12 px-3.5 py-2 text-xs tracking-wide text-[#f6eee8]/70 transition hover:border-white/30 hover:text-[#f6eee8]"
            >
              {sidebarOpen ? <X className="h-3.5 w-3.5" /> : <Menu className="h-3.5 w-3.5" />}
              panel
            </button>
          </div>
        </div>

        {error ? (
          <p className="pointer-events-none fixed inset-x-0 top-20 z-20 mx-auto max-w-md px-6 text-center text-xs font-light text-red-200/80">
            {error}
          </p>
        ) : null}
      </div>
    )
  }

  return (
    <div className="flex min-h-screen bg-[#14100f]">
      <SettingsPanel
        {...panel}
        overlay={false}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-3 border-b border-white/8 px-6 py-3">
          <button
            type="button"
            onClick={() => setSidebarOpen((open) => !open)}
            className="inline-flex items-center gap-2 rounded-md border border-white/12 px-3 py-1.5 text-xs tracking-wide text-[#f6eee8]/70 transition hover:border-white/25 hover:text-[#f6eee8] lg:hidden"
          >
            {sidebarOpen ? <X className="h-3.5 w-3.5" /> : <Menu className="h-3.5 w-3.5" />}
            panel
          </button>
          <p className="text-[10px] font-light tracking-[0.34em] text-[#f6eee8]/40 uppercase">
            the desk
          </p>
          <button
            type="button"
            onClick={() => {
              setStage("ask")
              setSidebarOpen(false)
            }}
            className="inline-flex items-center gap-2 rounded-md border border-white/12 px-3 py-1.5 text-xs tracking-wide text-[#f6eee8]/70 transition hover:border-[#e7a3ae]/50 hover:text-[#f6eee8]"
          >
            <Sparkles className="h-3.5 w-3.5" />
            ask
          </button>
          {result ? (
            <span className="ml-auto text-[11px] font-light text-[#f6eee8]/40">
              {result.poems.length} poem{result.poems.length === 1 ? "" : "s"} ·{" "}
              {result.seconds}s
            </span>
          ) : null}
        </div>

        {result ? <ResultMeta result={result} /> : null}

        {!result ? (
          <section className="flex min-h-[60vh] items-center justify-center px-6 py-16">
            <div className="w-full max-w-xl text-center">
              <p className="font-editorial text-[clamp(1.6rem,3.6vw,2.4rem)] leading-tight text-[#f6eee8]">
                Nothing written yet.
              </p>
              <button
                type="button"
                onClick={() => setStage("ask")}
                className="mt-6 inline-flex items-center gap-2 rounded-md bg-[#e7a3ae] px-5 py-2.5 text-sm font-medium tracking-wide text-[#2c1a1e] transition hover:bg-[#f0b4bd]"
              >
                <Sparkles className="h-4 w-4" />
                Give it a word
              </button>
            </div>
          </section>
        ) : null}

        {result && settings.view === "sheet" ? (
          <SheetView result={result} generatedAt={generatedAt} />
        ) : null}

        {result && settings.view === "poster"
          ? result.poems.map((poem, index) => (
              <SakuraEditorialPoster
                key={`${generatedAt.getTime()}-${index}`}
                title={posterTitle(result.prompt, poem)}
                sceneSrc={scene.src}
                sceneAlt={scene.alt}
                keywords={posterKeywords(result)}
                headline={poem.lines[0] ?? ""}
                body={poem.lines[1] ?? ""}
                subheadline={poem.lines[2] ?? ""}
                footerLeft={result.model_label.split(" - ")[0]}
                footerCenter={`${meterTag(poem)} · logprob ${poem.logprob.toFixed(3)}`}
                footerRight={stamp(generatedAt)}
                socialHandle={
                  result.prompt ? `seed: ${result.prompt}` : `seed ${result.seed ?? "random"}`
                }
                height="240vh"
              />
            ))
          : null}

        <footer className="mt-auto px-6 py-8 text-[11px] font-light text-[#f6eee8]/30">
          Poster component lives in{" "}
          <code>src/components/ui/sakura-editorial-poster.tsx</code> · chat orb in{" "}
          <code>src/components/ui/ai-thinking-orb-and-input.tsx</code> · model served by{" "}
          <code>backend/server.py</code>
        </footer>
      </main>
    </div>
  )
}

export default App
