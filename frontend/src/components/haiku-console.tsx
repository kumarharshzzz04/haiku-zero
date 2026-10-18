import { type ReactNode } from "react"

import {
  Dices,
  House,
  Layers,
  LayoutList,
  LoaderCircle,
  Sparkles,
} from "lucide-react"

import type { ModelInfo } from "@/lib/api"
import { SAKURA_SCENES } from "@/lib/sakura-scenes"
import { cn } from "@/lib/utils"

export type ViewMode = "poster" | "sheet"

export type ConsoleSettings = {
  prompt: string
  model: string
  count: number
  strict: boolean
  seed_first: boolean
  best_of: number
  temperature: number
  seed: string
  scene: number
  view: ViewMode
}

const PRESET_SEEDS = [
  "old pond",
  "first snow",
  "city rain",
  "mother's hands",
  "train window",
  "",
]

const COUNT_OPTIONS = [1, 2, 3, 4, 6]
const BEST_OF_OPTIONS = [1, 2, 4, 8, 16]

function Label({ children }: { children: ReactNode }) {
  return (
    <span className="text-[10px] font-light tracking-[0.22em] text-[#f6eee8]/45 uppercase">
      {children}
    </span>
  )
}

const CONTROL_CLASS =
  "w-full rounded-md border border-white/12 bg-white/[0.04] px-3 py-2 text-sm text-[#f6eee8] outline-none transition focus:border-[#e7a3ae]/60 focus:bg-white/[0.07]"

function Field({
  label,
  children,
  className,
}: {
  label: string
  children: ReactNode
  className?: string
}) {
  return (
    <label className={cn("flex flex-col gap-1.5", className)}>
      <Label>{label}</Label>
      {children}
    </label>
  )
}

function Toggle({
  label,
  title,
  on,
  onClick,
  onText,
  offText,
}: {
  label: string
  title?: string
  on: boolean
  onClick: () => void
  onText: string
  offText: string
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <Label>{label}</Label>
      <button
        type="button"
        title={title}
        onClick={onClick}
        className={cn(
          "flex items-center justify-between rounded-md border px-3 py-2 text-xs tracking-wide transition",
          on
            ? "border-[#e7a3ae]/60 bg-[#e7a3ae]/15 text-[#f6eee8]"
            : "border-white/12 bg-white/[0.03] text-[#f6eee8]/55",
        )}
      >
        {on ? onText : offText}
        <span
          className={cn("h-2 w-2 rounded-full", on ? "bg-[#e7a3ae]" : "bg-white/30")}
        />
      </button>
    </div>
  )
}

/** The desk sidebar: every option that used to sit in the wide console header. */
export function HaikuConsole({
  models,
  settings,
  loading,
  error,
  onPatch,
  onGenerate,
  onRandomize,
  onCover,
}: {
  models: ModelInfo[]
  settings: ConsoleSettings
  loading: boolean
  error: string | null
  onPatch: (patch: Partial<ConsoleSettings>) => void
  onGenerate: () => void
  onRandomize: () => void
  onCover?: () => void
}) {
  return (
    <div className="flex h-full flex-col gap-5 p-5">
      <header className="flex items-start justify-between gap-3">
        <div>
          <p className="text-[10px] font-light tracking-[0.34em] text-[#f6eee8]/45 uppercase">
            Haiku Zero
          </p>
          <p className="mt-1 text-[11px] font-light text-[#f6eee8]/35">
            local checkpoint sampler
          </p>
        </div>
        {onCover ? (
          <button
            type="button"
            onClick={onCover}
            title="Back to the cover"
            className="inline-flex items-center gap-1.5 rounded-md border border-white/12 px-2.5 py-1.5 text-[10px] tracking-[0.14em] text-[#f6eee8]/55 uppercase transition hover:border-white/25 hover:text-[#f6eee8]"
          >
            <House className="h-3 w-3" />
            cover
          </button>
        ) : null}
      </header>

      <form
        className="flex flex-col gap-4"
        onSubmit={(event) => {
          event.preventDefault()
          onGenerate()
        }}
      >
        <Field label="Seed phrase">
          <input
            className={CONTROL_CLASS}
            value={settings.prompt}
            onChange={(event) => onPatch({ prompt: event.target.value })}
            placeholder="leave empty for an unseeded poem"
            spellCheck={false}
          />
        </Field>

        <Field label="Checkpoint">
          <select
            className={CONTROL_CLASS}
            value={settings.model}
            onChange={(event) => onPatch({ model: event.target.value })}
          >
            {models.map((model) => (
              <option key={model.id} value={model.id} className="bg-[#14100f]">
                {model.label}
                {model.epoch != null ? ` · epoch ${model.epoch}` : ""}
                {model.val_loss != null ? ` · val ${model.val_loss.toFixed(3)}` : ""}
              </option>
            ))}
          </select>
        </Field>

        <div className="grid grid-cols-2 gap-4">
          <Field label="Poems">
            <select
              className={CONTROL_CLASS}
              value={settings.count}
              onChange={(event) => onPatch({ count: Number(event.target.value) })}
            >
              {COUNT_OPTIONS.map((n) => (
                <option key={n} value={n} className="bg-[#14100f]">
                  {n}
                </option>
              ))}
            </select>
          </Field>

          <Field label="Best of">
            <select
              className={CONTROL_CLASS}
              value={settings.best_of}
              onChange={(event) => onPatch({ best_of: Number(event.target.value) })}
            >
              {BEST_OF_OPTIONS.map((n) => (
                <option key={n} value={n} className="bg-[#14100f]">
                  {n}
                </option>
              ))}
            </select>
          </Field>
        </div>

        <Field label={`Temperature · ${settings.temperature.toFixed(2)}`}>
          <input
            type="range"
            min={0.4}
            max={1.5}
            step={0.05}
            value={settings.temperature}
            onChange={(event) => onPatch({ temperature: Number(event.target.value) })}
            className="mt-2 accent-[#e7a3ae]"
          />
        </Field>

        <Field label="Seed (optional)">
          <input
            className={CONTROL_CLASS}
            value={settings.seed}
            inputMode="numeric"
            placeholder="random"
            onChange={(event) =>
              onPatch({ seed: event.target.value.replace(/[^0-9-]/g, "") })
            }
          />
        </Field>

        <div className="flex flex-col gap-3">
          <Toggle
            label="Meter"
            on={settings.strict}
            onClick={() => onPatch({ strict: !settings.strict })}
            onText="strict 5-7-5"
            offText="free meter"
          />
          <Toggle
            label="Start with word"
            title="Write the seed as the poem's first word instead of hoping the model mentions it"
            on={settings.seed_first}
            onClick={() => onPatch({ seed_first: !settings.seed_first })}
            onText="seed leads"
            offText="model decides"
          />
        </div>

        <div className="flex flex-col gap-2">
          <Label>Scene</Label>
          <div className="flex gap-2">
            {SAKURA_SCENES.map((scene, index) => (
              <button
                key={scene.src}
                type="button"
                title={scene.alt}
                onClick={() => onPatch({ scene: index })}
                className={cn(
                  "h-10 w-10 overflow-hidden rounded-md border transition",
                  index === settings.scene
                    ? "border-[#e7a3ae] opacity-100"
                    : "border-white/15 opacity-55 hover:opacity-90",
                )}
              >
                <img
                  src={scene.src}
                  alt={scene.alt}
                  className="h-full w-full object-cover"
                />
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-col gap-2">
          <Label>View</Label>
          <div className="flex items-center gap-1 rounded-md border border-white/12 p-1">
            {(
              [
                ["poster", "Poster", Layers],
                ["sheet", "Sheet", LayoutList],
              ] as const
            ).map(([mode, text, Icon]) => (
              <button
                key={mode}
                type="button"
                onClick={() => onPatch({ view: mode })}
                className={cn(
                  "inline-flex flex-1 items-center justify-center gap-2 rounded px-3 py-1.5 text-xs tracking-wide transition",
                  settings.view === mode
                    ? "bg-white/12 text-[#f6eee8]"
                    : "text-[#f6eee8]/45 hover:text-[#f6eee8]/80",
                )}
              >
                <Icon className="h-3.5 w-3.5" />
                {text}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-col gap-2">
          <button
            type="submit"
            disabled={loading || models.length === 0}
            className="inline-flex items-center justify-center gap-2 rounded-md bg-[#e7a3ae] px-5 py-2.5 text-sm font-medium tracking-wide text-[#2c1a1e] transition hover:bg-[#f0b4bd] disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? (
              <LoaderCircle className="h-4 w-4 animate-spin" />
            ) : (
              <Sparkles className="h-4 w-4" />
            )}
            {loading ? "sampling…" : "Generate"}
          </button>

          <button
            type="button"
            onClick={onRandomize}
            className="inline-flex items-center justify-center gap-2 rounded-md border border-white/12 px-4 py-2.5 text-sm text-[#f6eee8]/75 transition hover:border-white/25 hover:text-[#f6eee8]"
          >
            <Dices className="h-4 w-4" />
            Random seed
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Label>Try</Label>
          {PRESET_SEEDS.map((seed) => (
            <button
              key={seed || "unseeded"}
              type="button"
              onClick={() => onPatch({ prompt: seed })}
              className="rounded-full border border-white/10 px-3 py-1 text-xs font-light text-[#f6eee8]/60 transition hover:border-[#e7a3ae]/50 hover:text-[#f6eee8]"
            >
              {seed || "(no seed)"}
            </button>
          ))}
        </div>

        {error ? (
          <p className="rounded-md border border-red-400/30 bg-red-500/10 px-3 py-2 text-xs text-red-200">
            {error}
          </p>
        ) : null}
      </form>
    </div>
  )
}

export default HaikuConsole
