"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

import { SAKURA_SCENES } from "@/lib/sakura-scenes";
import { cn } from "@/lib/utils";

export type SakuraEditorialKeyword = {
  label: string;
};

export type SakuraEditorialPosterProps = {
  title?: string;
  /** Small word set beside the big one, on the baseline: "HAIKU" + "ZERO". */
  titleTail?: string;
  /** How tall the wordmark may get, as a fraction of the hero. The cover wants it huge with the
   *  scene in front of it; a poem poster wants the poem readable, so it keeps the small default. */
  titleMaxFraction?: number;
  /** Rendered under the subheadline, inside the copy block (a start button, usually). */
  cta?: ReactNode;
  /** Copy-block scrim. "strong" is for a cover whose wordmark runs under the copy; the poem
   *  poster keeps the soft one so the photograph still shows through. */
  scrim?: "soft" | "strong";
  keywords?: SakuraEditorialKeyword[];
  headline?: string;
  body?: string;
  subheadline?: string;
  footerLeft?: string;
  footerCenter?: string;
  footerRight?: string;
  socialHandle?: string;
  sceneSrc?: string;
  sceneAlt?: string;
  foregroundSrc?: string | null;
  foregroundAlt?: string;
  height?: string;
  forceProgress?: number;
  preview?: boolean;
  /** Drop the page frame: no inset padding, no rounded card, no shadow — the scene reaches every edge. */
  fullBleed?: boolean;
  /** "compact" shrinks the copy block. The cover needs small type over a full-bleed photo;
   *  a poem poster needs the default, because those three lines are the thing being read. */
  copySize?: "default" | "compact";
  className?: string;
};

const FONT_LINK_ID = "sakura-editorial-poster-fonts";
const FONT_HREF =
  "https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Jost:wght@300;400;500;600&family=Saira+Extra+Condensed:wght@700;800&display=swap";

export const SAKURA_EDITORIAL_DEFAULT_KEYWORDS: SakuraEditorialKeyword[] = [
  { label: "Bloom" },
  { label: "Pause" },
  { label: "Return" },
];

const DEFAULT_BODY =
  "For a few still days the canopy turns pale pink, and the street below goes quiet. Walk while the color lasts — it is already leaving, petal by petal, into the wind.";

const FRAME_PAD_CLASS = "p-[clamp(1.25rem,4vmin,2.5rem)]";

// Width-fitting alone makes a short title gigantic: a five-letter word stretched to the row width
// ends up swallowing the whole poster and burying the poem under it. The title is capped to this
// fraction of the hero's height, so a long title still shrinks to fit but a short one stays a title.
const TITLE_MAX_HERO_FRACTION = 0.19;

// The wordmark is one fitted line, not two: the tail renders at a fixed fraction of the big size,
// so its width is measured at the probe size and folded into the same fit as the title.
const TITLE_TAIL_RATIO = 0.34;
const TITLE_TAIL_GAP_RATIO = 0.14;

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function getScrollParent(el: HTMLElement): HTMLElement | Window {
  let node: HTMLElement | null = el.parentElement;
  while (node) {
    const style = window.getComputedStyle(node);
    const oy = style.overflowY;
    const canScroll =
      (oy === "auto" || oy === "scroll" || oy === "overlay") &&
      node.scrollHeight > node.clientHeight + 1;
    if (canScroll) {
      if (node === document.documentElement || node === document.body) {
        return window;
      }
      return node;
    }
    node = node.parentElement;
  }
  return window;
}

function readScrollProgress(
  track: HTMLElement,
  scrollRoot: HTMLElement | Window,
): number {
  const useWindowScroll =
    !(scrollRoot instanceof HTMLElement) ||
    (typeof document !== "undefined" &&
      (scrollRoot === document.documentElement || scrollRoot === document.body));

  if (useWindowScroll) {
    const rect = track.getBoundingClientRect();
    const vh = window.innerHeight || 1;
    const scrollable = track.offsetHeight - vh;
    if (scrollable <= 0) return 1;
    return clamp01(-rect.top / scrollable);
  }

  const rootRect = scrollRoot.getBoundingClientRect();
  const trackRect = track.getBoundingClientRect();
  const scrollable = track.offsetHeight - scrollRoot.clientHeight;
  if (scrollable <= 0) return 1;
  return clamp01((rootRect.top - trackRect.top) / scrollable);
}

type TitleChar = {
  key: string;
  char: string;
  index: number;
  fromCenter: number;
};

function splitTitleChars(title: string): TitleChar[] {
  const chars = Array.from(title);
  const mid = Math.max(chars.length - 1, 1) / 2;
  return chars.map((char, index) => ({
    key: `${index}-${char === " " ? "sp" : char}`,
    char: char === " " ? "\u00A0" : char,
    index,
    fromCenter: mid <= 0 ? 0 : Math.abs(index - mid) / mid,
  }));
}

function charReveal(progress: number, fromCenter: number): number {
  const start = fromCenter * 0.55;
  const end = Math.min(1, start + 0.38);
  return clamp01((progress - start) / Math.max(0.001, end - start));
}

function useSakuraEditorialFonts() {
  useEffect(() => {
    if (typeof document === "undefined") return;
    if (document.getElementById(FONT_LINK_ID)) return;
    const link = document.createElement("link");
    link.id = FONT_LINK_ID;
    link.rel = "stylesheet";
    link.href = FONT_HREF;
    document.head.appendChild(link);
  }, []);
}

function SakuraFitTitle({
  title,
  tail,
  revealProgress,
  maxFraction,
}: {
  title: string;
  tail?: string;
  revealProgress: number;
  maxFraction: number;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const probeRef = useRef<HTMLSpanElement>(null);
  const tailProbeRef = useRef<HTMLSpanElement>(null);
  const [fontPx, setFontPx] = useState<number | null>(null);
  const chars = splitTitleChars(title);
  const titleProgress = clamp01(revealProgress / 0.4);
  const tailProgress = clamp01((revealProgress - 0.1) / 0.22);

  useEffect(() => {
    const wrap = wrapRef.current;
    const probe = probeRef.current;
    if (!wrap || !probe) return;

    const PROBE = 100;
    let cancelled = false;
    const fit = () => {
      if (cancelled) return;
      // both widths as a fraction of the font size they are drawn at
      const titleFraction = probe.scrollWidth / PROBE;
      const tailFraction = (tailProbeRef.current?.scrollWidth ?? 0) / PROBE;
      const denom =
        titleFraction +
        (tail ? TITLE_TAIL_GAP_RATIO + TITLE_TAIL_RATIO * tailFraction : 0);
      const widthFit = wrap.clientWidth / Math.max(0.0001, denom);
      if (!Number.isFinite(widthFit) || widthFit <= 0) return;
      const heroPx = wrap.parentElement?.clientHeight ?? 0;
      const cap = heroPx > 0 ? heroPx * maxFraction : widthFit;
      setFontPx(Math.min(widthFit, cap));
    };

    const ro = new ResizeObserver(fit);
    ro.observe(wrap);
    if (wrap.parentElement) ro.observe(wrap.parentElement);

    const fonts = document.fonts;
    const onFonts = () => {
      void fonts?.ready.then(fit);
    };
    fonts?.addEventListener?.("loadingdone", onFonts);
    void (async () => {
      try {
        await fonts?.load?.('800 100px "Saira Extra Condensed"');
      } catch {
        /* fallback metrics */
      }
      await fonts?.ready;
      fit();
    })();
    fit();

    return () => {
      cancelled = true;
      ro.disconnect();
      fonts?.removeEventListener?.("loadingdone", onFonts);
    };
  }, [title, tail, maxFraction]);

  const titleStyle = {
    fontFamily: '"Saira Extra Condensed", "Arial Narrow", sans-serif',
    fontWeight: 800,
    letterSpacing: "0.02em",
    WebkitFontSmoothing: "antialiased" as const,
    MozOsxFontSmoothing: "grayscale" as const,
    textRendering: "geometricPrecision" as const,
  };

  return (
    <div
      ref={wrapRef}
      className="absolute inset-x-[4%] top-[4%] z-20 overflow-visible"
    >
      <span
        ref={probeRef}
        aria-hidden
        className="pointer-events-none invisible absolute whitespace-nowrap uppercase leading-none"
        style={{ ...titleStyle, fontSize: 100 }}
      >
        {title}
      </span>
      <span
        ref={tailProbeRef}
        aria-hidden
        className="pointer-events-none invisible absolute whitespace-nowrap uppercase leading-none"
        style={{ ...titleStyle, fontSize: 100 }}
      >
        {tail ?? ""}
      </span>
      <h1
        className="m-0 flex items-end overflow-visible whitespace-nowrap text-left uppercase leading-none text-white"
        style={{
          ...titleStyle,
          fontSize: fontPx != null ? `${fontPx}px` : "min(36cqw, 52cqh)",
        }}
      >
        <span aria-hidden className="inline-block">
          {chars.map((item) => {
            const t = charReveal(titleProgress, item.fromCenter);
            const y = (1 - t) * (18 + item.fromCenter * 24);
            const side = item.index < chars.length / 2 ? 1 : -1;
            const x =
              (1 - t) *
              (item.fromCenter > 0.01 ? item.fromCenter * 16 * side : 0);
            return (
              <span
                key={item.key}
                aria-hidden
                className="inline-block"
                style={{
                  opacity: t,
                  transform: `translate3d(${x}px, ${y}px, 0)`,
                }}
              >
                {item.char}
              </span>
            );
          })}
        </span>
        {tail ? (
          <span
            aria-hidden
            className="inline-block uppercase leading-none"
            style={{
              marginLeft: `${TITLE_TAIL_GAP_RATIO}em`,
              fontSize: `${TITLE_TAIL_RATIO}em`,
              opacity: tailProgress,
              transform: `translate3d(0, ${(1 - tailProgress) * 16}px, 0)`,
            }}
          >
            {tail}
          </span>
        ) : null}
        <span className="sr-only">{tail ? `${title} ${tail}` : title}</span>
      </h1>
    </div>
  );
}

function SakuraHeroVisual({
  title,
  titleTail,
  titleMaxFraction,
  sceneSrc,
  sceneAlt,
  foregroundSrc,
  foregroundAlt,
  revealProgress,
}: {
  title: string;
  titleTail?: string;
  titleMaxFraction: number;
  sceneSrc: string;
  sceneAlt: string;
  foregroundSrc: string | null;
  foregroundAlt: string;
  revealProgress: number;
}) {
  return (
    <div className="relative h-full w-full overflow-hidden">
      <div className="absolute inset-0 z-0">
        <img
          src={sceneSrc}
          alt={sceneAlt}
          className="absolute inset-0 h-full w-full scale-105 object-cover object-center"
          draggable={false}
        />
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-[0.22] mix-blend-overlay"
          style={{
            backgroundImage:
              "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.55'/%3E%3C/svg%3E\")",
          }}
        />
      </div>

      <SakuraFitTitle
        title={title}
        tail={titleTail}
        revealProgress={revealProgress}
        maxFraction={titleMaxFraction}
      />

      {foregroundSrc ? (
        <img
          src={foregroundSrc}
          alt={foregroundAlt}
          className="pointer-events-none absolute bottom-0 left-1/2 z-30 h-auto w-[min(92%,78cqh)] -translate-x-[40%] object-contain object-bottom drop-shadow-[0_10px_28px_rgba(40,20,20,0.18)]"
          draggable={false}
        />
      ) : null}

      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 z-[35] bg-linear-to-b from-transparent via-transparent to-[#f5f5f0]/18"
      />
    </div>
  );
}

function SakuraEditorialCopy({
  keywordItems,
  headline,
  body,
  subheadline,
  cta,
  scrim,
  footerLeft,
  footerCenter,
  footerRight,
  socialHandle,
  dense,
}: {
  keywordItems: SakuraEditorialKeyword[];
  headline: string;
  body: string;
  subheadline: string;
  /** Cover-sized copy: noticeably smaller than the poem default. */
  dense: boolean;
  cta?: ReactNode;
  scrim: "soft" | "strong";
  footerLeft: string;
  footerCenter: string;
  footerRight: string;
  socialHandle?: string;
}) {
  return (
    <div
      className={cn(
        "relative flex min-h-[38%] flex-col border-0 bg-transparent p-[clamp(1.1rem,4.5cqw,2.25rem)] text-[#f6eee8]",
        // smaller type over a full-bleed photograph needs a lift off the flowers
        dense && "[text-shadow:0_1px_18px_rgba(24,10,14,0.55)]",
      )}
    >
      <div
        aria-hidden
        // four stops, not three: the copy band needs the dark held across its whole height, where
        // a from/via/to ramp is already down to a wash by the headline
        className={cn(
          "pointer-events-none absolute inset-0",
          scrim === "strong"
            ? "bg-[linear-gradient(to_top,rgba(31,19,23,0.93)_0%,rgba(39,24,29,0.82)_42%,rgba(62,40,46,0.42)_72%,rgba(62,40,46,0)_100%)]"
            : "bg-linear-to-t from-[#4a2c32]/55 via-[#c99aa0]/25 to-transparent",
        )}
      />
      <div className="relative z-10 flex items-start justify-between gap-3 text-[clamp(10px,1.5cqw,14px)] font-light tracking-[0.16em] text-[#f6eee8]/70">
        {keywordItems.map((item) => (
          <span key={item.label}>{item.label}</span>
        ))}
      </div>

      <h2
        className={cn(
          "relative z-10 mt-[clamp(0.7rem,2.2cqw,1.15rem)] font-semibold leading-[1.25] text-[#f6eee8]",
          dense ? "text-[clamp(1.05rem,2.05cqw,1.95rem)]" : "text-[clamp(1.3rem,3.2cqw,3.4rem)]",
        )}
        style={{
          fontFamily:
            '"Cormorant Garamond", "Hiragino Mincho ProN", "Yu Mincho", Georgia, serif',
        }}
      >
        {headline}
      </h2>

      <p
        className={cn(
          "relative z-10 mt-[clamp(0.5rem,1.6cqw,0.75rem)] max-w-[74%] font-light leading-[1.4] text-[#f6eee8]/85",
          dense ? "text-[clamp(0.85rem,1.35cqw,1.3rem)]" : "text-[clamp(1rem,2.3cqw,2.3rem)]",
        )}
      >
        {body}
      </p>

      <p
        className={cn(
          "relative z-10 mt-[clamp(0.65rem,2cqw,0.95rem)] font-medium leading-[1.3] text-[#f6eee8]",
          dense ? "text-[clamp(0.9rem,1.6cqw,1.5rem)]" : "text-[clamp(1.05rem,2.7cqw,2.7rem)]",
        )}
        style={{ fontFamily: '"Jost", ui-sans-serif, sans-serif' }}
      >
        {subheadline}
      </p>

      {cta ? (
        <div className="relative z-10 mt-[clamp(0.8rem,2.6cqw,1.35rem)]">{cta}</div>
      ) : null}

      <div className="relative z-10 mt-auto flex items-end justify-between gap-3 pt-[clamp(0.7rem,2.4cqw,1.1rem)] text-[clamp(10px,1.5cqw,14px)] font-light tracking-[0.08em] text-[#f6eee8]/75">
        <span>{footerLeft}</span>
        <span>{footerCenter}</span>
        <span>{footerRight}</span>
      </div>

      {socialHandle ? (
        <span className="absolute bottom-[clamp(0.35rem,1.2cqw,0.65rem)] right-[clamp(0.75rem,4.5cqw,2.25rem)] z-10 hidden text-[clamp(10px,1.8cqw,14px)] font-light tracking-[0.04em] text-[#f6eee8]/40 sm:block">
          {socialHandle}
        </span>
      ) : null}
    </div>
  );
}

export function SakuraEditorialPoster({
  title = "SAKURA",
  titleTail,
  titleMaxFraction = TITLE_MAX_HERO_FRACTION,
  cta,
  scrim = "soft",
  keywords = SAKURA_EDITORIAL_DEFAULT_KEYWORDS,
  headline = "Petals Hold the Light | 花びらが光を抱く。",
  body = DEFAULT_BODY,
  subheadline = "Stay for the fall. 散るまで、見ていて。",
  footerLeft = "DesignLayer",
  footerCenter = "Vol. 01",
  footerRight = "03.26 2026",
  socialHandle = "@designlayer",
  sceneSrc = SAKURA_SCENES[0].src,
  sceneAlt = SAKURA_SCENES[0].alt,
  foregroundSrc = null,
  foregroundAlt = "Cherry blossom branch in the foreground",
  height = "280vh",
  forceProgress,
  preview = false,
  fullBleed = false,
  copySize = "default",
  className,
}: SakuraEditorialPosterProps) {
  useSakuraEditorialFonts();

  const trackRef = useRef<HTMLElement>(null);
  const keywordItems = keywords.filter((item) => item.label.trim().length > 0);

  const locked = forceProgress != null && Number.isFinite(forceProgress);
  const [progress, setProgress] = useState(
    forceProgress != null ? clamp01(forceProgress) : 0,
  );
  const [stickyPx, setStickyPx] = useState<number | null>(null);

  const dense = copySize === "compact";
  const fillViewport = locked || preview;
  const trackHeight = fillViewport ? "auto" : height;
  const useSticky = !locked && !preview;

  useEffect(() => {
    // locked/preview read the prop directly when rendering, so the scroll state is not used at all
    // there - it used to be written anyway, which re-rendered on every mount for nothing
    if (locked || preview) return;

    const track = trackRef.current;
    if (!track) return;

    const reduceMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    if (reduceMotion) {
      setProgress(1);
      return;
    }

    const scrollRoot = getScrollParent(track);
    let target = 0;
    let current = 0;
    let raf = 0;

    const read = () => {
      const fromRoot = readScrollProgress(track, scrollRoot);
      if (scrollRoot === window) return fromRoot;
      const fromWindow = readScrollProgress(track, window);
      return Math.abs(fromWindow - fromRoot) > 0.02 ? fromWindow : fromRoot;
    };

    const loop = () => {
      const delta = target - current;
      current += Math.abs(delta) > 0.35 ? delta * 0.22 : delta * 0.14;
      if (Math.abs(delta) < 0.0008) current = target;
      setProgress(current);
      raf = window.requestAnimationFrame(loop);
    };

    const onScroll = () => {
      target = read();
    };

    const onResize = () => {
      if (scrollRoot === window) {
        setStickyPx(window.innerHeight);
      } else {
        setStickyPx((scrollRoot as HTMLElement).clientHeight);
      }
      target = read();
    };

    onResize();
    target = read();
    current = target;
    setProgress(current);

    const opts: AddEventListenerOptions = { passive: true };
    scrollRoot.addEventListener("scroll", onScroll, opts);
    window.addEventListener("resize", onResize);
    raf = window.requestAnimationFrame(loop);

    return () => {
      scrollRoot.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onResize);
      window.cancelAnimationFrame(raf);
    };
  }, [forceProgress, locked, preview]);

  const revealProgress = locked || preview ? clamp01(forceProgress ?? 0) : progress;
  const copyProgress = clamp01((revealProgress - 0.78) / 0.22);
  const copyOffset = `${(1 - copyProgress) * 100}%`;
  const panelHeight =
    useSticky && stickyPx != null
      ? stickyPx
      : fillViewport
        ? "100%"
        : ("100svh" as const);

  return (
    <section
      ref={trackRef}
      className={cn(
        "relative isolate w-full bg-[#ece8df]",
        fillViewport && "h-screen",
        className,
      )}
      style={{
        height: useSticky ? trackHeight : undefined,
        fontFamily: '"Jost", ui-sans-serif, sans-serif',
      }}
    >
      <div
        className={cn(
          "box-border w-full overflow-hidden",
          fullBleed ? "p-0" : FRAME_PAD_CLASS,
          useSticky ? "sticky top-0" : "relative",
        )}
        style={{ height: panelHeight }}
      >
        <article
          className={cn(
            "@container relative flex h-full w-full min-h-0 flex-col overflow-hidden bg-[#f5f5f0]",
            fullBleed ? "rounded-none" : "rounded-xl shadow-[0_24px_80px_rgba(80,50,50,0.12)]",
          )}
        >
          <div className="@container relative min-h-0 flex-1 overflow-hidden [container-type:size]">
            <SakuraHeroVisual
              title={title}
              titleTail={titleTail}
              titleMaxFraction={titleMaxFraction}
              sceneSrc={sceneSrc}
              sceneAlt={sceneAlt}
              foregroundSrc={foregroundSrc}
              foregroundAlt={foregroundAlt}
              revealProgress={revealProgress}
            />
          </div>

          <div
            className="absolute inset-x-0 bottom-0 z-30 will-change-transform"
            style={{ transform: `translate3d(0, ${copyOffset}, 0)` }}
          >
            <SakuraEditorialCopy
              keywordItems={keywordItems}
              headline={headline}
              body={body}
              subheadline={subheadline}
              cta={cta}
              scrim={scrim}
              dense={dense}
              footerLeft={footerLeft}
              footerCenter={footerCenter}
              footerRight={footerRight}
              socialHandle={socialHandle}
            />
          </div>
        </article>
      </div>
    </section>
  );
}

export default SakuraEditorialPoster;
