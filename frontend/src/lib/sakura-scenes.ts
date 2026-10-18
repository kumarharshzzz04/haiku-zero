/** Unsplash stock (verified reachable), used instead of the poster's original demo assets.
 *
 * Kept out of the component module so `sakura-editorial-poster.tsx` stays a pure component file
 * (React fast refresh only works when a module exports components alone).
 */
export const SAKURA_SCENES = [
  {
    src: "https://images.unsplash.com/photo-1522383225653-ed111181a951?w=1600&q=80",
    alt: "Cherry blossom branch against a soft sky",
  },
  {
    src: "https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=1600&q=80",
    alt: "Pink blossom petals in close-up",
  },
  {
    src: "https://images.unsplash.com/photo-1520763185298-1b434c919102?w=1600&q=80",
    alt: "Cherry blossoms filling the frame",
  },
  {
    src: "https://images.unsplash.com/photo-1552168324-d612d77725e3?w=1600&q=80",
    alt: "Blossom petals on a dark branch",
  },
] as const
