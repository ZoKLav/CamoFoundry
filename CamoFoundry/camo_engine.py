"""
Camo Foundry image generator engine.

Pure Pillow + NumPy. No GUI code lives here so it can be tested without Qt.
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

RGB = Tuple[int, int, int]

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - for ancient Pillow only
    RESAMPLE_BICUBIC = Image.BICUBIC
    RESAMPLE_NEAREST = Image.NEAREST
    RESAMPLE_BILINEAR = Image.BILINEAR

PATTERNS = [
    "Blob / Organic",
    "Woodland",
    "Tiger Stripe",
    "Spray Paint",
    "Digital",
    "Flecktarn",
    "Splinter",
    "Topographic",
    "Rain Streak",
    "Brush Stroke",
]

PALETTES: Dict[str, List[RGB]] = {
    "Woodland Classic": [(35, 47, 28), (75, 91, 50), (119, 113, 74), (58, 45, 34), (19, 20, 17)],
    "Jungle Deep": [(8, 27, 15), (23, 68, 34), (49, 104, 45), (100, 121, 58), (18, 17, 12)],
    "Desert 3-Color": [(212, 190, 140), (174, 136, 83), (119, 88, 56), (238, 224, 176), (83, 63, 45)],
    "Arid Digital": [(198, 174, 121), (151, 125, 80), (99, 80, 55), (222, 211, 170), (60, 56, 48)],
    "Urban Gray": [(30, 32, 35), (73, 78, 82), (126, 132, 136), (190, 195, 196), (12, 13, 15)],
    "Snow / Alpine": [(236, 238, 235), (194, 203, 203), (141, 155, 160), (72, 80, 83), (20, 23, 25)],
    "Flecktarn-ish": [(39, 56, 30), (83, 107, 49), (150, 127, 70), (95, 57, 34), (30, 21, 18)],
    "Tiger Jungle": [(29, 43, 22), (87, 107, 45), (145, 142, 80), (42, 34, 27), (10, 10, 8)],
    "Navy Blue": [(7, 18, 31), (19, 47, 73), (47, 82, 112), (83, 114, 137), (6, 8, 12)],
    "Rust / Industrial": [(41, 37, 33), (92, 70, 50), (142, 80, 44), (187, 123, 66), (22, 21, 19)],
    "High Contrast Test": [(0, 0, 0), (64, 64, 64), (128, 128, 128), (192, 192, 192), (255, 255, 255)],
    "Silly Pink Toy": [(49, 25, 44), (107, 48, 91), (177, 83, 138), (234, 151, 192), (252, 223, 239)],
}

DEFAULT_OPTIONS = {
    "pattern": "Blob / Organic",
    "palette": "Woodland Classic",
    "seed": 12345,
    "scale": 90,
    "detail": 4,
    "density": 50,
    "contrast": 55,
    "roughness": 55,
    "blur": 3,
    "edge_softness": 8,
    "stripe_width": 46,
    "stripe_spacing": 112,
    "stripe_wiggle": 90,
    "block_size": 28,
    "dot_size": 11,
    "speckle": 18,
    "background_noise": 12,
    "hsv_jitter": 0,
    "color_bleed": 0,
    "rotation": 0,
    "seamless": False,
    "invert": False,
    "shuffle_colors": False,
    "outline": False,
    "smooth_preview": True,
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clean_palette(colors: Sequence[RGB] | None) -> List[RGB]:
    if not colors:
        return list(PALETTES[DEFAULT_OPTIONS["palette"]])
    out: List[RGB] = []
    for color in colors:
        if color is None:
            continue
        if len(color) != 3:
            continue
        out.append(tuple(int(clamp(c, 0, 255)) for c in color))
    if len(out) < 2:
        out = list(PALETTES[DEFAULT_OPTIONS["palette"]])
    return out[:8]


def jitter_palette(colors: Sequence[RGB], rng: np.random.Generator, amount: int) -> List[RGB]:
    amount = int(clamp(amount, 0, 100))
    if amount <= 0:
        return list(colors)
    jittered: List[RGB] = []
    hue_shift = amount / 720.0
    sv_shift = amount / 180.0
    for r, g, b in colors:
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        h = (h + rng.uniform(-hue_shift, hue_shift)) % 1.0
        s = clamp(s + rng.uniform(-sv_shift, sv_shift), 0, 1)
        v = clamp(v + rng.uniform(-sv_shift, sv_shift), 0, 1)
        rr, gg, bb = colorsys.hsv_to_rgb(h, s, v)
        jittered.append((int(rr * 255), int(gg * 255), int(bb * 255)))
    return jittered


def _noise_layer(width: int, height: int, grid_w: int, grid_h: int, rng: np.random.Generator, seamless: bool) -> np.ndarray:
    grid_w = max(2, int(grid_w))
    grid_h = max(2, int(grid_h))
    grid = rng.random((grid_h + 1, grid_w + 1), dtype=np.float32)
    if seamless:
        grid[:, -1] = grid[:, 0]
        grid[-1, :] = grid[0, :]
    img = Image.fromarray(np.uint8(grid * 255), mode="L").resize((width, height), RESAMPLE_BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def fractal_noise(
    width: int,
    height: int,
    rng: np.random.Generator,
    patch_size: float,
    octaves: int,
    roughness: int,
    seamless: bool = False,
) -> np.ndarray:
    patch_size = max(3.0, float(patch_size))
    octaves = int(clamp(octaves, 1, 8))
    persistence = 0.35 + (clamp(roughness, 0, 100) / 100.0) * 0.45

    result = np.zeros((height, width), dtype=np.float32)
    total_amp = 0.0
    amp = 1.0
    for octave in range(octaves):
        current_patch = max(2.0, patch_size / (2 ** octave))
        grid_w = max(2, math.ceil(width / current_patch))
        grid_h = max(2, math.ceil(height / current_patch))
        result += _noise_layer(width, height, grid_w, grid_h, rng, seamless) * amp
        total_amp += amp
        amp *= persistence
    result /= max(total_amp, 0.0001)
    return normalize(result)


def normalize(a: np.ndarray) -> np.ndarray:
    amin = float(np.min(a))
    amax = float(np.max(a))
    if amax - amin < 1e-6:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - amin) / (amax - amin)).astype(np.float32)


def quantize_noise(noise: np.ndarray, colors: Sequence[RGB], options: Dict) -> np.ndarray:
    palette = np.asarray(colors, dtype=np.uint8)
    contrast = 0.35 + clamp(float(options.get("contrast", 50)), 0, 100) / 50.0
    density = (clamp(float(options.get("density", 50)), 0, 100) - 50) / 160.0

    n = (noise - 0.5) * contrast + 0.5 + density
    if options.get("invert", False):
        n = 1.0 - n
    n = np.clip(n, 0.0, 0.99999)
    idx = np.floor(n * len(palette)).astype(np.int16)
    idx = np.clip(idx, 0, len(palette) - 1)
    return palette[idx]


def apply_finish(img: Image.Image, options: Dict, rng: np.random.Generator) -> Image.Image:
    img = img.convert("RGB")

    bleed = int(clamp(options.get("color_bleed", 0), 0, 100))
    if bleed > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=bleed / 12.0))

    noise_amount = int(clamp(options.get("background_noise", 0), 0, 100))
    if noise_amount > 0:
        arr = np.asarray(img, dtype=np.int16)
        grit = rng.integers(-noise_amount, noise_amount + 1, size=arr.shape, dtype=np.int16)
        arr = np.clip(arr + grit, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, "RGB")

    angle = float(options.get("rotation", 0))
    if abs(angle) > 0.01:
        img = img.rotate(angle, resample=RESAMPLE_BICUBIC, expand=False)

    return img


def render_camo(size: int = 2048, options: Dict | None = None, colors: Sequence[RGB] | None = None) -> Image.Image:
    opts = dict(DEFAULT_OPTIONS)
    if options:
        opts.update(options)

    size = int(size)
    width = height = max(64, size)
    rng = np.random.default_rng(int(opts.get("seed", 1)) & 0xFFFFFFFF)
    palette = clean_palette(colors or PALETTES.get(str(opts.get("palette")), PALETTES[DEFAULT_OPTIONS["palette"]]))
    if opts.get("shuffle_colors", False):
        rng.shuffle(palette)
    palette = jitter_palette(palette, rng, int(opts.get("hsv_jitter", 0)))

    pattern = str(opts.get("pattern", DEFAULT_OPTIONS["pattern"]))
    if pattern == "Woodland":
        img = _woodland(width, height, palette, opts, rng)
    elif pattern == "Tiger Stripe":
        img = _tiger_stripe(width, height, palette, opts, rng)
    elif pattern == "Spray Paint":
        img = _spray_paint(width, height, palette, opts, rng)
    elif pattern == "Digital":
        img = _digital(width, height, palette, opts, rng)
    elif pattern == "Flecktarn":
        img = _flecktarn(width, height, palette, opts, rng)
    elif pattern == "Splinter":
        img = _splinter(width, height, palette, opts, rng)
    elif pattern == "Topographic":
        img = _topographic(width, height, palette, opts, rng)
    elif pattern == "Rain Streak":
        img = _rain_streak(width, height, palette, opts, rng)
    elif pattern == "Brush Stroke":
        img = _brush_stroke(width, height, palette, opts, rng)
    else:
        img = _blob(width, height, palette, opts, rng)

    img = apply_finish(img, opts, rng)
    return img


def _blob(width: int, height: int, palette: Sequence[RGB], opts: Dict, rng: np.random.Generator) -> Image.Image:
    scale = int(clamp(opts.get("scale", 90), 5, 400))
    detail = int(clamp(opts.get("detail", 4), 1, 8))
    soft = int(clamp(opts.get("edge_softness", 8), 0, 100))
    n = fractal_noise(width, height, rng, scale, detail, int(opts.get("roughness", 55)), bool(opts.get("seamless", False)))
    if soft > 0:
        n_img = Image.fromarray(np.uint8(n * 255), "L").filter(ImageFilter.GaussianBlur(radius=soft / 9.0))
        n = np.asarray(n_img, dtype=np.float32) / 255.0
    arr = quantize_noise(n, palette, opts)
    return Image.fromarray(arr, "RGB")


def _woodland(width: int, height: int, palette: Sequence[RGB], opts: Dict, rng: np.random.Generator) -> Image.Image:
    scale = int(clamp(opts.get("scale", 90), 5, 400))
    n1 = fractal_noise(width, height, rng, scale * 1.7, int(opts.get("detail", 4)), int(opts.get("roughness", 55)), bool(opts.get("seamless", False)))
    n2 = fractal_noise(width, height, rng, max(8, scale / 2), 3, 70, bool(opts.get("seamless", False)))
    n = normalize(n1 * 0.78 + n2 * 0.22)
    img = Image.fromarray(quantize_noise(n, palette, opts), "RGB")

    if opts.get("outline", False) and len(palette) > 2:
        draw = ImageDraw.Draw(img, "RGBA")
        count = max(6, int(width * height / 180_000))
        for _ in range(count):
            y = rng.integers(-height // 4, height + height // 4)
            points = []
            phase = rng.uniform(0, math.tau)
            amp = rng.uniform(20, 80) * width / 1024
            for x in range(-width // 6, width + width // 6, max(12, width // 80)):
                points.append((x, y + math.sin(x / width * math.tau * 2 + phase) * amp + rng.normal(0, amp / 5)))
            draw.line(points, fill=(*palette[-1], 175), width=max(3, width // 140))
    return img




def _paste_rgba_with_mask(base: Image.Image, mask: Image.Image, color: RGB, alpha: int = 255) -> None:
    """Composite a solid color into an RGBA image using a grayscale mask."""
    layer = Image.new("RGBA", base.size, (*color, int(clamp(alpha, 0, 255))))
    base.alpha_composite(Image.composite(layer, Image.new("RGBA", base.size, (0, 0, 0, 0)), mask))


def _paint_grid_rect(grid: np.ndarray, x0: int, y0: int, x1: int, y1: int, value: int, wrap: bool) -> None:
    """Paint a rectangle into a small digital-camo index grid, optionally wrapping edges."""
    h, w = grid.shape
    if not wrap:
        xa = max(0, min(w, x0))
        xb = max(0, min(w, x1))
        ya = max(0, min(h, y0))
        yb = max(0, min(h, y1))
        if xb > xa and yb > ya:
            grid[ya:yb, xa:xb] = value
        return

    # Split wrapped rectangles into ordinary pieces. This keeps the seamless toggle useful for pixel patterns.
    xs = list(range(x0, x1))
    ys = list(range(y0, y1))
    if not xs or not ys:
        return
    for yy in ys:
        grid[yy % h, [xx % w for xx in xs]] = value


def _irregular_fleck_points(x: float, y: float, radius: float, rng: np.random.Generator, stretch: float = 1.0) -> List[Tuple[float, float]]:
    sides = int(rng.integers(5, 10))
    start = rng.uniform(0, math.tau)
    pts: List[Tuple[float, float]] = []
    for i in range(sides):
        theta = start + (i / sides) * math.tau + rng.normal(0, 0.10)
        rr = radius * rng.uniform(0.45, 1.20)
        pts.append((x + math.cos(theta) * rr * stretch, y + math.sin(theta) * rr * rng.uniform(0.75, 1.20)))
    return pts


def _draw_fleck(draw: ImageDraw.ImageDraw, x: float, y: float, radius: float, color: RGB, rng: np.random.Generator, alpha: int = 255, stretch: float = 1.0) -> None:
    radius = max(0.65, float(radius))
    # Tiny flecks look more authentic as rough pixels/crumbs than as perfect circles.
    if radius <= 1.6:
        r = max(1, int(round(radius)))
        draw.rectangle([int(x) - r, int(y) - r, int(x) + r, int(y) + r], fill=(*color, alpha))
        return
    draw.polygon(_irregular_fleck_points(x, y, radius, rng, stretch), fill=(*color, alpha))


def _tapered_band_mask(width: int, height: int, rng: np.random.Generator, x0: float, y0: float, length: float, angle: float, thickness: float, wiggle: float, roughness: int) -> Image.Image:
    """Create one broken/tapered tiger-stripe band mask."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    segments = int(rng.integers(15, 28))
    phase = rng.uniform(0, math.tau)
    freq = rng.uniform(0.55, 1.35)
    centers: List[Tuple[float, float]] = []
    for i in range(segments):
        t = i / max(1, segments - 1)
        cx = x0 + math.cos(angle) * length * t
        cy = y0 + math.sin(angle) * length * t
        side = math.sin(t * math.tau * freq + phase) * wiggle + rng.normal(0, max(0.1, wiggle * 0.06))
        cx += math.cos(angle + math.pi / 2) * side
        cy += math.sin(angle + math.pi / 2) * side
        centers.append((cx, cy))

    left: List[Tuple[float, float]] = []
    right: List[Tuple[float, float]] = []
    for i, (cx, cy) in enumerate(centers):
        if i == 0:
            dx = centers[i + 1][0] - cx
            dy = centers[i + 1][1] - cy
        elif i == len(centers) - 1:
            dx = cx - centers[i - 1][0]
            dy = cy - centers[i - 1][1]
        else:
            dx = centers[i + 1][0] - centers[i - 1][0]
            dy = centers[i + 1][1] - centers[i - 1][1]
        mag = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / mag, dx / mag
        t = i / max(1, len(centers) - 1)
        taper = math.sin(math.pi * t) ** rng.uniform(0.24, 0.44)
        # Low-frequency width variation, not high-frequency noodle wobble.
        width_wave = 0.86 + 0.22 * math.sin(t * math.tau * rng.uniform(0.6, 1.4) + phase)
        half = max(1.0, thickness * taper * width_wave * rng.uniform(0.92, 1.08))
        left.append((cx + nx * half, cy + ny * half))
        right.append((cx - nx * half * rng.uniform(0.90, 1.12), cy - ny * half * rng.uniform(0.90, 1.12)))

    if len(left) + len(right) >= 3:
        draw.polygon(left + right[::-1], fill=255)

    # Bite notches out of the band. This makes broken cloth-print stripes, not smooth marker strokes.
    notch_count = int(rng.integers(1, 5 + max(1, int(roughness / 25))))
    for _ in range(notch_count):
        t = rng.uniform(0.08, 0.94)
        cx = x0 + math.cos(angle) * length * t + rng.normal(0, thickness * 0.38)
        cy = y0 + math.sin(angle) * length * t + rng.normal(0, thickness * 0.50)
        notch_len = rng.uniform(thickness * 0.7, thickness * 2.2)
        notch_w = rng.uniform(thickness * 0.20, thickness * 0.62)
        side = rng.choice([-1, 1])
        bx = math.cos(angle + math.pi / 2) * side
        by = math.sin(angle + math.pi / 2) * side
        ax = math.cos(angle)
        ay = math.sin(angle)
        pts = [
            (cx + bx * notch_w, cy + by * notch_w),
            (cx + ax * notch_len + bx * notch_w * 1.35, cy + ay * notch_len + by * notch_w * 1.35),
            (cx + ax * notch_len * 0.25 - bx * notch_w, cy + ay * notch_len * 0.25 - by * notch_w),
        ]
        draw.polygon(pts, fill=0)

    if roughness > 60:
        # Small edge speckle only. The previous full-mask noise made the stripes shatter into weird shards.
        specks = int((width * height / 120_000) * ((roughness - 55) / 45.0))
        for _ in range(max(0, specks)):
            sx = rng.uniform(0, width)
            sy = rng.uniform(0, height)
            rr = rng.uniform(max(1, thickness * 0.10), max(2, thickness * 0.28))
            draw.ellipse([sx - rr, sy - rr, sx + rr, sy + rr], fill=int(rng.choice([0, 255])))
    return mask


def _digital(width: int, height: int, palette: Sequence[RGB], opts: Dict, rng: np.random.Generator) -> Image.Image:
    """Generate CADPAT/MARPAT-ish digital camo.

    The old version was just a quantized square grid. This version uses a small pixel
    module, low-frequency region masks, and hundreds of L/T/block clusters so the
    output reads as pixelated camouflage instead of bathroom tile.
    """
    resolution = width / 2048.0
    requested_block = int(clamp(float(opts.get("block_size", 28)) * resolution, 2, 180))
    module = int(clamp(round(requested_block * 0.45), 2, 28))
    cw = max(8, math.ceil(width / module))
    ch = max(8, math.ceil(height / module))
    seamless = bool(opts.get("seamless", False))

    sub_opts = dict(opts)
    sub_opts["invert"] = False
    sub_opts["contrast"] = clamp(float(opts.get("contrast", 55)) + 12, 0, 100)
    scale = int(clamp(float(opts.get("scale", 90)) * resolution, 5, 400))
    detail = int(clamp(opts.get("detail", 4), 1, 8))
    roughness = int(clamp(opts.get("roughness", 55), 0, 100))

    macro = fractal_noise(cw, ch, rng, max(4, scale / module * 2.4), max(2, min(detail, 5)), roughness, seamless)
    mid = fractal_noise(cw, ch, rng, max(3, scale / module * 0.80), max(2, min(detail + 1, 6)), min(100, roughness + 15), seamless)
    micro = rng.random((ch, cw), dtype=np.float32)
    n = normalize(macro * 0.62 + mid * 0.30 + micro * 0.08)

    if opts.get("invert", False):
        n = 1.0 - n

    palette_arr = np.asarray(palette, dtype=np.uint8)
    idx = np.floor(np.clip(n, 0, 0.9999) * len(palette_arr)).astype(np.int16)

    density = clamp(float(opts.get("density", 50)), 0, 100) / 100.0
    speckle = clamp(float(opts.get("speckle", 18)), 0, 100) / 100.0
    cluster_count = int((cw * ch / 95.0) * (0.45 + density * 1.15) * (0.55 + speckle * 1.35))
    max_cluster_radius = max(2, int(scale / max(1, module) / 2.2))

    for _ in range(cluster_count):
        color_idx = int(rng.integers(0, len(palette)))
        cx = int(rng.integers(0, cw))
        cy = int(rng.integers(0, ch))
        pieces = int(rng.integers(2, 8 + int(speckle * 8)))
        radius = int(rng.integers(2, max(3, max_cluster_radius + 1)))
        for _piece in range(pieces):
            ox = int(round(rng.normal(0, radius * 0.55)))
            oy = int(round(rng.normal(0, radius * 0.55)))
            # Mostly small rectangles; occasional longer bars create the familiar pixel stair-steps.
            rw = int(rng.integers(1, max(2, min(8, radius + 2))))
            rh = int(rng.integers(1, max(2, min(8, radius + 2))))
            if rng.random() < 0.38:
                if rng.random() < 0.5:
                    rw *= int(rng.integers(2, 5))
                else:
                    rh *= int(rng.integers(2, 5))
            x0 = cx + ox - rw // 2
            y0 = cy + oy - rh // 2
            _paint_grid_rect(idx, x0, y0, x0 + rw, y0 + rh, color_idx, seamless)

    # Fine single/double modules stop large regions from looking like plain square chunks.
    salt_count = int(cw * ch * (0.010 + speckle * 0.055))
    for _ in range(salt_count):
        x = int(rng.integers(0, cw))
        y = int(rng.integers(0, ch))
        color_idx = int(rng.integers(0, len(palette)))
        w = int(rng.integers(1, 3))
        h = int(rng.integers(1, 3))
        _paint_grid_rect(idx, x, y, x + w, y + h, color_idx, seamless)

    if opts.get("shuffle_colors", False):
        # Already handled globally, but leave the index order stable here.
        pass

    small = Image.fromarray(palette_arr[np.clip(idx, 0, len(palette_arr) - 1)], "RGB")
    img = small.resize((width, height), RESAMPLE_NEAREST)
    return img.crop((0, 0, width, height))


def _tiger_stripe(width: int, height: int, palette: Sequence[RGB], opts: Dict, rng: np.random.Generator) -> Image.Image:
    """Generate tiger-stripe camo with tapered broken bands instead of sine-wave lines."""
    resolution = width / 2048.0
    base_color = palette[1] if len(palette) > 1 else palette[0]
    img = Image.new("RGBA", (width, height), (*base_color, 255))

    # Muted under-color swaths, kept low opacity so the dark stripes remain the star of the show.
    under_colors = [palette[0]]
    if len(palette) > 2:
        under_colors.append(palette[2])
    for i in range(max(10, int(height / max(18, 95 * resolution)))):
        mask = _tapered_band_mask(
            width,
            height,
            rng,
            rng.uniform(-width * 0.25, width * 0.80),
            rng.uniform(-height * 0.10, height * 1.10),
            rng.uniform(width * 0.24, width * 0.85),
            rng.uniform(-0.22, 0.22),
            float(opts.get("stripe_width", 46)) * resolution * rng.uniform(0.65, 1.25),
            float(opts.get("stripe_wiggle", 90)) * resolution * 0.12,
            15,
        ).filter(ImageFilter.GaussianBlur(radius=max(0.25, 1.0 * resolution)))
        _paste_rgba_with_mask(img, mask, under_colors[i % len(under_colors)], alpha=95)

    stripe_width = float(clamp(opts.get("stripe_width", 46), 3, 260)) * resolution
    spacing = max(10, int(float(clamp(opts.get("stripe_spacing", 112), 18, 420)) * resolution))
    wiggle = float(clamp(opts.get("stripe_wiggle", 90), 0, 300)) * resolution * 0.18
    density = clamp(float(opts.get("density", 50)), 0, 100) / 50.0
    roughness = int(clamp(opts.get("roughness", 55), 0, 100))
    edge_soft = float(clamp(opts.get("edge_softness", 8), 0, 100))

    dark = palette[-1]
    brown = palette[3] if len(palette) > 3 else dark
    step = max(9, int(spacing / max(0.55, density)))

    y = -step
    stripe_index = 0
    while y < height + step:
        bands_on_row = 1 + int(rng.random() < 0.28 * min(1.4, density))
        for _band in range(bands_on_row):
            length = rng.uniform(width * 0.36, width * 1.18)
            x0 = rng.uniform(-width * 0.42, width * 0.68)
            y0 = y + rng.normal(0, step * 0.30)
            angle = rng.uniform(-0.16, 0.16)
            thickness = max(2.0, stripe_width * rng.uniform(0.75, 1.75))
            mask = _tapered_band_mask(width, height, rng, x0, y0, length, angle, thickness, wiggle, roughness)
            if edge_soft > 0:
                mask = mask.filter(ImageFilter.GaussianBlur(radius=edge_soft / 55.0))

            if opts.get("outline", False) or rng.random() < 0.35:
                under = mask.filter(ImageFilter.MaxFilter(3))
                _paste_rgba_with_mask(img, under, brown, alpha=145)
            _paste_rgba_with_mask(img, mask, dark if stripe_index % 4 else brown, alpha=246)
            stripe_index += 1

            if rng.random() < 0.42:
                small_len = rng.uniform(width * 0.10, width * 0.30)
                small_x = x0 + rng.uniform(0, max(1, length * 0.85))
                small_y = y0 + rng.normal(step * 0.42, step * 0.24)
                small_mask = _tapered_band_mask(width, height, rng, small_x, small_y, small_len, angle + rng.normal(0, 0.07), max(1.5, thickness * rng.uniform(0.32, 0.65)), wiggle * 0.65, roughness)
                if edge_soft > 0:
                    small_mask = small_mask.filter(ImageFilter.GaussianBlur(radius=edge_soft / 65.0))
                _paste_rgba_with_mask(img, small_mask, dark, alpha=225)
        y += step * rng.uniform(0.78, 1.18)

    return img.convert("RGB")


def _spray_paint(width: int, height: int, palette: Sequence[RGB], opts: Dict, rng: np.random.Generator) -> Image.Image:
    bg = palette[0]
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img, "RGBA")
    density = clamp(float(opts.get("density", 50)), 0, 100) / 50.0
    dot_size = int(clamp(opts.get("dot_size", 11), 1, 90))
    scale = int(clamp(opts.get("scale", 90), 5, 400))
    count = min(140_000, int(width * height / max(45, 420 - density * 170)))
    for _ in range(count):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        r = max(1, int(abs(rng.normal(dot_size, dot_size * 0.8))))
        color = palette[int(rng.integers(1 if len(palette) > 1 else 0, len(palette)))]
        alpha = int(rng.integers(45, 210))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(*color, alpha))

    # Occasional soft clouds so it does not look like static unless requested.
    cloud = fractal_noise(width, height, rng, scale, int(opts.get("detail", 4)), int(opts.get("roughness", 55)), bool(opts.get("seamless", False)))
    overlay = Image.fromarray(quantize_noise(cloud, palette, opts), "RGB").filter(ImageFilter.GaussianBlur(radius=max(0.5, dot_size / 2)))
    return Image.blend(img, overlay, 0.28)


def _flecktarn(width: int, height: int, palette: Sequence[RGB], opts: Dict, rng: np.random.Generator) -> Image.Image:
    """Generate a Flecktarn-inspired field of clustered irregular flecks."""
    resolution = width / 2048.0
    density = clamp(float(opts.get("density", 50)), 0, 100) / 50.0
    dot = max(1.15, float(clamp(opts.get("dot_size", 11), 1, 90)) * resolution * 1.22)
    cluster_scale = max(8.0, float(clamp(opts.get("scale", 90), 8, 400)) * resolution)
    roughness = clamp(float(opts.get("roughness", 55)), 0, 100) / 100.0
    seamless = bool(opts.get("seamless", False))

    base_color = palette[1] if len(palette) > 1 else palette[0]
    img = Image.new("RGB", (width, height), base_color)
    draw = ImageDraw.Draw(img, "RGBA")

    # Flecktarn-ish order for a five-color palette: tan, dark green, brown, green, black.
    if len(palette) >= 5:
        layer_indices = [2, 0, 3, 1, 4]
    else:
        layer_indices = list(range(len(palette)))
    seen = set()
    layer_indices = [i for i in layer_indices if not (i in seen or seen.add(i)) and i < len(palette)]

    area = width * height
    for order, color_index in enumerate(layer_indices):
        color = palette[color_index]
        if color == base_color and order > 0:
            continue
        brightness = sum(color)
        is_dark = brightness < 115 or color_index == layer_indices[-1]
        if is_dark:
            weight = 0.58
            fleck_size = 0.62
        elif order == 0:
            weight = 0.92
            fleck_size = 1.05
        else:
            weight = 0.82
            fleck_size = 0.86

        clusters = int((area / 6_500.0) * density * weight)
        clusters = max(int(42 * max(1.0, width / 512.0) ** 2 * weight), clusters)
        for _ in range(clusters):
            cx = rng.uniform(-cluster_scale, width + cluster_scale)
            cy = rng.uniform(-cluster_scale, height + cluster_scale)
            spread_x = rng.uniform(cluster_scale * 0.24, cluster_scale * (1.15 + roughness * 0.55))
            spread_y = rng.uniform(cluster_scale * 0.20, cluster_scale * (0.88 + roughness * 0.45))
            flecks = int(rng.integers(16, 46) * (0.70 + density * 0.30) * (0.72 if is_dark else 1.0))
            for _f in range(flecks):
                x = cx + rng.normal(0, spread_x)
                y = cy + rng.normal(0, spread_y)
                radius = abs(rng.normal(dot * fleck_size, dot * (0.24 + roughness * 0.30)))
                radius = clamp(radius, max(0.65, dot * 0.15), max(1.4, dot * (1.55 if not is_dark else 1.05)))
                stretch = rng.uniform(0.75, 1.65)
                positions = [(x, y)]
                if seamless:
                    if x < cluster_scale: positions.append((x + width, y))
                    if x > width - cluster_scale: positions.append((x - width, y))
                    if y < cluster_scale: positions.append((x, y + height))
                    if y > height - cluster_scale: positions.append((x, y - height))
                for px, py in positions:
                    if -radius * 2 <= px <= width + radius * 2 and -radius * 2 <= py <= height + radius * 2:
                        _draw_fleck(draw, px, py, radius, color, rng, alpha=int(rng.integers(218, 256)), stretch=stretch)

    pepper_colors = [palette[-1]]
    if len(palette) > 3:
        pepper_colors.append(palette[3])
    pepper_count = int((area / 190.0) * clamp(density, 0.25, 2.0) * (0.60 + clamp(float(opts.get("speckle", 18)), 0, 100) / 120.0))
    pepper_count = min(70_000, pepper_count)
    for _ in range(pepper_count):
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        radius = rng.uniform(max(0.6, dot * 0.10), max(1.2, dot * 0.34))
        color = pepper_colors[int(rng.integers(0, len(pepper_colors)))]
        _draw_fleck(draw, x, y, radius, color, rng, alpha=int(rng.integers(190, 256)), stretch=rng.uniform(0.75, 1.45))

    edge_soft = float(clamp(opts.get("edge_softness", 8), 0, 100))
    if edge_soft > 0:
        # Very light softening only; Flecktarn should stay crumbly and crisp.
        img = img.filter(ImageFilter.GaussianBlur(radius=edge_soft / 110.0))
    return img.convert("RGB")


def _splinter(width: int, height: int, palette: Sequence[RGB], opts: Dict, rng: np.random.Generator) -> Image.Image:
    img = Image.new("RGB", (width, height), palette[0])
    draw = ImageDraw.Draw(img)
    density = clamp(float(opts.get("density", 50)), 0, 100) / 50.0
    scale = int(clamp(opts.get("scale", 90), 20, 500))
    count = int(28 * density + width * height / 90_000)
    for _ in range(count):
        x = int(rng.integers(-scale, width))
        y = int(rng.integers(-scale, height))
        length = int(rng.integers(scale, scale * 4))
        thickness = int(rng.integers(max(6, scale // 6), max(7, scale)))
        angle = rng.uniform(-math.pi, math.pi)
        dx = math.cos(angle) * length
        dy = math.sin(angle) * length
        px = -math.sin(angle) * thickness
        py = math.cos(angle) * thickness
        points = [(x, y), (x + dx, y + dy), (x + dx + px, y + dy + py), (x + px, y + py)]
        draw.polygon(points, fill=palette[int(rng.integers(1 if len(palette) > 1 else 0, len(palette)))])
    # Thin rain slashes, as on classic angular patterns.
    if opts.get("outline", False):
        for _ in range(int(90 * density)):
            x = int(rng.integers(0, width))
            y = int(rng.integers(0, height))
            length = int(rng.integers(scale // 2, scale * 2))
            draw.line([x, y, x + length // 3, y + length], fill=palette[-1], width=max(1, width // 800))
    return img


def _topographic(width: int, height: int, palette: Sequence[RGB], opts: Dict, rng: np.random.Generator) -> Image.Image:
    scale = int(clamp(opts.get("scale", 90), 5, 400))
    n = fractal_noise(width, height, rng, scale * 1.3, int(opts.get("detail", 4)), int(opts.get("roughness", 55)), bool(opts.get("seamless", False)))
    bands = 3 + int(clamp(opts.get("density", 50), 0, 100) / 8)
    wave = (np.sin(n * math.tau * bands) + 1) / 2
    mixed = normalize(n * 0.55 + wave * 0.45)
    arr = quantize_noise(mixed, palette, opts)
    if opts.get("outline", False):
        line_mask = (wave > 0.47) & (wave < 0.53)
        arr[line_mask] = np.asarray(palette[-1], dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _rain_streak(width: int, height: int, palette: Sequence[RGB], opts: Dict, rng: np.random.Generator) -> Image.Image:
    bg_noise = fractal_noise(width, height, rng, max(15, int(opts.get("scale", 90))), int(opts.get("detail", 4)), int(opts.get("roughness", 55)), bool(opts.get("seamless", False)))
    img = Image.fromarray(quantize_noise(bg_noise, palette[: max(2, len(palette) - 1)], opts), "RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    density = clamp(float(opts.get("density", 50)), 0, 100) / 50.0
    streaks = int(width * density * 1.8)
    max_len = int(clamp(opts.get("scale", 90), 10, 400) * 1.7)
    for _ in range(streaks):
        x = int(rng.integers(-width // 8, width + width // 8))
        y = int(rng.integers(-max_len, height))
        length = int(rng.integers(max(6, max_len // 5), max(7, max_len)))
        slant = int(rng.normal(0, max_len / 7))
        color = palette[int(rng.integers(0, len(palette)))]
        alpha = int(rng.integers(65, 210))
        draw.line([x, y, x + slant, y + length], fill=(*color, alpha), width=max(1, int(opts.get("stripe_width", 46) / 18)))
    return img


def _brush_stroke(width: int, height: int, palette: Sequence[RGB], opts: Dict, rng: np.random.Generator) -> Image.Image:
    img = _blob(width, height, palette, opts, rng).convert("RGBA")
    density = clamp(float(opts.get("density", 50)), 0, 100) / 50.0
    scale = int(clamp(opts.get("scale", 90), 5, 400))
    stroke_width = int(clamp(opts.get("stripe_width", 46), 2, 260))
    count = int((width * height / 75_000) * density + 25)
    for i in range(count):
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        x = int(rng.integers(-scale, width + scale))
        y = int(rng.integers(-scale, height + scale))
        angle = rng.uniform(-math.pi, math.pi)
        length = int(rng.integers(scale, scale * 5))
        points = []
        for t in np.linspace(0, 1, 10):
            wig = math.sin(t * math.tau * rng.uniform(0.5, 2.0) + rng.uniform(0, math.tau)) * stroke_width * rng.uniform(0.2, 1.2)
            px = x + math.cos(angle) * length * t + math.cos(angle + math.pi / 2) * wig
            py = y + math.sin(angle) * length * t + math.sin(angle + math.pi / 2) * wig
            points.append((px, py))
        draw.line(points, fill=255, width=max(1, int(stroke_width * rng.uniform(0.45, 1.4))), joint="curve")
        mask = mask.filter(ImageFilter.GaussianBlur(radius=clamp(opts.get("edge_softness", 8), 0, 100) / 24.0))
        color = palette[int(rng.integers(0, len(palette)))]
        stroke = Image.new("RGBA", (width, height), (*color, int(rng.integers(140, 256))))
        img.alpha_composite(Image.composite(stroke, Image.new("RGBA", (width, height), (0, 0, 0, 0)), mask))
    return img.convert("RGB")
