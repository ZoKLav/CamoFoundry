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


def _digital(width: int, height: int, palette: Sequence[RGB], opts: Dict, rng: np.random.Generator) -> Image.Image:
    block = int(clamp(opts.get("block_size", 28), 2, 180))
    gw = max(4, math.ceil(width / block))
    gh = max(4, math.ceil(height / block))
    sub_opts = dict(opts)
    sub_opts["edge_softness"] = 0
    n = fractal_noise(gw, gh, rng, max(2, int(opts.get("scale", 90)) / block), int(opts.get("detail", 4)), int(opts.get("roughness", 55)), bool(opts.get("seamless", False)))
    small = Image.fromarray(quantize_noise(n, palette, sub_opts), "RGB")
    img = small.resize((width, height), RESAMPLE_NEAREST)

    speckle = int(clamp(opts.get("speckle", 18), 0, 100))
    if speckle:
        draw = ImageDraw.Draw(img)
        count = int((width * height / max(1000, block * block * 3)) * speckle)
        for _ in range(count):
            x = int(rng.integers(0, width))
            y = int(rng.integers(0, height))
            bw = int(rng.integers(max(2, block // 3), max(3, block * 2)))
            bh = int(rng.integers(max(2, block // 3), max(3, block * 2)))
            x = (x // max(1, block // 2)) * max(1, block // 2)
            y = (y // max(1, block // 2)) * max(1, block // 2)
            draw.rectangle([x, y, x + bw, y + bh], fill=palette[int(rng.integers(0, len(palette)))])
    return img


def _tiger_stripe(width: int, height: int, palette: Sequence[RGB], opts: Dict, rng: np.random.Generator) -> Image.Image:
    base_noise = fractal_noise(width, height, rng, max(60, int(opts.get("scale", 90)) * 2), 3, 40, bool(opts.get("seamless", False)))
    base = Image.fromarray(quantize_noise(base_noise, palette[: max(2, len(palette) - 2)], opts), "RGB")
    img = base.convert("RGBA")

    width_slider = int(clamp(opts.get("stripe_width", 46), 2, 260))
    spacing = int(clamp(opts.get("stripe_spacing", 112), 10, 420))
    wiggle = int(clamp(opts.get("stripe_wiggle", 90), 0, 300))
    density = clamp(float(opts.get("density", 50)), 0, 100) / 50.0
    stripe_colors = list(palette[-2:] if len(palette) >= 4 else palette[1:])
    if not stripe_colors:
        stripe_colors = [palette[-1]]

    y_start = -height
    y_end = height * 2
    step = max(8, int(spacing / max(0.25, density)))
    for i, y0 in enumerate(range(y_start, y_end, step)):
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        phase = rng.uniform(0, math.tau)
        freq = rng.uniform(1.1, 3.5)
        lean = rng.uniform(-0.75, 0.75)
        points = []
        point_step = max(8, width // 120)
        for x in range(-width // 3, width + width // 3 + point_step, point_step):
            y = y0 + lean * x + math.sin((x / width) * math.tau * freq + phase) * wiggle
            y += rng.normal(0, max(1, wiggle / 8))
            points.append((x, y))
        draw.line(points, fill=255, width=max(1, int(width_slider * rng.uniform(0.65, 1.35))), joint="curve")
        rough = int(clamp(opts.get("roughness", 55), 0, 100))
        if rough:
            mask_noise = fractal_noise(width, height, rng, max(8, width_slider * 1.7), 2, rough, False)
            m = np.asarray(mask, dtype=np.float32) / 255.0
            m = np.clip(m + (mask_noise - 0.5) * (rough / 85.0), 0, 1)
            mask = Image.fromarray(np.uint8(m > 0.45) * 255, "L").filter(ImageFilter.GaussianBlur(radius=max(0, opts.get("edge_softness", 8) / 18)))
        color = stripe_colors[i % len(stripe_colors)]
        stripe = Image.new("RGBA", (width, height), (*color, 255))
        img.alpha_composite(Image.composite(stripe, Image.new("RGBA", (width, height), (0, 0, 0, 0)), mask))
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
    img = Image.new("RGB", (width, height), palette[0])
    draw = ImageDraw.Draw(img, "RGBA")
    density = clamp(float(opts.get("density", 50)), 0, 100) / 50.0
    dot = int(clamp(opts.get("dot_size", 11), 1, 70))
    cluster = int(clamp(opts.get("scale", 90), 5, 400))
    count = min(80_000, int(width * height / max(65, dot * 38) * density))
    for _ in range(count):
        cx = int(rng.integers(-cluster, width + cluster))
        cy = int(rng.integers(-cluster, height + cluster))
        dots_in_cluster = int(rng.integers(2, 8))
        color = palette[int(rng.integers(1 if len(palette) > 1 else 0, len(palette)))]
        for _ in range(dots_in_cluster):
            x = int(cx + rng.normal(0, cluster / 7))
            y = int(cy + rng.normal(0, cluster / 7))
            r = max(1, int(abs(rng.normal(dot, dot * 0.55))))
            if -r <= x < width + r and -r <= y < height + r:
                draw.ellipse([x - r, y - r, x + r, y + r], fill=(*color, int(rng.integers(150, 256))))
    if opts.get("edge_softness", 8):
        img = img.filter(ImageFilter.GaussianBlur(radius=clamp(opts.get("edge_softness", 8), 0, 100) / 28.0))
    return img


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
