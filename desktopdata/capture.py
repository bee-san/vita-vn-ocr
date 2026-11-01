"""The desktop capture model (docs/desktop-data.md): how a game frame of a given size ends up in a desktop capture.

A capture is the game's frame (an emulator's Vita / PSP frame, or a PC game at its own resolution) shown on a
desktop screen or in a window and captured there:

  setup      fullscreen on a 16:9, 16:10, 21:9 or 4:3 screen (the frame fitted with its aspect: letterbox or
             pillarbox bars), an emulator window at an integer or fractional multiple of the frame, a user-sized
             window, or the frame stretched to the screen (aspect not kept)
  resampler  how the game or emulator scales the frame: nearest, bilinear, bicubic, Lanczos, sharp bilinear (nearest
             to the largest integer multiple, then bilinear), bilinear plus a sharpening pass (FSR / CAS-like);
             INTER_AREA for part of the downscales
  dpi        a non-DPI-aware window at 125-175 % display scaling: rendered at the logical size, then stretched
             bilinear by Windows to the physical size
  capture    the capture itself: at full size, or downscaled (INTER_AREA), and saved as PNG or JPEG
             (quality 70-95, 4:2:0 or 4:4:4 chroma)

Every geometric step is an affine map per axis (x' = a x + b), so a box maps exactly: its edges go through the
composed map, start edges rounded down and end edges up (eval's and textspot's convention), clipped to the capture.
`Capture.apply(rgb)` returns the capture's RGB pixels and `Capture.box(b)` maps a box [x0, y0, x1, y1).

The mix (MIX) is fixed and recorded in each manifest; `sample(rng, w, h, platform)` draws one capture for a w x h
frame. Deterministic for a given numpy Generator.
"""
import math

import numpy as np

SCREENS = {
    "16:9": {"1280x720": 0.06, "1366x768": 0.05, "1600x900": 0.04, "1920x1080": 0.22, "2560x1440": 0.12,
             "3840x2160": 0.07},
    "16:10": {"1280x800": 0.015, "1440x900": 0.015, "1680x1050": 0.02, "1920x1200": 0.04, "2560x1600": 0.03},
    "21:9": {"2560x1080": 0.03, "3440x1440": 0.03},
    "4:3": {"800x600": 0.015, "1024x768": 0.03, "1280x960": 0.02, "1600x1200": 0.015},
}
MIX = {
    "setup": {"fullscreen": 0.80, "window_multiple": 0.10, "window_free": 0.06, "stretch": 0.04},
    "fullscreen": {f"{k}/{s}": w for k, v in SCREENS.items() for s, w in v.items()},
    # emulator windows: Vita3K / PPSSPP window sizes as multiples of the native frame (PSP multiples of 480 x 272)
    "window_multiple": {"vita": [1.25, 1.5, 2.0, 2.5, 3.0], "psp": [2.0, 3.0, 4.0, 5.0]},
    "window_free": {"width": [1024, 2560], "aspect_jitter": 0.06},
    "stretch": {"16:10/1920x1200": 0.4, "4:3/1280x960": 0.3, "16:10/2560x1600": 0.3},
    "resampler": {"bilinear": 0.33, "bicubic": 0.15, "lanczos": 0.14, "nearest": 0.12, "sharp_bilinear": 0.14,
                  "bilinear_sharpen": 0.12},
    "area_on_downscale": 0.5,
    "sharpen": {"amount": [0.4, 1.0], "sigma": [0.8, 1.5]},
    "dpi": {"p_window": 0.3, "factor": [1.25, 1.5, 1.75]},
    "bars": {"black": 0.85, "dark": 0.15, "dark_level": [6, 40]},
    "downscale": {"p": 0.10, "factor": [0.5, 0.8]},
    "jpeg": {"p": 0.25, "quality": [70, 95], "p444": 0.2},
}


def choose(rng, weights):
    keys = list(weights)
    p = np.asarray([weights[k] for k in keys], np.float64)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def wh(s):
    w, h = s.split("x")
    return int(w), int(h)


class Capture:
    """One capture of a w x h frame: the screen W x H, where the frame lands (ox, oy, cw, ch), the resampler, the DPI
    stretch, the capture's final size and its format. The composed per-axis map is x' = ax x + bx, y' = ay y + by."""

    def __init__(self, w, h, setup, screen, place, resampler, dpi=None, bars=(0, 0, 0), down=None, jpeg=None,
                 sharpen=None, area=False, name=None):
        self.w, self.h = int(w), int(h)
        self.setup, self.name = setup, name
        self.W, self.H = screen
        self.ox, self.oy, self.cw, self.ch = place
        self.resampler, self.dpi, self.bars, self.down, self.jpeg = resampler, dpi, tuple(bars), down, jpeg
        self.sharpen, self.area = sharpen, area
        # the logical window (before the DPI stretch) is what the game draws into
        if dpi:
            self.LW, self.LH = int(round(self.W / dpi)), int(round(self.H / dpi))
        else:
            self.LW, self.LH = self.W, self.H
        self.FW, self.FH = (int(round(self.W * down)), int(round(self.H * down))) if down else (self.W, self.H)

    # ---- geometry
    def logical_place(self):
        """(ox, oy, cw, ch) in the logical window: place scaled by LW / W (the frame is drawn there)."""
        if not self.dpi:
            return self.ox, self.oy, self.cw, self.ch
        kx, ky = self.LW / self.W, self.LH / self.H
        cw, ch = max(1, int(round(self.cw * kx))), max(1, int(round(self.ch * ky)))
        ox, oy = (self.LW - cw) // 2, (self.LH - ch) // 2
        if self.setup == "stretch":
            ox, oy, cw, ch = 0, 0, self.LW, self.LH
        return ox, oy, cw, ch

    def affine(self):
        """(ax, bx, ay, by): frame px -> capture px."""
        ox, oy, cw, ch = self.logical_place()
        ax, bx = cw / self.w, float(ox)
        ay, by = ch / self.h, float(oy)
        kx, ky = self.W / self.LW, self.H / self.LH            # the DPI stretch (1 without)
        dx, dy = self.FW / self.W, self.FH / self.H             # the capture's downscale (1 without)
        return ax * kx * dx, bx * kx * dx, ay * ky * dy, by * ky * dy

    def box(self, b):
        """A frame box [x0, y0, x1, y1) (float or int) -> capture px, start floor and end ceil, clipped; None when it
        falls outside the capture."""
        ax, bx, ay, by = self.affine()
        x0, y0 = int(math.floor(ax * b[0] + bx + 1e-9)), int(math.floor(ay * b[1] + by + 1e-9))
        x1, y1 = int(math.ceil(ax * b[2] + bx - 1e-9)), int(math.ceil(ay * b[3] + by - 1e-9))
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(self.FW, x1), min(self.FH, y1)
        return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None

    def content_rect(self):
        """Where the frame lies in the capture: [x0, y0, x1, y1)."""
        return self.box([0, 0, self.w, self.h])

    # ---- pixels
    def _resize(self, img, size):
        import cv2
        tw, th = size
        down = tw < img.shape[1] or th < img.shape[0]
        if self.area and down:
            return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        r = self.resampler
        if r == "nearest":
            return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST_EXACT)
        if r == "bicubic":
            return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        if r == "lanczos":
            return cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
        if r == "sharp_bilinear":
            k = int(min(tw / img.shape[1], th / img.shape[0]))
            if k >= 2:
                img = cv2.resize(img, (img.shape[1] * k, img.shape[0] * k), interpolation=cv2.INTER_NEAREST_EXACT)
            return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        out = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        if r == "bilinear_sharpen" and self.sharpen:
            amount, sigma = self.sharpen
            blur = cv2.GaussianBlur(out.astype(np.float32), (0, 0), sigma)
            out = np.clip(out.astype(np.float32) * (1 + amount) - blur * amount + 0.5, 0, 255).astype(np.uint8)
        return out

    def apply(self, rgb):
        """The capture's RGB pixels (uint8 FH x FW x 3) of an h x w x 3 RGB frame."""
        import cv2
        rgb = np.ascontiguousarray(rgb, np.uint8)
        if rgb.shape[:2] != (self.h, self.w):
            raise ValueError(f"a {self.w}x{self.h} capture got a {rgb.shape[1]}x{rgb.shape[0]} frame")
        ox, oy, cw, ch = self.logical_place()
        canvas = np.empty((self.LH, self.LW, 3), np.uint8)
        canvas[:] = np.asarray(self.bars, np.uint8)
        canvas[oy:oy + ch, ox:ox + cw] = self._resize(rgb, (cw, ch)) if (cw, ch) != (self.w, self.h) else rgb
        if self.dpi:
            canvas = cv2.resize(canvas, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if self.down:
            canvas = cv2.resize(canvas, (self.FW, self.FH), interpolation=cv2.INTER_AREA)
        if self.jpeg:
            q, sub = self.jpeg
            flag = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444 if sub == 444 else cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420
            ok, buf = cv2.imencode(".jpg", canvas[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(q),
                                                                cv2.IMWRITE_JPEG_SAMPLING_FACTOR, flag])
            if not ok:
                raise RuntimeError("JPEG encode failed")
            canvas = np.ascontiguousarray(cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1])
        return canvas

    def record(self):
        """The capture's settings, for a sample's meta (JSON)."""
        return {"setup": self.setup, "name": self.name, "frame": [self.w, self.h], "screen": [self.W, self.H],
                "place": [self.ox, self.oy, self.cw, self.ch], "resampler": self.resampler,
                "area_down": bool(self.area), "sharpen": [round(v, 3) for v in self.sharpen] if self.sharpen else None,
                "dpi": self.dpi, "bars": list(self.bars), "down": round(self.down, 4) if self.down else None,
                "jpeg": list(self.jpeg) if self.jpeg else None, "size": [self.FW, self.FH],
                "affine": [round(v, 6) for v in self.affine()]}


def fit_place(w, h, W, H):
    """(ox, oy, cw, ch): a w x h frame fitted into W x H with its aspect, centred (letterbox or pillarbox bars)."""
    s = min(W / w, H / h)
    cw, ch = max(1, int(round(w * s))), max(1, int(round(h * s)))
    cw, ch = min(cw, W), min(ch, H)
    return (W - cw) // 2, (H - ch) // 2, cw, ch


def sample(rng, w, h, platform=None, mix=None):
    """One capture of a w x h frame (platform 'vita' / 'psp' for an emulator's frame: its window multiples)."""
    M = mix or MIX
    setup = choose(rng, M["setup"])
    if setup == "window_multiple" and platform not in M["window_multiple"]:
        setup = "fullscreen"
    if setup == "fullscreen":
        name = choose(rng, M["fullscreen"])
        W, H = wh(name.split("/")[1])
        place = fit_place(w, h, W, H)
    elif setup == "stretch":
        name = choose(rng, M["stretch"])
        W, H = wh(name.split("/")[1])
        place = (0, 0, W, H)
    elif setup == "window_multiple":
        f = float(rng.choice(M["window_multiple"][platform]))
        W, H = int(round(w * f)), int(round(h * f))
        name = f"x{f:g}"
        place = (0, 0, W, H)
    else:
        lo, hi = M["window_free"]["width"]
        W = int(rng.integers(lo, hi + 1))
        j = M["window_free"]["aspect_jitter"]
        H = int(round(W / ((w / h) * float(rng.uniform(1 - j, 1 + j)))))
        name = "free"
        place = fit_place(w, h, W, H)
    resampler = choose(rng, M["resampler"])
    sharpen = None
    if resampler == "bilinear_sharpen":
        sharpen = (float(rng.uniform(*M["sharpen"]["amount"])), float(rng.uniform(*M["sharpen"]["sigma"])))
    area = bool(rng.random() < M["area_on_downscale"])
    dpi = None
    if setup in ("window_multiple", "window_free") and rng.random() < M["dpi"]["p_window"]:
        dpi = float(rng.choice(M["dpi"]["factor"]))
    if choose(rng, {k: v for k, v in M["bars"].items() if k != "dark_level"}) == "dark":
        g = int(rng.integers(M["bars"]["dark_level"][0], M["bars"]["dark_level"][1] + 1))
        bars = (g, g, g)
    else:
        bars = (0, 0, 0)
    down = float(rng.uniform(*M["downscale"]["factor"])) if rng.random() < M["downscale"]["p"] else None
    jpeg = None
    if rng.random() < M["jpeg"]["p"]:
        lo, hi = M["jpeg"]["quality"]
        jpeg = (int(rng.integers(lo, hi + 1)), 444 if rng.random() < M["jpeg"]["p444"] else 420)
    return Capture(w, h, setup, (W, H), place, resampler, dpi, bars, down, jpeg, sharpen, area, name)


def mean_pixels(w, h, platform, n=4000, seed=0, mix=None):
    """The mean capture size in pixels of n draws (for sizing a render)."""
    rng = np.random.default_rng(seed)
    tot = 0
    for _ in range(n):
        c = sample(rng, w, h, platform, mix)
        tot += c.FW * c.FH
    return tot / n
