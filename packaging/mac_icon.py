# Usage: python3 packaging/mac_icon.py <source.png> <out.png> [size, default 512]
# packaging/make-icons.sh runs it for assets/icon.png and packaging/voxely.icns.
# macOS (Big Sur+) app icon: art on an 824/1024 rounded square
# (superellipse, like Apple's continuous corners) with a soft shadow.
import sys
from PIL import Image, ImageDraw, ImageFilter
src, out = sys.argv[1], sys.argv[2]
size = int(sys.argv[3]) if len(sys.argv) > 3 else 512
C, T, SS = 1024, 824, 4            # canvas, tile, supersampling
off = (C - T) // 2
n = 5.0                            # superellipse exponent
big = T * SS
mask = Image.new("L", (big, big), 0)
pts = []
import math
for i in range(2000):
    t = 2 * math.pi * i / 2000
    c, s = math.cos(t), math.sin(t)
    x = math.copysign(abs(c) ** (2 / n), c)
    y = math.copysign(abs(s) ** (2 / n), s)
    pts.append(((x + 1) * big / 2, (y + 1) * big / 2))
ImageDraw.Draw(mask).polygon(pts, fill=255)
mask = mask.resize((T, T), Image.LANCZOS)
art = Image.open(src).convert("RGBA").resize((T, T), Image.LANCZOS)
tile = Image.new("RGBA", (T, T), (0, 0, 0, 0))
tile.paste(art, (0, 0), mask)
canvas = Image.new("RGBA", (C, C), (0, 0, 0, 0))
shadow = Image.new("RGBA", (C, C), (0, 0, 0, 0))
shadow.paste(Image.new("RGBA", (T, T), (0, 0, 0, 90)), (off, off + 12), mask)
canvas = Image.alpha_composite(canvas, shadow.filter(ImageFilter.GaussianBlur(14)))
canvas.alpha_composite(tile, (off, off))
canvas.resize((size, size), Image.LANCZOS).save(out, optimize=True)
