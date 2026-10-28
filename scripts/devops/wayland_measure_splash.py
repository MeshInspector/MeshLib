"""usage: wayland_measure_splash.py <max_width_px> <screenshot.png>...
Prints the bounding box of the dark (splash) pixels below the Weston panel in each screenshot and
fails when the first non-empty box is wider than <max_width_px>.
"""
import sys
from PIL import Image

max_width = int(sys.argv[1])
first = None
for path in sys.argv[2:]:
    im = Image.open(path).convert("L")
    w, h = im.size
    px = im.load()
    x0, y0, x1, y1 = w, h, -1, -1
    for y in range(100, h, 4):
        for x in range(0, w, 4):
            if px[x, y] < 30:
                x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)
    if x1 < 0:
        print(f"{path}: no dark pixels")
        continue
    print(f"{path}: dark box x {x0}..{x1} y {y0}..{y1}, {x1 - x0 + 4} x {y1 - y0 + 4} px of {w} x {h}")
    if first is None:
        first = (x1 - x0 + 4, y1 - y0 + 4)

if first is None:
    print("FAIL: no splash seen")
    sys.exit(1)
ok = first[0] <= max_width
print(("ok   " if ok else "FAIL ") + f"first splash box width {first[0]} <= {max_width}")
sys.exit(0 if ok else 1)
