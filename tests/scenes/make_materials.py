"""Writes materials.vox: one sphere per material type on a slab.

Used to eyeball glass, metal, plastic, emissive and blend shading, since
none of MagicaVoxel's bundled samples use them.
"""
import struct
import sys


def chunk(cid, content, children=b""):
    return cid + struct.pack("<ii", len(content), len(children)) + content + children


def string(s):
    b = s.encode()
    return struct.pack("<i", len(b)) + b


def dictionary(d):
    return struct.pack("<i", len(d)) + b"".join(string(k) + string(v) for k, v in d.items())


SIZE = (112, 40, 30)
voxels = {}
# Floor slab, index 1 (light gray).
for x in range(SIZE[0]):
    for y in range(SIZE[1]):
        voxels[(x, y, 0)] = 1

# (palette index, rgb, material dict)
spheres = [
    (10, (200, 230, 255), {"_type": "_glass", "_trans": "1", "_rough": "0", "_ri": "1.5"}),
    (11, (255, 160, 120), {"_type": "_glass", "_trans": "1", "_rough": "0.3", "_ri": "1.5"}),
    (12, (255, 200, 120), {"_type": "_metal", "_metal": "1", "_rough": "0.15"}),
    (13, (60, 120, 230), {"_type": "_metal", "_metal": "0", "_rough": "0.1", "_ri": "1.5"}),
    (14, (255, 240, 200), {"_type": "_emit", "_emit": "1", "_flux": "1"}),
    (15, (120, 230, 140), {"_type": "_blend", "_metal": "0.5", "_trans": "0.5", "_rough": "0.1"}),
    (16, (240, 240, 250), {"_type": "_media", "_d": "0.08", "_g": "0.3"}),
]
r = 6.5
for i, (idx, _, _) in enumerate(spheres):
    cx, cy, cz = 8 + i * 16, 20, 8
    for x in range(cx - 8, cx + 8):
        for y in range(cy - 8, cy + 8):
            for z in range(1, 16):
                if (x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 + (z + 0.5 - cz) ** 2 <= r * r:
                    voxels[(x, y, z)] = idx

palette = [(0, 0, 0, 255)] * 256
palette[1] = (210, 210, 210, 255)
for idx, rgb, _ in spheres:
    palette[idx] = (*rgb, 255)

body = chunk(b"SIZE", struct.pack("<iii", *SIZE))
body += chunk(b"XYZI", struct.pack("<i", len(voxels)) + b"".join(struct.pack("<BBBB", *p, c) for p, c in voxels.items()))
# RGBA entry k is color index k + 1.
body += chunk(b"RGBA", b"".join(struct.pack("<BBBB", *palette[k + 1 if k < 255 else 0]) for k in range(256)))
for idx, _, mat in spheres:
    body += chunk(b"MATL", struct.pack("<i", idx) + dictionary(mat))
body += chunk(b"rOBJ", dictionary({"_type": "_inf", "_i": "0.8", "_k": "255 255 255", "_angle": "45 120", "_area": "0.05"}))

out = sys.argv[1] if len(sys.argv) > 1 else "materials.vox"
with open(out, "wb") as f:
    f.write(b"VOX " + struct.pack("<i", 200) + chunk(b"MAIN", b"", body))
print("wrote", out, len(voxels), "voxels")
