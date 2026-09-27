#!/usr/bin/env python3
"""
png_inspect.py — validate and compare Forge worker PNGs. Standard library only
(zlib + struct), so it runs on a bare Ubuntu server.

  png_inspect.py stats FRAME.png [--expect WxH] [--json]
      Checks the file is a valid PNG, reports size, bit depth and per-channel
      statistics, and fails (exit 1) on a structurally bad frame: wrong size,
      empty, all black, all white, or flat (no spatial variation).

  png_inspect.py compare A.png B.png [--max-mean-abs X] [--min-psnr DB] [--json OUT.json]
      Per-channel mean absolute difference (MAE), RMS error (RMSE), PSNR, the
      maximum absolute difference and the fraction of pixels differing by more
      than 1/64, on a 0..1 scale. Fails if the MAE exceeds --max-mean-abs
      (default 0.02) or the PSNR is below --min-psnr (default: no floor).
      --json appends the metrics as one JSON line to OUT.json.

  png_inspect.py preview FRAME.png OUT.png [--scale N]
      Writes an 8-bit, nearest-neighbour enlarged copy for viewing.

Channel-order and orientation checks are part of `stats`: it reports the mean
colour of the top and bottom thirds (Forge's Mandelbrot / Newton frames are
sky-dark-blue above the relief's horizon and brighter terrain below) so a
vertical flip or an R/B swap is visible in the numbers, not only by eye.
"""
import json
import math
import struct
import sys
import zlib


def read_png(path):
    data = open(path, "rb").read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG (bad signature)")
    pos, idat, ihdr = 8, b"", None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])
        if zlib.crc32(ctype + body) & 0xFFFFFFFF != crc:
            raise ValueError(f"{path}: CRC mismatch in {ctype!r} chunk")
        if ctype == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif ctype == b"IDAT":
            idat += body
        elif ctype == b"IEND":
            break
        pos += 12 + length
    if ihdr is None:
        raise ValueError(f"{path}: no IHDR")
    width, height, depth, ctype, _, _, interlace = ihdr
    if interlace != 0 or ctype not in (2, 6) or depth not in (8, 16):
        raise ValueError(f"{path}: unsupported PNG (colour type {ctype}, depth {depth}, interlace {interlace})")
    channels = 3 if ctype == 2 else 4
    bpp = channels * depth // 8
    raw = zlib.decompress(idat)
    stride = width * bpp
    rows, prev = [], bytearray(stride)
    for y in range(height):
        f = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if f == 1:
                line[i] = (line[i] + a) & 255
            elif f == 2:
                line[i] = (line[i] + b) & 255
            elif f == 3:
                line[i] = (line[i] + (a + b) // 2) & 255
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(bytes(line))
        prev = line
    scale = float((1 << depth) - 1)
    pixels = []  # rows of (r, g, b) floats 0..1
    for line in rows:
        row = []
        for x in range(width):
            if depth == 16:
                vals = struct.unpack(">" + "H" * channels, line[x * bpp:(x + 1) * bpp])
            else:
                vals = line[x * bpp:(x + 1) * bpp]
            row.append(tuple(v / scale for v in vals[:3]))
        pixels.append(row)
    return {"width": width, "height": height, "depth": depth, "channels": channels, "pixels": pixels}


def region_mean(pixels, y0, y1):
    n, acc = 0, [0.0, 0.0, 0.0]
    for row in pixels[y0:y1]:
        for p in row:
            for c in range(3):
                acc[c] += p[c]
            n += 1
    return [a / max(n, 1) for a in acc]


def stats(path, expect=None, as_json=False):
    img = read_png(path)
    w, h, px = img["width"], img["height"], img["pixels"]
    flat = [c for row in px for p in row for c in p]
    finite = all(math.isfinite(v) for v in flat)
    mean = [sum(p[c] for row in px for p in row) / (w * h) for c in range(3)]
    mins = [min(p[c] for row in px for p in row) for c in range(3)]
    maxs = [max(p[c] for row in px for p in row) for c in range(3)]
    lum = [0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2] for row in px for p in row]
    lmean = sum(lum) / len(lum)
    lstd = math.sqrt(sum((v - lmean) ** 2 for v in lum) / len(lum))
    black = sum(1 for v in lum if v < 1 / 255) / len(lum)
    white = sum(1 for v in lum if v > 254 / 255) / len(lum)
    distinct = len({p for row in px for p in row})
    report = {
        "file": path, "width": w, "height": h, "bit_depth": img["depth"], "channels": img["channels"],
        "mean_rgb": mean, "min_rgb": mins, "max_rgb": maxs,
        "luminance_mean": lmean, "luminance_std": lstd,
        "fraction_black": black, "fraction_white": white, "distinct_colours": distinct,
        "top_third_mean_rgb": region_mean(px, 0, h // 3),
        "bottom_third_mean_rgb": region_mean(px, h - h // 3, h),
        "finite": finite,
    }
    problems = []
    if expect and (w, h) != expect:
        problems.append(f"size {w}x{h}, expected {expect[0]}x{expect[1]}")
    if black > 0.99:
        problems.append("all black")
    if white > 0.99:
        problems.append("all white")
    if lstd < 0.005 or distinct < 16:
        problems.append("flat image (no spatial variation)")
    if not finite:
        problems.append("non-finite values")
    report["problems"] = problems
    report["ok"] = not problems
    if as_json:
        print(json.dumps(report, indent=2))
    else:
        f3 = lambda v: "(" + ", ".join(f"{x:.3f}" for x in v) + ")"
        print(f"{path}: {w}x{h}, {img['depth']}-bit, {img['channels']} channels, valid PNG")
        print(f"  mean RGB {f3(mean)}  min {f3(mins)}  max {f3(maxs)}")
        print(f"  luminance mean {lmean:.3f} std {lstd:.3f}; black {black:.1%}, white {white:.1%}; {distinct} distinct colours")
        print(f"  top third mean {f3(report['top_third_mean_rgb'])}  bottom third mean {f3(report['bottom_third_mean_rgb'])}")
        print("  OK" if not problems else "  PROBLEMS: " + "; ".join(problems))
    return 0 if not problems else 1


def compare(a_path, b_path, max_mean_abs=0.02, min_psnr=None, json_out=None):
    a, b = read_png(a_path), read_png(b_path)
    if (a["width"], a["height"]) != (b["width"], b["height"]):
        print(f"size mismatch: {a['width']}x{a['height']} vs {b['width']}x{b['height']}")
        return 1
    n = a["width"] * a["height"]
    abs_sum, sq_sum, big, maxd = [0.0] * 3, [0.0] * 3, 0, 0.0
    for ra, rb in zip(a["pixels"], b["pixels"]):
        for pa, pb in zip(ra, rb):
            over = False
            for c in range(3):
                d = abs(pa[c] - pb[c])
                abs_sum[c] += d
                sq_sum[c] += d * d
                maxd = max(maxd, d)
                over = over or d > 1 / 64
            big += over
    mean_abs = [s / n for s in abs_sum]
    rms = math.sqrt(sum(sq_sum) / (3 * n))
    psnr = float("inf") if rms == 0 else 20 * math.log10(1 / rms)
    overall = sum(mean_abs) / 3
    print(f"{a_path}\n  vs {b_path}")
    print(f"  mean |diff| RGB ({mean_abs[0]:.5f}, {mean_abs[1]:.5f}, {mean_abs[2]:.5f}); overall {overall:.5f}")
    print(f"  RMS {rms:.5f}; PSNR {psnr:.2f} dB; max |diff| {maxd:.4f}; pixels off by >1/64: {big / n:.2%}")
    ok = overall <= max_mean_abs and (min_psnr is None or psnr >= min_psnr)
    if ok:
        print("  MATCH")
    elif overall > max_mean_abs:
        print(f"  DIFFERENT (overall mean |diff| > {max_mean_abs})")
    else:
        print(f"  DIFFERENT (PSNR below {min_psnr} dB)")
    if json_out:
        record = {"a": a_path, "b": b_path, "width": a["width"], "height": a["height"],
                  "mae": overall, "mae_rgb": mean_abs, "rmse": rms,
                  "psnr_db": None if psnr == float("inf") else psnr, "max_abs": maxd,
                  "fraction_over_1_64": big / n, "max_mean_abs_limit": max_mean_abs,
                  "min_psnr_limit": min_psnr, "match": ok}
        with open(json_out, "a") as f:
            f.write(json.dumps(record) + "\n")
    return 0 if ok else 1


def preview(path, out, scale=4):
    img = read_png(path)
    w, h = img["width"] * scale, img["height"] * scale
    raw = bytearray()
    for row in img["pixels"]:
        line = bytearray([0])
        for p in row:
            line += bytes(max(0, min(255, round(c * 255))) for c in p) * scale
        for _ in range(scale):
            raw += line

    def chunk(t, body):
        return struct.pack(">I", len(body)) + t + body + struct.pack(">I", zlib.crc32(t + body) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")
    open(out, "wb").write(png)
    print(f"wrote {out} ({w}x{h})")
    return 0


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    cmd = argv[1]
    if cmd == "stats":
        expect = None
        if "--expect" in argv:
            w, h = argv[argv.index("--expect") + 1].split("x")
            expect = (int(w), int(h))
        return stats(argv[2], expect, "--json" in argv)
    if cmd == "compare":
        limit = float(argv[argv.index("--max-mean-abs") + 1]) if "--max-mean-abs" in argv else 0.02
        floor = float(argv[argv.index("--min-psnr") + 1]) if "--min-psnr" in argv else None
        out = argv[argv.index("--json") + 1] if "--json" in argv else None
        return compare(argv[2], argv[3], limit, floor, out)
    if cmd == "preview":
        scale = int(argv[argv.index("--scale") + 1]) if "--scale" in argv else 4
        return preview(argv[2], argv[3], scale)
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
