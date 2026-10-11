# -*- coding: utf-8 -*-
"""按颜色 / 灰度分离并擦除手写笔迹（只在 diagnose_ink.py 判定可行时才用）。

两种模式:
  --by color   （默认，最稳）剔除**有色笔**（蓝笔/红笔/彩笔），保留中性印刷墨。
               纸面近中性、彩色笔彩度高，这个维度天然可分，基本无损。
  --by gray    （有风险）剔除某个灰度带的笔迹。只适用于手写明显比印刷体
               浅或深的场景；因为同色时两者灰度完全重叠，用错会啃掉印刷体。

用法:
  python separate_ink.py IMG --out OUT.jpg [--rotate cw90] [--by color]
                             [--chroma 40] [--overlay]
  python separate_ink.py IMG --out OUT.jpg --by gray --lo 130 --hi 200 --overlay
"""
import os
import sys
import argparse

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (imread_u, imwrite_u, rotate, inpaint_with_mask, vis_mask,
                     preview, banner, ensure_dir)


def color_mask(img, chroma_thr=40, rel_thr=18, min_area=12, dilate=1):
    """有色墨掩膜。彩度 = max(RGB)-min(RGB)，纸面近 0，彩笔显著 >0。
    再叠加"相对局部背景的彩度增量"，避免整张泛色纸被判满。"""
    mx = img.max(axis=2).astype(np.float32)
    mn = img.min(axis=2).astype(np.float32)
    chroma = mx - mn
    cq = np.clip(chroma, 0, 255).astype(np.uint8)
    bg = cv2.medianBlur(cq, 61).astype(np.float32)
    rel = chroma - bg

    m = ((chroma > chroma_thr) | (rel > rel_thr)).astype(np.uint8)
    # 去掉极暗的近中性像素（黑印刷体有时因噪点带一点彩度）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m[(gray < 80) & (chroma < chroma_thr + 20)] = 0

    if min_area > 1:
        n, lb, st, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros_like(m)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= min_area:
                keep[lb == i] = 1
        m = keep
    if dilate:
        m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=dilate)
    return m, {"chroma_p99": float(np.percentile(chroma, 99)),
               "colored_pct": float(m.mean() * 100)}


def gray_band_mask(img, lo, hi, min_area=12, dilate=1):
    """灰度带掩膜：保留 gray<lo 的深墨（印刷体），擦除 lo<=gray<=hi 的浅墨。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = ((gray >= lo) & (gray <= hi)).astype(np.uint8)
    if min_area > 1:
        n, lb, st, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros_like(m)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= min_area:
                keep[lb == i] = 1
        m = keep
    if dilate:
        m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=dilate)
    return m, {"band_pct": float(m.mean() * 100)}


def main():
    ap = argparse.ArgumentParser(description="按颜色/灰度擦除手写")
    ap.add_argument("image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rotate", default="none")
    ap.add_argument("--by", default="color", choices=["color", "gray"])
    ap.add_argument("--chroma", type=float, default=40, help="color 模式彩度阈值")
    ap.add_argument("--rel-thr", type=float, default=18, help="color 模式相对彩度增量")
    ap.add_argument("--lo", type=int, default=130, help="gray 模式灰带下界")
    ap.add_argument("--hi", type=int, default=200, help="gray 模式灰带上界")
    ap.add_argument("--min-area", type=int, default=12)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--overlay", action="store_true")
    args = ap.parse_args()

    img = imread_u(args.image)
    if img is None:
        raise SystemExit("读不到图: %s" % args.image)
    img = rotate(img, args.rotate)
    banner("分离并擦除手写（by=%s）" % args.by)

    if args.by == "color":
        mask, info = color_mask(img, args.chroma, args.rel_thr, args.min_area)
        print("彩度 P99 = %.1f" % info["chroma_p99"])
        if info["chroma_p99"] < 35:
            print("  ⚠️ 整张图彩度都很低 —— 手写很可能也是中性黑笔，"
                  "color 模式无效。回到 diagnose_ink.py 重新判定。")
    else:
        mask, info = gray_band_mask(img, args.lo, args.hi, args.min_area)
        print("⚠️ gray 模式会连带擦掉同灰度的印刷体，用前必须逐处放大复核。"
              "灰带 gray∈[%d,%d]" % (args.lo, args.hi))

    print("掩膜覆盖率 %.2f%%" % (mask.mean() * 100))
    out = inpaint_with_mask(img, mask, args.radius)

    imwrite_u(args.out, out, 95)
    print("已输出 %s  (%dx%d)" % (args.out, out.shape[1], out.shape[0]))

    od = os.path.join(os.path.dirname(os.path.abspath(args.out)), "_ink_check")
    ensure_dir(od)
    stem = os.path.splitext(os.path.basename(args.out))[0]
    imwrite_u(os.path.join(od, stem + "_overlay.jpg"),
              vis_mask(preview(img, 1400), preview(mask, 1400)), 90)
    if args.overlay:
        imwrite_u(os.path.join(od, stem + "_after.jpg"), preview(out, 1400), 90)
    print("核对图 → %s/%s_overlay.jpg" % (od, stem))
    print("复核要点：印刷体的笔画有没有被啃断/变细；手写是否清干净。")


if __name__ == "__main__":
    main()
