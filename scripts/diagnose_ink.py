# -*- coding: utf-8 -*-
"""手写答案可分离性判定 —— 整条链路的决策点。

背景（这是本技能存在的理由）：
  照片上"学生手写"与"印刷体"经常在 颜色 / 灰度 / 笔画粗细 三个维度上
  **完全重叠**（都用深色中性笔、墨色差 2~3 个灰阶、粗细差在噪声量级）。
  此时任何自动分离都会伤印刷体，硬做就是负优化。
  所以在动手擦之前，先花一分钟量化，得到一个明确结论。

用法:
  # A) 先出坐标网格，方便读坐标
  python diagnose_ink.py IMG --rotate cw90 --grid --outdir OUTDIR

  # B) 填好 regions.json 后做判定
  python diagnose_ink.py IMG --rotate cw90 --regions regions.json --outdir OUTDIR

regions.json 格式（坐标相对**转正后**的图；每项 [x, y, w, h]）:
  {
    "print": [[60,1085,1200,45], [60,930,1300,45]],
    "hand":  [[700,990,170,90],  [1010,930,140,70]]
  }
  建议每类各给 3~5 处，覆盖不同笔迹/不同题目，避免单点偶然。

判定结论:
  SEPARABLE_BY_COLOR   手写是有色笔（蓝/红笔）→ 用 separate_ink.py，几乎无损
  MAYBE_BY_GRAY        手写明显更浅或更深 → 可试阈值，必须逐处复核
  NOT_SEPARABLE        三维全重叠 → 不要自动擦，直接上交互擦除工具
"""
import os
import sys
import argparse

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (imread_u, imwrite_u, rotate, load_json, dump_json,
                     stroke_stats, label_strip, hstack_pad, vstack_pad,
                     preview, banner, ensure_dir)

GRID_STEP_DEFAULT = 200


def make_grid(img, step=GRID_STEP_DEFAULT, out_max=2000):
    """输出带坐标标尺的图：每 step 像素一条线并标注坐标，用于读 regions.json。"""
    im = preview(img, out_max).copy()
    h, w = im.shape[:2]
    scale = w / float(img.shape[1])
    # 细网格
    for x in range(0, img.shape[1], step):
        px = int(x * scale)
        if px < w:
            cv2.line(im, (px, 0), (px, h), (210, 225, 245), 1)
    for y in range(0, img.shape[0], step):
        py = int(y * scale)
        if py < h:
            cv2.line(im, (0, py), (w, py), (210, 225, 245), 1)
    # 主网格（每 5 格加粗并标数）
    for x in range(0, img.shape[1], step * 5):
        px = int(x * scale)
        if px < w:
            cv2.line(im, (px, 0), (px, h), (60, 90, 160), 2)
            cv2.putText(im, str(x), (px + 4, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (20, 40, 120), 2, cv2.LINE_AA)
    for y in range(0, img.shape[0], step * 5):
        py = int(y * scale)
        if py < h:
            cv2.line(im, (0, py), (w, py), (60, 90, 160), 2)
            cv2.putText(im, str(y), (4, py + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (20, 40, 120), 2, cv2.LINE_AA)
    return im


def _agg(stats_list):
    """按暗像素数加权聚合一组区域统计。"""
    if not stats_list:
        return None
    ws = np.array([s["dark_px"] for s in stats_list], np.float64)
    ws = ws / ws.sum()
    keys = ["gray_mean", "gray_min", "gray_p25", "chroma", "blue_frac",
            "dt_med", "dt_p90"]
    out = {}
    for k in keys:
        out[k] = float(np.dot(ws, [s[k] for s in stats_list]))
    out["bgr"] = [float(np.dot(ws, [s["bgr"][i] for s in stats_list]))
                  for i in range(3)]
    out["n_regions"] = len(stats_list)
    out["dark_px_total"] = int(sum(s["dark_px"] for s in stats_list))
    return out


def verdict_of(p, h):
    """核心判定。p=印刷体聚合，h=手写聚合。返回 (结论, 理由列表)。"""
    reasons = []
    # 1) 颜色分离（最强信号）
    if h["blue_frac"] > 0.35 and p["blue_frac"] < 0.05:
        return "SEPARABLE_BY_COLOR", [
            "手写区蓝笔像素占比 %.2f，印刷区仅 %.2f" % (h["blue_frac"], p["blue_frac"])]
    if h["chroma"] - p["chroma"] > 20:
        return "SEPARABLE_BY_COLOR", [
            "手写彩度 %.1f 明显高于印刷 %.1f（有色笔）" % (h["chroma"], p["chroma"])]

    # 2) 灰度分离
    d_gray = h["gray_mean"] - p["gray_mean"]
    d_p25 = h["gray_p25"] - p["gray_p25"]
    reasons.append("灰度均值 印刷 %.1f / 手写 %.1f （差 %+.1f）"
                   % (p["gray_mean"], h["gray_mean"], d_gray))
    reasons.append("灰度P25  印刷 %.1f / 手写 %.1f （差 %+.1f）"
                   % (p["gray_p25"], h["gray_p25"], d_p25))
    reasons.append("BGR 印刷 %.1f/%.1f/%.1f  手写 %.1f/%.1f/%.1f"
                   % (*p["bgr"], *h["bgr"]))
    reasons.append("笔画半宽 印刷 %.2f / 手写 %.2f"
                   % (p["dt_med"], h["dt_med"]))

    if abs(d_p25) >= 25 or abs(d_gray) >= 30:
        return "MAYBE_BY_GRAY", reasons + [
            "手写与印刷墨色差距够大，可试阈值分离，但必须逐处放大复核"]

    sw_ratio = abs(h["dt_med"] - p["dt_med"]) / max(1e-6, max(h["dt_med"], p["dt_med"]))
    reasons.append("笔画粗细差比 %.0f%%" % (sw_ratio * 100))
    if sw_ratio > 0.5 and abs(h["dt_med"] - p["dt_med"]) > 0.6:
        return "MAYBE_BY_STROKE", reasons + [
            "手写笔画明显更粗/更细，可试形态学开运算，但粗体印刷字会被误伤，必须复核"]

    return "NOT_SEPARABLE", reasons + [
        "颜色/灰度/粗细三个维度全部落在噪声量级 —— 不存在可靠的自动分离开关"]


def main():
    ap = argparse.ArgumentParser(description="手写可分离性判定")
    ap.add_argument("image")
    ap.add_argument("--rotate", default="none")
    ap.add_argument("--grid", action="store_true", help="只输出坐标网格图")
    ap.add_argument("--grid-step", type=int, default=GRID_STEP_DEFAULT)
    ap.add_argument("--regions", help="regions.json 路径")
    ap.add_argument("--outdir", default=".", help="输出目录")
    ap.add_argument("--thr", type=int, default=150, help="暗像素灰度阈值")
    ap.add_argument("--zoom", type=int, default=5, help="区域放大倍数")
    ap.add_argument("--pad", type=int, default=8, help="区域取坐标时外扩像素")
    args = ap.parse_args()

    img = imread_u(args.image)
    if img is None:
        raise SystemExit("读不到图: %s" % args.image)
    img = rotate(img, args.rotate)
    ensure_dir(args.outdir)
    base = os.path.splitext(os.path.basename(args.image))[0]
    banner("手写可分离性判定")
    print("源 %s  转正(%s) → %dx%d" % (args.image, args.rotate,
                                       img.shape[1], img.shape[0]))

    if args.grid or not args.regions:
        g = make_grid(img, args.grid_step)
        dst = os.path.join(args.outdir, "%s_grid.jpg" % base)
        imwrite_u(dst, g, 90)
        print("\n坐标网格 → %s" % dst)
        print("  蓝线每 %d px 一条，标的是**转正后原图**的像素坐标。" % args.grid_step)
        if not args.regions:
            print("\n下一步：照着网格读出坐标，写成 regions.json，再带 --regions 跑一次。")
            return

    spec = load_json(args.regions)
    rot_in_file = spec.get("rotate")
    if rot_in_file and rot_in_file != args.rotate:
        print("  ⚠️ regions.json 里写的 rotate=%s 与命令行 --rotate=%s 不一致，"
              "以命令行为准，请确认坐标是按哪个方向量的。" % (rot_in_file, args.rotate))

    def boxes(key):
        raw = spec.get(key) or spec.get(key.upper()) or []
        out = []
        for b in raw:
            x, y, w, h = [int(v) for v in b]
            out.append((x - args.pad, y - args.pad, w + 2 * args.pad, h + 2 * args.pad))
        return out

    pb, hb = boxes("print"), boxes("hand")
    if not pb or not hb:
        raise SystemExit("regions.json 必须同时含 print 和 hand（各 2 处以上更稳）")

    print("\n区域统计（暗像素阈值 gray<%d）:" % args.thr)
    ps, hs = [], []
    for tag, bx, store, col in (("PRINT", pb, ps, (40, 90, 200)),
                                ("HAND", hb, hs, (40, 160, 60))):
        for i, b in enumerate(bx):
            s = stroke_stats(img, b, args.thr)
            if s is None:
                print("  [%s #%d] %s → 暗像素不足，坐标可能取错，请照网格重取" % (tag, i + 1, list(b)))
                continue
            store.append(s)
            print("  [%s #%d] box=%s 暗%6d 灰度%6.1f 最暗%3d P25=%6.1f "
                  "BGR=%5.1f/%5.1f/%5.1f 彩度%5.1f 蓝笔率%.2f 笔画%.2f"
                  % (tag, i + 1, s["box"], s["dark_px"], s["gray_mean"], s["gray_min"],
                     s["gray_p25"], *s["bgr"], s["chroma"], s["blue_frac"], s["dt_med"]))

    if not ps or not hs:
        raise SystemExit("有效区域不足（print=%d hand=%d），无法判定。" % (len(ps), len(hs)))

    P, H = _agg(ps), _agg(hs)
    v, reasons = verdict_of(P, H)

    print("\n" + "-" * 68)
    print("结论: %s" % v)
    for r in reasons:
        print("  · " + r)
    print("-" * 68)
    guide = {
        "SEPARABLE_BY_COLOR": "→ 用 scripts/separate_ink.py 按颜色剔除手写，基本无损。",
        "MAYBE_BY_GRAY": "→ 可用 scripts/separate_ink.py --by gray 试阈值；"
                         "务必逐处放大复核印刷体是否受损。",
        "MAYBE_BY_STROKE": "→ 可试形态学开运算，风险高；建议直接走交互擦除工具。",
        "NOT_SEPARABLE": "→ 不要自动擦。交付 assets/eraser_tool.html + 已去红笔底图，"
                         "让用户交互擦除（薄笔画点一下就选中）。",
    }[v]
    print(guide)

    # 可视化：每个区域放大并左右对照
    zooms = []
    for tag, store, col in (("PRINT", ps, (40, 90, 200)), ("HAND", hs, (40, 160, 60))):
        row = []
        for s in store:
            x, y, w, h = s["box"]
            c = img[max(0, y):y + h, max(0, x):x + w]
            if c.size == 0:
                continue
            z = cv2.resize(c, (max(1, w * args.zoom), max(1, h * args.zoom)),
                           interpolation=cv2.INTER_NEAREST)
            z = preview(z, 900)
            row.append(vstack_pad([label_strip("%s %dx%d" % (tag, x, y), z.shape[1],
                                               bg=(255, 255, 255), fg=col), z], pad=0))
        if row:
            zooms.append(hstack_pad(row))
    if zooms:
        dst = os.path.join(args.outdir, "%s_diag_zoom.jpg" % base)
        imwrite_u(dst, vstack_pad(zooms), 92)
        print("\n区域放大对照 → %s" % dst)

    rep = os.path.join(args.outdir, "%s_diag.json" % base)
    dump_json(rep, {"image": args.image, "rotate": args.rotate, "verdict": v,
                    "reasons": reasons, "print": P, "hand": H,
                    "regions": {"print": ps, "hand": hs}})
    print("机器可读结论 → %s" % rep)
    print("\n⚠️ 结论必须配合 %s_diag_zoom.jpg 目视确认（先确认坐标框住的确实是那类内容），"
          "再决定走哪条路。" % base)


if __name__ == "__main__":
    main()
