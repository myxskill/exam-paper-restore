# -*- coding: utf-8 -*-
"""有空白原件时的最优解：几何对齐 + 相减，把"后加上去的墨"整片换回干净纸面。

这是唯一能做到**印刷体一个像素都不动**的路径：
空白原件本身就带正确的印刷内容，所以只要把它配准到"填写过的照片"的几何上，
凡是"填写版比空白版更暗"的地方，就是学生/老师后加上去的墨，直接取空白版覆盖。

用法:
  # 1) 先只做配准，看能不能对上
  python align_subtract.py FILLED BLANK --detect-only

  # 2) 对得上就出结果
  python align_subtract.py FILLED BLANK --out OUT.jpg \
         --rotate-filled cw90 --rotate-blank none --margin 22 --feather 2.5

  FILLED = 填写过的照片；BLANK = 空白件（照片或电子图都行）
"""
import os
import sys
import argparse

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (imread_u, imwrite_u, rotate, preview, label_strip,
                     vstack_pad, hstack_pad, banner, ensure_dir)


def _prep_gray(img, max_side=1600):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    k = min(1.0, float(max_side) / max(h, w))
    if k < 1.0:
        g = cv2.resize(g, (max(1, int(w * k)), max(1, int(h * k))),
                       interpolation=cv2.INTER_AREA)
    return g, k


def align_orb(filled_g, blank_g):
    """ORB 特征 + RANSAC 单应。返回 (H, 内点数, 匹配数)；失败返回 (None, 0, 0)。"""
    orb = cv2.ORB_create(nfeatures=8000, fastThreshold=8)
    k1, d1 = orb.detectAndCompute(filled_g, None)
    k2, d2 = orb.detectAndCompute(blank_g, None)
    if d1 is None or d2 is None or len(k1) < 30 or len(k2) < 30:
        return None, 0, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(d2, d1, k=2)          # blank -> filled
    good = [m for m, n in (p for p in knn if len(p) == 2)
            if m.distance < 0.78 * n.distance]
    if len(good) < 15:
        return None, 0, len(good)
    src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0, maxIters=4000,
                                confidence=0.995)
    n_in = int(inl.sum()) if inl is not None else 0
    return H, n_in, len(good)


def align_ecc(filled_g, blank_g, iters=200):
    """兜底：ECC 仿射配准，适合内容几乎一致、只有轻微错位的场景。"""
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-7)
    try:
        cc, warp = cv2.findTransformECC(filled_g, blank_g, warp,
                                        cv2.MOTION_AFFINE, criteria, None, 5)
        return warp, float(cc)
    except cv2.error:
        return None, 0.0


def main():
    ap = argparse.ArgumentParser(description="空白原件对齐相减")
    ap.add_argument("filled", help="填写过的照片")
    ap.add_argument("blank", help="空白原件（照片或电子图）")
    ap.add_argument("--out", help="输出图路径")
    ap.add_argument("--rotate-filled", default="none")
    ap.add_argument("--rotate-blank", default="none")
    ap.add_argument("--margin", type=int, default=22,
                    help="判定'后加墨'的灰度差阈值，越大越保守（越不容易误伤）")
    ap.add_argument("--feather", type=float, default=2.5, help="接缝羽化半径(px)")
    ap.add_argument("--detect-only", action="store_true", help="只做配准检查")
    ap.add_argument("--outdir", default=".", help="核对图输出目录")
    args = ap.parse_args()

    filled = imread_u(args.filled)
    blank = imread_u(args.blank)
    if filled is None:
        raise SystemExit("读不到填写版: %s" % args.filled)
    if blank is None:
        raise SystemExit("读不到空白版: %s" % args.blank)
    filled = rotate(filled, args.rotate_filled)
    blank = rotate(blank, args.rotate_blank)
    banner("空白原件对齐相减")
    print("填写版 %dx%d   空白版 %dx%d"
          % (filled.shape[1], filled.shape[0], blank.shape[1], blank.shape[0]))

    fg, kf = _prep_gray(filled)
    bg, kb = _prep_gray(blank)

    Hs = np.diag([1.0 / kf, 1.0 / kf, 1.0])       # filled 降采样 -> 原图
    Hb = np.diag([1.0 / kb, 1.0 / kb, 1.0])       # blank  降采样 -> 原图

    mode, H_full, quality = None, None, ""
    H, n_in, n_good = align_orb(fg, bg)
    if H is not None:
        mode, H_full = "ORB单应", Hs @ H @ np.linalg.inv(Hb)
        quality = "匹配对 %d / 内点 %d" % (n_good, n_in)
        print("\n[ORB] %s" % quality)
    else:
        print("\n[ORB] 失败（匹配对 %d），转 ECC 仿射…" % n_good)

    if H_full is None:
        warp, cc = align_ecc(fg, bg)
        if warp is not None:
            mode = "ECC仿射"
            A = np.vstack([warp, [0, 0, 1]])
            H_full = Hs @ A @ np.linalg.inv(Hb)
            quality = "相关系数 %.4f" % cc
            print("[ECC] %s" % quality)
        else:
            raise SystemExit("两种配准都失败：两份材料可能是不同的卷子/版式，"
                             "或空白件偏差太大。请换一张空白件，或改走擦除工具。")

    # 把空白版变换到填写版的几何
    h, w = filled.shape[:2]
    blank_aligned = cv2.warpPerspective(blank, H_full, (w, h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)

    with np.errstate(all="ignore"):
        if mode == "ECC仿射":
            valid = np.ones((h, w), bool)
        else:
            # 用单应把填写版的四角投回去，判断空白版覆盖范围
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            pts = cv2.perspectiveTransform(corners, np.linalg.inv(H_full)).reshape(-1, 2)
            x0, y0 = np.floor(pts.min(0)).astype(int)
            x1, y1 = np.ceil(pts.max(0)).astype(int)
            valid = np.zeros((h, w), bool)
            vx0, vy0 = max(0, x0), max(0, y0)
            vx1, vy1 = min(blank.shape[1], x1), min(blank.shape[0], y1)
            if vx1 > vx0 and vy1 > vy0:
                # 反向映射得到有效区
                m = np.full((blank.shape[0], blank.shape[1]), 255, np.uint8)
                mw = cv2.warpPerspective(m, H_full, (w, h), flags=cv2.INTER_NEAREST)
                valid = mw > 200

    f_gr = cv2.cvtColor(filled, cv2.COLOR_BGR2GRAY).astype(np.int16)
    b_gr = cv2.cvtColor(blank_aligned, cv2.COLOR_BGR2GRAY).astype(np.int16)
    added = (f_gr < b_gr - args.margin) & valid
    print("\n判定为'后加墨'的像素: %d (%.2f%%)"
          % (int(added.sum()), added.mean() * 100))
    if added.mean() < 0.0005:
        print("  ⚠️ 几乎没有检出后加墨 —— 要么两份材料本就一样，"
              "要么配准没对准（放大核对 diff 图）。")

    if args.detect_only:
        print("\n--detect-only：未输出成品。若上面内点数/相关系数不低，"
              "去掉该参数即可出图。")
        return

    if not args.out:
        raise SystemExit("请给 --out 指定输出路径")

    # 羽化合成：后加墨处取空白版内容，边缘高斯羽化避免硬接缝
    a = cv2.dilate(added.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
    if args.feather > 0:
        ksize = int(max(3, round(args.feather * 4)) | 1)
        a = cv2.GaussianBlur(a.astype(np.float32), (ksize, ksize), args.feather)
    a = np.clip(a, 0, 1)[..., None]
    out = (filled.astype(np.float32) * (1 - a) +
           blank_aligned.astype(np.float32) * a).astype(np.uint8)
    imwrite_u(args.out, out, 95)
    print("已输出 %s  (%dx%d)" % (args.out, w, h))

    # 核对图：填写版 / 空白版(变换后) / 检出掩膜 / 成品
    od = ensure_dir(args.outdir)
    stem = os.path.splitext(os.path.basename(args.out))[0]
    vis = filled.copy()
    vis[added] = (0, 0, 255)
    row = hstack_pad([
        vstack_pad([label_strip("1 填写版", 700), preview(filled, 700)], pad=0),
        vstack_pad([label_strip("2 空白版(已配准)", 700), preview(blank_aligned, 700)], pad=0),
    ])
    row2 = hstack_pad([
        vstack_pad([label_strip("3 检出的后加墨(红)", 700), preview(vis, 700)], pad=0),
        vstack_pad([label_strip("4 成品", 700), preview(out, 700)], pad=0),
    ])
    dst = os.path.join(od, stem + "_align_check.jpg")
    imwrite_u(dst, vstack_pad([row, row2]), 90)
    print("核对图 → %s" % dst)
    print("\n复核要点：① 印刷线条有没有重影（配准没对准会重影）；"
          "② 有没有漏擦的笔迹。若重影，换更清晰的空白件或调 --rotate-blank。")


if __name__ == "__main__":
    main()
