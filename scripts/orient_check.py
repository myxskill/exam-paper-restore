# -*- coding: utf-8 -*-
"""朝向判定：一次看清原图该怎么转。

为什么必须做这一步：手机拍的试卷经常是「一张纸的正反面各拍一次」，
每张图里左右两页并排，正确朝向往往是 90°，而不是想当然的 180°。
方向判错 → 后面所有坐标、裁切、掩膜全部白做。

用法:
  # 1) 出四方向联系表，人眼挑出正确的那个
  python orient_check.py IMG [IMG2 ...] --outdir OUTDIR

  # 2) 确认后，按指定方向写出全分辨率转正图
  python orient_check.py IMG --outdir OUTDIR --apply cw90
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (imread_u, imwrite_u, rotate, label_strip,
                     vstack_pad, hstack_pad, banner, ensure_dir)

DIRS = [("none", "0-原图"), ("cw90", "1-顺时针90"), ("180", "2-180"),
        ("ccw90", "3-逆时针90")]


def contact_sheet(img, tile=660):
    """四方向联系表：2×2 网格，每格带标签。"""
    tiles = []
    for spec, name in DIRS:
        r = rotate(img, spec)
        p = r
        h, w = p.shape[:2]
        k = min(1.0, float(tile) / max(h, w))
        if k < 1.0:
            import cv2
            p = cv2.resize(p, (max(1, int(w * k)), max(1, int(h * k))),
                           interpolation=cv2.INTER_AREA)
        tiles.append(vstack_pad([label_strip("%s   %dx%d" % (name, w, h), p.shape[1]),
                                 p], pad=0))
    row1 = hstack_pad(tiles[:2])
    row2 = hstack_pad(tiles[2:])
    return vstack_pad([row1, row2])


def main():
    ap = argparse.ArgumentParser(description="试卷朝向判定 / 转正")
    ap.add_argument("images", nargs="+", help="输入图片（可多个）")
    ap.add_argument("--outdir", default=".", help="输出目录")
    ap.add_argument("--apply", choices=[d[0] for d in DIRS],
                    help="按此方向写出全分辨率转正图")
    ap.add_argument("--tile", type=int, default=660, help="联系表单格边长")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    banner("朝向判定")
    for p in args.images:
        img = imread_u(p)
        if img is None:
            print("[跳过] 读不到图: %s" % p)
            continue
        h, w = img.shape[:2]
        base = os.path.splitext(os.path.basename(p))[0]
        print("\n%s\n  原始尺寸 %dx%d  宽高比 %.3f  (%.1f MB)"
              % (p, w, h, w / float(h), os.path.getsize(p) / 1048576.0))

        if args.apply:
            out = rotate(img, args.apply)
            dst = os.path.join(args.outdir, "%s_rot_%s.jpg" % (base, args.apply))
            imwrite_u(dst, out, 95)
            print("  已按 %s 转正 → %s  (%dx%d)"
                  % (args.apply, dst, out.shape[1], out.shape[0]))
        else:
            dst = os.path.join(args.outdir, "%s_orient.jpg" % base)
            imwrite_u(dst, contact_sheet(img, args.tile), 90)
            print("  四方向联系表 → %s" % dst)

    if not args.apply:
        print("\n读图判断方法：找出**所有文字都水平可读**的那一格，"
              "那个标签就是正确方向；再用 --apply 写全分辨率转正图。")
        print("注意：不要默认 180°，手机拍折叠试卷 90° 更常见。")


if __name__ == "__main__":
    main()
