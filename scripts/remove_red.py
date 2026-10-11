# -*- coding: utf-8 -*-
"""确定性去除老师红笔批改（分数、√、圈画、划线）。

这是整条链路里**唯一稳赢**的一步：红笔是彩色、印刷体与手写体是中性墨，
颜色维度天然可分。质量靠两点保证：
  1) 判据用「相对局部纸面的红度增量」，而不是绝对阈值 —— 纸张偏暖/发灰时
     绝对阈值会把整张纸判成红（实测有人踩过：掩膜覆盖 18%，印刷体被啃掉）。
  2) 出可视化叠加图 + 覆盖率数字，**必须目视复核**"红笔压在印刷体上"的区域。

用法:
  python remove_red.py IMG --out OUT.jpg [--rotate cw90] [--split 2]
                           [--k 9] [--radius 3] [--overlay]

  --split N   转正后按竖条切成 N 页（双页并排拍摄时用），文件名自动加 _p1/_p2
  --overlay   额外输出绿色掩膜叠加图，用于核对
"""
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (imread_u, imwrite_u, rotate, red_mask, red_sensitivity,
                     inpaint_with_mask, vis_mask, preview, banner, ensure_dir)


def split_pages(img, n):
    """按竖条切成 n 页；返回 [(name_suffix, sub_img), ...]。自动裁掉白边。"""
    h, w = img.shape[:2]
    out = []
    for i in range(n):
        x0 = int(round(w * i / float(n)))
        x1 = int(round(w * (i + 1) / float(n)))
        out.append(("_p%d" % (i + 1), img[:, x0:x1].copy()))
    return out


def main():
    ap = argparse.ArgumentParser(description="去除红笔批改")
    ap.add_argument("image")
    ap.add_argument("--out", required=True, help="输出图路径（.jpg/.png）")
    ap.add_argument("--rotate", default="none",
                    help="转正方向 none|cw90|ccw90|180（先用 orient_check.py 判定）")
    ap.add_argument("--split", type=int, default=0, help="转正后切成 N 页，0=不切")
    ap.add_argument("--k", type=float, default=9.0, help="a* 通道阈值，越小越敏感")
    ap.add_argument("--rel-thr", type=float, default=6.0, help="相对红度增量阈值")
    ap.add_argument("--min-area", type=int, default=10, help="连通域最小面积(去噪)")
    ap.add_argument("--radius", type=int, default=3, help="修复填充半径")
    ap.add_argument("--method", default="telea", choices=["telea", "ns"])
    ap.add_argument("--overlay", action="store_true", help="额外输出掩膜叠加图")
    args = ap.parse_args()

    img = imread_u(args.image)
    if img is None:
        raise SystemExit("读不到图: %s" % args.image)

    banner("去除红笔批改")
    img = rotate(img, args.rotate)
    h, w = img.shape[:2]
    print("输入 %s\n转正(%s)后尺寸 %dx%d" % (args.image, args.rotate, w, h))

    sens = red_sensitivity(img)
    print("\n红笔强度体检:")
    for k, v in sens.items():
        print("  %-20s %s" % (k, v))

    mask = red_mask(img, k=args.k, rel_thr=args.rel_thr,
                    min_area=args.min_area, dilate=1)
    cov = mask.mean() * 100.0
    print("\n掩膜: %d px  覆盖率 %.2f%%" % (int(mask.sum()), cov))
    if cov > 12:
        print("  ⚠️ 覆盖率偏高(>12%)：判据可能过松，会把纸面/印刷体一起吃掉。"
              "把 --k 提到 11~14 再试。")
    elif cov < 0.02:
        print("  ⚠️ 覆盖率极低(<0.02%)：判据可能过严，红笔没抓全。"
              "把 --k 降到 6~7、--rel-thr 降到 4 再试。")

    out = inpaint_with_mask(img, mask, args.radius, args.method)

    jobs = [("", out)] if args.split <= 0 else [
        (sfx, sub) for sfx, sub in split_pages(out, args.split)]
    base, ext = os.path.splitext(args.out)
    for sfx, sub in jobs:
        dst = base + sfx + ext
        imwrite_u(dst, sub, 95)
        print("  已输出 %s  (%dx%d)" % (dst, sub.shape[1], sub.shape[0]))

    # 可视化：原始 / 掩膜叠加 / 结果 —— 横向三联用于目视核对
    od = os.path.join(os.path.dirname(os.path.abspath(args.out)), "_red_check")
    ensure_dir(od)
    stem = os.path.basename(base)
    v = vis_mask(preview(img, 1200), preview(mask, 1200))
    imwrite_u(os.path.join(od, stem + "_overlay.jpg"), v, 88)
    imwrite_u(os.path.join(od, stem + "_after.jpg"), preview(out, 1200), 88)
    print("  核对图 → %s/%s_overlay.jpg（绿色=判定为红笔的像素）" % (od, stem))
    if args.overlay:
        print("\n复核要点：放大看红笔与印刷体重叠的地方，"
              "确认印刷线条没有被啃断、红痕是否还有淡残留。")


if __name__ == "__main__":
    main()
