# -*- coding: utf-8 -*-
"""exam-paper-restore 共享工具层。

只依赖 numpy + opencv-python。所有 IO 都走 np.fromfile/tofile，
因为 cv2.imread / imwrite 在 Windows 上无法处理含中文的路径。
"""
import os
import sys
import json
import numpy as np
import cv2


# ---------------------------------------------------------------- IO

def imread_u(path, flags=cv2.IMREAD_COLOR):
    """读图，兼容中文/Unicode 路径。返回 BGR ndarray，失败返回 None。"""
    if not os.path.isfile(path):
        return None
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, flags)
    except Exception:
        return None


def imwrite_u(path, img, quality=95):
    """写图，兼容中文/Unicode 路径。按扩展名选择编码器。"""
    ext = os.path.splitext(path)[1].lower() or ".png"
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif ext == ".png":
        # 无损，但给个压缩级别免得 3000px 图写太久
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(quality)]
    else:
        params = []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError("编码失败: " + path)
    buf.tofile(path)
    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- 几何

ROT_CHOICES = ("none", "cw90", "ccw90", "180")


def rotate(img, spec):
    """spec: none | cw90 | ccw90 | 180。cw90 == np.rot90(img, -1)。"""
    spec = (spec or "none").lower().replace(" ", "")
    if spec in ("none", "0", ""):
        return img
    if spec == "cw90":
        return np.rot90(img, -1)
    if spec == "ccw90":
        return np.rot90(img, 1)
    if spec == "180":
        return np.rot90(img, 2)
    raise ValueError("未知旋转: %r（可选 %s）" % (spec, "/".join(ROT_CHOICES)))


def preview(img, max_side=1500):
    """等比缩到最长边 <= max_side，用于预览/联系表。"""
    h, w = img.shape[:2]
    k = min(1.0, float(max_side) / max(h, w))
    if k >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))),
                      interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- 纸张自适应

def paper_bbox(img, thresh=235, margin=0):
    """估计纸张内容包围盒（去掉四周桌面色）。返回 (x0, y0, x1, y1)，左闭右开。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 纸面通常偏亮且低饱和，取"接近白"的像素
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.int16)
    paper = (g >= thresh) & (sat <= 60)
    if paper.mean() < 0.05:
        return (0, 0, img.shape[1], img.shape[0])
    ys, xs = np.where(paper)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    if margin:
        x0 = max(0, x0 - margin); y0 = max(0, y0 - margin)
        x1 = min(img.shape[1], x1 + margin); y1 = min(img.shape[0], y1 + margin)
    return (x0, y0, x1, y1)


# ---------------------------------------------------------------- 红笔掩膜

def red_mask(img, k=9.0, rel_thr=6.0, min_area=10, dilate=1,
             bg_kernel=61, restrict_to_paper=True):
    """提取"红笔批改"掩膜。

    原理：纸张本身常常偏暖/发灰，单靠绝对阈值会把整张纸判成红。
    所以用两条互补判据，且都以"相对局部背景的增量"为准：
      1) LAB 的 a* 通道（红-绿轴）偏离中性 128 超过 k；
      2) 红度 redness = R-(G+B)/2 相对局部中值背景的增量超过 rel_thr。

    返回 uint8 掩膜（0/1），已按连通域面积过滤 + 适度膨胀。
    另加返回覆盖率供调用方判断阈值是否过松/过紧。
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    A = lab[:, :, 1].astype(np.float32)

    b, g, r = (img[:, :, i].astype(np.float32) for i in range(3))
    redness = r - (g + b) / 2.0

    # 局部背景：大核中值。注意 medianBlur 只吃 uint8，先量化会损失精度，
    # 但红笔判据本来就是"增量"而非绝对值，够用且快。
    rq = np.clip(redness + 128, 0, 255).astype(np.uint8)
    bg = cv2.medianBlur(rq, bg_kernel).astype(np.float32) - 128.0
    rel = redness - bg

    m = ((A - 128.0) > k) | (rel > rel_thr)
    if restrict_to_paper:
        # 纸外（桌面/阴影）不参与，避免背景物被判成红笔
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m &= (gray > 60)
    m = m.astype(np.uint8)

    if min_area > 1:
        n, lb, st, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros_like(m)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= min_area:
                keep[lb == i] = 1
        m = keep

    if dilate > 0:
        m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=dilate)
    return m


def red_sensitivity(img):
    """量一下这张图的"红笔强度"，帮助选阈值。返回 dict。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    A = lab[:, :, 1].astype(np.float32) - 128.0
    b, g, r = (img[:, :, i].astype(np.float32) for i in range(3))
    redness = r - (g + b) / 2.0
    rq = np.clip(redness + 128, 0, 255).astype(np.uint8)
    bg = cv2.medianBlur(rq, 61).astype(np.float32) - 128.0
    rel = redness - bg
    return {
        "a_star_p99": float(np.percentile(A, 99)),
        "a_star_max": float(A.max()),
        "a_star_over_9_pct": float((A > 9).mean() * 100),
        "redness_rel_p99": float(np.percentile(rel, 99)),
        "advice": ("红度很弱(P99 < 12)，照片可能偏灰/红笔很淡，"
                   "建议 k 收到 7~9、rel_thr 保持 6，并目视复核残留"
                   if np.percentile(A, 99) < 12 else
                   "红度正常，默认参数即可"),
    }


# ---------------------------------------------------------------- 修复填充

def inpaint_with_mask(img, mask, radius=3, method="telea"):
    """用掩膜做修复填充。mask 可为 bool/uint8。"""
    m = (mask > 0).astype(np.uint8) * 255
    flags = cv2.INPAINT_TELEA if str(method).lower() != "ns" else cv2.INPAINT_NS
    return cv2.inpaint(img, m, int(radius), flags)


def vis_mask(img, mask, color=(0, 255, 0), alpha=1.0):
    """把掩膜涂成彩色以便目视核对。"""
    out = img.copy()
    sel = mask > 0
    if alpha >= 1.0:
        out[sel] = color
    else:
        out[sel] = (out[sel] * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)
    return out


# ---------------------------------------------------------------- 笔画度量

def stroke_stats(img_or_gray, box, thr=150):
    """量一个矩形区域内的笔画粗细/墨色。

    box = (x, y, w, h)。返回 dict：
      dark_px     暗像素数
      gray_mean   暗像素灰度均值（越低越黑）
      gray_min    最暗灰度
      gray_p25    暗像素灰度 25 分位（比均值更抗纸面起伏）
      bgr         暗像素 B/G/R 均值
      chroma      暗像素平均彩度（max-min，越大越"有色"）
      blue_frac   B-R > 25 的占比（判断是不是蓝笔）
      dt_med      距离变换中位半径（≈笔画半宽，越小越细）
      dt_p90      距离变换 90 分位
    区域过小/无暗像素时返回 None。
    """
    g = img_or_gray if img_or_gray.ndim == 2 else cv2.cvtColor(img_or_gray, cv2.COLOR_BGR2GRAY)
    x, y, w, h = box
    H, W = g.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + w)), min(H, int(y + h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    sub = g[y0:y1, x0:x1]
    m = (sub < thr).astype(np.uint8)
    if int(m.sum()) < 20:
        return None

    if img_or_gray.ndim == 3:
        px = img_or_gray[y0:y1, x0:x1][m > 0].astype(np.float32)
        bgr = [float(px[:, i].mean()) for i in range(3)]
        chroma = float((px.max(axis=1) - px.min(axis=1)).mean())
        blue_frac = float(((px[:, 0] - px[:, 2]) > 25).mean())
    else:
        bgr = [float(sub[m > 0].mean())] * 3
        chroma = 0.0
        blue_frac = 0.0

    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    vals = dt[m > 0]
    return {
        "box": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
        "dark_px": int(m.sum()),
        "gray_mean": float(sub[m > 0].mean()),
        "gray_min": int(sub.min()),
        "gray_p25": float(np.percentile(sub[m > 0], 25)),
        "bgr": [round(v, 1) for v in bgr],
        "chroma": round(chroma, 1),
        "blue_frac": round(blue_frac, 3),
        "dt_med": round(float(np.median(vals)), 2),
        "dt_p90": round(float(np.percentile(vals, 90)), 2),
    }


# ---------------------------------------------------------------- 小工具

def ensure_dir(p):
    if p:
        os.makedirs(p, exist_ok=True)
    return p


def eprint(*a):
    print(*a, file=sys.stderr)


def banner(t):
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


def label_strip(text, width, height=34, bg=(255, 255, 255), fg=(180, 20, 20)):
    """生成一条带文字的白底标签条（用于拼联系表）。"""
    strip = np.full((height, max(1, width), 3), bg, np.uint8)
    cv2.putText(strip, text, (10, int(height * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, fg, 2, cv2.LINE_AA)
    return strip


def hstack_pad(imgs, pad=8, bg=245):
    """等高横向拼接，自动补白。"""
    H = max(i.shape[0] for i in imgs)
    out = []
    for i, im in enumerate(imgs):
        if im.shape[0] < H:
            im = np.vstack([im, np.full((H - im.shape[0], im.shape[1], 3), bg, np.uint8)])
        out.append(im)
        if i != len(imgs) - 1:
            out.append(np.full((H, pad, 3), bg, np.uint8))
    return np.hstack(out)


def vstack_pad(imgs, pad=8, bg=245):
    """等宽纵向拼接，自动补白。"""
    W = max(i.shape[1] for i in imgs)
    out = []
    for i, im in enumerate(imgs):
        if im.shape[1] < W:
            im = np.hstack([im, np.full((im.shape[0], W - im.shape[1], 3), bg, np.uint8)])
        out.append(im)
        if i != len(imgs) - 1:
            out.append(np.full((pad, W, 3), bg, np.uint8))
    return np.vstack(out)
