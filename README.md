# exam-paper-restore · 试卷还原 / 手写擦除

把「考过、老师批过」的试卷照片，还原成可以重新打印的**空白卷**。

A skill that restores a photographed **used exam paper** into a clean, printable **blank version** — by deterministically removing teacher red-pen marks, diagnosing whether student handwriting is auto-separable from print, and delivering an interactive eraser tool when it is not.

---

## 它解决什么

| 内容 | 能否自动去除 | 原因 |
|---|---|---|
| 老师红笔批改（分数 / √ / 圈画 / 划线） | ✅ 能，质量高 | 红笔是彩色的，与中性油墨在颜色维度天然可分 |
| 学生手写答案 | ⚠️ 取决于笔色 | 深色中性笔与印刷油墨常在**颜色 / 灰度 / 笔画粗细**三项上完全重叠，不存在可靠分离开关 |
| 照片朝向 | ✅ 能判定 | 出四方向联系表，人眼一秒可辨 |
| 有空白原件的场景 | ✅ 能，**零改动** | 几何对齐后相减，印刷体一个像素都不动 |

> **核心态度：不夸大、不假装。** 能自动做的做扎实，做不到的给出根因和最快的手工路径，而不是假装成功。

## 工作流

```
第1步 盘点输入
第2步 判朝向（强制，别默认 180°）
第3步 去红笔批改（确定性，稳赢）
第4步 诊断手写可分离性（决策点） ──┬─ SEPARABLE_BY_COLOR → 按颜色擦
                                     ├─ MAYBE_BY_GRAY/STROKE → 按灰度擦（高风险，必须复核）
                                     └─ NOT_SEPARABLE        → 交互擦除工具
第5步 有空白原件 → 对齐相减（唯一零改动路径）
第6步 交付：干净底图 + 工具/成品 + 一句实话
```

完整流程见 **[`SKILL.md`](SKILL.md)**；实测数据、判据阈值、环境坑见 **[`references/field-notes.md`](references/field-notes.md)**。

## 快速开始

```bash
pip install numpy opencv-python-headless
SK="path/to/exam-paper-restore/scripts"
PY=python

# 1. 判朝向（先看联系表，挑文字水平可读的那格）
"$PY" "$SK/orient_check.py" 试卷.jpg --outdir work
"$PY" "$SK/orient_check.py" 试卷.jpg --outdir work --apply cw90

# 2. 去红笔批改（--split 2 表示双页并排切成两页）
"$PY" "$SK/remove_red.py" 试卷.jpg --rotate cw90 --split 2 --out work/p1_nored.jpg

# 3. 诊断手写能否自动分离
"$PY" "$SK/diagnose_ink.py" work/p1_nored.jpg --rotate cw90 --grid --outdir work
#    照网格写 regions.json 后：
"$PY" "$SK/diagnose_ink.py" work/p1_nored.jpg --rotate cw90 --regions work/regions.json --outdir work

# 4. 有空白原件 → 对齐相减（零改动）
"$PY" "$SK/align_subtract.py" 填写版.jpg 空白版.jpg --detect-only --outdir work
"$PY" "$SK/align_subtract.py" 填写版.jpg 空白版.jpg --margin 22 --out out/空白.jpg --outdir work
```

不可自动分离时，把 `assets/eraser_tool.html` 复制给用户 —— 单文件、纯前端、离线可用，点一下选中整段笔画再擦。

## 目录结构

| 路径 | 用途 |
|---|---|
| `SKILL.md` | 技能主文档：完整工作流与硬规则 |
| `references/field-notes.md` | 实测数据、判据阈值、环境坑、SOP 清单 |
| `scripts/_common.py` | 共享层：Unicode 路径 IO、旋转、红笔掩膜、笔画度量 |
| `scripts/orient_check.py` | 四方向联系表；`--apply` 写全分辨率转正图 |
| `scripts/remove_red.py` | 确定性去红笔；`--split` 切页、`--overlay` 出核对图 |
| `scripts/diagnose_ink.py` | `--grid` 出坐标网格；`--regions` 出可分离性判定 |
| `scripts/separate_ink.py` | 按颜色 / 灰度分离并擦除手写 |
| `scripts/align_subtract.py` | 空白原件对齐相减（零改动路径） |
| `assets/eraser_tool.html` | 交互式擦除工具（纯前端单文件，无需联网） |

## 关键设计取舍

- **确定性优先**：红笔去除、朝向纠正这类可判定的步骤绝不交给启发式猜测。
- **诚实交付**：当三个维度全部重叠时，明确判定 `NOT_SEPARABLE`，绝不硬做自动擦除去啃印刷体。
- **Windows 中文路径**：一律走 `_common.imread_u / imwrite_u`；`cv2.imread` 遇中文路径会**静默返回 None**。
- **拒绝负优化**：实测通用 AI 影像擦除模型对试卷「几乎不动手写、还压分辨率、改宽高比」，不采用。

## 环境

- Python 3.9+
- `numpy`、`opencv-python-headless`

## 许可

[MIT](LICENSE)
