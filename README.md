# 浣熊三视图与灰模

V25 是已确认的侧视、背视美术参考。V28 是当前待审的后脑外缘与脸颊灰模，使用同一站姿、正交镜头和脚底线。原画、图稿、模型和渲染结果按版本归档在 [raccoon-turnaround](raccoon-turnaround)。

| 版本 | 审阅资料 |
| --- | --- |
| V28 灰模 | [五角度叠线总览](raccoon-turnaround/head_outline_3d_v28/gray-overlays.png) · [0°](raccoon-turnaround/head_outline_3d_v28/overlays/000.png) · [45°](raccoon-turnaround/head_outline_3d_v28/overlays/045.png) · [90°](raccoon-turnaround/head_outline_3d_v28/overlays/090.png) · [135°](raccoon-turnaround/head_outline_3d_v28/overlays/135.png) · [180°](raccoon-turnaround/head_outline_3d_v28/overlays/180.png) |
| V27 后脑校准 | [轮廓叠线](raccoon-turnaround/head_shape_3d_v27/head-outline.png) · [五角度彩色静帧](raccoon-turnaround/head_shape_3d_v27/five-views.png) |
| V26 接缝小样 | [五角度彩色静帧](raccoon-turnaround/turnaround_3d_v26/five-views.png) · [45° 和 135° 接缝放大](raccoon-turnaround/turnaround_3d_v26/junctions-2x.png) |
| V25 定稿 | [侧视原稿](raccoon-turnaround/side_back_art_v25/side-090.png) · [背视原稿](raccoon-turnaround/side_back_art_v25/back-180.png) · [游戏尺寸对照](raccoon-turnaround/side_back_art_v25/native-preview.png) |
| V14 历史草图 | [三视图审阅页](raccoon-turnaround/walk_3d_v14/shape_review/three_views.html) · [站姿参考](raccoon-turnaround/walk_3d_v14/shape_review/standing_views.png) · [动作草图](raccoon-turnaround/walk_3d_v14/shape_review/action_views.jpg) |

V28 每张灰模为 480×256 px。黄线为 V27，白线为 V28，90° 和 180° 中的青线为 V25。正面眼鼻嘴保护区的顶点、UV 和灰模像素均通过一致性核验。背面右缘相对 V25 的平均偏差从 13.7 px 降到 9.8 px。原始数据见 [灰模核验](raccoon-turnaround/head_outline_3d_v28/review-check.json) 和 [网格核验](raccoon-turnaround/head_outline_3d_v28/mesh-check.json)。原正面见 [ART21 正面原画](raccoon-turnaround/source/art21-idle-front.png)。

## 在 Windows 复现 V28

从仓库根目录运行。需要 Godot 4.6.3、Python，以及系统的 Microsoft YaHei 字体。可将命令中的 `godot` 换成本机 Godot 可执行文件路径。GLB 可直接在三维工具中打开，静站组装使用 V28 头部和 V26 身体与两条裤管。

```powershell
python -m pip install numpy opencv-python pillow
$sample = 'raccoon-turnaround/head_outline_3d_v28'
python "$sample/fit_outline.py"
godot --path $sample --script res://render_gray.gd -- --baseline
godot --path $sample --script res://render_gray.gd
godot --path $sample --script res://render_gray.gd -- --pieces
python "$sample/verify_gray.py"
```

V14 的交互审阅页可在克隆后用浏览器打开本地 HTML。历史版本中的 Python 文件也为 V28 的网格拟合提供读取和坐标工具。
