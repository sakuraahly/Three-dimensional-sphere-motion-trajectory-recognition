# legacy/ —— 早期草稿（保留备查，不影响主流程）

这两个脚本是本课题**最开始的尝试**：直接用 Canny 边缘检测看"边缘"。
它们已经被 `sphere_tracker/` 取代，但仍**可以运行**，保留的理由是：

1. 论文 / 答辩里可以做"改进前 vs 改进后"的对照；
2. 里面的**摄像头扫描、选择与真实帧率实测**思路后来被吸收进了 `sphere_tracker/sources.py`，
   留着方便追溯；
3. 随时可以回看最初的思路。

| 文件 | 原文件名 | 作用 |
|---|---|---|
| `01_canny_single_image.py` | `test.py` | 对单张图片做 Canny 边缘检测，matplotlib 并排显示原图与边缘图 |
| `02_canny_realtime_edges.py` | `video_edge_detect.py` | 实时摄像头 Canny 边缘；含摄像头扫描/选择、实测帧率、录制边缘视频 |

## 怎么运行

```powershell
conda activate myenv
cd <项目根目录>          # 注意：必须在项目根目录下运行

# 单张图片：读 assets/images/test.jpg，结果存 outputs/images/edges_result.jpg
python legacy/01_canny_single_image.py

# 实时边缘：按 q 退出，结果存 outputs/videos/output_edges.mp4
python legacy/02_canny_realtime_edges.py
```

> 这两个脚本内部用 `Path(__file__).resolve().parents[1]` 定位项目根目录，
> 所以**不管从哪里调用**都能找到输入与输出目录；但仍然建议在根目录下运行，
> 因为 `cd legacy` 之后 `python 01_canny_single_image.py` 会因为导入路径不同而行为不一致。

## 为什么它们不够用

Canny 给的是"画面里所有边缘"——桌子边、阴影、反光、背景花纹全在里面，
拿到的是一堆像素点，**而不是"球心 + 半径"**。轨迹识别需要的是后者。

要真正检测球体、标记轨迹、预测未来轨迹，请用：

```powershell
python apps/run_realtime.py --camera 0 --method fusion
```

详细对比见 `README.md` 第一节"先说你原来那套为什么不行"。
