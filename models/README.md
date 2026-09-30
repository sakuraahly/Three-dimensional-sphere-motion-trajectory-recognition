# models/ —— 推理权重目录

权重都是**可下载的二进制**，不进版本库（见 `.gitignore` 的 `*.pt` / `*.onnx`）。

| 文件 | 用在哪 | 获取方式 |
|---|---|---|
| `yolo11n.pt` | `--method yolo`（COCO 零样本检测，含 sports ball） | 见 `README.md` 第六节：国内代理下载命令 |
| `yolo26n-seg.pt` / `yoloe-26n-seg.pt` | 需要实例掩膜或开放词汇检测时 | 同上，改文件名即可 |
| `sam2.1_t.pt` | `--method yolo_sam`（像素级精细边缘） | 首次运行时 ultralytics 自动下载 |

代码里 `DetectorConfig.yolo_weights` 默认就是 `models/yolo11n.pt`；
如果该文件不存在，会自动退回裸文件名，由 ultralytics 自己去找或下载。
