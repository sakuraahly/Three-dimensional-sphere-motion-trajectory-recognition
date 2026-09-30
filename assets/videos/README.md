# assets/videos/

把你**要处理的视频**放在这个目录，然后交给离线程序处理：

```powershell
python apps/run_video.py assets/videos/ball.mp4 --out outputs/videos/annotated.mp4 --csv outputs/traj.csv
```

- 支持的格式：OpenCV 能解码的都可以（`.mp4` / `.avi` / `.mov` / `.mkv` …），推荐 `.mp4`（H.264）。
- 这个目录里的视频**默认不进版本库**（体积太大），见 `.gitignore`；
  只有本文件会被跟踪，用来保留目录结构。
- 拍摄建议见 `README.md` 与开题报告：分辨率 1920×1080、60 fps、固定曝光与白平衡、固定机位。

> 目录里的 `assets/images/` 放的是示例图片（单图调试用），
> 例如 `assets/images/test.jpg`。
