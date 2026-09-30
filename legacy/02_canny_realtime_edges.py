"""【早期草稿 · 保留备查】实时摄像头 Canny 边缘检测（原文件名 video_edge_detect.py）。

这是本课题最开始的实时尝试：扫描并选择摄像头、逐帧 Canny、可录制边缘视频。
其中"实测摄像头真实帧率"的思路后来被吸收进了 sphere_tracker/sources.py。
结论是 Canny 得不到"球心 + 半径"，因此主流程已被 apps/run_realtime.py 取代。

运行（在项目根目录下）::

    python legacy/02_canny_realtime_edges.py     # 按 q 退出
"""

from __future__ import annotations
import cv2
import sys
import time
from pathlib import Path

# ================= 配置区 =================
# 是否保存处理后的视频
SAVE_OUTPUT = True
# 【已修复】原来写的是 "../outputs/videos/output_edges.mp4"：
# 那是"项目的上一级目录"，那里并没有 outputs 文件夹，VideoWriter 会创建失败，
# 而且失败时只打印一句警告就静默跳过，很容易以为在录、其实什么都没存。
# 改成相对本文件所在的项目根目录，并在写入前自动建目录。
# 本脚本在 legacy/ 下，所以项目根目录要往上退一级
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = str(_PROJECT_ROOT / "outputs" / "videos" / "output_edges.mp4")

# 【说明】Canny 边缘图只是"所有边缘"，并不是"球"。
# 想要球心、半径、轨迹和预测，请看同目录下的 sphere_tracker/ 与 README.md。

# 扫描摄像头的最大编号（一般不会超过 5 个）
MAX_CAM_INDEX = 5
# ==========================================


def scan_cameras(max_index=MAX_CAM_INDEX, warmup_frames=3):
    """
    扫描可用的摄像头。
    返回一个列表 [(index, width, height, fps), ...]
    注意：在 Windows 上，打开摄像头后要 release，且可能需要一点延时才能再次打开。
    """
    available = []
    print("正在扫描摄像头，请稍候...")
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # Windows 上用 DSHOW 更快更稳
        if not cap.isOpened():
            cap.release()
            continue

        # 尝试读几帧，有些摄像头打开后前几帧是黑的或读不到
        ok = False
        for _ in range(warmup_frames):
            ret, frame = cap.read()
            if ret and frame is not None:
                ok = True
                break
            time.sleep(0.05)

        if ok:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            available.append((idx, w, h, fps))
            print(f"  [发现] 摄像头编号 {idx}  ->  {w}x{h} @ {fps:.1f} fps")
        else:
            print(f"  [跳过] 编号 {idx} 打开成功但读取不到画面")

        cap.release()
        time.sleep(0.1)  # 给系统一点时间释放设备

    return available


def choose_camera():
    """
    自动扫描 + 用户选择，返回选中的摄像头编号。
    如果没有任何摄像头，返回 None。
    """
    cameras = scan_cameras()

    if not cameras:
        print("\n没有检测到任何可用摄像头。")
        return None

    print("\n可用的摄像头：")
    for i, (idx, w, h, fps) in enumerate(cameras):
        print(f"  [{i}] 编号 {idx}  ({w}x{h} @ {fps:.1f} fps)")

    while True:
        choice = input(
            f"\n请输入要使用的摄像头序号 (0-{len(cameras)-1}，直接回车默认 0): "
        ).strip()
        if choice == "":
            selected = cameras[0][0]
            break
        if choice.isdigit():
            i = int(choice)
            if 0 <= i < len(cameras):
                selected = cameras[i][0]
                break
        print("输入无效，请重新输入。")

    print(f"\n已选择摄像头编号: {selected}")
    return selected


def measure_camera_fps(cap, duration=5.0, warmup=10):
    """
    测量已打开摄像头的实际帧率。

    参数:
        cap: cv2.VideoCapture 对象，必须已经打开
        duration: 测量时长，单位秒，默认 5 秒
        warmup: 预热帧数，默认先读 10 帧，让摄像头稳定

    返回:
        实际 FPS (float)
        如果失败，返回 None
    """
    if not cap.isOpened():
        print("摄像头未打开")
        return None

    # 预热：先读几帧，避免刚打开时帧率不稳定
    for _ in range(warmup):
        ret, _ = cap.read()
        if not ret:
            print("预热阶段读取失败")
            return None

    frame_count = 0
    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取帧失败")
            break

        frame_count += 1
        elapsed = time.perf_counter() - start_time

        if elapsed >= duration:
            break

    elapsed = time.perf_counter() - start_time

    if frame_count == 0 or elapsed <= 0:
        return None

    fps = frame_count / elapsed
    return fps


def measure_camera_fps_by_index(index=0, duration=5.0, warmup=10):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"无法打开摄像头 {index}")
        return None

    fps = measure_camera_fps(cap, duration=duration, warmup=warmup)
    cap.release()
    return fps


def main():
    # 1. 扫描并选择摄像头
    cam_index = choose_camera()
    if cam_index is None:
        print("程序退出。")
        sys.exit(1)

    # 2. 打开选中的摄像头
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"无法打开摄像头编号 {cam_index}，请检查设备是否被占用。")
        sys.exit(1)

    # 3. 获取视频参数（保存视频时需要用到）
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == -1:
        fps = measure_camera_fps(cap)
        if fps is None:
            fps = 30.0
        print(f"检测到摄像头未返回有效帧率，已测得实际帧率为 {fps}")
    # 如果摄像头返回的 fps 为 0，手动设一个合理值
    if fps == 0 or fps != fps:  # fps != fps 判断 NaN
        fps = 30.0
        print(f"检测到无效帧率，已自动设为 {fps}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频分辨率: {width}x{height}, 帧率: {fps}")

    # 4. 创建 VideoWriter 用于保存结果
    out = None
    if SAVE_OUTPUT:
        Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter.fourcc(*"XVID")
        # 【已修复】XVID 编码器不支持单通道灰度，用 isColor=False 写出来的
        # 多半是坏文件。这里统一按三通道写入（灰度图转成 BGR 即可）。
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height), isColor=True)
        if not out.isOpened():
            print(f"警告：无法创建输出文件 {OUTPUT_PATH}，将不会保存视频。")
            out = None

    # 5. 逐帧循环处理
    print("\n开始处理，按 q 键退出。")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频读取结束或读取失败")
            break

        # ---------- 对每一帧进行处理 ----------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        # -------------------------------------

        # 6. 显示原视频和边缘视频
        cv2.imshow("Original", frame)
        cv2.imshow("Edges", edges)

        # 7. 保存边缘帧（单通道灰度 -> 三通道，配合上面的 isColor=True）
        if out is not None:
            out.write(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

        # 8. 按 q 键退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 9. 释放资源
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print("处理完成，结果已保存。" if out is not None else "处理完成。")


if __name__ == "__main__":
    main()
