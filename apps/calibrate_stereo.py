"""自动双目标定：把标定板在两台相机前挥一遍就行。

用法::

    # 1) 两台实时相机（推荐：能实时看到收帧进度）
    python apps/calibrate_stereo.py --left 0 --right 1 --cols 8 --rows 5 --square 25

    # 2) 两段视频（指导文档里的流程：先分别录，再拿来标定）
    python apps/calibrate_stereo.py --left-video cam1.mp4 --right-video cam2.mp4 \
        --cols 8 --rows 5 --square 25

    # 3) 两个图片文件夹（每张都是独立位姿；此时不要求"板静止"）
    python apps/calibrate_stereo.py --left-dir imgs/left --right-dir imgs/right

程序会：
  - 自动判断每一帧该不该收（清晰度、大小、贴边、曝光、位姿多样性、覆盖均匀）
  - 收够之后自动标定，并用**极线误差**等指标判定能不能用
  - 把结果存成 config/stereo_calibration.npz + .json（含人读的质量报告）

关于同步：**标定时不需要左右视频同步**。程序只收"板停稳"的帧，
静止目标在任意时刻看起来都一样，所以时间偏移不影响标定。
（真正需要同步的是后续"拍运动的小球"，那是另一回事。）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from calib import AutoPairCollector, BoardSpec, CalibConfig, StereoRig, calibrate_stereo
from calib.board import detect_board
from sphere_tracker.sources import CameraSource, LatestFrameCamera

WINDOW = "stereo calibration"
PREVIEW_W = 720          # 单路预览宽度（两路并排 + 覆盖率图，正好放得下）
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------
# 帧源：摄像头 / 两个视频 / 两个图片文件夹
# --------------------------------------------------------------------------
class PairSource:
    def __init__(self, args: argparse.Namespace) -> None:
        self.kind = ""
        self.caps: list[cv2.VideoCapture] = []
        self.files: list[Path] = []
        self.files_r: list[Path] = []
        self.cam: list[CameraSource] = []
        self.index = 0
        self.total = 0

        if args.left is not None and args.right is not None:
            self.kind = "camera"
            # 每路一个后台采集线程：主循环不再被"等下一帧"拖住，
            # 处理得慢就自动丢帧，永远处理最新画面
            self.cam = [
                LatestFrameCamera(CameraSource(index=args.left, width=args.width,
                                               height=args.height, fps=args.fps)),
                LatestFrameCamera(CameraSource(index=args.right, width=args.width,
                                               height=args.height, fps=args.fps)),
            ]
            self._primed = False
            print(f"相机 {args.left}: {self.cam[0].info.width}x{self.cam[0].info.height} @ {self.cam[0].info.fps:.1f} fps")
            print(f"相机 {args.right}: {self.cam[1].info.width}x{self.cam[1].info.height} @ {self.cam[1].info.fps:.1f} fps")
        elif args.left_video and args.right_video:
            self.kind = "video"
            self.caps = [cv2.VideoCapture(args.left_video), cv2.VideoCapture(args.right_video)]
            if not all(c.isOpened() for c in self.caps):
                raise RuntimeError("视频打不开，检查路径")
            self.total = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in self.caps)
            print(f"视频：{args.left_video} + {args.right_video}，各 {self.total} 帧")
        elif args.left_dir and args.right_dir:
            self.kind = "dir"
            ld, rd = Path(args.left_dir), Path(args.right_dir)
            self.files = sorted(p for p in ld.iterdir() if p.suffix.lower() in IMG_EXT)
            self.files_r = sorted(p for p in rd.iterdir() if p.suffix.lower() in IMG_EXT)
            if len(self.files) != len(self.files_r):
                raise RuntimeError(f"两个文件夹图片数不一致：{len(self.files)} vs {len(self.files_r)}")
            if not self.files:
                raise RuntimeError("文件夹里没有图片")
            self.total = len(self.files)
            print(f"图片：左右各 {self.total} 张（按文件名排序一一对应）")
        else:
            raise RuntimeError("请指定 --left/--right 或 --left-video/--right-video 或 --left-dir/--right-dir")

    @property
    def is_live(self) -> bool:
        return self.kind == "camera"

    def read(self):
        """返回 (ok, left_bgr, right_bgr)。"""
        if self.kind == "camera":
            if not self._primed:
                first = [c.wait_first() for c in self.cam]
                self._primed = True
                # 报告的分辨率和实际帧尺寸可能不一致（实测内置相机
                # 报告 1920x1080、实际给 1280x1080），所以按真实帧再说一次
                for i, f in enumerate(first):
                    if f is not None:
                        real = f"{f.shape[1]}x{f.shape[0]}"
                        rep = f"{self.cam[i].info.width}x{self.cam[i].info.height}"
                        if real != rep:
                            print(f"  （相机 {i} 实际输出 {real}，与报告的 {rep} 不一致，已按实际情况处理）")
            _s1, a = self.cam[0].latest()
            _s2, b = self.cam[1].latest()
            return (a is not None and b is not None), a, b
        if self.kind == "video":
            ok1, a = self.caps[0].read()
            ok2, b = self.caps[1].read()
            self.index += 1
            return (ok1 and ok2), a, b
        if self.index >= len(self.files):
            return False, None, None
        a = cv2.imread(str(self.files[self.index]))
        b = cv2.imread(str(self.files_r[self.index]))
        self.index += 1
        return (a is not None and b is not None), a, b

    def release(self) -> None:
        for c in self.cam:
            c.stop()
        for c in self.caps:
            c.release()


# --------------------------------------------------------------------------
def build_report_figure(rig: StereoRig, pairs, spec: BoardSpec, out_path: Path) -> None:
    """画一张验证图：把左右两路的角点放到同一张画布上（左半 / 右半）。

    极线误差是个抽象像素数，这张图让它变得可见：
    校正正确时，**左右对应的角点应该落在同一条水平线上**，
    所以这里把每一对对应角点用横线连起来 —— 线是水平的就说明校正对了。
    """
    if not pairs:
        return
    panels = []
    for k, pair in enumerate(pairs[:3]):
        pl, pr = rig.rectify_points(pair.corners_left, pair.corners_right)
        h, w = rig.image_size[1], rig.image_size[0]
        canvas = np.zeros((h, w * 2 + 10, 3), np.uint8)
        for y in range(0, h, 60):
            cv2.line(canvas, (0, y), (canvas.shape[1], y), (60, 60, 66), 1)
        n = min(len(pl), len(pr))
        max_dy = 0.0
        for i in range(n):
            x1, y1 = pl[i]
            x2, y2 = pr[i]
            max_dy = max(max_dy, abs(y1 - y2))
            cv2.line(canvas, (int(x1), int(y1)), (int(x2) + w + 10, int(y2)), (90, 200, 90), 1)
        for pts, off in ((pl, 0), (pr, w + 10)):
            for x, y in pts:
                cv2.circle(canvas, (int(x) + off, int(y)), 3, (0, 220, 255), -1)
        cv2.putText(canvas, f"pair {k + 1}: left | right   max |dy| = {max_dy:.3f} px",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
        panels.append(canvas)
    fig = np.vstack(panels) if len(panels) > 1 else panels[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), fig)
    print(f"验证图已保存：{out_path}（左右对应角点连成的线越水平，校正越准）")


def main() -> int:
    ap = argparse.ArgumentParser(description="自动双目标定")
    ap.add_argument("--left", type=int, default=None, help="左相机索引")
    ap.add_argument("--right", type=int, default=None, help="右相机索引")
    ap.add_argument("--left-video", default=None)
    ap.add_argument("--right-video", default=None)
    ap.add_argument("--left-dir", default=None)
    ap.add_argument("--right-dir", default=None)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--cols", type=int, default=8, help="棋盘格内角点列数")
    ap.add_argument("--rows", type=int, default=5, help="棋盘格内角点行数")
    ap.add_argument("--square", type=float, default=25.0, help="方格实际边长（毫米）")
    ap.add_argument("--target", type=int, default=25, help="目标收多少对")
    ap.add_argument("--min-pairs", type=int, default=10)
    ap.add_argument("--baseline-mm", type=float, default=None, help="预期基线（毫米），用于核对")
    ap.add_argument("--out", default="config/stereo_calibration", help="输出前缀")
    ap.add_argument("--save-pairs", default=None, help="把收下的图像对另存到这个目录（便于复查）")
    ap.add_argument("--report-png", default=None, help="输出验证图路径")
    ap.add_argument("--no-show", action="store_true", help="不弹窗（批处理）")
    ap.add_argument("--max-frames", type=int, default=0, help=">0 时最多处理这么多帧")
    args = ap.parse_args()

    spec = BoardSpec(cols=args.cols, rows=args.rows, square_mm=args.square)
    print(f"标定板：{spec.describe()}")
    print("提示：把板举到**两路都能看到**的位置，慢慢移动，每个姿势停一下（程序只收停稳的帧）")
    print()

    src = PairSource(args)
    from calib import CaptureConfig
    cap_cfg = CaptureConfig(
        target_pairs=args.target,
        min_pairs=args.min_pairs,
        # 独立静帧场景不要求"板静止"（帧间本来就是不同位姿）
        require_stationary=(src.kind != "dir"),
    )
    collector = AutoPairCollector(spec, cap_cfg)

    cov_cache = None
    cov_cache_n = -1
    det_ms = 0.0
    t_prev = time.perf_counter()
    fps_frames = 0
    pairs_dir: Path | None = Path(args.save_pairs) if args.save_pairs else None
    if pairs_dir:
        (pairs_dir / "left").mkdir(parents=True, exist_ok=True)
        (pairs_dir / "right").mkdir(parents=True, exist_ok=True)

    frames = 0
    try:
        while True:
            ok, left, right = src.read()
            if not ok or left is None or right is None:
                if src.is_live:
                    continue
                break
            frames += 1
            if args.max_frames and frames > args.max_frames:
                break

            t0 = time.perf_counter()
            gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            accepted, reason = collector.offer(gl, gr)
            det_ms = 0.9 * det_ms + 0.1 * (time.perf_counter() - t0) * 1000.0
            fps_frames += 1

            if accepted and pairs_dir:
                i = collector.count
                cv2.imwrite(str(pairs_dir / "left" / f"{i:03d}.png"), left)
                cv2.imwrite(str(pairs_dir / "right" / f"{i:03d}.png"), right)

            if not args.no_show:
                # 直接复用粗检结果画框 —— 之前这里又检测了一遍，
                # 每帧 4 次全图精确检测（约 1.2 秒）就是卡顿的根源。
                #
                # 注意两路分辨率可能不同（实测外置 1280x720、内置实际给 1280x1080），
                # 所以每路按自己的宽度独立缩放，再补齐到同一高度 ——
                # 按"统一宽度"缩放会因为宽高比不同而得到不同高度，hstack 直接报错。
                def _prep(img):
                    s = PREVIEW_W / float(img.shape[1])
                    out = cv2.resize(img, (PREVIEW_W, max(1, int(round(img.shape[0] * s)))),
                                     interpolation=cv2.INTER_LINEAR)
                    return out, s

                a, scale_l = _prep(left)
                b, scale_r = _prep(right)
                hh = max(a.shape[0], b.shape[0])
                if a.shape[0] != hh:
                    a = np.vstack([a, np.zeros((hh - a.shape[0], a.shape[1], 3), np.uint8)])
                if b.shape[0] != hh:
                    b = np.vstack([b, np.zeros((hh - b.shape[0], b.shape[1], 3), np.uint8)])
                for img, obs, sc in ((a, collector.last_obs[0], scale_l),
                                     (b, collector.last_obs[1], scale_r)):
                    if obs is None or not obs.found or obs.corners is None:
                        continue
                    colour = (0, 220, 0) if collector.last_accepted else (0, 140, 255)
                    quad = obs.quad()
                    if quad is not None:
                        cv2.polylines(img, [(quad * sc).astype(np.int32)], True, colour, 2)
                    pts = obs.corners.reshape(-1, 2) * sc
                    for x, y in pts:
                        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 255), -1)
                top = np.hstack([a, b])

                # 覆盖率图只在收帧数变化时重画（原来每帧都重建一次画布）
                if collector.count != cov_cache_n:
                    if collector.image_size:
                        # 缩放到约 200 px 高，宽度按原比例跟着走
                        cov_scale = 200.0 / collector.image_size[1]
                    else:
                        cov_scale = 0.2
                    cov_cache = collector.coverage_image(scale=cov_scale)
                    cov_cache_n = collector.count
                cov = cov_cache
                if cov is None:
                    cov = np.zeros((top.shape[0], 180, 3), np.uint8)
                if cov.shape[0] < top.shape[0]:
                    cov = np.vstack([cov, np.zeros((top.shape[0] - cov.shape[0], cov.shape[1], 3), np.uint8)])
                panel = np.hstack([top, cov[: top.shape[0]]])

                lines = collector.status_lines()
                lines.append(f"detect {det_ms:.0f} ms/frame  ({1000.0 / max(det_ms, 1e-3):.0f} fps)")
                for i, line in enumerate(lines):
                    cv2.putText(panel, line, (8, panel.shape[0] - 78 + i * 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(WINDOW, panel)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    collector.reset()
                    print("已清空重新收集")

            if collector.full:
                print(f"已收满 {collector.count} 对，开始标定…")
                break
            if src.is_live and frames % 60 == 0:
                now = time.perf_counter()
                fps = fps_frames / max(now - t_prev, 1e-6)
                t_prev = now
                fps_frames = 0
                print(f"  已收 {collector.count}/{args.target}  {fps:5.1f} fps  "
                      f"检测 {det_ms:.0f} ms/帧  ({collector.last_reason})")
    finally:
        src.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    print()
    print(f"共处理 {frames} 帧，收下 {collector.count} 对")
    if collector.rejected:
        print("剔除原因统计：")
        for r, n in sorted(collector.rejected.items(), key=lambda kv: -kv[1]):
            print(f"  {n:5d}  {r}")

    if not collector.enough:
        print(f"\n[失败] 只收到 {collector.count} 对，少于 {args.min_pairs} 对，无法标定。")
        print("建议：让板更大更清晰、移动慢一点、每个姿势停稳 1 秒、覆盖画面各个区域。")
        return 1

    objp = spec.object_points()
    ip1 = [p.corners_left for p in collector.pairs]
    ip2 = [p.corners_right for p in collector.pairs]
    img_size = collector.image_size

    cfg = CalibConfig(expected_baseline_mm=args.baseline_mm)
    res = calibrate_stereo(objp, ip1, ip2, img_size, cfg, spec)
    print(res.report())

    if args.report_png:
        try:
            rig_tmp = StereoRig.from_result(res)
            build_report_figure(rig_tmp, collector.pairs, spec, Path(args.report_png))
        except ValueError:
            pass

    if not res.ok:
        print("标定未通过质量判定，未保存结果。请按上面的提示重拍。")
        return 1

    rig = StereoRig.from_result(res, meta={
        "board": spec.describe(),
        "source": src.kind,
        "left": args.left if args.left is not None else (args.left_video or args.left_dir),
        "right": args.right if args.right is not None else (args.right_video or args.right_dir),
        "image_size": list(img_size),
    })
    npz, js = rig.save(args.out)
    print(f"已保存：{npz}")
    print(f"        {js}")
    print()
    print("下一步：")
    print("  - 三角测量时用 calib.StereoRig.load('config/stereo_calibration') 读取参数")
    print("  - 装置如果被碰动，用 apps/verify_stereo.py 复检极线误差")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
