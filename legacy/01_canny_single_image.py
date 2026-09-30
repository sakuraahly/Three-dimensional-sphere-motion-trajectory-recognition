"""【早期草稿 · 保留备查】对单张图片做 Canny 边缘检测。

这是本课题最开始的尝试（原文件名 test.py）：用 Canny 看"边缘"。
结论是 Canny 给的是"画面里所有边缘"，不是"球"，因此已被 sphere_tracker/ 取代。
保留它是为了在论文/答辩里做"改进前 vs 改进后"的对照，它本身仍可运行。

运行（在项目根目录下）::

    python legacy/01_canny_single_image.py

读取 assets/images/test.jpg，结果存到 outputs/images/edges_result.jpg。
主程序见 apps/run_realtime.py，说明见 README.md 第一节。
"""

from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 本脚本在 legacy/ 下，所以项目根目录要往上退一级
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 【已修复】原来直接 cv2.imread("test.jpg")：路径不对，而且读不到时
# imread 返回 None，紧接着 cvtColor(None) 会直接抛异常崩掉。
IMAGE_PATH = PROJECT_ROOT / "assets" / "images" / "test.jpg"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "images" / "edges_result.jpg"

# 另外提醒：assets/images/test.jpg 的真实格式其实是 PNG（带 alpha 通道），
# 只是扩展名写成了 .jpg。OpenCV 能正常读，但建议改名成 .png 以免别的工具报错。


def main() -> int:
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        print(f"[错误] 读不到图片：{IMAGE_PATH}")
        print("       请确认文件存在，或把 IMAGE_PATH 改成你自己的图片。")
        return 1
    print(f"已读取：{IMAGE_PATH}  尺寸 {image.shape[1]}x{image.shape[0]}")

    # 含 alpha 通道的 PNG 要转成 BGR 才能送进 Canny
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap="gray")
    plt.title("Canny Edges")
    plt.axis("off")
    plt.tight_layout()

    # 【已修复】原来存到当前工作目录，和项目 outputs/ 的约定不一致
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), edges)
    print(f"边缘检测结果已保存：{OUTPUT_PATH}")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
