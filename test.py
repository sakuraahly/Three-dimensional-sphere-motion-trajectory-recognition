"""Canny 边缘检测的最小示例（原始小脚本，已修好路径问题）。

注意：Canny 给的是"画面里所有边缘"，不是"球"。
      要真正识别球体、标记轨迹、预测未来轨迹，请用 sphere_tracker 包，
      见 README.md 和 apps/run_realtime.py。
"""

from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent

# 【已修复】原来直接 cv2.imread("test.jpg")：
# 1) 图片其实在 material_images/ 下，路径不对；
# 2) 读不到时 imread 返回 None，紧接着 cvtColor(None) 会直接抛异常崩掉。
IMAGE_PATH = PROJECT_ROOT / "material_images" / "test.jpg"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "images" / "edges_result.jpg"

# 另外提醒：material_images/test.jpg 的真实格式其实是 PNG（带 alpha 通道），
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
