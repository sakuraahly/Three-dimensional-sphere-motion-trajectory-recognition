import cv2

# ================= 配置区 =================
# 视频源：0 表示默认摄像头；也可以换成视频文件路径，如 "input.mp4"
VIDEO_SOURCE = 1

# 是否保存处理后的视频
SAVE_OUTPUT = True
OUTPUT_PATH = "output_edges.avi"  # 建议用 .avi，编码器兼容性好
# ==========================================

# 1. 打开视频源
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("无法打开视频源，请检查摄像头或视频文件路径")
    exit()

# 2. 获取视频参数（保存视频时需要用到）
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"视频分辨率: {width}x{height}, 帧率: {fps}")

# 如果摄像头返回的 fps 为 0，手动设一个合理值
if fps == 0:
    fps = 50.0

# 3. 创建 VideoWriter 用于保存结果
if SAVE_OUTPUT:
    # XVID 编码，保存为 .avi；若想保存 .mp4，可改用 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # 边缘图是单通道灰度图，所以 isColor=False
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height), isColor=False)

# 4. 逐帧循环处理
while True:
    ret, frame = cap.read()
    if not ret:
        print("视频读取结束或读取失败")
        break

    # ---------- 对每一帧进行处理 ----------
    # 转灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny 边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    # -------------------------------------

    # 5. 显示原视频和边缘视频
    cv2.imshow("Original", frame)
    cv2.imshow("Edges", edges)

    # 6. 保存边缘帧
    if SAVE_OUTPUT:
        out.write(edges)

    # 7. 按 q 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 8. 释放资源
cap.release()
if SAVE_OUTPUT:
    out.release()
cv2.destroyAllWindows()
print("处理完成，结果已保存。" if SAVE_OUTPUT else "处理完成。")
