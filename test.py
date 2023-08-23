import pyrealsense2 as rs
import numpy as np
import cv2

# 创建一个管道
pipeline = rs.pipeline()

# 创建一个配置对象，用于配置管道
config = rs.config()

# 配置管道以产生彩色流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 开始流
pipeline.start(config)

try:
    while True:
        # 获取彩色帧
        color_frame = pipeline.wait_for_frames().get_color_frame()

        # 检查彩色帧是否可用
        if not color_frame:
            continue

        # 将彩色帧转化为 numpy 数组以便显示
        color_image = np.asanyarray(color_frame.get_data())

        # 使用 OpenCV 显示彩色帧
        cv2.imshow('Color Frame', color_image)

        # 使用 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 停止管道
    pipeline.stop()