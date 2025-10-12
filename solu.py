"可以调整视频前后帧，"
"可以调整播放速度，"
"可以暂停播放，"
"选择区域后可识别颜色并输出指令，只要出现红色、蓝色、紫色就会输出对应指令，"

import cv2
import numpy as np
from collections import Counter

# 全局变量
paused = False
current_frame = 0
total_frames = 0
roi_points = []  # 用于手动选择区域
selecting_roi = False
current_mouse_pos = (0, 0)  # 当前鼠标位置
roi_color = None  # 存储ROI区域的颜色信息
last_color_output = ""  # 用于跟踪上次输出的颜色信息
speed_factor = 1.0  # 速度因子，1.0为正常速度
current_instruction = ""  # 当前显示的指令
last_dominant_color = ""  # 上一次检测到的主体颜色

# 颜色范围定义 (HSV格式)
color_ranges = {
    "Red": [(0, 120, 70), (10, 255, 255), (170, 120, 70), (180, 255, 255)],
    "Green": [(40, 40, 40), (80, 255, 255)],
    "Blue": [(100, 150, 0), (140, 255, 255)],
    "Yellow": [(20, 100, 100), (30, 255, 255)],
    "Purple": [(125, 50, 50), (155, 255, 255)],
    "Orange": [(10, 100, 100), (20, 255, 255)],
    "Pink": [(150, 50, 50), (170, 255, 255)]
}

# 颜色对应的指令
color_instructions = {
    "Red": "Red detected!",
    "Blue": "Blue detected!",
    "Purple": "Purple detected!"
}

# 鼠标回调函数，用于手动选择区域
def select_roi(event, x, y, flags, param):
    global roi_points, selecting_roi, current_mouse_pos, roi_color
    
    # 更新当前鼠标位置
    current_mouse_pos = (x, y)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points = [(x, y)]
        selecting_roi = True
        roi_color = None  # 重置颜色信息
        
    elif event == cv2.EVENT_LBUTTONUP:
        roi_points.append((x, y))
        selecting_roi = False
        
        # 计算选择的区域大小
        x1, y1 = roi_points[0]
        x2, y2 = roi_points[1]
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        if width > 10 and height > 10:  # 只有当区域足够大时才分析颜色
            print(f"Selected ROI: {roi_points}")
        else:
            print("Selected area is too small, please select again")
            roi_points = []

# 分析ROI区域的颜色
def analyze_roi_color(frame):
    global roi_points, roi_color, current_instruction, last_dominant_color
    
    if len(roi_points) != 2:
        return None
    
    # 确保坐标正确
    x1, y1 = roi_points[0]
    x2, y2 = roi_points[1]
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    
    # 提取ROI区域
    roi = frame[y_min:y_max, x_min:x_max]
    
    if roi.size == 0:
        return None
    
    # 转换为HSV颜色空间
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 计算平均HSV值
    avg_h = np.mean(hsv_roi[:,:,0])
    avg_s = np.mean(hsv_roi[:,:,1])
    avg_v = np.mean(hsv_roi[:,:,2])
    
    # 根据HSV值判断颜色
    detected_color = classify_color(avg_h, avg_s, avg_v)
    
    # 计算各颜色在ROI中的占比，并找到占比最大的颜色
    color_distribution = calculate_color_distribution(hsv_roi)
    
    # 找到占比最大的颜色
    dominant_color = max(color_distribution.items(), key=lambda x: x[1])[0] if color_distribution else "Unknown"
    
    # 检查是否需要更新指令
    if dominant_color in color_instructions and dominant_color != last_dominant_color:
        current_instruction = color_instructions[dominant_color]
        last_dominant_color = dominant_color
        print(f"Instruction updated: {current_instruction}")
    
    roi_color = {
        "dominant": dominant_color,  # 使用占比最大的颜色
        "distribution": color_distribution,
        "hsv": (avg_h, avg_s, avg_v)
    }
    
    return roi_color

# 根据HSV值分类颜色
def classify_color(h, s, v):
    # 根据色调(H)判断颜色
    if s < 50 or v < 50:  # 饱和度或亮度太低，可能是灰色或黑色
        if v < 30:
            return "Black"
        elif s < 30:
            return "White" if v > 200 else "Gray"
    
    if (0 <= h <= 10) or (170 <= h <= 180):
        return "Red"
    elif 11 <= h <= 25:
        return "Orange"
    elif 26 <= h <= 35:
        return "Yellow"
    elif 36 <= h <= 85:
        return "Green"
    elif 86 <= h <= 125:
        return "Blue"
    elif 126 <= h <= 155:
        return "Purple"
    elif 156 <= h <= 169:
        return "Pink"
    
    return "Unknown"

# 计算ROI中各颜色的分布
def calculate_color_distribution(hsv_roi):
    height, width = hsv_roi.shape[:2]
    total_pixels = height * width
    
    # 如果区域太大，抽样处理以提高性能
    if total_pixels > 10000:
        step = int(np.sqrt(total_pixels / 1000))
        hsv_roi = hsv_roi[::step, ::step]
        height, width = hsv_roi.shape[:2]
        total_pixels = height * width
    
    color_count = {}
    
    # 对每个像素进行分类
    for y in range(height):
        for x in range(width):
            h, s, v = hsv_roi[y, x]
            color = classify_color(h, s, v)
            color_count[color] = color_count.get(color, 0) + 1
    
    # 计算百分比
    color_distribution = {}
    for color, count in color_count.items():
        color_distribution[color] = (count / total_pixels) * 100
    
    return color_distribution

# 实时输出颜色信息到终端
def output_color_info(color_info, frame_num):
    global last_color_output
    
    if not color_info:
        return
    
    # 构建输出字符串
    output = f"Frame {frame_num}: Dominant Color: {color_info['dominant']} | "
    output += f"HSV: ({color_info['hsv'][0]:.1f}, {color_info['hsv'][1]:.1f}, {color_info['hsv'][2]:.1f}) | "
    
    # 添加主要颜色分布
    dist_str = ""
    for color, percentage in sorted(color_info['distribution'].items(), key=lambda x: x[1], reverse=True):
        if percentage > 5:  # 只显示占比超过5%的颜色
            dist_str += f"{color}: {percentage:.1f}% "
    
    output += f"Distribution: {dist_str}"
    
    # 只有当颜色信息发生变化时才输出
    if output != last_color_output:
        print(output)
        last_color_output = output

# 打开视频文件
video_path = "res/output.avi"
vc = cv2.VideoCapture(video_path)

if not vc.isOpened():
    print("Cannot open video file")
    exit()

# 获取视频信息
total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vc.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30
wait_time = int(1000 / (fps * speed_factor))  # 根据速度因子调整等待时间

print(f"Video info: total frames={total_frames}, fps={fps}")
print("Select an area by clicking and dragging the mouse")
print("Press 'q' to quit, 'r' to reset ROI, SPACE to pause/play")
print("1: backward 50 frames, 2: forward 50 frames")
print("+/-: increase/decrease speed, 0: reset speed")
print("Instructions: Red=STOP, Blue=MOVE, Purple=TURN")

# 创建窗口并设置鼠标回调
cv2.namedWindow('Video with Color Detection')
cv2.setMouseCallback('Video with Color Detection', select_roi)

# 读取第一帧
ret, frame = vc.read()
if not ret:
    print("Cannot read video frame")
    exit()

# 主循环
while True:
    if not paused:
        ret, frame = vc.read()
        if not ret:
            # 视频结束，重置到开头
            current_frame = 0
            vc.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = vc.read()
            if not ret:
                break
    
    if ret:
        current_frame = int(vc.get(cv2.CAP_PROP_POS_FRAMES))
        
        # 显示原始帧
        display_frame = frame.copy()
        
        # 如果有选择的区域，绘制矩形并分析颜色
        if len(roi_points) == 2:
            x1, y1 = roi_points[0]
            x2, y2 = roi_points[1]
            
            # 确保坐标正确
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            # 绘制矩形
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # 实时分析ROI区域的颜色
            color_info = analyze_roi_color(frame)
            
            # 在终端输出颜色信息
            output_color_info(color_info, current_frame)
            
            # 在视频上显示颜色信息
            if color_info:
                color_text = f"Color: {color_info['dominant']}"
                hsv_text = f"HSV: ({color_info['hsv'][0]:.1f}, {color_info['hsv'][1]:.1f}, {color_info['hsv'][2]:.1f})"
                
                cv2.putText(display_frame, color_text, 
                           (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, hsv_text, 
                           (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # 在另一个窗口显示ROI
                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    roi_display = roi.copy()
                    cv2.putText(roi_display, color_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(roi_display, hsv_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow('ROI Area', roi_display)
        
        # 如果正在选择区域，绘制临时矩形
        if selecting_roi and len(roi_points) == 1:
            # 使用当前鼠标位置和起始点绘制矩形
            cv2.rectangle(display_frame, roi_points[0], current_mouse_pos, (255, 0, 0), 2)
        
        # 显示指令（如果有）
        if current_instruction:
            # 在视频顶部中央显示指令
            text_size = cv2.getTextSize(current_instruction, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = (display_frame.shape[1] - text_size[0]) // 2
            text_y = 50
            
            # 添加背景框使文本更清晰
            cv2.rectangle(display_frame, 
                         (text_x - 10, text_y - text_size[1] - 10), 
                         (text_x + text_size[0] + 10, text_y + 10), 
                         (0, 0, 0), -1)
            
            # 根据指令类型设置文本颜色
            if "Red" in current_instruction:
                text_color = (0, 0, 255)  # 红色
            elif "Blue" in current_instruction:
                text_color = (255, 0, 0)  # 蓝色
            elif "Purple" in current_instruction:
                text_color = (128, 0, 128)  # 紫色
            else:
                text_color = (255, 255, 255)  # 白色
            
            cv2.putText(display_frame, current_instruction, 
                       (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
        
        # 显示帧信息和速度信息
        cv2.putText(display_frame, f"Frame: {current_frame}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Speed: {speed_factor:.1f}x", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, "Click & drag to select ROI for color analysis", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, "SPACE: pause/play, R: reset ROI, Q: quit", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, "1: backward 50 frames, 2: forward 50 frames", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, "+/-: speed, 0: reset speed", 
                   (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Video with Color Detection', display_frame)
    
    # 键盘控制
    key = cv2.waitKey(wait_time if not paused else 1)
    
    if key == -1:  # 没有按键
        pass
    elif key == ord('q') or key == ord('Q'):  # 退出
        break
    elif key == ord(' '):  # 暂停/继续
        paused = not paused
    elif key == ord('r') or key == ord('R'):  # 重置ROI
        roi_points = []
        roi_color = None
        last_color_output = ""
        current_instruction = ""
        last_dominant_color = ""
        cv2.destroyWindow('ROI Area')
        print("ROI reset")
    elif key == ord('1'):  # 后退50帧
        current_frame = max(0, current_frame - 50)
        vc.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        print(f"Backward 50 frames, current frame: {current_frame}")
    elif key == ord('2'):  # 前进50帧
        current_frame = min(total_frames - 1, current_frame + 50)
        vc.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        print(f"Forward 50 frames, current frame: {current_frame}")
    elif key == ord('+') or key == ord('='):  # 加速
        speed_factor = min(speed_factor * 1.5, 10.0)  # 最大10倍速
        wait_time = int(1000 / (fps * speed_factor))
        print(f"Speed up: {speed_factor:.1f}x")
    elif key == ord('-'):  # 减速
        speed_factor = max(speed_factor / 1.5, 0.1)  # 最小0.1倍速
        wait_time = int(1000 / (fps * speed_factor))
        print(f"Speed down: {speed_factor:.1f}x")
    elif key == ord('0'):  # 重置速度
        speed_factor = 1.0
        wait_time = int(1000 / fps)
        print("Speed reset to normal")

vc.release()
cv2.destroyAllWindows()