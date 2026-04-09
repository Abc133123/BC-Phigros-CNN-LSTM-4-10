#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phigros AI 玩家 - 推理脚本 v4
修复: PhigrosNet 结构与训练脚本完全对齐
      去掉 predict_frames 冗余, forward 直接返回单 tensor
      IMAGE_SIZE=128 与训练一致
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import time
from collections import deque
from pathlib import Path

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("[!] 未安装 keyboard 库，无法使用 I+K+L 暂停功能")

# ===== 配置 =====
MODEL_PATH = Path(r"G:\1255\AutoPhigros2\models\best_v5.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQUENCE_LENGTH = 15
IMAGE_SIZE = 128               # ★ 与训练一致

MUMU_WINDOW_LEFT = 0
MUMU_WINDOW_TOP = 270
MUMU_WINDOW_RIGHT = 960
MUMU_WINDOW_BOTTOM = 800
MUMU_WIDTH = MUMU_WINDOW_RIGHT - MUMU_WINDOW_LEFT
MUMU_HEIGHT = MUMU_WINDOW_BOTTOM - MUMU_WINDOW_TOP

CONFIDENCE_THRESHOLD = 0.5
DEAD_ZONE = 0.02
X_OFFSET = 0.0
Y_OFFSET = 0.0


# ============================================================
# ★★★ 模型结构：与训练脚本 1:1 对齐 ★★★
# ============================================================
class PhigrosNet(nn.Module):
    def __init__(self, sequence_length=15):
        super().__init__()
        self.sequence_length = sequence_length

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6))          # 128 * 36 = 4608
        )

        self.lstm = nn.LSTM(
            input_size=128 * 36 + 5,              # 4613
            hidden_size=512,
            num_layers=2,                           # ★ 2层, 与训练一致
            dropout=0.2,
            batch_first=True
        )

        self.fc = nn.Sequential(                    # ★ Sequential FC, 与训练一致
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)                       # [is_active, x, y, is_tap, is_slide]
        )

    def forward(self, input_frames, input_gestures):
        batch_size, seq_len = input_frames.shape[:2]
        frames_flat = input_frames.reshape(batch_size * seq_len, *input_frames.shape[2:])
        cnn_features = self.cnn(frames_flat)
        cnn_features = cnn_features.reshape(batch_size, seq_len, -1)
        combined = torch.cat([cnn_features, input_gestures], dim=-1)
        lstm_out, _ = self.lstm(combined)
        return self.fc(lstm_out[:, -1])             # ★ 直接返回单 tensor, 不再套列表


# ============================================================
# PC 控制器 (不变)
# ============================================================
class PCController:
    def __init__(self):
        import mss
        import pyautogui
        self.sct = mss.mss()
        self.pyautogui = pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.001
        self.monitor = {
            'left': MUMU_WINDOW_LEFT,
            'top': MUMU_WINDOW_TOP,
            'width': MUMU_WIDTH,
            'height': MUMU_HEIGHT
        }
        print(f"[OK] 截图区域: ({MUMU_WINDOW_LEFT},{MUMU_WINDOW_TOP})->"
              f"({MUMU_WINDOW_RIGHT},{MUMU_WINDOW_BOTTOM}) {MUMU_WIDTH}x{MUMU_HEIGHT}")

    def screenshot(self):
        try:
            shot = self.sct.grab(self.monitor)
            return Image.frombytes('RGB', shot.size, shot.rgb)
        except Exception as e:
            print(f"[!] 截图失败: {e}")
            return None

    def mouse_down(self, x, y):
        self.pyautogui.mouseDown(int(MUMU_WINDOW_LEFT + x),
                                 int(MUMU_WINDOW_TOP + y))

    def mouse_up(self):
        self.pyautogui.mouseUp()

    def move_to(self, x, y):
        self.pyautogui.moveTo(int(MUMU_WINDOW_LEFT + x),
                              int(MUMU_WINDOW_TOP + y), duration=0)


# ============================================================
# AI 代理
# ============================================================
class PhigrosAgent:
    def __init__(self, model_path):
        self.device = DEVICE
        print(f"[*] 加载模型: {model_path}")

        # ★ 构建模型 (不再传 predict_frames)
        self.model = PhigrosNet(SEQUENCE_LENGTH).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                info = f"Epoch {checkpoint.get('epoch','?')}, Val Loss: {checkpoint.get('val_loss','?')}"
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                info = "格式: state_dict"
            else:
                state_dict = checkpoint
                info = "直接 dict"
        else:
            state_dict = checkpoint
            info = "直接 state_dict"

        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[OK] 模型加载成功 ({info})")

        self.controller = PCController()
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.gesture_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self._init_buffers()

        self.last_active = False
        self.paused = False
        self.pause_print_time = 0
        self.action_count = 0
        self.frame_count = 0
        self.debug_counter = 0

        print("[OK] v4: 反馈环断开 + 模型结构对齐")
        if KEYBOARD_AVAILABLE:
            try:
                keyboard.add_hotkey('i+k+l', self.toggle_pause)
                print("[OK] 热键 I+K+L 暂停/恢复")
            except Exception:
                pass

    def _init_buffers(self):
        for _ in range(SEQUENCE_LENGTH):
            self.frame_buffer.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32))
            self.gesture_buffer.append(np.zeros(5, dtype=np.float32))

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            print(f"\n[!!!] AI 已暂停")
        else:
            if self.last_active:
                self.controller.mouse_up()
                self.last_active = False
            print(f"\n[!!!] AI 已恢复")
        self.paused = self.paused  # just to be clear

    def preprocess_frame(self, frame):
        img = frame.resize((IMAGE_SIZE, IMAGE_SIZE))
        return np.array(img, dtype=np.float32) / 255.0

    def predict(self):
        frames = np.array(list(self.frame_buffer), dtype=np.float32)
        gestures = np.array(list(self.gesture_buffer), dtype=np.float32)

        # shape: (1, seq_len, 3, H, W)
        frames_tensor = torch.from_numpy(frames).unsqueeze(0).permute(0, 1, 4, 2, 3).to(self.device)
        gestures_tensor = torch.from_numpy(gestures).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(frames_tensor, gestures_tensor)

        # ★ 直接返回 (5,) 不再套列表
        return pred[0].cpu().numpy()

    def decode_action(self, pred):
        """
        v3 核心修复: gesture_buffer 永远写 idle, 断开自回归反馈环
        训练时 LSTM 吃 ground truth, 推理时不能吃自己的预测(会误差爆炸)
        """
        is_active = pred[0]
        raw_x = pred[1]
        raw_y = pred[2]
        is_tap = pred[3]
        is_slide = pred[4]

        # ★★★ 核心改动：永远只写 idle ★★★
        self.gesture_buffer.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))

        if is_active < CONFIDENCE_THRESHOLD:
            return None

        safe_x = float(np.clip(raw_x, DEAD_ZONE, 1.0 - DEAD_ZONE))
        safe_y = float(np.clip(raw_y, DEAD_ZONE, 1.0 - DEAD_ZONE))

        comp_x = float(np.clip(safe_x - X_OFFSET, DEAD_ZONE, 1.0 - DEAD_ZONE))
        comp_y = float(np.clip(safe_y - Y_OFFSET, DEAD_ZONE, 1.0 - DEAD_ZONE))

        window_x = comp_x * MUMU_WIDTH
        window_y = comp_y * MUMU_HEIGHT

        self.debug_counter += 1
        if self.debug_counter % 90 == 0:
            print(f"\n[DBG] raw=({raw_x:.3f},{raw_y:.3f}) "
                  f"screen=({window_x:.0f},{window_y:.0f}) "
                  f"conf={is_active:.2f}")

        return {
            'x': window_x,
            'y': window_y,
            'is_tap': float(np.clip(is_tap, 0, 1)) > float(np.clip(is_slide, 0, 1)),
            'confidence': is_active
        }

    def execute_action(self, action):
        if action is None:
            if self.last_active:
                self.controller.mouse_up()
                self.last_active = False
            return

        x, y = action['x'], action['y']
        if not self.last_active:
            self.controller.mouse_down(x, y)
            self.last_active = True
            self.action_count += 1
            print(f"\r[Action {self.action_count}] DOWN ({x:.0f},{y:.0f}) "
                  f"conf={action['confidence']:.2f} ", end='')
        else:
            self.controller.move_to(x, y)

    def run(self):
        print("\n" + "=" * 70)
        print("[*] Phigros AI v4 启动")
        print("=" * 70)
        print("[*] 按 Ctrl+C 停止 | I+K+L 暂停/恢复")
        print("[*] 请确保模拟器窗口在指定位置...\n")

        for i in range(3, 0, -1):
            print(f"\r[*] {i}秒后开始...", end='', flush=True)
            time.sleep(1)
        print("\n[*] 开始运行!\n")

        last_time = time.time()

        try:
            while True:
                if self.paused:
                    now = time.time()
                    if now - self.pause_print_time > 0.5:
                        print("\r[!!!] 已暂停, I+K+L 恢复...", end='', flush=True)
                        self.pause_print_time = now
                    time.sleep(0.05)
                    continue

                frame = self.controller.screenshot()
                if frame is None:
                    time.sleep(0.01)
                    continue

                self.frame_buffer.append(self.preprocess_frame(frame))
                pred = self.predict()
                action = self.decode_action(pred)
                self.execute_action(action)

                self.frame_count += 1
                now = time.time()
                if now - last_time >= 1.0:
                    fps = self.frame_count
                    self.frame_count = 0
                    last_time = now
                    print(f"\r[FPS:{fps}] Actions:{self.action_count}  ", end='')

        except KeyboardInterrupt:
            print("\n\n[*] 停止")
            if self.last_active:
                self.controller.mouse_up()
            print(f"[*] 总动作数: {self.action_count}")
            print("=" * 70)


def main():
    if not MODEL_PATH.exists():
        print(f"[!] 模型不存在: {MODEL_PATH}")
        return
    agent = PhigrosAgent(MODEL_PATH)
    agent.run()


if __name__ == "__main__":
    main()
