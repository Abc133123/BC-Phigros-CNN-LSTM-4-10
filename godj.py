#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phigros AI 玩家 - PC端推理脚本 (反馈环断开版 v3)
修复:
  v2 的所有修复
  + ★★★ v3 核心：推理时 gesture_buffer 永远填 idle，断开自回归反馈环 ★★★
    训练时 LSTM 吃的是 ground truth（teacher forcing）
    推理时如果喂模型自己的预测 → 误差累积 → hidden state 漂移 → 坐标跑飞
    所以推理时让 LSTM 只看画面序列做决策，不看自己的历史动作
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import time
import io
from collections import deque
from pathlib import Path

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("[!] 未安装 keyboard 库，无法使用 I+K+L 暂停功能。请执行: pip install keyboard")

# ===== 配置 =====
MODEL_PATH = Path(r"G:\1255\AutoPhigros2\models\best1.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQUENCE_LENGTH = 15
PREDICT_FRAMES = 3
IMAGE_SIZE = 128

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


class PhigrosNet(nn.Module):
    def __init__(self, sequence_length=15, predict_frames=3):
        super().__init__()
        self.sequence_length = sequence_length
        self.predict_frames = predict_frames
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
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.lstm = nn.LSTM(
            input_size=128*4*4 + 5,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )
        self.fc_gesture = nn.Linear(256, 5)

    def forward(self, input_frames, input_gestures):
        batch_size, seq_len = input_frames.shape[:2]
        frames_flat = input_frames.reshape(batch_size * seq_len, *input_frames.shape[2:])
        cnn_features = self.cnn(frames_flat)
        cnn_features = cnn_features.reshape(batch_size, seq_len, -1)
        combined = torch.cat([cnn_features, input_gestures], dim=-1)
        lstm_out, _ = self.lstm(combined)
        predictions = []
        for t in range(self.predict_frames):
            gesture_pred = self.fc_gesture(lstm_out[:, -1])
            predictions.append(gesture_pred)
        return predictions


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
        print(f"[✓] PC 控制器初始化成功")
        print(f" 截图区域: ({MUMU_WINDOW_LEFT}, {MUMU_WINDOW_TOP}) -> ({MUMU_WINDOW_RIGHT}, {MUMU_WINDOW_BOTTOM})")
        print(f" 窗口尺寸: {MUMU_WIDTH}x{MUMU_HEIGHT}")

    def screenshot(self):
        try:
            shot = self.sct.grab(self.monitor)
            img = Image.frombytes('RGB', shot.size, shot.rgb)
            return img
        except Exception as e:
            print(f"[!] 截图失败: {e}")
            return None

    def mouse_down(self, x, y):
        abs_x = int(MUMU_WINDOW_LEFT + x)
        abs_y = int(MUMU_WINDOW_TOP + y)
        self.pyautogui.mouseDown(abs_x, abs_y)

    def mouse_up(self):
        self.pyautogui.mouseUp()

    def move_to(self, x, y):
        abs_x = int(MUMU_WINDOW_LEFT + x)
        abs_y = int(MUMU_WINDOW_TOP + y)
        self.pyautogui.moveTo(abs_x, abs_y, duration=0)


class PhigrosAgent:
    def __init__(self, model_path):
        self.device = DEVICE

        print(f"[*] 加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = PhigrosNet(SEQUENCE_LENGTH, PREDICT_FRAMES).to(self.device)

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                epoch_info = f"Epoch {checkpoint.get('epoch', '?')}, Val Loss: {checkpoint.get('val_loss', '?')}"
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                epoch_info = f"Epoch {checkpoint.get('epoch', '?')}, Loss: {checkpoint.get('loss', '?')}"
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                epoch_info = "格式: model"
            else:
                state_dict = checkpoint
                epoch_info = "直接保存的 state_dict"
        else:
            state_dict = checkpoint
            epoch_info = "直接保存的 state_dict"

        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[✓] 模型加载成功 ({epoch_info})")

        self.controller = PCController()

        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.gesture_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self._init_buffers()

        self.last_valid_x = 0.5
        self.last_valid_y = 0.5

        self.frame_count = 0
        self.action_count = 0
        self.last_active = False

        self.paused = False
        self.pause_print_time = 0
        self.debug_counter = 0

        print("[✓] v3: 反馈环断开模式已开启 (gesture_buffer 永远填 idle)")

        if KEYBOARD_AVAILABLE:
            try:
                keyboard.add_hotkey('i+k+l', self.toggle_pause)
                print("[✓] 已注册热键：同时按下 I + K + L 可暂停/恢复 AI")
            except Exception as e:
                print(f"[!] 注册热键失败: {e}")

    def _init_buffers(self):
        dummy_frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        dummy_gesture = np.zeros(5, dtype=np.float32)
        for _ in range(SEQUENCE_LENGTH):
            self.frame_buffer.append(dummy_frame)
            self.gesture_buffer.append(dummy_gesture)
        print("[*] 缓冲区初始化完成")

    def toggle_pause(self):
        self.paused = not self.paused
        status = "暂停" if self.paused else "恢复"
        print(f"\n[!!!] AI 已{status} !!!")
        if not self.paused and self.last_active:
            self.controller.mouse_up()
            self.last_active = False

    def preprocess_frame(self, frame):
        img = frame.resize((IMAGE_SIZE, IMAGE_SIZE))
        return np.array(img, dtype=np.float32) / 255.0

    def predict(self):
        frames = np.array(list(self.frame_buffer), dtype=np.float32)
        gestures = np.array(list(self.gesture_buffer), dtype=np.float32)
        frames_tensor = torch.from_numpy(frames).unsqueeze(0).permute(0, 1, 4, 2, 3).to(self.device)
        gestures_tensor = torch.from_numpy(gestures).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predictions = self.model(frames_tensor, gestures_tensor)
        return predictions[0][0].cpu().numpy()

    # ================================================================
    # ★★★ v3 唯一改动的函数：decode_action ★★★
    # ================================================================
    def decode_action(self, pred):
        """
        ★★★ v3 核心修复：断开自回归反馈环 ★★★

        问题原因：
          训练时 LSTM 吃的是 ground truth 手势序列 (teacher forcing)
          推理时如果喂模型自己的预测 → 预测有误差 → 误差进buffer →
          下帧LSTM看到脏数据 → 预测更偏 → 误差指数爆炸 → 坐标跑飞

        修复方案：
          推理时 gesture_buffer 永远只写 idle [0, 0, 0, 0, 0]
          LSTM 纯靠画面序列来做决策，不看自己的历史动作
          这样推理时的输入分布 ≈ 训练时（训练时空闲帧就是idle）
        """
        is_active = pred[0]
        raw_x = pred[1]
        raw_y = pred[2]
        is_tap = pred[3]
        is_slide = pred[4]

        # ★★★ v3 核心改动：无论预测什么，buffer 永远写 idle ★★★
        #     断开 "模型预测 → 写回buffer → 下帧LSTM读取" 的反馈环
        idle = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.gesture_buffer.append(idle)

        if is_active < CONFIDENCE_THRESHOLD:
            return None

        # 以下完全不变：解析坐标、clip、返回 action
        safe_x = float(np.clip(raw_x, DEAD_ZONE, 1.0 - DEAD_ZONE))
        safe_y = float(np.clip(raw_y, DEAD_ZONE, 1.0 - DEAD_ZONE))

        self.last_valid_x = safe_x
        self.last_valid_y = safe_y

        c_tap = float(np.clip(is_tap, 0.0, 1.0))
        c_slide = float(np.clip(is_slide, 0.0, 1.0))

        comp_x = float(np.clip(safe_x - X_OFFSET, DEAD_ZONE, 1.0 - DEAD_ZONE))
        comp_y = float(np.clip(safe_y - Y_OFFSET, DEAD_ZONE, 1.0 - DEAD_ZONE))

        window_x = comp_x * MUMU_WIDTH
        window_y = comp_y * MUMU_HEIGHT

        # DEBUG 输出
        self.debug_counter += 1
        if self.debug_counter % 90 == 0:
            print(f"\n[DBG] raw=({raw_x:.3f},{raw_y:.3f}) "
                  f"safe=({safe_x:.3f},{safe_y:.3f}) "
                  f"scr=({window_x:.0f},{window_y:.0f}) "
                  f"conf={is_active:.2f} "
                  f"[反馈环:已断开]")

        return {
            'x': window_x,
            'y': window_y,
            'is_tap': c_tap > c_slide,
            'is_slide': c_slide > c_tap,
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
            print(f"\r[Action {self.action_count}] DOWN ({x:.0f}, {y:.0f}) conf={action['confidence']:.2f} ", end='')
        else:
            self.controller.move_to(x, y)

    def run(self):
        print("\n" + "=" * 70)
        print("[*] Phigros AI 玩家启动（v3 反馈环断开版）")
        print("=" * 70)
        print(f"[*] 反馈环断开: ON")
        print(f"[*] 偏移补偿: X={X_OFFSET}, Y={Y_OFFSET}")
        print(f"[*] 死区: {DEAD_ZONE}")
        print("[*] 按 Ctrl+C 停止")
        if KEYBOARD_AVAILABLE:
            print("[*] 同时按下 I + K + L 可暂停/恢复 AI")
        print("[*] 请确保 MuMu 模拟器窗口在指定位置，并开始游戏...")
        print()

        for i in range(3, 0, -1):
            print(f"\r[*] {i} 秒后开始...", end='', flush=True)
            time.sleep(1)
        print("\n[*] 开始运行!\n")

        last_time = time.time()
        fps = 0
        try:
            while True:
                if self.paused:
                    now = time.time()
                    if now - self.pause_print_time > 0.5:
                        print("\r[!!!] AI 已暂停，按下 I+K+L 恢复运行...", end='', flush=True)
                        self.pause_print_time = now
                    time.sleep(0.05)
                    continue

                frame = self.controller.screenshot()
                if frame is None:
                    time.sleep(0.01)
                    continue

                processed_frame = self.preprocess_frame(frame)
                self.frame_buffer.append(processed_frame)

                pred = self.predict()
                action = self.decode_action(pred)
                self.execute_action(action)

                self.frame_count += 1
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    fps = self.frame_count
                    self.frame_count = 0
                    last_time = current_time
                    print(f"\r[FPS: {fps}] Running... Actions: {self.action_count} ", end='')

        except KeyboardInterrupt:
            print("\n\n[*] 停止运行")
            print(f"[*] 总动作数: {self.action_count}")
            print("=" * 70)


def main():
    if not MODEL_PATH.exists():
        print(f"[!] 模型文件不存在: {MODEL_PATH}")
        print("[!] 请先运行训练脚本")
        return
    agent = PhigrosAgent(MODEL_PATH)
    agent.run()


if __name__ == "__main__":
    main()
