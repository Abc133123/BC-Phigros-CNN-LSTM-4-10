#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phigros 终极数据采集器 v5
- winAPI 鼠标操作 (SetCursorPos + mouse_event)
- 单指采集 (Phigros 只需单指)
- timeline 兼容 main.py (gesture key, 单对象/null)
- 不依赖 pyautogui

坐标映射:
  norm_x = 1 - (phone_y / PHONE_MAX_Y)    手机Y翻转
  norm_y = phone_x / PHONE_MAX_X            手机X映射
"""

import subprocess
import threading
import time
import json
import re
import queue
from pathlib import Path
import io
import traceback
import ctypes

# ===== 配置 =====
PHONE_ADB = r"G:\AZIDE\platform-tools\adb.exe"
PHONE_DEVICE = "ADB读取设备ID放这"
PHONE_MIN_X = 0
PHONE_MAX_X = 10527
PHONE_MIN_Y = 0
PHONE_MAX_Y = 22655
MUMU_WINDOW_LEFT = 0
MUMU_WINDOW_TOP = 270
MUMU_WINDOW_RIGHT = 960
MUMU_WINDOW_BOTTOM = 800
TARGET_FPS = 20
JPEG_QUALITY = 65


# ============================
# winAPI 鼠标
# ============================

class WinMouse:
    """
    Windows API 鼠标控制
    SetCursorPos: 定位 (像素坐标)
    mouse_event:  按键 (LEFTDOWN/LEFTUP)
    比 pyautogui 快 10 倍, 无内置延迟
    """
    LEFTDOWN = 0x0002
    LEFTUP   = 0x0004

    @staticmethod
    def down(x, y):
        ctypes.windll.user32.SetCursorPos(int(x), int(y))
        ctypes.windll.user32.mouse_event(WinMouse.LEFTDOWN, 0, 0, 0, 0)

    @staticmethod
    def up():
        ctypes.windll.user32.mouse_event(WinMouse.LEFTUP, 0, 0, 0, 0)

    @staticmethod
    def move(x, y):
        ctypes.windll.user32.SetCursorPos(int(x), int(y))


# ============================
# 采集器
# ============================

class PhigrosCollector:

    def __init__(self):
        self.mumu_w = MUMU_WINDOW_RIGHT - MUMU_WINDOW_LEFT
        self.mumu_h = MUMU_WINDOW_BOTTOM - MUMU_WINDOW_TOP
        self.output_dir = Path(r"C:\Users\Administrator\Desktop\AIC3103\AutoPhigros2\phigros_data")
        self.output_dir.mkdir(exist_ok=True)
        self.recording = False
        self._stop = False
        self.start_time = None
        self.gestures = []
        self.session_dir = None
        self.frames_dir = None
        self.frame_queue = queue.Queue(maxsize=120)
        self.click_queue = queue.Queue(maxsize=200)
        self.capture_times = []
        self.click_times = []
        self.sct = None
        self.monitor = None

    # ---------- 初始化 ----------

    def _init_capture(self):
        try:
            import mss
            self.sct = mss.mss()
            self.monitor = {
                'left': MUMU_WINDOW_LEFT, 'top': MUMU_WINDOW_TOP,
                'width': self.mumu_w, 'height': self.mumu_h
            }
            print(f"[*] mss 初始化成功")
            print(f"    区域: ({MUMU_WINDOW_LEFT},{MUMU_WINDOW_TOP}) -> ({MUMU_WINDOW_RIGHT},{MUMU_WINDOW_BOTTOM})")
            print(f"    尺寸: {self.mumu_w}x{self.mumu_h}")
            return True
        except Exception as e:
            print(f"[!] mss 初始化失败: {e}")
            traceback.print_exc()
            return False

    def _capture(self):
        try:
            from PIL import Image
            shot = self.sct.grab(self.monitor)
            img = Image.frombytes('RGB', shot.size, shot.rgb)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=JPEG_QUALITY, optimize=True)
            return buf.getvalue()
        except:
            return None

    # ---------- 坐标 ----------

    def _to_norm(self, px, py):
        nx = 1.0 - (py / PHONE_MAX_Y)
        ny = px / PHONE_MAX_X
        return max(0, min(1, nx)), max(0, min(1, ny))

    def _to_screen(self, nx, ny):
        return int(MUMU_WINDOW_LEFT + nx * self.mumu_w), int(MUMU_WINDOW_TOP + ny * self.mumu_h)

    # ---------- 录制 ----------

    def record(self):
        if self.recording:
            print("[!] 已在录制中")
            return
        if not self._init_capture():
            return

        n = len(list(self.output_dir.glob('session_*')))
        self.session_dir = self.output_dir / f"session_{n:03d}"
        self.frames_dir = self.session_dir / "frames"
        self.session_dir.mkdir(exist_ok=True)
        self.frames_dir.mkdir(exist_ok=True)

        self.recording = True
        self._stop = False
        self.start_time = time.time()
        self.gestures = []
        self.capture_times = []
        self.click_times = []
        for q in (self.frame_queue, self.click_queue):
            while not q.empty():
                try: q.get_nowait()
                except: pass

        print()
        print("=" * 60)
        print("[*] Phigros 终极采集器 v5")
        print("[*] winAPI 鼠标 | 单指 | main.py 兼容")
        print("=" * 60)
        print(f"  目录: {self.session_dir}")
        print(f"  手机: {PHONE_DEVICE}")
        print()
        print("  手机触摸 -> MuMu 模拟点击")
        print("  Ctrl+C 停止")
        print()

        threading.Thread(target=self._save_loop, daemon=True).start()
        threading.Thread(target=self._click_loop, daemon=True).start()
        threading.Thread(target=self._touch_loop, daemon=True).start()
        self._capture_loop()

        self.recording = False
        self._stop = True
        print("\n[*] 刷新队列...")
        time.sleep(1.5)
        self._save_data()

    def stop(self):
        self._stop = True

    # ---------- 截图 (主线程) ----------

    def _capture_loop(self):
        fid = 0
        interval = 1.0 / TARGET_FPS
        while not self._stop:
            t0 = time.time()
            try:
                data = self._capture()
                dt = time.time() - t0
                self.capture_times.append(dt)
                if data and len(data) > 100:
                    ts = time.time() - self.start_time
                    try:
                        self.frame_queue.put({'id': fid, 'ts': ts, 'data': data}, timeout=0.1)
                        fid += 1
                        fps = fid / ts if ts > 0 else 0
                        avg = sum(self.capture_times[-20:]) / min(20, len(self.capture_times)) * 1000
                        print(f"\r[CAP] 帧:{fid:4d} 时间:{ts:6.1f}s "
                              f"FPS:{fps:5.1f} 截图:{avg:4.0f}ms 手势:{len(self.gestures):3d}",
                              end='', flush=True)
                    except queue.Full:
                        pass
            except:
                pass
            sleep = interval - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    # ---------- 触摸监听 ----------

    def _touch_loop(self):
        print("  [触摸] 启动...")
        cmd = [PHONE_ADB, '-s', PHONE_DEVICE, 'shell', 'getevent', '-lt']
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, bufsize=1)
            print("  [触摸] 已启动")
        except Exception as e:
            print(f"  [触摸] 启动失败: {e}")
            return

        cx = cy = None
        g_start = None
        g_sx = g_sy = None
        moves = []
        last_mv = 0

        while not self._stop:
            try:
                line = proc.stdout.readline()
                if not line:
                    break
                ts = time.time() - self.start_time

                if 'ABS_MT_POSITION_X' in line:
                    m = re.search(r'([0-9a-fA-F]{4,})\s*$', line)
                    if m:
                        cx = int(m.group(1), 16)

                elif 'ABS_MT_POSITION_Y' in line:
                    m = re.search(r'([0-9a-fA-F]{4,})\s*$', line)
                    if m:
                        cy = int(m.group(1), 16)

                elif 'BTN_TOUCH' in line and 'DOWN' in line:
                    if cx is not None and cy is not None:
                        g_start = ts
                        g_sx, g_sy = cx, cy
                        moves = []
                        last_mv = 0
                        nx, ny = self._to_norm(cx, cy)
                        sx, sy = self._to_screen(nx, ny)
                        print(f"\n  [DOWN] {ts:.3f}s | ({cx},{cy})->({sx},{sy})")
                        self.click_queue.put({'t': 'down', 'x': sx, 'y': sy})

                elif 'BTN_TOUCH' in line and 'UP' in line:
                    if g_start is not None and g_sx is not None:
                        dur = ts - g_start
                        snx, sny = self._to_norm(g_sx, g_sy)
                        enx, eny = self._to_norm(cx, cy)
                        ssx, ssy = self._to_screen(snx, sny)
                        esx, esy = self._to_screen(enx, eny)
                        dist = ((enx - snx)**2 + (eny - sny)**2) ** 0.5
                        gtype = 'slide' if dist > 0.05 else ('long_press' if dur > 0.15 else 'tap')
                        gid = len(self.gestures)
                        self.gestures.append({
                            'gesture_id': gid,
                            'start_time': round(g_start, 3),
                            'end_time': round(ts, 3),
                            'duration_ms': round(dur * 1000, 1),
                            'type': gtype,
                            'phone_start': {'x': g_sx, 'y': g_sy},
                            'phone_end': {'x': cx, 'y': cy},
                            'normalized_start': {'x': round(snx, 4), 'y': round(sny, 4)},
                            'normalized_end': {'x': round(enx, 4), 'y': round(eny, 4)},
                            'screen_command': {
                                'start': {'x': ssx, 'y': ssy},
                                'end': {'x': esx, 'y': esy}
                            },
                            'distance': round(dist, 4),
                            'move_positions': moves.copy(),
                            'is_primary': True
                        })
                        self.click_queue.put({'t': 'up'})
                        print(f"  [UP] #{gid}: {gtype} | {dur*1000:.0f}ms")
                        g_start = g_sx = g_sy = None
                        moves = []

                elif 'SYN_REPORT' in line or 'SYN_MT_REPORT' in line:
                    if g_start is not None and cx is not None and cy is not None and ts - last_mv > 0.03:
                        last_mv = ts
                        nx, ny = self._to_norm(cx, cy)
                        sx, sy = self._to_screen(nx, ny)
                        moves.append({
                            'time': round(ts, 3),
                            'phone': {'x': cx, 'y': cy},
                            'normalized': {'x': round(nx, 4), 'y': round(ny, 4)},
                            'screen': {'x': sx, 'y': sy}
                        })
                        self.click_queue.put({'t': 'move', 'x': sx, 'y': sy})

            except Exception as e:
                if not self._stop:
                    print(f"\n  [ERR] {e}")
                break

        proc.terminate()
        print("  [触摸] 已停止")

    # ---------- winAPI 点击执行 ----------

    def _click_loop(self):
        print("  [点击] 启动 (winAPI)")
        while not self._stop or not self.click_queue.empty():
            try:
                ev = self.click_queue.get(timeout=0.1)
                t0 = time.time()
                if ev['t'] == 'down':
                    WinMouse.down(ev['x'], ev['y'])
                elif ev['t'] == 'up':
                    WinMouse.up()
                elif ev['t'] == 'move':
                    WinMouse.move(ev['x'], ev['y'])
                self.click_times.append((time.time() - t0) * 1000)
                self.click_queue.task_done()
            except queue.Empty:
                continue
            except:
                pass
        print("  [点击] 已停止")

    # ---------- 保存线程 ----------

    def _save_loop(self):
        print("  [保存] 启动")
        count = 0
        while not self._stop or not self.frame_queue.empty():
            try:
                item = self.frame_queue.get(timeout=0.5)
                fname = f"{item['id']:06d}_{item['ts']:.3f}.jpg"
                try:
                    with open(self.frames_dir / fname, 'wb') as f:
                        f.write(item['data'])
                    count += 1
                except Exception as e:
                    print(f"\n  [ERR] 保存失败: {e}")
                self.frame_queue.task_done()
            except queue.Empty:
                continue
        print(f"  [保存] 结束, 共 {count} 帧")

    # ---------- 数据保存 ----------

    def _save_data(self):
        if not self.session_dir:
            return
        dur = time.time() - self.start_time
        files = sorted(self.frames_dir.glob('*.jpg'))
        print(f"[*] 找到 {len(files)} 张图片")

        frames = []
        for f in files:
            try:
                p = f.stem.split('_')
                if len(p) == 2:
                    frames.append({'frame_id': int(p[0]), 'timestamp': float(p[1]), 'filename': f.name})
            except:
                pass

        mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
        avg_cap = sum(self.capture_times) / len(self.capture_times) * 1000 if self.capture_times else 0
        avg_clk = sum(self.click_times) / len(self.click_times) if self.click_times else 0
        taps = sum(1 for g in self.gestures if g['type'] == 'tap')
        lps = sum(1 for g in self.gestures if g['type'] == 'long_press')
        sld = sum(1 for g in self.gestures if g['type'] == 'slide')

        data = {
            'source': 'phigros_ultimate_winapi',
            'config': {
                'phone_device': PHONE_DEVICE,
                'phone_coordinate_range': {'x': [PHONE_MIN_X, PHONE_MAX_X], 'y': [PHONE_MIN_Y, PHONE_MAX_Y]},
                'coordinate_transform': {'formula': {'norm_x': '1 - phone_y_norm', 'norm_y': 'phone_x_norm'}},
                'mumu_window_area': {
                    'left': MUMU_WINDOW_LEFT, 'top': MUMU_WINDOW_TOP,
                    'right': MUMU_WINDOW_RIGHT, 'bottom': MUMU_WINDOW_BOTTOM,
                    'width': self.mumu_w, 'height': self.mumu_h
                },
                'capture_settings': {'target_fps': TARGET_FPS, 'jpeg_quality': JPEG_QUALITY},
                'mouse_backend': 'winAPI (ctypes SetCursorPos + mouse_event)'
            },
            'summary': {
                'duration_sec': round(dur, 2),
                'frames': len(frames),
                'gestures': len(self.gestures),
                'gesture_stats': {'tap': taps, 'long_press': lps, 'slide': sld},
                'storage_mb': round(mb, 2),
                'fps_avg': round(len(frames) / dur, 2) if dur > 0 else 0,
                'performance': {'avg_capture_ms': round(avg_cap, 1), 'avg_click_exec_ms': round(avg_clk, 1)}
            },
            'frame_data': frames,
            'gesture_data': self.gestures,
            'timeline': self._build_timeline(frames)
        }

        with open(self.session_dir / "data.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print()
        print("=" * 60)
        print("[OK] 保存完成")
        print("=" * 60)
        print(f"  时长: {int(dur//60)}:{int(dur%60):02d}")
        print(f"  帧数: {len(frames)} | FPS: {data['summary']['fps_avg']}")
        print(f"  手势: {len(self.gestures)} (tap:{taps} long:{lps} slide:{sld})")
        print(f"  存储: {mb:.2f} MB")
        print(f"  截图: {avg_cap:.1f}ms/帧 | 点击: {avg_clk:.1f}ms/次")
        print(f"  目录: {self.session_dir}")
        print("=" * 60)

    def _build_timeline(self, frames):
        """timeline 兼容 main.py: gesture 为单对象或 null"""
        tl = []
        for frame in frames:
            ft = frame['timestamp']
            gesture = None
            for g in self.gestures:
                if g['start_time'] <= ft <= (g['end_time'] or ft):
                    pos = None
                    for p in g['move_positions']:
                        if p['time'] <= ft:
                            pos = p
                    gesture = {
                        'gesture_id': g['gesture_id'],
                        'type': g['type'],
                        'normalized': g['normalized_start'],
                        'current_position': pos
                    }
                    break
            tl.append({'frame': frame['filename'], 'time': ft, 'gesture': gesture})
        return tl


# ============================
# 入口
# ============================

_collector = None

def _sig(sig, frame):
    global _collector
    print("\n\n[*] Ctrl+C 停止...")
    if _collector:
        _collector.stop()

def main():
    global _collector
    import signal
    signal.signal(signal.SIGINT, _sig)
    print()
    print("=" * 60)
    print("[*] Phigros 终极采集器 v5")
    print("[*] winAPI 鼠标 | 单指 | main.py 兼容")
    print("=" * 60)
    _collector = PhigrosCollector()
    print(f"\n手机: {PHONE_DEVICE}")
    print(f"窗口: ({MUMU_WINDOW_LEFT},{MUMU_WINDOW_TOP}) -> ({MUMU_WINDOW_RIGHT},{MUMU_WINDOW_BOTTOM})")
    print(f"截图: {_collector.mumu_w}x{_collector.mumu_h} @ {TARGET_FPS}FPS")
    print()
    print("命令: start | test | quit")
    while True:
        try:
            cmd = input("\n>>> ").strip().lower()
            if cmd == 'start':
                _collector.record()
            elif cmd == 'test':
                print("[test] 3秒后测试鼠标点击 MuMu 窗口中央...")
                time.sleep(3)
                cx = (MUMU_WINDOW_LEFT + MUMU_WINDOW_RIGHT) // 2
                cy = (MUMU_WINDOW_TOP + MUMU_WINDOW_BOTTOM) // 2
                WinMouse.down(cx, cy)
                time.sleep(0.2)
                WinMouse.up()
                print(f"[test] 已点击 ({cx},{cy})")
            elif cmd in ['quit', 'q', 'exit']:
                break
            else:
                print("命令: start | test | quit")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
