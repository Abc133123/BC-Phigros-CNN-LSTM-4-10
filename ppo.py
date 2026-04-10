#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phigros BC->PPO 在线微调
- 全网络反向传播, 分层学习率
- 真实 gesture 输入 (和BC训练分布一致)
- 音频上升沿检测, 命中打印在终端
- PPO更新预建序列 + 进度条
- 最有价值的一集
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from PIL import Image
import mss
import ctypes
import time
from collections import deque
from pathlib import Path
import scipy.signal as signal

try:
    import keyboard; KEYBOARD_AVAILABLE = True
except:
    KEYBOARD_AVAILABLE = False

try:
    import pyaudio; PYAUDIO_AVAILABLE = True
except:
    PYAUDIO_AVAILABLE = False

# ==================== 配置 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MUMU_LEFT, MUMU_TOP = 0, 270
MUMU_W, MUMU_H = 960, 530
IMAGE_SIZE = 128
SEQUENCE_LENGTH = 15

BC_MODEL_PATH = Path(r"G:\1255\AutoPhigros2\models\best_v5.pth")
PPO_SAVE_DIR = Path(r"G:\1255\AutoPhigros2\ppo_models")
PPO_SAVE_DIR.mkdir(exist_ok=True)

CONFIDENCE_THRESHOLD = 0.5
DEAD_ZONE = 0.02
AUDIO_THRESHOLD = 0.04

# PPO
GAMMA = 0.99
LAMBDA_GAE = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.015
VALUE_COEF = 0.5
GRAD_CLIP = 0.5
UPDATE_EVERY = 512
MINI_BATCH_SIZE = 64
EPOCHS_PER_UPDATE = 3

# 判定窗口
MATCH_WINDOW = 5
TIMEOUT_FRAMES = 8


# ==================== PPO 模型 ====================
class PPOAgent(nn.Module):
    def __init__(self, seq_len=15):
        super().__init__()
        self.seq_len = seq_len
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.lstm = nn.LSTM(
            128 * 36 + 5, 512,
            num_layers=2, dropout=0.2, batch_first=True
        )
        self.actor = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, 5)
        )
        self.critic = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        self.log_std = nn.Parameter(torch.tensor([0.0, -1.9, -1.9]))

    def forward(self, frames, gestures):
        bs, sl = frames.shape[:2]
        ff = frames.reshape(bs * sl, *frames.shape[2:])
        c = self.cnn(ff).reshape(bs, sl, -1)
        x, _ = self.lstm(torch.cat([c, gestures], dim=-1))
        features = x[:, -1]
        return self.actor(features), self.critic(features), features

    def std(self):
        return torch.exp(self.log_std.clamp(min=-3.0, max=1.0))

    def load_from_bc(self, path, device):
        ck = torch.load(path, map_location=device, weights_only=False)
        sd = (ck.get('model_state_dict', ck.get('state_dict', ck))
              if isinstance(ck, dict) else ck)
        mapping = {}
        for k, v in sd.items():
            nk = 'actor.' + k[3:] if k.startswith('fc.') else k
            mapping[nk] = v
        self.load_state_dict(mapping, strict=False)
        print("[OK] BC权重 -> PPO (backbone+actor从BC, critic随机)")


# ==================== 音频监听 ====================
class AudioListener:
    def __init__(self):
        self.hit_flag = False
        self.current_rms = 0.0
        self.peak_rms = 0.0
        self._above = False
        self.b, self.a = signal.butter(4, [2000, 10000], btype='band', fs=48000)
        self.zi = signal.lfilter_zi(self.b, self.a) * 0
        self.p = self.stream = None

    def _cb(self, in_data, fc, ti, st):
        a = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        f, self.zi = signal.lfilter(self.b, self.a, a, zi=self.zi)
        r = np.sqrt(np.mean(f ** 2))
        self.current_rms = r
        if r > self.peak_rms:
            self.peak_rms = r
        na = r > AUDIO_THRESHOLD
        self.hit_flag = na and not self._above
        self._above = na
        return (in_data, pyaudio.paContinue)

    def start(self):
        if not PYAUDIO_AVAILABLE:
            print("[!] 未安装 pyaudio"); return
        self.p = pyaudio.PyAudio()
        dev = None
        for i in range(self.p.get_device_count()):
            d = self.p.get_device_info_by_index(i)
            if d['maxInputChannels'] > 0 and any(
                    k in d['name'] for k in ['CABLE', 'Virtual', '立体声']):
                dev = i
                print(f"[*] 音频设备: {d['name']}")
                break
        if dev is None:
            print("[!] 找不到 VB-Cable 设备"); return
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16, channels=1, rate=48000,
                input=True, input_device_index=dev,
                stream_callback=self._cb, frames_per_buffer=1024)
            self.stream.start_stream()
            print("[OK] 音频监听启动")
        except Exception as e:
            print(f"[!] 音频失败: {e}")

    def consume_hit(self):
        hit = self.hit_flag
        self.hit_flag = False
        return hit

    def stop(self):
        if self.stream:
            self.stream.stop_stream(); self.stream.close()
        if self.p:
            self.p.terminate()


# ==================== 进度条 ====================
class PPOProgress:
    """简易终端进度条"""
    def __init__(self, total):
        self.total = total
        self.done = 0
        self.t0 = time.time()
        self.bar_w = 40

    def step(self, msg=""):
        self.done += 1
        pct = self.done / self.total
        filled = int(self.bar_w * pct)
        elapsed = time.time() - self.t0
        eta = elapsed / pct * (1 - pct) if pct > 0 else 0
        bar = "█" * filled + "░" * (self.bar_w - filled)
        print(f"\r  [{bar}] {pct:5.1%} "
              f"{self.done}/{self.total} "
              f"{elapsed:.1f}s eta:{eta:.1f}s {msg}",
              end="", flush=True)
        if self.done == self.total:
            print()

    def finish(self):
        if self.done < self.total:
            self.done = self.total
            self.step()


# ==================== 经验缓冲区 ====================
class RolloutBuffer:
    def __init__(self):
        self.clear()
    def clear(self):
        self.frames = []
        self.gestures = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []


# ==================== 主训练器 ====================
class PPOTrainer:
    def __init__(self):
        print(f"[*] 设备: {DEVICE}")

        self.model = PPOAgent(SEQUENCE_LENGTH).to(DEVICE)
        self.model.load_from_bc(BC_MODEL_PATH, DEVICE)
        self._build_optimizer()

        self.ppo_step = 0
        ckpt = PPO_SAVE_DIR / "latest_ppo.pth"
        if ckpt.exists():
            self.model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            self._build_optimizer()
            print("[OK] PPO 进度已加载")

        self.sct = mss.mss()
        self.mon = {'left': MUMU_LEFT, 'top': MUMU_TOP,
                    'width': MUMU_W, 'height': MUMU_H}

        self.fb = deque(maxlen=SEQUENCE_LENGTH)
        self.gb = deque(maxlen=SEQUENCE_LENGTH)
        for _ in range(SEQUENCE_LENGTH):
            self.fb.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32))
            self.gb.append(np.zeros(5, dtype=np.float32))

        self.prev_gesture = np.zeros(5, dtype=np.float32)
        self.audio = AudioListener()
        self.last_active = False
        self.paused = False
        self._stopped = False
        self.audio_hit_count = 0

        if KEYBOARD_AVAILABLE:
            keyboard.add_hotkey('i+k+l', self._toggle)

        self.pending = deque()
        self.hits = self.misses = self.empties = 0
        self.buffer = RolloutBuffer()

    def _build_optimizer(self):
        self.optimizer = optim.Adam([
            {'params': self.model.cnn.parameters(),    'lr': 1e-6},
            {'params': self.model.lstm.parameters(),   'lr': 5e-6},
            {'params': self.model.actor.parameters(),  'lr': 2e-4},
            {'params': self.model.critic.parameters(), 'lr': 2e-4},
            {'params': self.model.log_std,            'lr': 1e-5},
        ])

    def _toggle(self):
        if not self.paused:
            self.paused = True; self._release()
            print("\n[!!!] 已暂停 | 再按 I+K+L 退出")
        else:
            self._stopped = True

    def _get_frame(self):
        s = self.sct.grab(self.mon)
        return np.array(
            Image.frombytes('RGB', s.size, s.rgb).resize(
                (IMAGE_SIZE, IMAGE_SIZE)),
            dtype=np.float32) / 255.0

    def _release(self):
        if self.last_active:
            ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
            self.last_active = False

    def _forward_no_grad(self):
        f = np.array(list(self.fb), dtype=np.float32)
        g = np.array(list(self.gb), dtype=np.float32)
        ft = torch.from_numpy(f).unsqueeze(0).permute(0, 1, 4, 2, 3).to(DEVICE)
        gt = torch.from_numpy(g).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            self.model.eval()
            logits, val, _ = self.model(ft, gt)
        return logits.squeeze(0), val.squeeze().item()

    def _reward(self, step, audio_hit):
        tp = 0.0
        while self.pending and step - self.pending[0] > TIMEOUT_FRAMES:
            self.pending.popleft(); tp -= 0.2; self.empties += 1
        if audio_hit:
            mi = -1
            for i, s in enumerate(self.pending):
                if step - s <= MATCH_WINDOW: mi = i; break
            if mi != -1:
                del self.pending[mi]; self.hits += 1
                return 1.0 + tp
            else:
                self.misses += 1
                return -0.5 + tp
        return -0.01 + tp

    def _build_sequences(self, all_f, all_g):
        """一次性预建所有序列, 返回 (N, seq_len, ...) 的大 tensor"""
        n = len(all_f)
        seqs_f = []
        seqs_g = []
        for i in range(n):
            ss = max(0, i - SEQUENCE_LENGTH + 1)
            sf = torch.stack(all_f[ss:i + 1])
            sg = torch.stack(all_g[ss:i + 1])
            if len(sf) < SEQUENCE_LENGTH:
                pad_f = torch.zeros(SEQUENCE_LENGTH - len(sf),
                                    3, IMAGE_SIZE, IMAGE_SIZE)
                pad_g = torch.zeros(SEQUENCE_LENGTH - len(sg), 5)
                sf = torch.cat([pad_f, sf])
                sg = torch.cat([pad_g, sg])
            seqs_f.append(sf)
            seqs_g.append(sg)
        return torch.stack(seqs_f), torch.stack(seqs_g)

    def _update(self):
        n = len(self.buffer.rewards)
        if n < UPDATE_EVERY:
            return

        print(f"\n[*] PPO 更新 ({n}帧) "
              f"命中:{self.hits} 漏:{self.misses} 空:{self.empties}")

        # 1. 预建序列 (一次性, 不再重复堆叠)
        print("  [1/3] 预建序列...", end="", flush=True)
        t0 = time.time()
        all_f = self.buffer.frames
        all_g = self.buffer.gestures
        seq_f, seq_g = self._build_sequences(all_f, all_g)
        # 一次性搬GPU
        seq_f = seq_f.to(DEVICE)
        seq_g = seq_g.to(DEVICE)
        print(f" {time.time()-t0:.1f}s ({seq_f.shape[0]}x{seq_f.shape[1]})")

        # 2. GAE
        old_actions = torch.stack(self.buffer.actions).to(DEVICE)
        old_log_p = torch.stack(self.buffer.log_probs).to(DEVICE)
        vals = torch.tensor(self.buffer.values, dtype=torch.float32).to(DEVICE)

        rets, gae = [], 0.0
        for t in reversed(range(n)):
            nv = 0.0 if t == n - 1 else vals[t + 1].item()
            d = self.buffer.rewards[t] + GAMMA * nv - vals[t].item()
            gae = d + GAMMA * LAMBDA_GAE * gae
            rets.insert(0, gae + vals[t].item())
        rets = torch.tensor(rets, dtype=torch.float32).to(DEVICE)
        adv = rets - vals
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # 3. PPO 多轮训练 (带进度条)
        self.model.train()
        total_batches = EPOCHS_PER_UPDATE * ((n + MINI_BATCH_SIZE - 1) // MINI_BATCH_SIZE)
        prog = PPOProgress(total_batches)

        ent_last = a_last = c_last = 0.0
        for ep in range(EPOCHS_PER_UPDATE):
            idx = torch.randperm(n)
            for s in range(0, n, MINI_BATCH_SIZE):
                bi = idx[s:s + MINI_BATCH_SIZE].tolist()

                bf = seq_f[bi]
                bg = seq_g[bi]

                logits, value, _ = self.model(bf, bg)
                rl = logits[:, :3]
                sd = self.model.std()
                dist = Normal(rl, sd)

                nlp = dist.log_prob(old_actions[bi]).sum(-1)
                ent = dist.entropy().sum(-1).mean()

                ratio = torch.exp(nlp - old_log_p[bi])
                s1 = ratio * adv[bi]
                s2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv[bi]
                a_loss = -torch.min(s1, s2).mean()
                c_loss = nn.MSELoss()(value.squeeze(), rets[bi])

                loss = a_loss + VALUE_COEF * c_loss - ENTROPY_COEF * ent

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                self.optimizer.step()

                ent_last = ent.item()
                a_last = a_loss.item()
                c_last = c_loss.item()

                prog.step(f"ep{ep+1}")

        # 4. 保存
        self.ppo_step += 1
        torch.save(self.model.state_dict(), PPO_SAVE_DIR / "latest_ppo.pth")
        sd = self.model.std()
        print(f"  [OK] PPO #{self.ppo_step} | "
              f"std:[{sd[0]:.2f},{sd[1]:.2f},{sd[2]:.2f}] | "
              f"Ent:{ent_last:.3f} | "
              f"Actor:{a_last:.4f} Critic:{c_last:.4f}")
        self.buffer.clear()


    def run(self):
        print("\n" + "=" * 60)
        print("[*] Phigros BC->PPO (全网络反向传播 + 真实gesture)")
        print("[*] 分层LR: CNN 1e-6 | LSTM 5e-6 | Actor 2e-4")
        print("[*] 判定: 响了+1 | 没响-0.2 | 漏了-0.5")
        print("[*] I+K+L 暂停退出")
        print("=" * 60 + "\n")
        time.sleep(3)
        self.audio.start()

        step = 0
        try:
            while not self._stopped:
                if self.paused:
                    time.sleep(0.1); continue
                t0 = time.time()

                raw = self._get_frame()
                self.fb.append(raw)
                self.gb.append(self.prev_gesture.copy())

                logits, val = self._forward_no_grad()

                rl_logits = logits[:3]
                sd = self.model.std()
                dist = Normal(rl_logits, sd)
                noisy = dist.sample()
                lp = dist.log_prob(noisy).sum()

                bc_prob = torch.sigmoid(logits[0]).item()
                rl_prob = torch.sigmoid(noisy[0]).item()
                did_click = rl_prob > CONFIDENCE_THRESHOLD

                cur_gesture = np.zeros(5, dtype=np.float32)
                if did_click:
                    norm_x = float(np.clip(
                        torch.sigmoid(noisy[1]).item(), DEAD_ZONE, 1 - DEAD_ZONE))
                    norm_y = float(np.clip(
                        torch.sigmoid(noisy[2]).item(), DEAD_ZONE, 1 - DEAD_ZONE))
                    x = norm_x * MUMU_W
                    y = norm_y * MUMU_H
                    if not self.last_active:
                        ctypes.windll.user32.SetCursorPos(
                            int(MUMU_LEFT + x), int(MUMU_TOP + y))
                        ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
                        self.last_active = True
                        self.pending.append(step)
                    else:
                        ctypes.windll.user32.SetCursorPos(
                            int(MUMU_LEFT + x), int(MUMU_TOP + y))
                    cur_gesture = np.array(
                        [1.0, norm_x, norm_y, 1.0, 0.0], dtype=np.float32)
                else:
                    if self.last_active:
                        self._release()

                self.prev_gesture = cur_gesture

                ah = self.audio.consume_hit()
                if ah:
                    self.audio_hit_count += 1
                    rms = self.audio.current_rms
                    print(f"\n  [AUDIO #{self.audio_hit_count}] "
                          f"S:{step} | {step/30:.1f}s | RMS:{rms:.4f}")

                rw = self._reward(step, ah)

                self.buffer.frames.append(
                    torch.tensor(raw, dtype=torch.float32).permute(2, 0, 1))
                self.buffer.gestures.append(
                    torch.tensor(self.gb[-1].copy(), dtype=torch.float32))
                self.buffer.actions.append(noisy.detach().cpu())
                self.buffer.log_probs.append(lp.detach().cpu())
                self.buffer.rewards.append(rw)
                self.buffer.values.append(val)

                step += 1

                if step % 30 == 0:
                    print(
                        f"\rS:{step:>5} | "
                        f"BC:{bc_prob:.2f}->RL:{rl_prob:.2f} | "
                        f"R:{rw:>5.2f} RMS:{self.audio.current_rms:.3f} | "
                        f"H:{self.hits} M:{self.misses} E:{self.empties} | "
                        f"音频:{self.audio_hit_count} | "
                        f"Buf:{len(self.buffer.rewards)}/{UPDATE_EVERY}",
                        end="", flush=True)

                if len(self.buffer.rewards) >= UPDATE_EVERY:
                    self._update()

                time.sleep(max(0, 1 / 30 - (time.time() - t0)))

        except KeyboardInterrupt:
            pass
        finally:
            self._release()
            self.audio.stop()
            if len(self.buffer.rewards) > 64:
                self._update()
            print(f"\n[*] 结束 | H:{self.hits} M:{self.misses} E:{self.empties} | "
                  f"音频脉冲:{self.audio_hit_count} | 峰值RMS:{self.audio.peak_rms:.4f}")
            print(f"[*] PPO 更新 {self.ppo_step} 次")


if __name__ == "__main__":
    PPOTrainer().run()
