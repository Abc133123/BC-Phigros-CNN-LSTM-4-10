#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phigros 模型训练器 - v6 轻量版
[ARCH]   1300万→~35万参数, 匹配25K样本
[METRIC] 加 recall/precision/f1
[SIZE]   IMAGE_SIZE=64
"""

import json, math, random, time, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image

# ===== 配置 =====
DATA_DIR = Path(r"G:\1255\AutoPhigros2\phigros_data")
MODEL_SAVE_DIR = Path(r"G:\1255\AutoPhigros2\models")
MODEL_SAVE_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 100
SEQUENCE_LENGTH = 15
IMAGE_SIZE = 64
CACHE_DATA = True
WARMUP_EPOCHS = 5
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 15
SAMPLE_STEP = 3
DELAY_COMPENSATION = 3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()
print(f"[*] 设备: {DEVICE}, AMP: {'ON' if USE_AMP else 'OFF'}")


def print_bar(iteration, total, prefix='', suffix='', length=40, fill='█'):
    percent = "{0:.1f}".format(100 * iteration / float(total))
    filled = int(length * iteration // total)
    bar = fill * filled + '-' * (length - filled)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()


class PhigrosLoss(nn.Module):
    def __init__(self, pos_weight=3.0, w_active=4.0, w_xy=2.0, w_type=1.0):
        super().__init__()
        self.register_buffer('_pos_weight', torch.tensor([pos_weight]))
        self.bce_active = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight)
        self.mse = nn.MSELoss()
        self.bce_type = nn.BCEWithLogitsLoss()
        self.w_active = w_active
        self.w_xy = w_xy
        self.w_type = w_type

    def forward(self, pred, target):
        loss_active = self.bce_active(pred[:, 0:1], target[:, 0:1])
        active_mask = (target[:, 0] > 0.5)
        n_active = active_mask.sum().item()

        if n_active > 0:
            loss_xy = self.mse(pred[active_mask, 1:3], target[active_mask, 1:3])
            loss_type = self.bce_type(pred[active_mask, 3:5], target[active_mask, 3:5])
        else:
            loss_xy = torch.tensor(0.0, device=pred.device)
            loss_type = torch.tensor(0.0, device=pred.device)

        total = (self.w_active * loss_active + self.w_xy * loss_xy + self.w_type * loss_type)

        with torch.no_grad():
            pred_sig = torch.sigmoid(pred[:, 0])
            correct = ((pred_sig > 0.5) == (target[:, 0] > 0.5)).float().mean()

            active_pred = (pred_sig > 0.5)
            active_true = (target[:, 0] > 0.5)
            tp = (active_pred & active_true).sum().item()
            fp = (active_pred & ~active_true).sum().item()
            fn = (~active_pred & active_true).sum().item()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 0.001)

        return total, {
            'total': total.item(),
            'active': loss_active.item(),
            'xy': loss_xy.item() if n_active > 0 else -1.0,
            'type': loss_type.item() if n_active > 0 else -1.0,
            'active_acc': correct.item(),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_active': n_active
        }


class PhigrosNet(nn.Module):
    def __init__(self, sequence_length=15):
        super().__init__()
        self.sequence_length = sequence_length

        # 64x64 -> 32x32 -> 16x16 -> Pool(4,4) => 32*4*4=512
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # 512+5=517 -> 128, 1层
        self.lstm = nn.LSTM(
            input_size=512 + 5,
            hidden_size=128,
            num_layers=1,
            dropout=0.0,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, input_frames, input_gestures):
        batch_size, seq_len = input_frames.shape[:2]
        frames_flat = input_frames.reshape(batch_size * seq_len, *input_frames.shape[2:])
        cnn_features = self.cnn(frames_flat)
        cnn_features = cnn_features.reshape(batch_size, seq_len, -1)
        combined = torch.cat([cnn_features, input_gestures], dim=-1)
        lstm_out, _ = self.lstm(combined)
        return self.fc(lstm_out[:, -1])


class PhigrosDataset(Dataset):
    def __init__(self, sessions, sequence_length=15, delay_comp=3, cache=True, augment=False):
        self.sequence_length = sequence_length
        self.delay_comp = delay_comp
        self.cache = cache
        self.augment = augment
        self.image_cache = {}
        self.sessions = sessions
        self.samples = self._create_samples()
        if cache:
            self._preload_images()

    @staticmethod
    def load_sessions(data_dir):
        sessions = []
        for session_dir in sorted(Path(data_dir).glob('session_*')):
            json_path = session_dir / 'data.json'
            if not json_path.exists():
                continue
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            sessions.append({
                'dir': session_dir,
                'frames': data.get('frame_data', []),
                'gestures': data.get('gesture_data', []),
                'timeline': data.get('timeline', []),
                'frames_dir': session_dir / 'frames'
            })
            print(f"    加载: {session_dir.name} - {len(data.get('timeline', []))} 帧")
        return sessions

    def _create_samples(self):
        samples = []
        required_len = self.sequence_length + 1 + self.delay_comp
        for session in self.sessions:
            timeline = session['timeline']
            frames_dir = session['frames_dir']
            if len(timeline) < required_len:
                continue
            for i in range(0, len(timeline) - required_len + 1, SAMPLE_STEP):
                target_idx = i + self.sequence_length + self.delay_comp
                samples.append({
                    'frames_dir': frames_dir,
                    'input_frames': timeline[i:i + self.sequence_length],
                    'input_gestures': self._extract_gestures(timeline[i:i + self.sequence_length]),
                    'target_gesture': self._get_gesture(timeline[target_idx])
                })
        return samples

    def _preload_images(self):
        print(f"\n    预加载图片 ({IMAGE_SIZE}x{IMAGE_SIZE})...")
        all_paths = set()
        for session in self.sessions:
            frames_dir = session['frames_dir']
            for item in session['timeline']:
                p = str(frames_dir / item['frame'])
                if p not in self.image_cache:
                    all_paths.add(p)

        total = len(all_paths)
        if total == 0:
            return

        start = time.time()
        errors = 0
        for i, img_path in enumerate(sorted(all_paths)):
            try:
                img = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
                self.image_cache[img_path] = np.array(img, dtype=np.float32) / 255.0
            except Exception:
                self.image_cache[img_path] = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
                errors += 1
            if i % 500 == 0 or i == total - 1:
                elapsed = time.time() - start
                speed = (i + 1) / max(elapsed, 0.001)
                eta = (total - i - 1) / speed
                print_bar(i + 1, total, prefix='    加载:',
                          suffix=f'{i + 1}/{total} {speed:.0f}张/s 剩余{eta:.0f}s')

        elapsed = time.time() - start
        mem = len(self.image_cache) * IMAGE_SIZE * IMAGE_SIZE * 3 * 4 / 1024 ** 2
        print(f'\n    ✓ {len(self.image_cache)}张 错误{errors}张 耗时{elapsed:.1f}s ~{mem:.0f}MB')

    def _extract_gestures(self, timeline_segment):
        result = []
        for item in timeline_segment:
            g = item.get('gesture')
            if g:
                result.append([1, g['normalized']['x'], g['normalized']['y'],
                               1 if g.get('type') == 'tap' else 0,
                               1 if g.get('type') == 'slide' else 0])
            else:
                result.append([0, 0, 0, 0, 0])
        return result

    def _get_gesture(self, timeline_item):
        g = timeline_item.get('gesture')
        if g:
            return [1, g['normalized']['x'], g['normalized']['y'],
                    1 if g.get('type') == 'tap' else 0,
                    1 if g.get('type') == 'slide' else 0]
        return [0, 0, 0, 0, 0]

    def _augment(self, img_array):
        img = img_array.copy()
        img = img + random.uniform(-0.03, 0.03)
        if random.random() < 0.3:
            img = img + np.random.normal(0, 0.015, img.shape).astype(np.float32)
        if random.random() < 0.3:
            img = img * random.uniform(0.95, 1.05)
        return np.clip(img, 0.0, 1.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_frames = []
        for frame_info in sample['input_frames']:
            img_path = str(sample['frames_dir'] / frame_info['frame'])
            img = self.image_cache.get(img_path, np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32))
            if self.augment:
                img = self._augment(img)
            input_frames.append(img)

        input_frames = torch.from_numpy(np.array(input_frames)).permute(0, 3, 1, 2)
        input_gestures = torch.FloatTensor(sample['input_gestures'])
        target_gesture = torch.FloatTensor(sample['target_gesture'])

        return {
            'input_frames': input_frames,
            'input_gestures': input_gestures,
            'target_gesture': target_gesture
        }


def get_lr_lambda(epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train():
    print("\n" + "=" * 70)
    print("[*] Phigros 模型训练器 - v6 轻量版")
    print("=" * 70)
    print(f"[*] 延迟补偿: DELAY_COMPENSATION = {DELAY_COMPENSATION} 帧")
    print(f"[*] 图片尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}")

    print("\n[*] 加载数据...")
    all_sessions = PhigrosDataset.load_sessions(DATA_DIR)
    print(f"    共 {len(all_sessions)} 个session")

    if not all_sessions:
        print("[!] 没有数据！")
        return

    random.seed(42)
    random.shuffle(all_sessions)
    split_idx = max(1, int(0.85 * len(all_sessions)))
    train_sessions = all_sessions[:split_idx]
    val_sessions = all_sessions[split_idx:]
    print(f"    训练sessions: {len(train_sessions)} | 验证sessions: {len(val_sessions)}")

    print("\n[*] 构建训练集...")
    train_dataset = PhigrosDataset(train_sessions, SEQUENCE_LENGTH, DELAY_COMPENSATION, CACHE_DATA, augment=True)
    print("[*] 构建验证集...")
    val_dataset = PhigrosDataset(val_sessions, SEQUENCE_LENGTH, DELAY_COMPENSATION, CACHE_DATA, augment=False)

    print(f"\n    训练样本: {len(train_dataset)} | 验证样本: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = PhigrosNet(SEQUENCE_LENGTH).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[*] 模型参数量: {total_params:,}")

    criterion = PhigrosLoss(pos_weight=3.0, w_active=4.0, w_xy=2.0, w_type=1.0).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: get_lr_lambda(e, WARMUP_EPOCHS, EPOCHS))
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

    best_val_loss = float('inf')
    patience_counter = 0
    epoch_times = []

    print(f"\n[*] 开始训练 (最多{EPOCHS}轮, early stop={EARLY_STOP_PATIENCE})")
    print("=" * 70)

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()

        sum_loss, sum_acc, n_batch = 0, 0, 0
        sum_recall, sum_precision, sum_f1 = 0, 0, 0

        for batch_idx, batch in enumerate(train_loader):
            input_frames = batch['input_frames'].to(DEVICE, non_blocking=True)
            input_gestures = batch['input_gestures'].to(DEVICE, non_blocking=True)
            target = batch['target_gesture'].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if USE_AMP:
                with torch.amp.autocast('cuda'):
                    pred = model(input_frames, input_gestures)
                    loss, ld = criterion(pred, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(input_frames, input_gestures)
                loss, ld = criterion(pred, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            sum_loss += ld['total']
            sum_acc += ld['active_acc']
            sum_recall += ld['recall']
            sum_precision += ld['precision']
            sum_f1 += ld['f1']
            n_batch += 1

            if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:
                lr_now = optimizer.param_groups[0]['lr']
                elapsed = time.time() - epoch_start
                speed = (batch_idx + 1) / max(elapsed, 0.001)
                eta = (len(train_loader) - batch_idx - 1) / speed
                print_bar(batch_idx + 1, len(train_loader),
                          prefix=f'[Epoch {epoch + 1:3d}]',
                          suffix=f'L:{ld["total"]:.3f} f1:{ld["f1"]:.2f} recall:{ld["recall"]:.2f} {eta:.0f}s')

        avg_loss = sum_loss / max(n_batch, 1)
        avg_recall = sum_recall / max(n_batch, 1)
        avg_f1 = sum_f1 / max(n_batch, 1)

        model.eval()
        v_sum_loss, v_sum_acc, v_n_batch = 0, 0, 0
        v_sum_recall, v_sum_precision, v_sum_f1 = 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                input_frames = batch['input_frames'].to(DEVICE, non_blocking=True)
                input_gestures = batch['input_gestures'].to(DEVICE, non_blocking=True)
                target = batch['target_gesture'].to(DEVICE, non_blocking=True)

                if USE_AMP:
                    with torch.amp.autocast('cuda'):
                        pred = model(input_frames, input_gestures)
                        loss, ld = criterion(pred, target)
                else:
                    pred = model(input_frames, input_gestures)
                    loss, ld = criterion(pred, target)

                v_sum_loss += ld['total']
                v_sum_acc += ld['active_acc']
                v_sum_recall += ld['recall']
                v_sum_precision += ld['precision']
                v_sum_f1 += ld['f1']
                v_n_batch += 1

        v_avg_loss = v_sum_loss / max(v_n_batch, 1)
        v_avg_recall = v_sum_recall / max(v_n_batch, 1)
        v_avg_f1 = v_sum_f1 / max(v_n_batch, 1)

        scheduler.step()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_time = sum(epoch_times[-10:]) / len(epoch_times[-10:])
        remaining = avg_time * (min(EPOCHS, epoch + EARLY_STOP_PATIENCE) - epoch - 1)
        rm, rs = int(remaining // 60), int(remaining % 60)

        print(f'\n  [{epoch + 1:3d}] {epoch_time:4.0f}s '
              f'| T:loss={avg_loss:.4f} f1={avg_f1:.3f} recall={avg_recall:.3f} '
              f'| V:loss={v_avg_loss:.4f} f1={v_avg_f1:.3f} recall={v_avg_recall:.3f} '
              f'| LR={optimizer.param_groups[0]["lr"]:.1e} '
              f'| ETA {rm}m{rs}s')

        if v_avg_loss < best_val_loss:
            best_val_loss = v_avg_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': v_avg_loss,
                'val_f1': v_avg_f1,
                'val_recall': v_avg_recall,
                'config': {
                    'sequence_length': SEQUENCE_LENGTH,
                    'image_size': IMAGE_SIZE,
                    'delay_compensation': DELAY_COMPENSATION,
                    'version': 'v6'
                }
            }, MODEL_SAVE_DIR / "best_v6.pth")
            print(f'    ★ 保存最佳模型 Val Loss: {best_val_loss:.4f}')
        else:
            patience_counter += 1
            print(f'    (未改善 {patience_counter}/{EARLY_STOP_PATIENCE})')

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f'\n[*] Early Stopping! {EARLY_STOP_PATIENCE}轮无改善')
            break

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), MODEL_SAVE_DIR / f"v6_epoch{epoch + 1}.pth")

    print("\n" + "=" * 70)
    print(f"[OK] 训练完成! 最佳 Val Loss: {best_val_loss:.4f}")
    print(f"模型保存: {MODEL_SAVE_DIR / 'best_v6.pth'}")
    print("=" * 70)


if __name__ == "__main__":
    train()
