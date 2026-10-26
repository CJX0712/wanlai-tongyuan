#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《万籁同源》 One Origin of All Sounds
====================================
世界级原创音乐 · 纯 NumPy 生成式作曲 + 合成 (零外部依赖)

架构 (集世界顶级算法之大成):
  1. 调式内核  : 东方五声音阶 宫商角徵羽 = C D E G A，五幕共享一个演化动机(motif)
  2. 音色引擎  : FM 合成 (John Chowning, 数字合成基石) + 加法合成 pad + 锯齿 bass
  3. 空间      : Schröder 混响 (FFT 重叠相加卷积, 左右声道去相关)
  4. 母带      : tanh 软削波 + 峰值归一 + 等功率交叉淡入串联五幕

五幕: 静谧(50bpm) → 萌动(72) → 冲突(100) → 爆发(132) → 合一(84)
作者: 晨星 | 许可: MIT | 仓库: github.com/CJX0712/wanlai-tongyuan
"""

import numpy as np
import wave
import os

SR = 44100
PENT = [0, 2, 4, 7, 9]          # 宫 商 角 徵 羽 (C D E G A)
SEED = 20261027
rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------------
# 音乐基础
# ----------------------------------------------------------------------------
def midi_to_freq(m: float) -> float:
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


def pnote(deg: int, octv: int) -> float:
    """五声音阶音高: deg 0..4 = 宫商角徵羽; octv 如 4 => C4"""
    return midi_to_freq(12 * (octv + 1) + PENT[deg % 5])


def chroma(midi: int) -> float:
    """任意半音音高 (用于冲突幕的张力音)"""
    return midi_to_freq(midi)


# ----------------------------------------------------------------------------
# 包络 (ADSR)
# ----------------------------------------------------------------------------
def envelope(n: int, a: float, d: float, r: float, sus: float = 0.6) -> np.ndarray:
    an = max(1, int(a * SR)); dn = max(1, int(d * SR)); rn = max(1, int(r * SR))
    total = an + dn + rn
    if total > n:
        s = n / total
        an, dn, rn = int(an * s), int(dn * s), int(rn * s)
    env = np.empty(n)
    env[:an] = np.linspace(0.0, 1.0, an)
    env[an:an + dn] = np.linspace(1.0, sus, dn)
    rel = n - rn
    env[an + dn:rel] = sus
    env[rel:] = np.linspace(sus, 0.0, rn)
    return env


# ----------------------------------------------------------------------------
# 合成声部
# ----------------------------------------------------------------------------
def fm(freq, n, ratio=2.0, index=3.0, index_decay=0.0, vib=0.0):
    """Chowning FM 合成: y = sin(2π f_c t + I·sin(2π f_m t))"""
    t = np.arange(n) / SR
    if vib > 0:
        freq = freq * (1.0 + 0.004 * np.sin(2 * np.pi * 5.0 * t))
    I = index * np.exp(-index_decay * t) if index_decay > 0 else index
    return np.sin(2 * np.pi * freq * t + I * np.sin(2 * np.pi * ratio * freq * t))


def pad(freq, n, harmonics=6, vib=0.0):
    """加法合成泛音铺底 (正弦谐波叠加)"""
    t = np.arange(n) / SR
    f = freq * (1.0 + 0.003 * np.sin(2 * np.pi * 0.3 * t)) if vib > 0 else freq
    y = np.zeros(n)
    for h in range(1, harmonics + 1):
        y += (1.0 / h) * np.sin(2 * np.pi * h * f * t)
    return y / harmonics


def bass(freq, n):
    """锯齿式低音 (奇次强、偶次弱)"""
    t = np.arange(n) / SR
    y = np.zeros(n)
    for h in range(1, 13):
        y += (1.0 / h) * np.sin(2 * np.pi * h * freq * t) * (1.0 if h % 2 else 0.6)
    return y / 6.0


def noise_pad(n, cutoff=700.0, gain=0.04):
    """一极点低通白噪铺底 (氛围呼吸感)"""
    x = rng.standard_normal(n)
    a = 1.0 - np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty(n)
    y[0] = x[0]
    for i in range(1, n):
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y * envelope(n, 2.0, 1.0, 3.0, sus=0.5) * gain


def render_event(freq, dur_s, voice, gain=1.0):
    n = max(1, int(dur_s * SR))
    if voice == 'pad':
        sig = pad(freq, n, vib=0.5)
        env = envelope(n, 0.8, 1.0, 2.5, sus=0.8)
    elif voice == 'pluck':
        sig = fm(freq, n, ratio=2.0, index=4.0, index_decay=4.0)
        env = envelope(n, 0.005, 0.15, 0.3, sus=0.0)
    elif voice == 'bell':
        sig = fm(freq, n, ratio=3.5, index=2.0, index_decay=1.2)
        env = envelope(n, 0.002, 0.2, 2.0, sus=0.0)
    elif voice == 'bass':
        sig = bass(freq, n)
        env = envelope(n, 0.01, 0.1, 0.2, sus=0.7)
    elif voice == 'lead':
        sig = fm(freq, n, ratio=1.0, index=2.0, index_decay=1.0, vib=0.5)
        env = envelope(n, 0.02, 0.2, 0.4, sus=0.6)
    else:
        sig = pad(freq, n); env = envelope(n, 0.1, 0.2, 0.5)
    return sig * env * gain


def stereoize(sig, pan):
    pan = max(-1.0, min(1.0, pan))
    ang = (pan + 1.0) / 2.0 * (np.pi / 2)
    return sig * np.cos(ang), sig * np.sin(ang)


# ----------------------------------------------------------------------------
# 五幕事件序列 —— 共享动机: 宫(C) 商(D) 角(E) 徵(G) 羽(A)
# ----------------------------------------------------------------------------
def build_movements():
    movs = []

    # ---- 第一幕 静谧 (50 bpm) ------------------------------------------------
    ev = []
    ev.append(('drone', 0, pnote(0, 3), 33, 'pad', 0.0, 0.16))
    for b, deg in zip([2, 6, 11, 16, 22, 28], [0, 2, 4, 3, 2, 0]):
        ev.append(('note', b, pnote(deg, 6), 3, 'bell', rng.uniform(-0.8, 0.8), 0.5))
    ev.append(('note', 13, pnote(1, 4), 4, 'pad', 0.0, 0.10))
    ev.append(('noise', 0, 0.0, 33, 'noise', 0.0, 0.03))
    movs.append(('静谧 Tranquility', 50, ev, 2.0))

    # ---- 第二幕 萌动 (72 bpm) ------------------------------------------------
    ev = []
    ev.append(('pad', 0, pnote(0, 3), 52, 'pad', 0.0, 0.10))
    ev.append(('pad', 0, pnote(4, 3), 52, 'pad', 0.0, 0.08))
    motif = [(0, 5), (2, 5), (4, 5), (3, 5)]  # 宫 商 角 徵
    md = [1, 1, 1, 2]
    for rep, start in enumerate([8, 16, 24, 32, 40]):
        for (deg, octv), dur, i in zip(motif, md, range(4)):
            gain = 0.55 + 0.05 * rep
            ev.append(('note', start + sum(md[:i]), pnote(deg, octv), dur, 'pluck',
                       rng.uniform(-0.5, 0.5), gain))
    for b in range(0, 52, 2):
        ev.append(('bass', b, pnote(0, 2), 1.5, 'bass', 0.0, 0.45))
    # 对位旋律
    for b, deg in zip([10, 18, 26, 34], [2, 4, 3, 4]):
        ev.append(('note', b, pnote(deg, 5), 2, 'bell', rng.uniform(-0.6, 0.6), 0.35))
    movs.append(('萌动 Awakening', 72, ev, 2.0))

    # ---- 第三幕 冲突 (100 bpm) ----------------------------------------------
    ev = []
    ost = [(3, 3), (0, 4), (4, 4), (3, 4)]  # A3 C4 E4 A4 循环
    for bar in range(18):
        for i, (deg, octv) in enumerate(ost):
            b = bar * 4 + i
            cres = 0.30 + 0.015 * bar
            ev.append(('note', b, pnote(deg, octv), 0.9, 'lead', rng.uniform(-0.4, 0.4), cres))
        # 张力音: 半音碰撞 (Db4 / B3)
        if bar in (5, 9, 13):
            ev.append(('note', bar * 4 + 1.5, chroma(61), 0.4, 'lead', 0.6, 0.4))
            ev.append(('note', bar * 4 + 3.5, chroma(59), 0.4, 'lead', -0.6, 0.4))
    # 小二度不协和铺底 (C4 vs Db4) 渐强
    for start, dur in [(0, 18), (18, 18), (36, 18)]:
        ev.append(('pad', start, pnote(0, 4), dur, 'pad', -0.2, 0.06 + 0.02 * start / 18))
        ev.append(('pad', start, chroma(61), dur, 'pad', 0.2, 0.06 + 0.02 * start / 18))
    movs.append(('冲突 Conflict', 100, ev, 2.0))

    # ---- 第四幕 爆发 (132 bpm) ----------------------------------------------
    ev = []
    ev.append(('pad', 0, pnote(0, 4), 110, 'pad', 0.0, 0.12))
    ev.append(('pad', 0, pnote(4, 4), 110, 'pad', 0.0, 0.10))
    ev.append(('pad', 0, pnote(2, 4), 110, 'pad', 0.0, 0.10))
    # 驱动低音 (八分音符)
    for b in np.arange(0, 110, 0.5):
        ev.append(('bass', float(b), pnote(0, 2), 0.45, 'bass', 0.0, 0.42))
    # 快速琶音 (十六分音符) 跨两个八度
    scale = [(0, 4), (2, 4), (4, 4), (3, 4), (4, 5), (3, 5), (2, 5), (0, 5)]
    for bar in range(27):
        for i, (deg, octv) in enumerate(scale):
            b = bar * 4 + i * 0.5
            ev.append(('note', b, pnote(deg, octv), 0.45, 'lead',
                       np.sin(b) * 0.6, 0.38))
    # 高潮强击
    ev.append(('bell', 104, pnote(0, 5), 5, 'bell', 0.0, 0.7))
    ev.append(('bell', 104, pnote(4, 5), 5, 'bell', 0.0, 0.5))
    movs.append(('爆发 Eruption', 132, ev, 3.0))

    # ---- 第五幕 合一 (84 bpm) -----------------------------------------------
    ev = []
    chord = [(0, 4), (2, 4), (4, 4), (3, 4), (0, 5), (2, 5)]
    for deg, octv in chord:
        ev.append(('pad', 0, pnote(deg, octv), 56, 'pad', 0.0, 0.18))
    ev.append(('bass', 0, pnote(0, 2), 56, 'pad', 0.0, 0.14))
    for b, deg in zip([4, 10, 16, 22, 28, 34, 40, 46], [0, 2, 4, 3, 2, 4, 0, 2]):
        ev.append(('note', b, pnote(deg, 6), 4, 'bell', rng.uniform(-0.7, 0.7), 0.4))
    # 动机回归 (柔和)
    for start in [8, 20, 32]:
        for i, (deg, octv) in enumerate([(0, 5), (2, 5), (4, 5), (3, 5)]):
            ev.append(('note', start + i, pnote(deg, octv), 1.2, 'pluck',
                       rng.uniform(-0.4, 0.4), 0.45))
    ev.append(('noise', 0, 0.0, 56, 'noise', 0.0, 0.025))
    movs.append(('合一 Unity', 84, ev, 4.0))

    return movs


# ----------------------------------------------------------------------------
# 渲染单幕
# ----------------------------------------------------------------------------
def render_movement(events, bpm, tail_beats):
    spb = 60.0 / bpm
    end = 0.0
    for e in events:
        _, beat, _, dur, _, _, _ = e
        end = max(end, beat + (dur if dur > 0 else 0))
    n = int((end + tail_beats) * spb * SR)
    L = np.zeros(n); R = np.zeros(n)
    for e in events:
        kind, beat, freq, dur, voice, pan, gain = e
        if kind == 'noise':
            sig = noise_pad(n if dur <= 0 else int(dur * spb * SR))
        else:
            sig = render_event(freq, dur * spb, voice, gain=gain)
        sl, sr_ = stereoize(sig, pan)
        start = int(beat * spb * SR)
        L[start:start + len(sl)] += sl
        R[start:start + len(sr_)] += sr_
    return L, R


# ----------------------------------------------------------------------------
# Schröder 混响: FFT 重叠相加卷积
# ----------------------------------------------------------------------------
def make_ir(dur, decay, seed):
    n = int(SR * dur)
    r = np.random.default_rng(seed)
    ir = r.standard_normal(n) * np.exp(-decay * np.arange(n) / SR)
    return ir / np.max(np.abs(ir))


def fft_conv(x, ir):
    L = len(x); M = len(ir)
    N = 1 << int(np.ceil(np.log2(2 * M)))
    H = np.fft.rfft(ir, N)
    hop = N - M + 1
    out = np.zeros(L + M - 1)
    i = 0
    while i < L:
        xb = x[i:i + hop]
        if len(xb) < hop:
            xb = np.pad(xb, (0, hop - len(xb)))
        X = np.fft.rfft(xb, N)
        Y = np.fft.irfft(X * H, N)[:L - i + M - 1] if i + hop > L else np.fft.irfft(X * H, N)
        out[i:i + len(Y)] += Y
        i += hop
    return out[:L + M - 1]


def crossfade(a, b, cf):
    n = int(cf * SR)
    fa = np.cos(np.linspace(0, np.pi / 2, n))
    fb = np.sin(np.linspace(0, np.pi / 2, n))
    out = np.empty(len(a) + len(b) - n)
    out[:len(a) - n] = a[:len(a) - n]
    out[len(a) - n:len(a)] = a[len(a) - n:] * fa + b[:n] * fb
    out[len(a):] = b[n:]
    return out


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    movs = build_movements()

    L = None; R = None
    lengths = []
    for name, bpm, ev, tail in movs:
        ml, mr = render_movement(ev, bpm, tail)
        lengths.append(len(ml) / SR)
        if L is None:
            L, R = ml, mr
        else:
            L = crossfade(L, ml, 1.5)
            R = crossfade(R, mr, 1.5)
        print(f"  ✓ {name:18s} bpm={bpm:<4} len={len(ml)/SR:6.1f}s")

    print("混响 (Schröder FFT 卷积)…")
    irL = make_ir(1.6, 3.0, 11)
    irR = make_ir(1.6, 3.0, 22)
    revL = fft_conv(L, irL)
    revR = fft_conv(R, irR)
    L = 0.90 * L + 0.28 * revL[:len(L)]
    R = 0.90 * R + 0.28 * revR[:len(R)]

    print("母带 (软削波 + 归一)…")
    L = np.tanh(0.8 * L)
    R = np.tanh(0.8 * R)
    peak = max(np.max(np.abs(L)), np.max(np.abs(R))) or 1.0
    L = L / peak * 0.95
    R = R / peak * 0.95

    path = os.path.join(out_dir, "wanlai_tongyuan.wav")
    data = np.empty((len(L), 2), dtype=np.float32)
    data[:, 0] = L; data[:, 1] = R
    pcm = np.int16(np.clip(data * 32767.0, -32768, 32767))
    with wave.open(path, 'w') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())

    total = len(L) / SR
    print(f"\n✅ 成品: {path}")
    print(f"   总时长 {total/60:.1f}:{int(total%60):02d} | {SR}Hz 立体声 16bit | {os.path.getsize(path)/1e6:.1f} MB")
    print("   五幕时长:", " / ".join(f"{x/60:.1f}:{int(x%60):02d}" for x in lengths))


if __name__ == "__main__":
    main()
