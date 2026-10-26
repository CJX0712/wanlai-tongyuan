# -*- coding: utf-8 -*-
"""
《万籁同源》程序化合成引擎  —  作者：晨星
纯 Python + numpy，零外部依赖。把五幕总谱渲染为 44.1kHz/16-bit 立体声 WAV。

算法清单：
  · FM 合成 (Chowning)        —— 钟磬 / 马林巴 / 钢琴
  · 失谐加性锯齿 + 低通       —— 史诗弦乐 Pad
  · 共振峰合成 (谐振器)        —— 无词吟唱 / 占位主唱
  · ADSR 参数化包络
  · FFT 重叠相加卷积混响 (OLA) —— 空间化 Hall
  · 等功率声像 + mid/side 宽度
  · 母带链：软削波 + 峰值归一 + TPDF 抖动

运行：python synth.py  ->  万籁同源.wav + score.json
"""
import json, math, wave, struct, random
import numpy as np

SR = 44100
TWOP = 2 * math.pi

# ---------------------------------------------------------------------------
# 1. 音名 -> 频率 (十二平均律, A4=440)
# ---------------------------------------------------------------------------
_NOTE = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,'F#':6,'Gb':6,
          'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}
def freq(name):
    m = __import__('re').match(r'([A-G][#b]?)(\d)', name)
    semi = (int(m.group(2)) + 1) * 12 + _NOTE[m.group(1)]
    return 440.0 * 2 ** ((semi - 69) / 12.0)

def f0_from_ratio(base, ratio):
    return base * (2 ** (ratio / 12.0))

# ---------------------------------------------------------------------------
# 2. ADSR 包络
# ---------------------------------------------------------------------------
def adsr(n, a, d, s, r, sus=0.7, sr=SR):
    """返回长度 n 的包络 (0..1)。a/d/r 为秒；分段长度恒等于 n。"""
    na = min(int(a*sr), max(n-3, 0))
    nd = min(int(d*sr), max(n-na-2, 0))
    nr = min(int(r*sr), max(n-na-nd-1, 0))
    ns = n - na - nd - nr
    out = np.zeros(n)
    i = 0
    if na: out[i:i+na] = np.linspace(0, 1, na); i += na
    if nd: out[i:i+nd] = np.linspace(1, sus, nd); i += nd
    if ns: out[i:i+ns] = sus; i += ns
    if nr: out[i:i+nr] = np.linspace(sus, 0, nr)
    return out

# ---------------------------------------------------------------------------
# 3. 乐器合成 (均返回单声道 np.ndarray, 长度 = int(dur*sr) + 释放尾)
# ---------------------------------------------------------------------------
def fm_bell(f, dur, index=3.0, ratio=2.0, decay=4.0, a=0.002, r=0.4, gain=1.0, sr=SR):
    n = int(dur*sr)
    t = np.arange(n)/sr
    # 2-op FM: 载波被调制器调相，指数衰减调制深度 -> 钟磬/马林巴
    mod = index * f * np.exp(-t*decay) * np.sin(TWOP*ratio*f*t)
    sig = np.sin(TWOP*f*t + mod)
    env = adsr(n, a, 0.05, 0.0, r, sus=0.0, sr=sr)
    return sig * env * gain

def fm_pluck(f, dur, index=1.5, ratio=1.0, decay=8.0, a=0.001, r=0.25, gain=1.0, sr=SR):
    """钢琴/马林巴式快衰拨奏。"""
    n = int(dur*sr)
    t = np.arange(n)/sr
    mod = index * f * np.exp(-t*decay) * np.sin(TWOP*ratio*f*t)
    sig = np.sin(TWOP*f*t + mod)
    # 叠加少量高次谐波增亮
    sig = sig + 0.3*np.sin(TWOP*2*f*t + mod*0.5)
    env = adsr(n, a, min(0.08, dur*0.5), 0.0, r, sus=0.0, sr=sr)
    return sig * env * gain

def pad_saw(f, dur, detune=6.0, cutoff=1800.0, a=0.6, r=0.8, gain=1.0, sr=SR):
    """3 个微失谐锯齿低通 -> 弦乐铺底。"""
    n = int(dur*sr)
    t = np.arange(n)/sr
    cents = [-detune, 0.0, detune]
    sig = np.zeros(n)
    for c in cents:
        ff = f * (2 ** (c/1200.0))
        sig += np.sin(TWOP*ff*t) + 0.5*np.sin(TWOP*2*ff*t)
    sig /= 3.0
    # 一极低通 (阻尼高频, 柔和)
    a_lp = 1.0 - min(cutoff/(0.5*sr), 0.95)
    y = np.empty(n); y[0] = (1-a_lp)*sig[0]
    y[1:] = a_lp*y[:-1] + (1-a_lp)*sig[1:]
    env = adsr(n, a, 0.2, 0.8, r, sus=0.8, sr=sr)
    return y * env * gain

def bass(f, dur, a=0.01, r=0.1, gain=1.0, sr=SR):
    """正弦 sub + 低通锯齿。"""
    n = int(dur*sr)
    t = np.arange(n)/sr
    sig = np.sin(TWOP*f*t) + 0.4*np.sin(TWOP*2*f*t)
    a_lp = 0.6
    y = np.empty(n); y[0] = (1-a_lp)*sig[0]
    y[1:] = a_lp*y[:-1] + (1-a_lp)*sig[1:]
    env = adsr(n, a, 0.05, 0.9, r, sus=0.9, sr=sr)
    return y * env * gain

def _resonator(x, f, bw, sr=SR):
    """2 极点谐振器 (带通)，中心 f、带宽 bw。"""
    r = math.exp(-math.pi*bw/sr)
    th = TWOP*f/sr
    c = 2*r*math.cos(th); g = (1-r*r)
    y = np.empty_like(x)
    y[0] = g*x[0]; y[1] = c*y[0] + g*x[1]
    # 向量化递推
    for i in range(2, len(x)):
        y[i] = c*y[i-1] - r*r*y[i-2] + g*x[i]
    return y

def vocal(f, dur, vowel='a', a=0.05, r=0.3, gain=1.0, sr=SR):
    """共振峰合成：谐波声源 → 频域 Lorentz 共振峰（啊/喔/呃）。"""
    n = int(dur*sr)
    t = np.arange(n)/sr
    # 声门源：前 14 次谐波 1/n
    src = np.zeros(n)
    for h in range(1, 15):
        src += (1.0/h) * np.sin(TWOP*h*f*t)
    src /= 14.0
    # 元音形式ant (中心频率, 带宽)
    FORM = {
        'a': [(800,140),(1150,150),(2900,200)],
        'o': [(500,120),(900,140),(2500,200)],
        'e': [(600,130),(1800,180),(2700,220)],
    }[vowel]
    # 频域共振峰强调 (Lorentz 峰)，向量化、无逐样本循环
    S = np.fft.rfft(src)
    fr = np.fft.rfftfreq(n, 1.0/sr)
    mag = np.ones_like(fr)
    for fc, bw in FORM:
        mag += 6.0 / (1.0 + ((fr - fc) / (bw/2.0))**2)
    mag /= max(mag.max(), 1.0)
    out = np.fft.irfft(S * mag, n)
    env = adsr(n, a, 0.1, 0.8, r, sus=0.8, sr=sr)
    return out[:n] * env * gain

def kick(t0s=0.0, dur=0.3, gain=1.0, sr=SR):
    n = int(dur*sr); t = np.arange(n)/sr
    f = 120*np.exp(-t*30) + 45
    sig = np.sin(TWOP*np.cumsum(f)/sr)
    env = np.exp(-t*12)
    return sig*env*gain

def snare(dur=0.2, gain=1.0, sr=SR):
    n = int(dur*sr); t = np.arange(n)/sr
    noise = np.random.randn(n)
    a_lp = 0.5
    y = np.empty(n); y[0]=(1-a_lp)*noise[0]
    y[1:] = a_lp*y[:-1]+(1-a_lp)*noise[1:]
    tone = np.sin(TWOP*180*t)*np.exp(-t*30)
    env = np.exp(-t*18)
    return (y*0.6 + tone*0.4)*env*gain

def hat(dur=0.05, gain=0.5, sr=SR):
    n = int(dur*sr); t = np.arange(n)/sr
    noise = np.random.randn(n)
    a_hp = 0.85
    y = np.empty(n); y[0] = (1-a_hp)*noise[0]
    y[1:] = a_hp*y[:-1]+(1-a_hp)*noise[1:]
    env = np.exp(-t*120)
    return y*env*gain

def taiko(dur=0.4, gain=1.0, sr=SR):
    n = int(dur*sr); t = np.arange(n)/sr
    f = 80*np.exp(-t*8) + 55
    sig = np.sin(TWOP*np.cumsum(f)/sr)
    noise = np.random.randn(n)*np.exp(-t*40)*0.4
    env = np.exp(-t*9)
    return (sig+noise)*env*gain

# ---------------------------------------------------------------------------
# 4. 混响：FFT 重叠相加卷积 (OLA)
# ---------------------------------------------------------------------------
def make_ir(sr=SR, dur=2.2, decay=2.5, damp=0.5, seed=0):
    rng = np.random.default_rng(seed)
    n = int(sr*dur)
    ir = rng.standard_normal(n)
    env = np.exp(-np.arange(n)/(sr*decay))
    ir *= env
    # 一极阻尼 (暗尾)
    a = damp; y = np.empty(n); y[0]=(1-a)*ir[0]
    y[1:] = a*y[:-1]+(1-a)*ir[1:]
    y /= (np.max(np.abs(y)) + 1e-9)
    return y

def conv_ola(x, ir, block=8192):
    M = len(ir)
    L = block + M - 1
    L2 = 1 << (int(np.ceil(np.log2(L))))
    H = np.fft.rfft(ir, L2)
    out = np.zeros(len(x) + L - 1)
    i = 0
    while i < len(x):
        seg = x[i:i+block]
        if len(seg) < block:
            seg = np.pad(seg, (0, block-len(seg)))
        X = np.fft.rfft(seg, L2)
        y = np.fft.irfft(X*H, L2)[:L]
        out[i:i+len(y)] += y
        i += block
    return out[:len(x)+M-1]

# ---------------------------------------------------------------------------
# 5. 声像 / 宽度
# ---------------------------------------------------------------------------
def pan(stereo, theta):
    """theta: -1(全左)..0(中)..1(全右)，等功率。"""
    th = (theta+1)/2 * (math.pi/2)
    l = math.cos(th); r = math.sin(th)
    return stereo * np.array([l, r])

# ---------------------------------------------------------------------------
# 6. 总谱事件表 + 渲染
# ---------------------------------------------------------------------------
events = []          # (instr, note_or_None, t0_sec, dur_sec, pan, gain, send, kw)
chords = []          # (freqs_list, t0, dur, pan, gain, send)

def add(instr, note, t0, dur, p=0.0, g=1.0, s=0.3, **kw):
    events.append((instr, note, t0, dur, p, g, s, kw))

def add_chord(freqs, t0, dur, p=0.0, g=0.5, s=0.5):
    chords.append((freqs, t0, dur, p, g, s))

# ---- 标志性动机 (16 个八分音符) ----
LONELY = ['D4','F4','G4','A4','C5','A4','G4','F4','D4',None,'G4','A4','C5','A4','G4','D4']
UNITY  = ['D4','E4','F#4','A4','B4','A4','F#4','E4','D4',None,'F#4','A4','B4','A4','F#4','D4']

def place_motif(seq, t0, bpm, eighth=None, instr='bell', p=0.0, g=0.7, s=0.4, detune_steps=0):
    beat = 60.0/bpm
    e = beat/2 if eighth is None else eighth
    for i, nt in enumerate(seq):
        if nt is None:
            continue
        f = freq(nt) * (2**(detune_steps/12.0))
        add(instr, nt if False else f, t0 + i*e, e*0.95, p=p, g=g, s=s)

# ===========================================================================
# 曲式编排（五幕 + 引子 + 尾声），严格对应总谱文档
# ===========================================================================
timeline = []
t = 0.0

# ---- 引子：静谧 (56 BPM, D 羽) ----
bpm = 56; beat = 60/bpm
# 钟磬孤独动机 (极慢, 稀疏)
place_motif(LONELY, t, bpm, instr='bell', p=0.0, g=0.7, s=0.7)
# 空五度 drone D+A
add_chord([freq('D3'), freq('A3')], t, 8*beat, p=0.0, g=0.35, s=0.7)
# 无词吟唱 啊 (宽)
add('vocal', freq('A3'), t+2*beat, 4*beat, p=-0.7, g=0.5, s=0.8, vowel='a')
add('vocal', freq('D4'), t+4*beat, 4*beat, p=0.7, g=0.5, s=0.8, vowel='o')
t += 16*beat

# ---- A 幕：萌动 (72 BPM, D 羽) ----
bpm = 72; beat = 60/bpm
place_motif(LONELY, t, bpm, instr='piano', p=0.0, g=0.7, s=0.3)
# 钢琴高八度回声
place_motif(LONELY, t+0.18, bpm, instr='piano', p=0.4, g=0.35, s=0.4, detune_steps=12)
# bass 脉冲 (二分音符) + 渐进律动 (2 对 3 种子)
for b in range(8):
    add('bass', freq('D2'), t+b*2*beat, 1.6*beat, p=0.0, g=0.7, s=0.1)
# 电子 pulse (三连，制造 2:3)
for b in range(12):
    add('bass', freq('A2')*2, t+b*beat, 0.2*beat, p=0.6, g=0.18, s=0.1)
# pad 和弦 Dm Bb F C
for k,(cf,ch) in enumerate([('D3',['D3','F3','A3']),('Bb2',['Bb2','D3','F3']),
                            ('F2',['F2','A2','C3']),('C3',['C3','E3','G3'])]):
    add_chord([freq(x) for x in ch], t+k*4*beat, 4*beat, p=0.0, g=0.28, s=0.4)
# 主歌人声 (孤独感) —— 占位 vocal 跟旋律
verse = ['D4','F4','G4','A4','C5','A4','G4','F4']
for i,nt in enumerate(verse):
    add('vocal', freq(nt), t+ (2+i*beat), beat*0.9, p=0.0, g=0.4, s=0.4, vowel='e')
t += 32*beat

# ---- B 幕：冲突 (96 BPM, D 羽→离调, 复合节奏 3:4) ----
bpm = 96; beat = 60/bpm
# 钟磬 3 连音 vs 电子 4 分 —— 错位撕扯
for k in range(4):
    base = t + k*4*beat
    for j in range(3):
        add('bell', freq('C5'), base + j*(4*beat/3), 0.3*beat, p=(-0.6 if j%2 else 0.6), g=0.5, s=0.5)
    for j in range(4):
        add('bass', freq('A2'), base + j*beat, 0.18*beat, p=0.5, g=0.22, s=0.1)
# 失谐弦乐簇 Dm + A#(三全音) 对位
add_chord([freq('D3'),freq('F3'),freq('A3'),freq('A#3')], t, 8*beat, p=-0.5, g=0.4, s=0.5)
add_chord([freq('G3'),freq('Bb3'),freq('D4')], t+8*beat, 8*beat, p=0.5, g=0.4, s=0.5)
# 动机加速变形 (失真钢琴)
place_motif(LONELY, t, bpm, instr='piano', p=0.2, g=0.5, s=0.3, detune_steps=1)
t += 32*beat

# ---- C 幕：爆发 (132 BPM, 转 D 宫, 全编制) ----
bpm = 132; beat = 60/bpm
# 鼓组：底鼓每拍、军鼓 2&4、镲八分；复合 4:3 碎拍
for b in range(16):
    add('kick', None, t+b*beat, 0.3, p=0.0, g=0.9, s=0.05)
    if b % 2 == 1:
        add('snare', None, t+b*beat, 0.2, p=0.0, g=0.6, s=0.2)
    for h in range(2):
        add('hat', None, t+b*beat+h*beat/2, 0.05, p=0.0, g=0.35, s=0.05)
# bass 推进 D2
for b in range(16):
    add('bass', freq('D2'), t+b*beat, 0.9*beat, p=0.0, g=0.8, s=0.05)
# 强力和弦 D-G-A-Bm-D (D 宫)
prog = [['D3','A3','D4'],['G2','D3','G3'],['A2','E3','A3'],['B2','F#3','B3'],['D3','A3','D4']]
for k,ch in enumerate(prog):
    add_chord([freq(x) for x in ch], t+k*4*beat, 4*beat, p=0.0, g=0.45, s=0.3)
# 标志动机 合一版 (主唱 + 钟磬齐鸣)，升八度一次
place_motif(UNITY, t, bpm, instr='piano', p=0.0, g=0.85, s=0.3)
place_motif(UNITY, t+8*beat, bpm, instr='bell', p=0.0, g=0.7, s=0.4, detune_steps=12)
# 中文主唱 (占位 vocal 跟副歌)
chorus = ['D4','E4','F#4','A4','B4','A4','F#4','E4']
for i,nt in enumerate(chorus):
    add('vocal', freq(nt), t+i*beat, beat*0.85, p=0.0, g=0.7, s=0.3, vowel='a')
for i,nt in enumerate(chorus):
    add('vocal', freq(nt), t+8*beat+i*beat, beat*0.85, p=0.2, g=0.7, s=0.3, vowel='a')
# 宽合唱 啊 (合一预兆)
add_chord([freq('D4'),freq('A4'),freq('D5')], t, 16*beat, p=0.0, g=0.3, s=0.7)
t += 32*beat

# ---- A' 幕：合一 (108 BPM 起, D 宫, 长尾) ----
bpm = 108; beat = 60/bpm
# 长混响合唱铺底
add_chord([freq('D4'),freq('F#4'),freq('A4'),freq('E5')], t, 24*beat, p=0.0, g=0.4, s=0.85)
# 钟磬 合一动机 (首尾呼应)
place_motif(UNITY, t, bpm, instr='bell', p=0.0, g=0.7, s=0.7)
place_motif(UNITY, t+8*beat, bpm, instr='bell', p=0.3, g=0.5, s=0.7, detune_steps=12)
# 主唱回归 + 吟唱宽
for i,nt in enumerate(chorus):
    add('vocal', freq(nt), t+i*beat, beat*0.9, p=-0.3, g=0.6, s=0.6, vowel='a')
add('vocal', freq('D4'), t+14*beat, 6*beat, p=-0.8, g=0.6, s=0.85, vowel='a')
add('vocal', freq('A4'), t+14*beat, 6*beat, p=0.8, g=0.6, s=0.85, vowel='o')
t += 24*beat

# ---- 尾声：同源 (自由渐隐) ----
bpm = 80; beat = 60/bpm
add_chord([freq('D4'),freq('A4'),freq('D5'),freq('E5')], t, 10*beat, p=0.0, g=0.4, s=0.9)
add('bell', freq('D5'), t, 4*beat, p=0.0, g=0.6, s=0.9)
add('vocal', freq('D4'), t+2*beat, 8*beat, p=0.0, g=0.5, s=0.9, vowel='a')
add('vocal', freq('A4'), t+2*beat, 8*beat, p=0.6, g=0.4, s=0.9, vowel='o')
CONTENT = t + 10*beat

# ===========================================================================
# 渲染
# ===========================================================================
def render():
    ir_dur = 2.2
    N = int((CONTENT + ir_dur) * SR)
    dry = np.zeros((N, 2), dtype=np.float64)
    send = np.zeros((N, 2), dtype=np.float64)

    def place(sig_mono, t0, p, g, s):
        i0 = int(t0*SR)
        n = len(sig_mono)
        if i0 + n > N:
            n = N - i0
            sig_mono = sig_mono[:n]
        if n <= 0:
            return
        # 干声 (等功率声像)
        th = (p+1)/2*(math.pi/2)
        l, r = math.cos(th)*g, math.sin(th)*g
        dry[i0:i0+n, 0] += sig_mono*l
        dry[i0:i0+n, 1] += sig_mono*r
        # 混响发送
        sl, sr2 = math.cos(th)*(g*s), math.sin(th)*(g*s)
        send[i0:i0+n, 0] += sig_mono*sl
        send[i0:i0+n, 1] += sig_mono*sr2

    # 单音事件
    for (instr, note, t0, dur, p, g, s, kw) in events:
        if instr in ('kick',):
            sig = kick(sr=SR); place(sig, t0, p, g, s); continue
        if instr == 'snare':
            sig = snare(sr=SR); place(sig, t0, p, g, s); continue
        if instr == 'hat':
            sig = hat(sr=SR); place(sig, t0, p, g, s); continue
        if note is None:
            continue
        if instr == 'bell':
            sig = fm_bell(note, dur, gain=1.0, **kw); place(sig, t0, p, g, s)
        elif instr == 'piano':
            sig = fm_pluck(note, dur, gain=1.0, **kw); place(sig, t0, p, g, s)
        elif instr == 'bass':
            sig = bass(note, dur, gain=1.0, **kw); place(sig, t0, p, g, s)
        elif instr == 'vocal':
            sig = vocal(note, dur, gain=1.0, **kw); place(sig, t0, p, g, s)
        else:
            sig = fm_bell(note, dur, gain=1.0); place(sig, t0, p, g, s)

    # 和弦事件 (多音叠加)
    for (freqs, t0, dur, p, g, s) in chords:
        sig = np.zeros(int(dur*SR))
        for f in freqs:
            sig += pad_saw(f, dur, gain=1.0)
        sig /= max(len(freqs),1)
        place(sig, t0, p, g, s)

    # 混响 (OLA 卷积)，左右用不同 IR 增宽
    ir_l = make_ir(SR, seed=1); ir_r = make_ir(SR, seed=2)
    wet_l = conv_ola(send[:,0], ir_l)
    wet_r = conv_ola(send[:,1], ir_r)
    wet = np.stack([wet_l[:N], wet_r[:N]], axis=1)
    out = dry + 0.9*wet

    # mid/side 轻微加宽干声
    mid = (out[:,0]+out[:,1])*0.5
    side = (out[:,0]-out[:,1])*0.5
    side *= 1.15
    out = np.stack([mid+side, mid-side], axis=1)

    # 母带：软削波 + 峰值归一 + TPDF 抖动
    out = np.tanh(out*1.5)/math.tanh(1.5)
    peak = np.max(np.abs(out))
    out = out * (0.96/peak) if peak > 0 else out
    dither = (np.random.rand(*out.shape) - 0.5)/32768.0
    out = np.clip(out + dither, -1.0, 1.0)
    return out

def write_wav(path, x, sr=SR):
    xi = (x*32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(xi.tobytes())

if __name__ == '__main__':
    random.seed(42); np.random.seed(42)
    print("渲染中… 内容时长 %.1f s" % CONTENT)
    out = render()
    write_wav('万籁同源.wav', out)
    print("已写出 万籁同源.wav  总采样 %d  时长 %.1f s" % (len(out), len(out)/SR))
    # 机器可读总谱
    score = {
        "title": "万籁同源", "author": "晨星", "bpm_plan": [56,72,96,132,108,80],
        "modes": {"lonely":"D 羽 (D F G A C)","unity":"D 宫 (D E F# A B)"},
        "signature_lonely": LONELY, "signature_unity": UNITY,
        "events": len(events), "chords": len(chords), "sample_rate": SR,
    }
    with open('score.json','w',encoding='utf-8') as f:
        json.dump(score, f, ensure_ascii=False, indent=2)
    print("已写出 score.json  事件数 %d  和弦段 %d" % (len(events), len(chords)))
