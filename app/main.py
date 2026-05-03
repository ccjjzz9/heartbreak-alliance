#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
呓 v2.0 — 原生 Windows 桌面应用
全部路径相对化 + 动态扫描，插入U盘即用。
"""

import sys, os, json, subprocess, threading, queue, time, re, shutil
from pathlib import Path
from datetime import datetime

# ====== 路径初始化 ======
if getattr(sys, 'frozen', False):
    ROOT_DIR = Path(sys.executable).resolve().parent
else:
    ROOT_DIR = Path(__file__).resolve().parent.parent

CONFIG_FILE = ROOT_DIR / "toolbox_config.json"
HF_MIRROR = "https://hf-mirror.com"

def load_config():
    if CONFIG_FILE.exists():
        try: return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except: pass
    return {}
def save_config(c): CONFIG_FILE.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")

CFG = load_config()

def _resolve(key, default_rel):
    saved = CFG.get(key, "")
    if saved:
        p = Path(saved)
        if p.exists(): return p
    p = ROOT_DIR / default_rel
    return p if p.exists() else (Path(saved) if saved else p)

LLAMA_FACTORY_DIR = _resolve("llama_factory", "LLaMA-Factory")
MODELS_DIR = _resolve("models_dir", "models")
AI_ENV_DIR = Path(CFG.get("ai_env_dir", str(ROOT_DIR / "ai-env")))
PYTORCH_ENV = ROOT_DIR / "pytorch-env" / "python.exe"
MEMOTRACE = ROOT_DIR / "MemoTrace.exe"
DATA_DIR = LLAMA_FACTORY_DIR / "data"

def _scan_output_dirs():
    """扫描所有可能包含 LoRA 适配器的 output 目录"""
    dirs = []
    for base in [LLAMA_FACTORY_DIR]:
        p = base / "output"
        if p.exists() and p not in dirs: dirs.append(p)
    for drv in ["D:", "C:"]:
        p = Path(drv) / "LLaMA-Factory" / "output"
        if p.exists() and p not in dirs: dirs.append(p)
    p = ROOT_DIR / "LLaMA-Factory" / "output"
    if p.exists() and p not in dirs: dirs.append(p)
    return dirs

def _scan_datasets():
    """动态扫描所有可用数据集"""
    datasets = []
    for d in [DATA_DIR, ROOT_DIR / "LLaMA-Factory" / "data",
              Path("D:") / "LLaMA-Factory" / "data"]:
        if d.exists():
            for f in d.glob("*.json"):
                if f.name != "dataset_info.json" and f.stem not in datasets:
                    datasets.append(f.stem)
    return sorted(datasets) if datasets else ["custom_train"]

def _scan_adapters():
    """动态扫描所有 LoRA 适配器"""
    results, seen = [], set()
    for base in _scan_output_dirs():
        if not base.exists(): continue
        for root, dirs, files in os.walk(str(base)):
            if "checkpoint-" in str(root): continue
            if any(f in files for f in ("adapter_model.safetensors","adapter_model.bin","adapter_config.json")):
                rel = Path(root).relative_to(base); name = str(rel).replace("\\","/")
                if name == ".": name = Path(root).name
                if name not in seen:
                    seen.add(name); results.append({"name":name,"path":str(Path(root))})
    return sorted(results, key=lambda x: x["name"])

if not CFG.get("llama_factory"):
    CFG.update({"llama_factory":str(LLAMA_FACTORY_DIR),"models_dir":str(MODELS_DIR)})
    save_config(CFG)

# ====== 全局状态 ======
training_proc = None; training_log_queue = queue.Queue()
infer_proc = None; chat_history = []; loaded_models = {}; active_model_name = None
FILTER_PATTERNS = ["[表情包]","[图片]","[视频]","[语音]","[文件]","[链接]","[小程序]","[红包]","[转账]","[位置]","[名片]","[收藏]","[合并转发]"]
SYSTEM_KEYWORDS = ["撤回了一条消息","朋友验证","你已添加了","以上是打招呼的内容","开启了朋友验证"]
_TQDM_RE = re.compile(r"^\s*\d+%\|[█▏▎▍▌▋▊▉ ╶─═]+\|\s*\d+/\d+\s*\[")

def _is_tqdm_line(line): return bool(_TQDM_RE.match(line))
def find_python():
    for p in [PYTORCH_ENV, Path(sys.executable)]:
        if p.exists(): return p
    return Path(sys.executable)

def detect_vram_gb():
    try:
        r = subprocess.run([str(find_python()),"-c","import torch; print(torch.cuda.get_device_properties(0).total_memory//(1024**3) if torch.cuda.is_available() else 0)"],
            capture_output=True, text=True, timeout=15)
        return int(r.stdout.strip()) if r.returncode == 0 else 0
    except: return 0

# ====== 隐藏子进程控制台窗口 ======
import subprocess as _sp
_CREATE_NO_WINDOW = 0x08000000
_ORIG_POPEN = _sp.Popen
_ORIG_RUN = _sp.run

def _no_console_popen(*args, **kwargs):
    kwargs.setdefault("creationflags", _CREATE_NO_WINDOW)
    return _ORIG_POPEN(*args, **kwargs)

def _no_console_run(*args, **kwargs):
    kwargs.setdefault("creationflags", _CREATE_NO_WINDOW)
    return _ORIG_RUN(*args, **kwargs)

_sp.Popen = _no_console_popen
_sp.run = _no_console_run

# ====== TKINTER ======
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox

def setup_ttk_style(root, c):
    style = ttk.Style(); style.theme_use("clam")
    style.configure(".", background=c["bg"], foreground=c["text"], fieldbackground=c["input_bg"])
    style.configure("TCombobox",
        fieldbackground=c["input_bg"], background=c["input_bg"],
        foreground=c["text"], arrowcolor=c["text2"], borderwidth=1,
        selectbackground=c["accent_bg"], selectforeground=c["text"])
    style.map("TCombobox",
        fieldbackground=[("readonly", c["input_bg"]), ("focus", c["input_focus"])],
        foreground=[("readonly", c["text"])])
    # 下拉弹出列表：深色背景 + 亮色文字
    root.option_add("*TCombobox*Listbox.background", c["card"])
    root.option_add("*TCombobox*Listbox.foreground", c["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", c["accent_bg"])
    root.option_add("*TCombobox*Listbox.selectForeground", c["accent"])
    root.option_add("*TCombobox*Listbox.font", ("Microsoft YaHei UI", 10))
    return style

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("呓 v2.0"); self.root.geometry("1160x760"); self.root.minsize(940, 620)
        self.root.configure(bg="#0d0d14")

        from tkinter import font
        avail = font.families()
        self._ff = next((n for n in ["Microsoft YaHei UI","Microsoft YaHei","PingFang SC","SimHei","TkDefaultFont"] if n in avail), "TkDefaultFont")

        # 科技感配色
        self.c = {
            "bg":"#08080f","sidebar":"#0a0a14","card":"#0f0f1c","card2":"#141424",
            "border":"#1a1a30","border_soft":"#181830","border_active":"#222244",
            "text":"#e8e8f0","text2":"#8888a0","text3":"#555570",
            "accent":"#00e4d4","accent_dim":"#00b8a8","accent_bg":"#061a18",
            "accent_glow":"#003330","red":"#f45050","green":"#44c8a0","yellow":"#e8c878","blue":"#5ea8f0",
            "input_bg":"#0a0a16","input_focus":"#121224",
        }
        self.fonts = {
            "body":(self._ff,10),"sm":(self._ff,9),"xs":(self._ff,8),
            "title":(self._ff,15,"bold"),"h3":(self._ff,11,"bold"),
            "mono":("Cascadia Code",10),"mono_sm":("Cascadia Code",9),
            "logo":(self._ff,22,"bold"),
        }
        self._glow_jobs = {}  # track active glow animations
        self.current_panel = None; self.toasts = []
        setup_ttk_style(self.root, self.c)
        self.build_sidebar(); self.build_content(); self.switch_panel("env")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        for n in list(loaded_models.keys()):
            try: loaded_models[n]["proc"].terminate()
            except: pass
        if infer_proc:
            try: infer_proc.terminate()
            except: pass
        if training_proc:
            try: training_proc.terminate()
            except: pass
        self.root.destroy()

    # ── 发光动画 ──
    def _glow_to(self, widget, attr, target, steps=6, interval=16):
        """平滑过渡颜色"""
        key = str(widget)+attr
        if key in self._glow_jobs:
            self.root.after_cancel(self._glow_jobs[key])
        try:
            cur = widget.cget(attr)
        except: return
        if cur == target: return
        def _step(s=0):
            if s >= steps:
                try: widget.configure(**{attr: target})
                except: pass
                self._glow_jobs.pop(key, None); return
            # 线性插值
            r1,g1,b1 = int(cur[1:3],16),int(cur[3:5],16),int(cur[5:7],16)
            r2,g2,b2 = int(target[1:3],16),int(target[3:5],16),int(target[5:7],16)
            r = int(r1+(r2-r1)*s/steps); g = int(g1+(g2-g1)*s/steps); b = int(b1+(b2-b1)*s/steps)
            try: widget.configure(**{attr: f"#{r:02x}{g:02x}{b:02x}"})
            except: pass
            self._glow_jobs[key] = self.root.after(interval, _step, s+1)
        _step()

    # ── 组件构建 ──
    def build_sidebar(self):
        sb = tk.Frame(self.root, bg=self.c["sidebar"], width=215)
        sb.pack(side="left", fill="y"); sb.pack_propagate(False)

        lf = tk.Frame(sb, bg=self.c["sidebar"]); lf.pack(fill="x", padx=18, pady=(22,16))
        tk.Label(lf, text="呓", font=self.fonts["logo"], fg=self.c["accent"], bg=self.c["sidebar"]).pack(side="left")
        tk.Label(lf, text="v2.0", font=self.fonts["xs"], fg=self.c["text3"], bg=self.c["sidebar"]).pack(side="right", pady=(12,0))

        # 细线分隔
        sep = tk.Canvas(sb, height=1, bg=self.c["sidebar"], highlightthickness=0)
        sep.create_line(14,0,201,0, fill=self.c["border"], width=1)
        sep.pack(fill="x", padx=0)

        nav = tk.Frame(sb, bg=self.c["sidebar"]); nav.pack(fill="both", expand=True, padx=10, pady=10)
        self.nb, self.ni, self._nav_bg = {}, {}, {}
        panels = [("env","🖥","环境检测"),("config","⚙","自动配置"),("wechat","💬","微信工具"),
                  ("convert","📝","聊天记录转换"),("finetune","🔥","模型微调"),("chat","🤖","模型对话"),("system","📊","系统状态")]
        for pid, icon, label in panels:
            cf = tk.Frame(nav, bg=self.c["sidebar"]); cf.pack(fill="x", pady=1)
            # 发光指示器
            ind = tk.Canvas(cf, width=4, height=32, bg=self.c["sidebar"], highlightthickness=0)
            ind.pack(side="left", padx=(0,6))
            self.ni[pid] = ind
            # 按钮
            btn = tk.Button(cf, text=f"{icon}  {label}", anchor="w", font=self.fonts["body"],
                bg=self.c["sidebar"], fg=self.c["text2"], bd=0, padx=12, pady=10,
                activebackground="#111122", activeforeground=self.c["text"],
                cursor="hand2", command=lambda p=pid: self.switch_panel(p))
            btn.pack(side="left", fill="x", expand=True)
            self._nav_bg[pid] = self.c["sidebar"]
            def make_hover(b, pid):
                b.bind("<Enter>", lambda e: self._glow_to(b, "background", "#111122"))
                b.bind("<Leave>", lambda e: self._glow_to(b, "background", self.c["accent_bg"] if self.current_panel==pid else self.c["sidebar"]))
            make_hover(btn, pid)
            self.nb[pid] = btn

        # 底部状态
        sf = tk.Frame(sb, bg="#060610"); sf.pack(fill="x", side="bottom")
        self.sdot = tk.Label(sf, text="●", font=self.fonts["xs"], fg=self.c["green"], bg="#060610")
        self.sdot.pack(side="left", padx=(18,4), pady=12)
        self.slabel = tk.Label(sf, text="就绪", font=self.fonts["sm"], fg=self.c["text3"], bg="#060610")
        self.slabel.pack(side="left", pady=12)

    def build_content(self):
        self.cf = tk.Frame(self.root, bg=self.c["bg"]); self.cf.pack(side="right", fill="both", expand=True)
        hdr = tk.Frame(self.cf, bg=self.c["bg"]); hdr.pack(fill="x", padx=32, pady=(20,0))
        self.ptitle = tk.Label(hdr, text="", font=self.fonts["title"], fg=self.c["text"], bg=self.c["bg"])
        self.ptitle.pack(side="left")
        self.aframe = tk.Frame(hdr, bg=self.c["bg"]); self.aframe.pack(side="right")
        # accent 细线
        sep2 = tk.Canvas(self.cf, height=2, bg=self.c["bg"], highlightthickness=0)
        sep2.create_line(0,0,80,0, fill=self.c["accent"], width=2)
        sep2.create_line(80,0,9999,0, fill=self.c["border"], width=1)
        sep2.pack(fill="x", padx=32, pady=(12,0))
        self.canvas = tk.Canvas(self.cf, bg=self.c["bg"], highlightthickness=0)
        sb2 = tk.Scrollbar(self.cf, orient="vertical", command=self.canvas.yview)
        self.pc = tk.Frame(self.canvas, bg=self.c["bg"])
        self.pc.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0,0), window=self.pc, anchor="nw", tags="inner")
        self.canvas.configure(yscrollcommand=sb2.set)
        self.canvas.pack(side="left", fill="both", expand=True, padx=(32,0), pady=16)
        sb2.pack(side="right", fill="y", pady=16)
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-1*(e.delta//120),"units"))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig("inner", width=e.width))
        # Canvas 内嵌 Frame 的焦点修复
        self.canvas.bind("<Button-1>", lambda e: self.root.focus_get() or None)

    def _ft_refresh_ds(self):
        """刷新微调面板的数据集下拉列表"""
        ds = _scan_datasets()
        if hasattr(self, '_ft_ds_combo') and self._ft_ds_combo.winfo_exists():
            self._ft_ds_combo.configure(values=ds)
            if ds and self.ft_vars.get("ft_dataset"):
                cur = self.ft_vars["ft_dataset"].get()
                if cur not in ds and ds:
                    self.ft_vars["ft_dataset"].set(ds[0])
        self.toast(f"已刷新: {len(ds)} 个数据集", "info")

    # ── 面板切换过渡 ──
    def _panel_transition(self, pid, step=0):
        """渐变切换面板：快速闪烁效果模拟过渡"""
        if step == 0:
            self.pc.configure(bg="#141420")
            self.root.after(40, lambda: self._panel_transition(pid, 1))
        elif step == 1:
            self.pc.configure(bg="#101018")
            self.root.after(40, lambda: self._panel_transition(pid, 2))
        elif step == 2:
            self.pc.configure(bg=self.c["bg"])
            self._render_panel(pid)

    def _render_panel(self, pid):
        for w in self.pc.winfo_children(): w.destroy()
        for w in self.aframe.winfo_children(): w.destroy()
        titles = {"env":"环境检测","config":"自动配置","wechat":"微信工具","convert":"聊天记录转换",
                   "finetune":"模型微调","chat":"模型对话","system":"系统状态"}
        self.ptitle.configure(text=titles.get(pid,pid))
        getattr(self, f"panel_{pid}", lambda: None)()
        # 强制更新 Canvas 滚动区域
        self.pc.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.yview_moveto(0)

    def switch_panel(self, pid):
        self.current_panel = pid
        for bid, btn in self.nb.items():
            if bid == pid:
                self._glow_to(btn, "background", self.c["accent_bg"])
                btn.configure(fg=self.c["accent"])
                c = self.ni[bid]; c.delete("all")
                c.create_rectangle(0,0,4,32, fill=self.c["accent"], outline="")
            else:
                self._glow_to(btn, "background", self.c["sidebar"])
                btn.configure(fg=self.c["text2"])
                self.ni[bid].delete("all")
        self._panel_transition(pid)

    def set_status(self, text, color=None):
        self.slabel.configure(text=text); self.sdot.configure(fg=color or self.c["green"])

    # ── Toast ──
    def toast(self, text, tp="info"):
        cl = {"info":self.c["accent"],"success":self.c["green"],"error":self.c["red"]}.get(tp,self.c["text2"])
        t = tk.Label(self.cf, text=f"  {text}  ", font=self.fonts["sm"], fg=cl, bg="#1c1c28", padx=12, pady=6)
        t.place(relx=0.98, y=50+len(self.toasts)*38, anchor="ne"); self.toasts.append(t)
        self.root.after(3000, lambda t0=t: (t0.place_forget(), self.toasts.remove(t0)) if t0 in self.toasts else None)

    # ── 组件工厂 ──
    def card(self, parent, title=None):
        # 外层带 accent 顶线的卡片
        outer = tk.Frame(parent, bg=self.c["bg"]); outer.pack(fill="x", pady=6)
        # accent 顶部细线
        top = tk.Canvas(outer, height=2, bg=self.c["bg"], highlightthickness=0)
        top.create_line(0,1,60,1, fill=self.c["accent"], width=2)
        top.create_line(60,1,9999,1, fill=self.c["border"], width=1)
        top.pack(fill="x")
        f = tk.Frame(outer, bg=self.c["card"], bd=0, highlightthickness=1, highlightbackground=self.c["border"])
        f.pack(fill="both", expand=True, ipadx=18, ipady=14)
        if title:
            tk.Label(f, text=title, font=self.fonts["h3"], fg=self.c["text"], bg=self.c["card"]).pack(anchor="w", pady=(0,10))
        inner = tk.Frame(f, bg=self.c["card"]); inner.pack(fill="both", expand=True, padx=4, pady=4); return inner

    def label(self, p, t): return tk.Label(p, text=t, font=self.fonts["sm"], fg=self.c["text2"], bg=p["bg"])

    def select(self, p, options, default=""):
        v = tk.StringVar(value=default or (options[0] if options else ""))
        s = ttk.Combobox(p, textvariable=v, values=options, state="readonly", font=self.fonts["body"])
        return v, s

    def textarea(self, p, default="", h=4):
        f = tk.Frame(p, bg="#1a1a30", highlightthickness=2, highlightbackground=self.c["accent_dim"])
        t = tk.Text(f, font=self.fonts["body"], bg="#111118", fg=self.c["text"],
            insertbackground=self.c["accent"], relief="flat", bd=0, height=h, width=60, wrap="word",
            padx=10, pady=8, selectbackground=self.c["accent_bg"], selectforeground=self.c["text"])
        t.insert("1.0", default); t.pack(fill="both", expand=True, padx=3, pady=3)
        def on_focus_in(e): f.configure(highlightbackground=self.c["accent"])
        def on_focus_out(e): f.configure(highlightbackground=self.c["accent_dim"])
        t.bind("<FocusIn>", on_focus_in); t.bind("<FocusOut>", on_focus_out)
        # 确保可以被点击和输入
        t.configure(state="normal", takefocus=True)
        return f, t

    def entry(self, p, default="", w=45):
        v = tk.StringVar(value=default)
        f = tk.Frame(p, bg="#1a1a30", highlightthickness=1, highlightbackground=self.c["accent_dim"])
        e = tk.Entry(f, textvariable=v, font=self.fonts["body"], bg="#111118", fg=self.c["text"],
            insertbackground=self.c["accent"], relief="flat", bd=0,
            selectbackground=self.c["accent_bg"], selectforeground=self.c["text"])
        e.pack(fill="x", ipady=4, padx=8, pady=2)
        def on_focus_in(ev): f.configure(highlightbackground=self.c["accent"])
        def on_focus_out(ev): f.configure(highlightbackground=self.c["accent_dim"])
        e.bind("<FocusIn>", on_focus_in); e.bind("<FocusOut>", on_focus_out)
        return v, e

    def _make_entry(self, parent, var, **kw):
        """单行输入框工厂（用于嵌入已有布局）"""
        e = tk.Entry(parent, textvariable=var, font=self.fonts["body"], bg=self.c["card2"], fg=self.c["text"],
            insertbackground=self.c["accent"], relief="flat", bd=0,
            selectbackground=self.c["accent_bg"], selectforeground=self.c["text"], **kw)
        return e

    def btn(self, p, text, cmd, accent=False, danger=False, small=False):
        if accent:
            bg, hbg = self.c["accent"], self.c["accent_dim"]
            fg = "#000"
        elif danger:
            bg, hbg = "#2d1518", "#3d1a1e"
            fg = self.c["red"]
        else:
            bg, hbg = self.c["card2"], "#1e1e34"
            fg = self.c["text"]
        pd = (8,4) if small else (15,8)
        b = tk.Button(p, text=text, command=cmd, font=self.fonts["sm"] if small else self.fonts["body"],
            bg=bg, fg=fg, bd=0, padx=pd[0], pady=pd[1], cursor="hand2",
            activebackground=hbg, activeforeground=fg)
        def on_enter(): self._glow_to(b, "background", hbg)
        def on_leave(): self._glow_to(b, "background", bg)
        b.bind("<Enter>", lambda e: on_enter())
        b.bind("<Leave>", lambda e: on_leave())
        return b

    def console(self, p, h=12):
        t = scrolledtext.ScrolledText(p, font=self.fonts["mono_sm"], bg=self.c["card2"], fg=self.c["text2"],
            wrap="word", height=h, relief="flat", bd=0, highlightthickness=1, highlightbackground=self.c["border_soft"])
        return t

    def grid2(self, p):
        f = tk.Frame(p, bg=p["bg"]); f.columnconfigure(0,weight=1); f.columnconfigure(1,weight=1); return f

    # ====== 面板: 环境检测 ======
    def panel_env(self):
        c = self.card(self.pc, "组件检测结果")
        self.et = self.console(c, 18); self.et.pack(fill="x")
        self.et.insert("end", "点击「开始检测」扫描环境...\n","dim")
        for tag, fg in [("ok",self.c["green"]),("err",self.c["red"]),("dim",self.c["text3"])]:
            self.et.tag_configure(tag, foreground=fg)
        self.btn(self.aframe, "开始检测", self._env_run, accent=True).pack(side="left", padx=4)

    def _env_run(self):
        self.set_status("检测中...", self.c["yellow"]); self.et.delete("1.0","end"); py = find_python()
        def add(n,d,ok):
            self.et.insert("end",f"  {n}\n","ok" if ok else "err")
            self.et.insert("end",f"    {d}\n\n","dim")
        add("Python",f"v{sys.version.split()[0]} ({py})",True)
        r=subprocess.run([str(py),"-c","import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"],
            capture_output=True, text=True, timeout=60, env={**os.environ,"HF_ENDPOINT":HF_MIRROR})
        add("PyTorch+CUDA",r.stdout.strip() if r.returncode==0 else r.stderr.strip()[:200],r.returncode==0)
        r=subprocess.run([str(py),"-c","import transformers; print(f'transformers={transformers.__version__}')"],capture_output=True,text=True,timeout=30)
        add("Transformers",r.stdout.strip() if r.returncode==0 else "未安装",r.returncode==0)
        lf=(LLAMA_FACTORY_DIR/"src"/"llamafactory").exists(); add("LLaMA-Factory",str(LLAMA_FACTORY_DIR),lf)
        gm=(MODELS_DIR/"glm-4-9b-chat"/"config.json").exists(); add("GLM-4-9B",str(MODELS_DIR/"glm-4-9b-chat"),gm)
        mt=MEMOTRACE.exists(); add("MemoTrace",f"{MEMOTRACE} ({MEMOTRACE.stat().st_size/(1024*1024):.1f}MB)" if mt else "未找到",mt)
        self.set_status("检测完成"); self.toast("检测完成","success")

    # ====== 面板: 自动配置 ======
    def panel_config(self):
        c = self.card(self.pc, "一键自动配置")
        self.cgt = self.console(c, 16); self.cgt.pack(fill="x"); self.cgt.insert("end","点击按钮开始配置...\n","dim")
        self.cgt.tag_configure("dim", foreground=self.c["text3"])
        self.btn(self.aframe, "一键配置全部", self._cfg_run, accent=True).pack(side="left", padx=4)
        self.btn(self.aframe, "仅复制文件", self._cfg_copy).pack(side="left", padx=4)
        self.btn(self.aframe, "仅安装依赖", self._cfg_deps).pack(side="left", padx=4)

    def _cglog(self, m):
        self.cgt.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {m}\n"); self.cgt.see("end")

    def _cfg_run(self): threading.Thread(target=self._cfg_full, daemon=True).start()
    def _cfg_copy(self): self._cglog("LLaMA-Factory 已在当前目录"); self.toast("文件已就绪","info")
    def _cfg_deps(self): threading.Thread(target=self._cfg_full, daemon=True).start()

    def _cfg_full(self):
        self.root.after(0, lambda: self.set_status("配置中...", self.c["yellow"]))
        py = find_python(); ae = ROOT_DIR / "ai-env"; aep = ae / "Scripts" / "python.exe"
        if not aep.exists():
            self.root.after(0, lambda: self._cglog("创建 venv..."))
            subprocess.run([str(py),"-m","venv",str(ae)], capture_output=True, timeout=300)
        self.root.after(0, lambda: self._cglog("LLaMA-Factory: "+str(LLAMA_FACTORY_DIR)))
        if aep.exists():
            for lb, pk in [("基础包",["torch","transformers","accelerate","peft","huggingface_hub"]),
                           ("训练框架",["datasets","einops","bitsandbytes","scipy","sentencepiece","protobuf"])]:
                self.root.after(0, lambda l=lb: self._cglog(f"安装 {l}..."))
                subprocess.run([str(aep),"-m","pip","install"]+pk, capture_output=True, timeout=1800,
                    env={**os.environ,"HF_ENDPOINT":HF_MIRROR})
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        dsi = DATA_DIR / "dataset_info.json"
        ds = json.loads(dsi.read_text(encoding="utf-8")) if dsi.exists() else {}
        ds["custom_train"] = {"file_name":"custom_train.json","formatting":"sharegpt",
            "columns":{"messages":"messages"},"tags":{"role_tag":"role","content_tag":"content",
            "user_tag":"user","assistant_tag":"assistant","system_tag":"system"}}
        dsi.write_text(json.dumps(ds, ensure_ascii=False, indent=2), encoding="utf-8")
        self.root.after(0, lambda: self._cglog("配置完成!"))
        self.root.after(0, lambda: self.set_status("就绪")); self.root.after(0, lambda: self.toast("配置完成","success"))

    # ====== 面板: 微信工具 ======
    def panel_wechat(self):
        c = self.card(self.pc, "MemoTrace 微信记录提取")
        ok = MEMOTRACE.exists()
        tk.Label(c, text=f"MemoTrace: {'✓ 已找到' if ok else '✗ 未找到'}", font=self.fonts["body"],
            fg=self.c["green"] if ok else self.c["red"], bg=self.c["card"]).pack(anchor="w",pady=(0,6))
        tk.Label(c, text=str(MEMOTRACE), font=self.fonts["sm"], fg=self.c["text2"], bg=self.c["card"]).pack(anchor="w")
        self.btn(self.aframe, "启动 MemoTrace", self._mt_launch, accent=True).pack(side="left", padx=4)
        self.btn(self.aframe, "检查微信状态", self._wx_check).pack(side="left", padx=4)

    def _mt_launch(self):
        try: subprocess.Popen([str(MEMOTRACE)], cwd=str(MEMOTRACE.parent)); self.toast("已启动","success")
        except Exception as e: messagebox.showerror("错误",str(e))
    def _wx_check(self):
        r = subprocess.run(["tasklist","/FI","IMAGENAME eq WeChat.exe"], capture_output=True, text=True)
        messagebox.showinfo("微信状态",f"微信进程: {'运行中' if 'WeChat.exe' in r.stdout else '未运行'}")

    # ====== 面板: 聊天记录转换 ======
    def panel_convert(self):
        c = self.card(self.pc, "聊天记录转换")
        self.label(c, "聊天记录文件 (.txt)").pack(anchor="w")
        ff = tk.Frame(c, bg=self.c["card"]); ff.pack(fill="x", pady=(2,10))
        self.cf_var = tk.StringVar()
        tk.Entry(ff, textvariable=self.cf_var, font=self.fonts["body"], bg=self.c["card2"],
            fg=self.c["text"], width=45, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=self.c["border_soft"]).pack(side="left", fill="x", expand=True)
        self.btn(ff, "浏览...", self._cv_browse, small=True).pack(side="left", padx=6)

        # 发送者名称
        row1 = tk.Frame(c, bg=self.c["card"]); row1.pack(fill="x", pady=3)
        self.label(row1, "发送者名称 (此人为 user)").pack(anchor="w")
        sf = tk.Frame(row1, bg=self.c["card"]); sf.pack(fill="x")
        self.cv_sender = tk.StringVar()
        tk.Entry(sf, textvariable=self.cv_sender, font=self.fonts["body"], bg=self.c["card2"],
            fg=self.c["text"], relief="flat", bd=0, highlightthickness=1,
            highlightbackground=self.c["border_soft"]).pack(side="left", fill="x", expand=True)
        self.btn(sf, "自动检测", self._cv_detect_sender, small=True).pack(side="left", padx=6)

        # 数据集选择 + 新建
        ds_list = _scan_datasets()
        row2 = tk.Frame(c, bg=self.c["card"]); row2.pack(fill="x", pady=3)
        self.label(row2, "保存到数据集").pack(anchor="w")
        dsf = tk.Frame(row2, bg=self.c["card"]); dsf.pack(fill="x")
        self.cv_dataset = tk.StringVar(value="custom_train")
        cb = ttk.Combobox(dsf, textvariable=self.cv_dataset, values=ds_list, font=self.fonts["body"])
        cb.pack(side="left", fill="x", expand=True)
        self._cv_combo = cb
        self.btn(dsf, "+ 新建", self._cv_new_dataset, small=True).pack(side="left", padx=6)
        self.btn(dsf, "↻", self._cv_refresh_ds, small=True).pack(side="left", padx=2)

        # 人设输入区 — 用独立卡片包裹确保可见
        pc = self.card(self.pc, "人设 (System Prompt)")
        tk.Label(pc, text="这段文字会放在每个训练样本开头，定义 AI 的性格和回复风格",
            font=self.fonts["xs"], fg=self.c["text3"], bg=self.c["card"]).pack(anchor="w", pady=(0,6))
        _, self.cv_persona = self.textarea(pc, "你是一个友好的AI助手。", h=4)

        # 手动问答对
        qc = self.card(self.pc, "手动问答对（可选）")
        tk.Label(qc, text="格式: user: 问题\\nassistant: 回答\\n支持多轮，中英文冒号均可",
            font=self.fonts["xs"], fg=self.c["text3"], bg=self.c["card"]).pack(anchor="w", pady=(0,6))
        _, self.cv_qa = self.textarea(qc, "", h=5)

        self.cv_prev = self.console(c, 10); self.cv_prev.pack(fill="x", pady=(8,0))
        for t,cmd,ac in [("预览",self._cv_preview,True),("转换并保存",self._cv_run,True),("检测序列长度",self._cv_analyze,False)]:
            self.btn(self.aframe, t, cmd, accent=ac).pack(side="left", padx=4)

    def _cv_browse(self):
        f = filedialog.askopenfilename(filetypes=[("Text","*.txt"),("All","*.*")])
        if f: self.cf_var.set(f)

    def _cv_detect_sender(self):
        c = self._cv_get_content()
        if not c: return
        lines = c.splitlines()[:200]; counts = {}
        pat = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2}\s+(.+)$")
        for line in lines:
            m = pat.match(line.strip())
            if m: n = m.group(1).strip(); counts[n] = counts.get(n,0)+1
        if counts:
            top = max(counts, key=counts.get)
            self.cv_sender.set(top); self.toast(f"检测到: {top}","info")

    def _cv_new_dataset(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("新建数据集"); dlg.geometry("360x160"); dlg.resizable(False, False)
        dlg.configure(bg=self.c["card"])
        dlg.transient(self.root); dlg.grab_set()
        # 居中
        dlg.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width()-360)//2
        y = self.root.winfo_y() + (self.root.winfo_height()-160)//2
        dlg.geometry(f"+{x}+{y}")

        tk.Label(dlg, text="输入数据集名称（不含 .json 后缀）", font=self.fonts["body"],
            fg=self.c["text"], bg=self.c["card"]).pack(pady=(20,10))
        v = tk.StringVar()
        e = tk.Entry(dlg, textvariable=v, font=self.fonts["body"], bg=self.c["card2"], fg=self.c["text"],
            insertbackground=self.c["text"], relief="flat", bd=0, highlightthickness=1,
            highlightbackground=self.c["border_soft"], highlightcolor=self.c["accent"], width=30)
        e.pack(padx=20, pady=(0,16)); e.focus()

        def submit():
            n = v.get().strip()
            if n:
                ds = _scan_datasets()
                if n not in ds: ds.append(n)
                self._cv_combo.configure(values=sorted(ds))
                self.cv_dataset.set(n); self.toast(f"已创建: {n}","success")
            dlg.destroy()

        bf = tk.Frame(dlg, bg=self.c["card"]); bf.pack()
        self.btn(bf, "取消", dlg.destroy, small=True).pack(side="left", padx=4)
        self.btn(bf, "创建", submit, accent=True, small=True).pack(side="left", padx=4)
        e.bind("<Return>", lambda ev: submit())
        dlg.bind("<Escape>", lambda ev: dlg.destroy())

    def _cv_refresh_ds(self):
        ds = _scan_datasets()
        self._cv_combo.configure(values=ds)
        cur = self.cv_dataset.get()
        if cur not in ds and ds: self.cv_dataset.set(ds[0])
        self.toast(f"已刷新: {len(ds)} 个","info")

    def _cv_get_content(self):
        fp = self.cf_var.get()
        return Path(fp).read_text(encoding="utf-8", errors="ignore") if fp and Path(fp).exists() else None

    def _cv_preview(self):
        data = self._cv_get_all_data()
        if not data: return
        self.cv_prev.delete("1.0","end")
        self.cv_prev.insert("end", f"共 {len(data)} 个训练样本\n\n")
        for i, item in enumerate(data[:3]):
            self.cv_prev.insert("end", f"── 样本 {i+1} ──\n")
            for m in item["messages"][:6]: self.cv_prev.insert("end", f"  [{m['role']}] {m['content'][:200]}\n")
            self.cv_prev.insert("end", "\n")
        self.toast(f"预览: {len(data)} 个样本","info")

    def _cv_get_all_data(self):
        """获取全部训练数据：聊天文件转换 + 手动问答对"""
        persona = self.cv_persona.get("1.0","end-1c").strip()
        data = []
        c = self._cv_get_content()
        if c:
            data = self._convert(c)
        # 合并手动问答对
        qa = self.cv_qa.get("1.0","end-1c").strip()
        if qa:
            for q,a in self._parse_qa(qa):
                if q and a:
                    data.append({"messages":[
                        {"role":"system","content":persona},
                        {"role":"user","content":q},
                        {"role":"assistant","content":a}]})
        return data

    def _cv_run(self):
        data = self._cv_get_all_data()
        if not data:
            return messagebox.showerror("错误","没有可保存的数据。请选择聊天记录文件或添加手动问答对。")
        self.set_status("转换中...", self.c["yellow"])
        ds = (self.cv_dataset.get().strip() or "custom_train")
        if not ds.endswith(".json"): ds += ".json"
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR/ds).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self._cv_register(ds.replace(".json",""))
        # 保存人设提示词到数据目录
        if data:
            sp = data[0]["messages"][0]["content"]
            (DATA_DIR/f"{ds.replace('.json','')}_prompt.txt").write_text(sp, encoding="utf-8")
        self.cv_prev.insert("end", f"\n✅ 转换完成! {len(data)} 个样本 → {ds}\n")
        self.set_status("转换完成"); self.toast(f"已保存到 {ds}","success")
        # 刷新数据集下拉列表
        self._cv_combo.configure(values=_scan_datasets())

    def _cv_analyze(self):
        data = self._cv_get_all_data()
        if not data: return
        self.set_status("分析中...", self.c["yellow"])
        self.cv_prev.delete("1.0","end"); self.cv_prev.insert("end","正在分析...\n")
        threading.Thread(target=self._cv_analyze_bg, args=(data,), daemon=True).start()

    def _cv_analyze_bg(self, data):
        tmp = LLAMA_FACTORY_DIR / "_az.json"; tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        code = f'''import json; from transformers import AutoTokenizer
data=json.load(open(r"{tmp}","r",encoding="utf-8"))
tok=AutoTokenizer.from_pretrained(r"{MODELS_DIR/'glm-4-9b-chat'}",trust_remote_code=True)
lens=[len(tok.encode(tok.apply_chat_template(d["messages"],tokenize=False,add_generation_prompt=False))) for d in data]; lens.sort()
n=len(lens); import json; print(json.dumps(dict(samples=n,min=lens[0],max=lens[-1],p50=lens[n//2],p80=lens[int(n*.8)],p90=lens[int(n*.9)],p95=lens[int(n*.95)])))'''
        sp = LLAMA_FACTORY_DIR / "_az.py"; sp.write_text(code, encoding="utf-8")
        r = subprocess.run([str(find_python()), str(sp)], capture_output=True, text=True, timeout=120,
            env={**os.environ,"HF_ENDPOINT":HF_MIRROR})
        if r.returncode == 0:
            st = json.loads(r.stdout.strip()); p95 = st["p95"]
            rec = 512 if p95<=512 else (768 if p95<=768 else (1024 if p95<=1024 else 2048))
            def show():
                self.cv_prev.insert("end",f"\n📊 {st['samples']}样本 | P50:{st['p50']} | P80:{st['p80']} | P90:{st['p90']} | P95:{st['p95']}\n✅ 推荐 cutoff_len={rec}\n")
                self.set_status("分析完成")
            self.root.after(0, show)

    # ── 转换核心逻辑 ──
    def _convert(self, content):
        self._cv_auto_sender = None  # 每次转换重置自动检测
        msgs = self._parse_chat(content); sender = self.cv_sender.get().strip()
        persona = self.cv_persona.get("1.0","end-1c").strip()
        all_s = set()
        for m in msgs:
            n,_,_ = self._extract_sender(m["content"].strip(), None)
            if n: all_s.add(n)
        cl = []
        for m in msgs:
            c = m["content"].strip()
            if self._filter_msg(c): continue
            n, bd, _ = self._extract_sender(c, all_s)
            fc = self._clean_content(bd if n and bd else c)
            if not fc: continue
            # 角色分配：sender=user，非sender=assistant。未指定sender时自动取第一个人为user
            role = "user"
            if n:
                if sender:
                    role = "user" if n == sender else "assistant"
                else:
                    if self._cv_auto_sender is None:
                        self._cv_auto_sender = n  # 第一个人自动成为user
                    role = "user" if n == self._cv_auto_sender else "assistant"
            cl.append({"role":role, "content":fc, "time":m.get("time","")})
        if not cl: return []
        mg = []
        for m in cl:
            if mg and mg[-1]["role"]==m["role"]: mg[-1]["content"]+="\n"+m["content"]
            else: mg.append(m)
        rs, cur, last = [], [], None
        for m in mg:
            if cur and m["role"]==last: rs.append(cur); cur=[]
            cur.append(m); last=m["role"]
        if cur: rs.append(cur)
        data = []
        for seg in rs:
            # assistant 开头 → 补 user 消息使格式合法。用短问候语尽量少干扰训练
            if seg[0]["role"] == "assistant":
                seg.insert(0, {"role": "user", "content": "在吗"})
            while seg and seg[-1]["role"]!="assistant": seg.pop()
            if len(seg)<2: continue
            start=0
            while start<len(seg)-1:
                w=seg[start:start+12]
                while w and w[0]["role"]!="user": start+=1; w=seg[start:start+12]
                if len(w)<2: break
                if w[-1]["role"]!="assistant": w.pop()
                if len(w)>=2:
                    data.append({"messages":[{"role":"system","content":persona}]+[
                        {"role":m["role"],"content":m["content"]} for m in w]})
                start+=2
        return data

    def _parse_chat(self, text):
        pat = re.compile(r"^(\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2})\s+(.+)$")
        msgs, cur = [], None
        for line in text.splitlines():
            line = line.rstrip("\n\r")
            if not line: continue
            m = pat.match(line)
            if m and cur: msgs.append(cur); cur={"time":m.group(1),"content":m.group(2)}
            elif m: cur={"time":m.group(1),"content":m.group(2)}
            elif cur: cur["content"]+="\n"+line
        if cur: msgs.append(cur)
        return msgs

    def _filter_msg(self, c):
        c=c.strip()
        if not c: return True
        for p in FILTER_PATTERNS:
            if c==p or c.startswith(p): return True
        for kw in SYSTEM_KEYWORDS:
            if kw in c: return True
        return False

    def _extract_sender(self, c, known):
        for sep in (': ',':','\n',' '):
            i=c.find(sep)
            if i>0:
                n=c[:i].strip()
                if n and (not known or n in known): return n,c[i+len(sep):].strip(),sep
        return None,c,None

    def _clean_content(self, t):
        for p in FILTER_PATTERNS: t=t.replace(p,"")
        return re.sub(r"\n{3,}","\n\n", re.sub(r" {2,}"," ",t)).strip()

    def _parse_qa(self, raw):
        pairs, role, content = [], None, []
        for line in raw.split("\n"):
            m = re.match(r"^(user|assistant)\s*[：:]\s*(.*)$", line, re.I)
            if m:
                if role and content:
                    t = "\n".join(content).strip()
                    if role=="user": pairs.append([t,""])
                    elif pairs and not pairs[-1][1]: pairs[-1][1]=t
                role=m.group(1).lower(); content=[m.group(2)]
            elif role and line.strip(): content.append(line.strip())
        if role and content:
            t="\n".join(content).strip()
            if role=="user": pairs.append([t,""])
            elif pairs and not pairs[-1][1]: pairs[-1][1]=t
        return [(q,a) for q,a in pairs if q and a]

    def _cv_register(self, name):
        p = DATA_DIR / "dataset_info.json"
        ds = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        if name not in ds:
            ds[name] = {"file_name":f"{name}.json","formatting":"sharegpt",
                "columns":{"messages":"messages"},"tags":{"role_tag":"role","content_tag":"content",
                "user_tag":"user","assistant_tag":"assistant","system_tag":"system"}}
            p.write_text(json.dumps(ds, ensure_ascii=False, indent=2), encoding="utf-8")

    # ====== 面板: 模型微调 ======
    def _ft_render_fields(self, parent, fields):
        """在 parent (2列 grid) 中渲染字段, 返回 (vars_dict, widgets_dict)"""
        vd, wd = {}, {}
        for i, (label, key, default, ftype, options) in enumerate(fields):
            row, col = i // 2, i % 2
            ff = tk.Frame(parent, bg=self.c["card"])
            ff.grid(row=row, column=col, sticky="ew", padx=(0, 8), pady=3)
            self.label(ff, label).pack(anchor="w")
            var = tk.StringVar(value=str(default))
            if ftype == "combo":
                w = ttk.Combobox(ff, textvariable=var, values=list(options),
                                state="readonly", font=self.fonts["body"])
            elif ftype == "bool":
                w = ttk.Combobox(ff, textvariable=var, values=["True", "False"],
                                state="readonly", font=self.fonts["body"], width=10)
            else:
                w = tk.Entry(ff, textvariable=var, font=self.fonts["body"],
                            bg=self.c["card2"], fg=self.c["text"], relief="flat",
                            bd=0, highlightthickness=1,
                            highlightbackground=self.c["border_soft"])
            w.pack(fill="x")
            vd[key] = var
            wd[key] = w
        return vd, wd

    def panel_finetune(self):
        ds_list = _scan_datasets()
        vram = detect_vram_gb()
        if vram >= 24:
            cutoff, batch, ga, qbit = 2048, 4, 4, "none"
            dq = "False"
        elif vram >= 16:
            cutoff, batch, ga, qbit = 1024, 2, 4, "4"
            dq = "False"
        else:
            cutoff, batch, ga, qbit = 512, 1, 8, "4"
            dq = "False"
        ds0 = ds_list[0] if ds_list else "custom_train"
        TEMPLATES = ["glm4", "default", "qwen", "qwen3", "llama3", "baichuan2", "chatglm3", "yi", "mistral", "gemma"]
        STAGES = ["sft", "dpo", "rm", "ppo", "kto", "pt"]
        FT_TYPES = ["lora", "freeze", "full", "oft"]
        SCHEDULERS = ["cosine", "linear", "constant", "constant_with_warmup", "polynomial", "inverse_sqrt"]
        OPTIMS = ["adamw_torch", "adamw_8bit", "paged_adamw_8bit", "adafactor", "sgd"]
        ATTNS = ["auto", "fa2", "fa3", "sdpa", "disabled"]
        LORA_TARGETS = ["all", "q_proj,v_proj", "q_proj,k_proj,v_proj,o_proj", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"]

        self.ft_vars = {}

        # ── 基础配置 ──
        c1 = self.card(self.pc, "基础配置")
        g1 = self.grid2(c1); g1.pack(fill="x")
        v1, w1 = self._ft_render_fields(g1, [
            ("模型路径", "ft_model", str(MODELS_DIR/"glm-4-9b-chat"), "str", None),
            ("数据集", "ft_dataset", ds0, "combo", ds_list),
            ("输出目录", "ft_output", "custom_lora", "str", None),
            ("对话模板", "ft_template", "glm4", "combo", TEMPLATES),
            ("训练阶段", "ft_stage", "sft", "combo", STAGES),
            ("微调方式", "ft_finetuning_type", "lora", "combo", FT_TYPES),
            ("验证集比例", "ft_val_size", "0.0", "float", None),
        ])
        self.ft_vars.update(v1)
        self._ft_ds_combo = w1["ft_dataset"]
        self._ft_ds_list = ds_list

        # ── LoRA 配置 ──
        c2 = self.card(self.pc, "LoRA 配置")
        g2 = self.grid2(c2); g2.pack(fill="x")
        v2, _ = self._ft_render_fields(g2, [
            ("LoRA Rank", "ft_lora_rank", "8", "int", None),
            ("LoRA Alpha", "ft_lora_alpha", "16", "int", None),
            ("LoRA Dropout", "ft_lora_dropout", "0.0", "float", None),
            ("LoRA Target", "ft_lora_target", "all", "combo", LORA_TARGETS),
            ("使用 DoRA", "ft_use_dora", "False", "bool", None),
            ("使用 RSLoRA", "ft_use_rslora", "False", "bool", None),
            ("额外训练模块", "ft_additional_target", "", "str", None),
        ])
        self.ft_vars.update(v2)

        # ── 训练超参 ──
        c3 = self.card(self.pc, "训练超参")
        g3 = self.grid2(c3); g3.pack(fill="x")
        v3, _ = self._ft_render_fields(g3, [
            ("训练轮数", "ft_epochs", "3", "float", None),
            ("批次大小", "ft_batch", str(batch), "int", None),
            ("梯度累积", "ft_grad", str(ga), "int", None),
            ("学习率", "ft_lr", "1e-4", "float", None),
            ("序列长度", "ft_cutoff", str(cutoff), "int", None),
            ("学习率调度器", "ft_lr_scheduler", "cosine", "combo", SCHEDULERS),
            ("预热比例", "ft_warmup_ratio", "0.05", "float", None),
            ("最大梯度范数", "ft_max_grad_norm", "1.0", "float", None),
            ("权重衰减", "ft_weight_decay", "0.0", "float", None),
            ("最大训练步数", "ft_max_steps", "-1", "int", None),
            ("随机种子", "ft_seed", "42", "int", None),
        ])
        self.ft_vars.update(v3)

        # ── 优化与量化 ──
        c4 = self.card(self.pc, "优化与量化")
        g4 = self.grid2(c4); g4.pack(fill="x")
        v4, _ = self._ft_render_fields(g4, [
            ("量化位数", "ft_quantization_bit", qbit, "combo", ["none", "4", "8"]),
            ("双重量化", "ft_double_quantization", dq, "bool", None),
            ("Flash Attention", "ft_flash_attn", "auto", "combo", ATTNS),
            ("使用 bf16", "ft_bf16", "True", "bool", None),
            ("Liger Kernel", "ft_enable_liger_kernel", "False", "bool", None),
            ("优化器", "ft_optim", "adamw_torch", "combo", OPTIMS),
        ])
        self.ft_vars.update(v4)

        # ── 高级设置 ──
        c5 = self.card(self.pc, "高级设置")
        g5 = self.grid2(c5); g5.pack(fill="x")
        v5, _ = self._ft_render_fields(g5, [
            ("日志步数", "ft_logging_steps", "20", "int", None),
            ("保存步数", "ft_save_steps", "200", "int", None),
            ("最多保存数", "ft_save_total_limit", "3", "int", None),
            ("覆盖输出目录", "ft_overwrite_output_dir", "True", "bool", None),
            ("覆盖缓存", "ft_overwrite_cache", "False", "bool", None),
            ("预处理进程数", "ft_preprocessing_num_workers",
             str(min(8, max(2, (os.cpu_count() or 4) // 2))), "int", None),
            ("断点续训路径", "ft_resume_from_checkpoint", "", "str", None),
        ])
        self.ft_vars.update(v5)

        # 刷新数据集按钮
        r2 = tk.Frame(self.pc, bg=self.c["bg"]); r2.pack(fill="x", pady=(4, 0))
        self.btn(r2, "↻ 刷新数据集列表", self._ft_refresh_ds, small=True).pack(side="left")
        tk.Label(r2, text=f"当前 {len(ds_list)} 个可用", font=self.fonts["sm"],
            fg=self.c["text3"], bg=self.c["bg"]).pack(side="left", padx=8)

        self.ft_log = self.console(self.pc, 15); self.ft_log.pack(fill="x", pady=(10, 0))
        self.ft_log.insert("end", "就绪 — 点击「开始训练」\n")
        for t, f in [("ok", self.c["green"]), ("err", self.c["red"])]:
            self.ft_log.tag_configure(t, foreground=f)

        self.fsb = self.btn(self.aframe, "开始训练", self._ft_start, accent=True); self.fsb.pack(side="left", padx=4)
        self.fsp = self.btn(self.aframe, "停止训练", self._ft_stop, danger=True); self.fsp.pack(side="left", padx=4)
        self.fsp.configure(state="disabled")
        self.btn(self.aframe, "生成配置", self._ft_gen).pack(side="left", padx=4)

    def _ft_cfg(self):
        v = self.ft_vars
        def g(k, d=""): return v[k].get() if k in v else d
        def gi(k, d=0):
            try: return int(g(k, str(d)))
            except: return d
        def gf(k, d=0.0):
            try: return float(g(k, str(d)))
            except: return d
        def gb(k, d=False): return g(k, str(d)) == "True"
        cfg = {
            "model_name_or_path": g("ft_model"),
            "dataset": g("ft_dataset", "custom_train"),
            "output_dir": str(LLAMA_FACTORY_DIR / "output" / g("ft_output", "custom_lora")),
            "template": g("ft_template", "glm4"),
            "stage": g("ft_stage", "sft"),
            "finetuning_type": g("ft_finetuning_type", "lora"),
            "do_train": True,
            "lora_rank": gi("ft_lora_rank", 8),
            "lora_alpha": gi("ft_lora_alpha", 16),
            "lora_dropout": gf("ft_lora_dropout", 0.0),
            "lora_target": g("ft_lora_target", "all"),
            "use_dora": gb("ft_use_dora"),
            "use_rslora": gb("ft_use_rslora"),
            "num_train_epochs": gf("ft_epochs", 3.0),
            "per_device_train_batch_size": gi("ft_batch", 1),
            "gradient_accumulation_steps": gi("ft_grad", 8),
            "learning_rate": gf("ft_lr", 1e-4),
            "cutoff_len": gi("ft_cutoff", 512),
            "lr_scheduler_type": g("ft_lr_scheduler", "cosine"),
            "warmup_ratio": gf("ft_warmup_ratio", 0.05),
            "max_grad_norm": gf("ft_max_grad_norm", 1.0),
            "weight_decay": gf("ft_weight_decay", 0.0),
            "seed": gi("ft_seed", 42),
            "val_size": gf("ft_val_size", 0.0),
            "bf16": gb("ft_bf16", True),
            "flash_attn": g("ft_flash_attn", "auto"),
            "enable_liger_kernel": gb("ft_enable_liger_kernel"),
            "optim": g("ft_optim", "adamw_torch"),
            "logging_steps": gi("ft_logging_steps", 20),
            "save_steps": gi("ft_save_steps", 200),
            "save_total_limit": gi("ft_save_total_limit", 3),
            "overwrite_output_dir": gb("ft_overwrite_output_dir", True),
            "overwrite_cache": gb("ft_overwrite_cache"),
            "preprocessing_num_workers": gi("ft_preprocessing_num_workers",
                min(8, max(2, (os.cpu_count() or 4) // 2))),
            "dataloader_num_workers": 0,
            "tokenized_path": str(LLAMA_FACTORY_DIR / "data" / "tokenized" / g("ft_dataset", "custom_train")),
        }
        qbit = g("ft_quantization_bit", "4")
        if qbit and qbit != "none":
            cfg["quantization_bit"] = int(qbit)
        cfg["double_quantization"] = gb("ft_double_quantization")
        ms = gi("ft_max_steps", -1)
        if ms > 0:
            cfg["max_steps"] = ms
        at = g("ft_additional_target", "")
        if at:
            cfg["additional_target"] = at
        rp = g("ft_resume_from_checkpoint", "")
        if rp:
            cfg["resume_from_checkpoint"] = rp
        return cfg

    def _ft_gen(self):
        cfg = self._ft_cfg(); self.ft_log.delete("1.0","end")
        self.ft_log.insert("end", json.dumps(cfg, ensure_ascii=False, indent=2)); self.toast("配置已生成","info")

    def _ft_start(self):
        global training_proc
        if training_proc and isinstance(training_proc, subprocess.Popen) and training_proc.poll() is None:
            return messagebox.showerror("错误","训练已在运行中")
        cfg = self._ft_cfg(); cp = LLAMA_FACTORY_DIR / "train_config.yaml"
        try:
            import yaml; cp.write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True), encoding="utf-8")
        except ImportError:
            cp = LLAMA_FACTORY_DIR / "train_config.json"; cp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        env = {**os.environ,"PYTHONPATH":str(LLAMA_FACTORY_DIR/"src"),"DISABLE_VERSION_CHECK":"1",
               "PYTHONIOENCODING":"utf-8","HF_ENDPOINT":HF_MIRROR,"PYTHONUNBUFFERED":"1"}
        self.ft_log.delete("1.0","end"); self.ft_log.insert("end","启动训练...\n")
        self.fsb.configure(state="disabled"); self.fsp.configure(state="normal")
        self.set_status("训练中...", self.c["yellow"])
        def _run():
            global training_proc
            try:
                training_proc = subprocess.Popen([str(find_python()),"-m","llamafactory.cli","train",str(cp)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                    errors="replace", cwd=str(LLAMA_FACTORY_DIR), env=env)
                batch, lf, lp = [], time.time(), None
                for line in training_proc.stdout:
                    s = line.rstrip("\r\n")
                    if not s: continue
                    if _is_tqdm_line(s): lp=s; continue
                    if lp: batch.append(lp); lp=None
                    batch.append(s)
                    if len(batch)>=50 or (time.time()-lf)>1.0:
                        self.root.after(0, lambda b=batch[:]: self._ft_append("\n".join(b)+"\n"))
                        batch.clear(); lf=time.time()
                if lp: batch.append(lp)
                if batch: self.root.after(0, lambda b=batch: self._ft_append("\n".join(b)+"\n"))
                training_proc.wait(); rc=training_proc.returncode
                msg = "训练完成!" if rc==0 else f"训练退出 (code {rc})"
                self.root.after(0, lambda: self._ft_append(msg+"\n","ok" if rc==0 else "err"))
                self.root.after(0, lambda: self.toast(msg, "success" if rc==0 else "error"))
                if rc==0:
                    od=cfg.get("output_dir","")
                    if od and os.path.isdir(od):
                        for n in os.listdir(od):
                            if n.startswith("checkpoint-"): shutil.rmtree(os.path.join(od,n), ignore_errors=True)
                        dn=cfg.get("dataset",""); dfn=DATA_DIR/f"{dn}.json"
                        if dfn.exists():
                            try:
                                dd=json.loads(dfn.read_text(encoding="utf-8"))
                                if dd and dd[0]["messages"][0]["role"]=="system":
                                    Path(od,"system_prompt.txt").write_text(dd[0]["messages"][0]["content"],encoding="utf-8")
                            except: pass
            except Exception as e: self.root.after(0, lambda: self._ft_append(f"错误: {e}\n","err"))
            finally:
                self.root.after(0, lambda: self.fsb.configure(state="normal"))
                self.root.after(0, lambda: self.fsp.configure(state="disabled"))
                self.root.after(0, lambda: self.set_status("就绪")); training_proc = None
        threading.Thread(target=_run, daemon=True).start()

    def _ft_append(self, t, tag=None):
        if tag: self.ft_log.insert("end", t, tag)
        else: self.ft_log.insert("end", t)
        self.ft_log.see("end")

    def _ft_stop(self):
        global training_proc
        if training_proc and isinstance(training_proc, subprocess.Popen):
            training_proc.terminate(); self.set_status("已停止"); self.toast("已停止","info")

    # ====== 面板: 模型对话 ======
    def panel_chat(self):
        # 顶部模型栏
        bar = tk.Frame(self.pc, bg=self.c["sidebar"]); bar.pack(fill="x", pady=(0,8))
        tk.Label(bar, text=" 基础模型", font=self.fonts["sm"], fg=self.c["text2"], bg=self.c["sidebar"]).pack(side="left")
        self.ch_mv = tk.StringVar(value=str(MODELS_DIR/"glm-4-9b-chat"))
        tk.Entry(bar, textvariable=self.ch_mv, font=self.fonts["sm"], bg=self.c["card2"], fg=self.c["text"],
            width=30, relief="flat", bd=0, highlightthickness=1, highlightbackground=self.c["border_soft"]).pack(side="left", padx=4)

        tk.Label(bar, text="LoRA", font=self.fonts["sm"], fg=self.c["text2"], bg=self.c["sidebar"]).pack(side="left", padx=(8,0))
        adapters = _scan_adapters()
        names = ["(基础模型)"] + [a["name"] for a in adapters]
        self.ch_lv = tk.StringVar(value="(基础模型)")
        self.ch_lp = {"(基础模型)":None}
        for a in adapters: self.ch_lp[a["name"]] = a["path"]
        s = ttk.Combobox(bar, textvariable=self.ch_lv, values=names, state="readonly", font=self.fonts["sm"], width=18)
        s.pack(side="left", padx=4)

        self.btn(bar, "加载", self._ch_load, accent=True, small=True).pack(side="left", padx=4)
        self.btn(bar, "卸载", self._ch_unload, danger=True, small=True).pack(side="left", padx=4)

        # 状态
        sf = tk.Frame(bar, bg=self.c["sidebar"]); sf.pack(side="right")
        self.ch_st = tk.Label(sf, text="未加载", font=self.fonts["sm"], fg=self.c["text3"], bg=self.c["sidebar"])
        self.ch_st.pack(side="right", padx=(0,4))

        # ── 生成参数栏 ──
        gen_card = tk.Frame(self.pc, bg=self.c["card"], highlightthickness=1, highlightbackground=self.c["border"])
        gen_card.pack(fill="x", pady=(0, 6))
        gen_inner = tk.Frame(gen_card, bg=self.c["card"]); gen_inner.pack(fill="both", expand=True, padx=12, pady=8)
        tk.Label(gen_inner, text="生成参数", font=self.fonts["h3"], fg=self.c["text"], bg=self.c["card"]).pack(anchor="w", pady=(0, 6))
        gr1 = tk.Frame(gen_inner, bg=self.c["card"]); gr1.pack(fill="x")
        gr2 = tk.Frame(gen_inner, bg=self.c["card"]); gr2.pack(fill="x", pady=(4, 0))

        def _gen_entry(parent, label, default, width=8):
            f = tk.Frame(parent, bg=self.c["card"]); f.pack(side="left", padx=(0, 12))
            tk.Label(f, text=label, font=self.fonts["xs"], fg=self.c["text2"], bg=self.c["card"]).pack(anchor="w")
            var = tk.StringVar(value=str(default))
            e = tk.Entry(f, textvariable=var, font=self.fonts["sm"], bg=self.c["card2"], fg=self.c["text"],
                        relief="flat", bd=0, highlightthickness=1, highlightbackground=self.c["border_soft"], width=width)
            e.pack(ipady=2)
            return var

        def _gen_combo(parent, label, values, default, width=8):
            f = tk.Frame(parent, bg=self.c["card"]); f.pack(side="left", padx=(0, 12))
            tk.Label(f, text=label, font=self.fonts["xs"], fg=self.c["text2"], bg=self.c["card"]).pack(anchor="w")
            var = tk.StringVar(value=str(default))
            cb = ttk.Combobox(f, textvariable=var, values=values, state="readonly", font=self.fonts["sm"], width=width)
            cb.pack()
            return var

        self.ch_max_new_tokens = _gen_entry(gr1, "max_new_tokens", "256")
        self.ch_temperature = _gen_entry(gr1, "temperature", "0.9")
        self.ch_top_p = _gen_entry(gr1, "top_p", "0.95")
        self.ch_top_k = _gen_entry(gr1, "top_k", "50")
        self.ch_repetition_penalty = _gen_entry(gr1, "repetition_penalty", "1.15")
        self.ch_do_sample = _gen_combo(gr1, "do_sample", ["True", "False"], "True")
        self.ch_num_beams = _gen_entry(gr2, "num_beams", "1", 6)
        tk.Label(gr2, text="    system prompt", font=self.fonts["xs"], fg=self.c["text2"], bg=self.c["card"]).pack(side="left", padx=(4, 6))
        self.ch_system_prompt = tk.StringVar(value="")
        tk.Entry(gr2, textvariable=self.ch_system_prompt, font=self.fonts["sm"], bg=self.c["card2"], fg=self.c["text"],
                relief="flat", bd=0, highlightthickness=1, highlightbackground=self.c["border_soft"], width=40).pack(side="left", ipady=2)
        tk.Label(gr2, text="  (留空=自动加载训练人设)", font=self.fonts["xs"], fg=self.c["text3"], bg=self.c["card"]).pack(side="left")
        self.btn(gr2, "自动检测", self._ch_auto_detect, small=True).pack(side="left", padx=(8, 0))

        # 对话区
        self.ch_area = scrolledtext.ScrolledText(self.pc, font=self.fonts["body"], bg=self.c["bg"],
            fg=self.c["text"], wrap="word", height=18, relief="flat", bd=0, state="disabled")
        self.ch_area.tag_configure("usend", foreground=self.c["blue"], font=(self._ff,9,"bold"), spacing3=12)
        self.ch_area.tag_configure("ubub", foreground=self.c["text"], font=self.fonts["body"],
            lmargin1=20, lmargin2=20, background="#0f1722", spacing1=3, spacing3=6)
        self.ch_area.tag_configure("aisend", foreground=self.c["accent"], font=(self._ff,9,"bold"), spacing3=4)
        self.ch_area.tag_configure("aibub", foreground=self.c["text"], font=self.fonts["body"],
            lmargin1=20, lmargin2=20, background="#0f1f1e", spacing1=3, spacing3=6)
        self.ch_area.tag_configure("errmsg", foreground=self.c["red"], font=self.fonts["sm"], lmargin1=20)
        self.ch_area.pack(fill="both", expand=True, pady=(0,6))

        # 输入区
        ir = tk.Frame(self.pc, bg=self.c["card2"], highlightthickness=1, highlightbackground=self.c["border_soft"])
        ir.pack(fill="x")
        self.ch_in = tk.Text(ir, font=self.fonts["body"], bg=self.c["card2"], fg=self.c["text"],
            insertbackground=self.c["accent"], height=3, relief="flat", bd=0, wrap="word")
        self.ch_in.pack(side="left", fill="x", expand=True, padx=(10,6), pady=8)
        self.ch_in.bind("<Return>", lambda e: (self._ch_send() if not e.state&1 else None))
        tk.Label(ir, text="Enter 发送  Shift+Enter 换行", font=self.fonts["xs"], fg=self.c["text3"], bg=self.c["card2"]).pack(side="left", padx=(0,4))
        self.btn(ir, "发送 ➤", self._ch_send, accent=True).pack(side="left", padx=(0,8), pady=8)

    def _ch_load(self):
        global infer_proc, active_model_name, chat_history
        if infer_proc:
            try: infer_proc.terminate()
            except: pass
        for n in list(loaded_models.keys()):
            try: loaded_models[n]["proc"].terminate()
            except: pass
        loaded_models.clear(); infer_proc=None; active_model_name=None; chat_history.clear()
        self.ch_st.configure(text="加载中...", fg=self.c["yellow"]); self.root.update()
        mp = self.ch_mv.get(); ap = self.ch_lp.get(self.ch_lv.get())
        script = self._ch_build_infer(mp, ap)
        (LLAMA_FACTORY_DIR/"_chat_server.py").write_text(script, encoding="utf-8")
        env = {**os.environ,"PYTHONPATH":str(LLAMA_FACTORY_DIR/"src"),"HF_ENDPOINT":HF_MIRROR,
               "PYTHONIOENCODING":"utf-8","PYTHONUNBUFFERED":"1"}
        def _load():
            global infer_proc, active_model_name
            try:
                infer_proc = subprocess.Popen([str(find_python()), str(LLAMA_FACTORY_DIR/"_chat_server.py")],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
                for raw in infer_proc.stdout:
                    line = raw.decode("utf-8",errors="replace").strip()
                    if line.startswith("READY:"):
                        name = Path(mp).name if ap is None else Path(ap).name
                        active_model_name=name; loaded_models[name]={"proc":infer_proc}
                        self.root.after(0, lambda: self.ch_st.configure(text=f" {name} · 在线", fg=self.c["green"]))
                        self.root.after(0, lambda: self.toast("模型加载完成","success"))
                        return
                err = infer_proc.stderr.read().decode("utf-8",errors="replace")[:500] if infer_proc.stderr else ""
                self.root.after(0, lambda: self.ch_st.configure(text=" 加载失败", fg=self.c["red"]))
                if err: self.root.after(0, lambda: messagebox.showerror("加载失败", err))
            except Exception as e:
                self.root.after(0, lambda: self.ch_st.configure(text=" 加载失败", fg=self.c["red"]))
        threading.Thread(target=_load, daemon=True).start()

    def _ch_unload(self):
        global infer_proc, active_model_name, chat_history
        if infer_proc:
            try: infer_proc.terminate()
            except: pass
        infer_proc=None; active_model_name=None; chat_history.clear(); loaded_models.clear()
        self.ch_st.configure(text=" 未加载", fg=self.c["text3"])
        self.ch_area.configure(state="normal"); self.ch_area.delete("1.0","end"); self.ch_area.configure(state="disabled")
        self.toast("模型已卸载","info")

    def _ch_send(self):
        global chat_history
        if not infer_proc or infer_proc.poll() is not None:
            return messagebox.showwarning("提示","请先加载模型")
        msg = self.ch_in.get("1.0","end-1c").strip()
        if not msg: return
        self._ch_add("user", msg); self.ch_in.delete("1.0","end")
        self.ch_st.configure(text=" 对方正在输入...", fg=self.c["yellow"])
        def _send():
            try:
                req = json.dumps({"message":msg,"history":chat_history[-10:]}, ensure_ascii=False)
                infer_proc.stdin.write((req+"\n").encode("utf-8")); infer_proc.stdin.flush()
                raw = infer_proc.stdout.readline()
                resp = json.loads(raw.decode("utf-8",errors="replace").strip())
                if "error" in resp:
                    self.root.after(0, lambda: self._ch_add("error", resp["error"]))
                else:
                    r = resp.get("response","")
                    chat_history.append({"role":"user","content":msg})
                    chat_history.append({"role":"assistant","content":r})
                    self.root.after(0, lambda: self._ch_add("assistant", r))
                self.root.after(0, lambda: self.ch_st.configure(text=" 在线", fg=self.c["green"]))
            except Exception as e:
                self.root.after(0, lambda: self._ch_add("error", str(e)))
        threading.Thread(target=_send, daemon=True).start()

    def _ch_add(self, role, text):
        if not hasattr(self,'ch_area') or not self.ch_area.winfo_exists(): return
        self.ch_area.configure(state="normal")
        if role=="user":
            self.ch_area.insert("end","\n我\n","usend"); self.ch_area.insert("end",f"{text}\n","ubub")
        elif role=="assistant":
            self.ch_area.insert("end","\n呓\n","aisend"); self.ch_area.insert("end",f"{text}\n","aibub")
        else: self.ch_area.insert("end",f"\n⚠ {text}\n","errmsg")
        self.ch_area.see("end"); self.ch_area.configure(state="disabled")

    def _ch_auto_detect(self):
        """检测硬件并自动推荐生成参数 (torch → nvidia-smi → CPU)"""
        vram = detect_vram_gb()
        if vram <= 0:
            # torch didn't find GPU, try nvidia-smi as fallback
            import shutil
            nvsmi = shutil.which("nvidia-smi")
            if not nvsmi:
                for p in [r"C:\Windows\System32\nvidia-smi.exe",
                          r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"]:
                    if Path(p).exists(): nvsmi = p; break
            if nvsmi:
                try:
                    r = subprocess.run([nvsmi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=10)
                    if r.returncode == 0 and r.stdout.strip():
                        vram = round(float(r.stdout.strip()) / 1024, 1)
                except: pass
        if vram >= 24:
            max_nt, note = 1024, "显存充足"
        elif vram >= 16:
            max_nt, note = 768, "显存充裕"
        elif vram >= 12:
            max_nt, note = 512, "显存适中"
        elif vram >= 8:
            max_nt, note = 256, "显存有限"
        elif vram > 0:
            max_nt, note = 128, "显存较小"
        else:
            max_nt, note = 64, "未检测到GPU, CPU推理"
        # 应用推荐值
        def _set(attr, val):
            try:
                var = getattr(self, attr, None)
                if var: var.set(str(val))
            except: pass
        _set("ch_max_new_tokens", max_nt)
        _set("ch_temperature", 0.9)
        _set("ch_top_p", 0.95)
        _set("ch_top_k", 50)
        _set("ch_repetition_penalty", 1.15)
        _set("ch_do_sample", "True")
        _set("ch_num_beams", 1)
        self.toast(f"已应用推荐: max_new_tokens={max_nt} ({note}, {vram}GB VRAM)", "success")

    def _ch_gen_param(self, attr, cast, default):
        """读取生成参数, 支持类型转换"""
        try:
            v = getattr(self, attr).get()
            return cast(v)
        except: return default

    def _ch_build_infer(self, mp, ap):
        mp = mp.replace("\\","/"); ac=""; mc=""
        if ap:
            ap=ap.replace("\\","/"); ac=f'adapter_path="{ap}"'
            mc="\nfrom peft import PeftModel\nmodel=PeftModel.from_pretrained(model,adapter_path)\nprint('LoRA loaded',flush=True)"
        # 读取生成参数
        max_nt = self._ch_gen_param("ch_max_new_tokens", int, 256)
        temp = self._ch_gen_param("ch_temperature", float, 0.9)
        top_p = self._ch_gen_param("ch_top_p", float, 0.95)
        top_k = self._ch_gen_param("ch_top_k", int, 50)
        rep_pen = self._ch_gen_param("ch_repetition_penalty", float, 1.15)
        do_samp = self._ch_gen_param("ch_do_sample", lambda x: x=="True", True)
        n_beams = self._ch_gen_param("ch_num_beams", int, 1)
        sys_prompt = self._ch_gen_param("ch_system_prompt", str, "")
        return f'''import sys,json,torch,os
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
mp="{mp}";{ac}
tok=AutoTokenizer.from_pretrained(mp,trust_remote_code=True)
sp="你是一个友好的AI助手。"
try:
 if ap:
  sf=os.path.join(ap,"system_prompt.txt")
  if os.path.exists(sf): sp=open(sf,"r",encoding="utf-8").read().strip()
except: pass
# 用户自定义 system prompt 优先
user_sp={json.dumps(sys_prompt) if sys_prompt else '""'}
if user_sp: sp=user_sp
cu=torch.cuda.is_available();qk={{}};ak={{}}
if cu:
 try: import flash_attn;ai="flash_attention_2"
 except ImportError: ai="sdpa"
 dt=torch.bfloat16;ak={{"attn_implementation":ai}}
 qk={{"quantization_config":BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=dt,bnb_4bit_use_double_quant=False)}}
model=AutoModelForCausalLM.from_pretrained(mp,device_map="auto" if cu else None,trust_remote_code=True,**ak,**qk)
{mc}
print("READY:"+mp,flush=True)
for line in sys.stdin:
 try:
  req=json.loads(line.strip());um=req.get("message","");hi=req.get("history",[])
  msgs=[{{"role":"system","content":sp}}]
  for h in hi: msgs.append({{"role":h["role"],"content":h["content"]}})
  msgs.append({{"role":"user","content":um}})
  tx=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
  inp=tok(tx,return_tensors="pt")
  if cu: inp={{k:v.cuda() for k,v in inp.items()}}
  with torch.no_grad():
   gen_kwargs={{k:v for k,v in [("max_new_tokens",{max_nt}),("do_sample",{do_samp}),("temperature",{temp}),("top_p",{top_p}),("top_k",{top_k}),("repetition_penalty",{rep_pen}),("num_beams",{n_beams})]}}
   out=model.generate(**inp,**gen_kwargs)
  rp=tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True)
  print(json.dumps({{"response":rp}},ensure_ascii=False),flush=True)
 except Exception as e: print(json.dumps({{"error":str(e)}},ensure_ascii=False),flush=True)
'''

    # ====== 面板: 系统状态 ======
    def panel_system(self):
        c = self.card(self.pc, "系统状态")
        self.sys_txt = self.console(c, 18); self.sys_txt.pack(fill="x")
        self.btn(self.aframe, "刷新", self._sys_refresh, accent=True).pack(side="left", padx=4)
        self._sys_refresh()

    def _sys_refresh(self):
        self.sys_txt.delete("1.0","end")
        try:
            import psutil; cpu=psutil.cpu_percent(interval=0.1); mem=psutil.virtual_memory()
            self.sys_txt.insert("end",f"CPU: {cpu}%  ({psutil.cpu_count()} 核心)\n内存: {mem.percent}%  ({mem.used//(1024**3)}GB/{mem.total//(1024**3)}GB)\n\n")
        except ImportError: self.sys_txt.insert("end","(psutil 未安装)\n\n")
        try:
            r = subprocess.run(["nvidia-smi","--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu","--format=csv,noheader"],
                capture_output=True, text=True, timeout=10)
            self.sys_txt.insert("end",f"GPU:\n{r.stdout}\n")
        except: self.sys_txt.insert("end","nvidia-smi 不可用\n")
        self.sys_txt.insert("end",f"\n显存: {detect_vram_gb()} GB\n模型: {MODELS_DIR}\n框架: {LLAMA_FACTORY_DIR}\nPython: {find_python()}\n")
        self.sys_txt.insert("end",f"\n已发现适配器: {len(_scan_adapters())} 个\n可用数据集: {len(_scan_datasets())} 个\n")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    App().run()
