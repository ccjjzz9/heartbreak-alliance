# -*- coding: utf-8 -*-
"""
AI Toolbox Server - Backend for the dark-themed web UI.
Uses only stdlib, no extra dependencies.
"""
import os, sys, json, subprocess, threading, time, re, logging, ctypes, queue
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

USB_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = USB_ROOT / "pytorch-env" / "python.exe"
MEMO_TRACE = USB_ROOT / "MemoTrace.exe"
LOG_DIR = USB_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
HF_MIRROR = "https://hf-mirror.com"
_CONFIG_FILE = USB_ROOT / "toolbox_config.json"

def _find_system_python():
    """Find a usable system Python on this machine (not the USB pytorch-env)."""
    # Check common locations
    for drv in ["D:", "C:", "E:"]:
        p = Path(f"{drv}/python.exe")
        if p.exists():
            return p
    # Search PATH
    import shutil as _shutil
    for name in ["python", "python3"]:
        found = _shutil.which(name)
        if found:
            p = Path(found)
            if p.exists() and USB_ROOT not in p.parents:
                return p
    return Path(sys.executable)

def _find_best_local_drive():
    """Find the best local drive for large files (LLaMA-Factory, models).
    Prefers D:, falls back to C:, then USB root."""
    for drv in ["D:", "C:"]:
        p = Path(f"{drv}/")
        if p.exists():
            return p
    return USB_ROOT

SYSTEM_PYTHON = _find_system_python()
LOCAL_DRIVE = _find_best_local_drive()

logging.basicConfig(
    filename=str(LOG_DIR / "toolbox.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("AIToolbox")

_BS = chr(92)

def _load_config():
    if _CONFIG_FILE.exists():
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_config(cfg):
    with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

_saved = _load_config()

# Priority: saved config > USB drive > local drive
LLAMA_FACTORY = Path(_saved.get("llama_factory",
    str(USB_ROOT / "LLaMA-Factory") if (USB_ROOT / "LLaMA-Factory").exists()
    else str(LOCAL_DRIVE / "LLaMA-Factory")))
MODELS_DIR = Path(_saved.get("models_dir",
    str(USB_ROOT / "models") if (USB_ROOT / "models").exists()
    else str(LOCAL_DRIVE / "models")))
AI_ENV_DIR = Path(_saved.get("ai_env_dir", str(LOCAL_DRIVE / "ai-env")))
GLM4_MODEL = MODELS_DIR / "glm-4-9b-chat"
AI_ENV_PYTHON = AI_ENV_DIR / "Scripts" / "python.exe"
AI_CHAT_PYTHON = AI_ENV_PYTHON

# Global state
training_proc = None
training_log_queue = queue.Queue()
infer_proc = None
infer_lock = threading.Lock()
chat_history = []
loaded_models = {}
active_model_name = None
event_queues = []

FILTER_PATTERNS = [
    "[表情包]", "[图片]", "[视频]", "[语音]", "[文件]",
    "[链接]", "[小程序]", "[红包]", "[转账]",
    "[位置]", "[名片]", "[收藏]", "[合并转发]",
]
SYSTEM_KEYWORDS = [
    "撤回了一条消息", "朋友验证", "你已添加了",
    "以上是打招呼的内容", "开启了朋友验证",
]

def broadcast_event(event_type, data):
    msg = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
    dead = []
    for q in event_queues:
        try:
            q.put_nowait(msg)
        except Exception:
            dead.append(q)
    for q in dead:
        try:
            event_queues.remove(q)
        except ValueError:
            pass

# ── API Handlers ──────────────────────────────────────────────

def api_status():
    return {"status": "ok", "version": "2.0.0", "timestamp": datetime.now().isoformat()}

def _find_python_exe():
    """Find the best available Python executable with AI packages.
    Priority: pytorch-env > D:\\python.exe (3.13) > ai-env > current process"""
    candidates = [PYTHON_EXE, SYSTEM_PYTHON, AI_CHAT_PYTHON, Path(sys.executable)]
    for p in candidates:
        if p.exists():
            return p
    return Path(sys.executable)

def api_env_check():
    checks = []
    python_exe = _find_python_exe()

    # 1. Python (current process)
    checks.append({"name": "Python (工具箱)", "status": "ok",
                   "detail": f"Python {sys.version.split()[0]} ({sys.executable})"})

    # 2. pytorch-env Python
    checks.append({"name": "pytorch-env Python", "status": "ok" if PYTHON_EXE.exists() else "error",
                   "detail": str(PYTHON_EXE) if PYTHON_EXE.exists() else f"not found: {PYTHON_EXE}"})

    # 3. PyTorch + CUDA
    r = subprocess.run([str(python_exe), "-c",
        "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "HF_ENDPOINT": HF_MIRROR})
    checks.append({"name": "PyTorch + CUDA", "status": "ok" if r.returncode == 0 else "error",
                   "detail": r.stdout.strip() if r.returncode == 0 else r.stderr.strip()[:300]})

    # 4. Transformers
    r = subprocess.run([str(python_exe), "-c",
        "import transformers; print(f'transformers={transformers.__version__}')"],
        capture_output=True, text=True, timeout=30)
    checks.append({"name": "Transformers", "status": "ok" if r.returncode == 0 else "error",
                   "detail": r.stdout.strip() if r.returncode == 0 else r.stderr.strip()[:200]})

    # 5. HuggingFace Hub
    r = subprocess.run([str(python_exe), "-c",
        "import huggingface_hub; print(f'hf_hub={huggingface_hub.__version__}')"],
        capture_output=True, text=True, timeout=30)
    checks.append({"name": "HuggingFace Hub", "status": "ok" if r.returncode == 0 else "error",
                   "detail": r.stdout.strip() if r.returncode == 0 else r.stderr.strip()[:200]})

    # 6. Accelerate
    r = subprocess.run([str(python_exe), "-c",
        "import accelerate; print(f'accelerate={accelerate.__version__}')"],
        capture_output=True, text=True, timeout=30)
    checks.append({"name": "Accelerate", "status": "ok" if r.returncode == 0 else "error",
                   "detail": r.stdout.strip() if r.returncode == 0 else r.stderr.strip()[:200]})

    # 7. ai-env
    if AI_ENV_PYTHON.exists():
        checks.append({"name": "ai-env", "status": "ok", "detail": str(AI_ENV_DIR)})
    else:
        checks.append({"name": "ai-env", "status": "error",
                       "detail": f"Python not found: {AI_ENV_PYTHON} (dir: {AI_ENV_DIR})"})

    # 8. LLaMA-Factory
    lf_src = LLAMA_FACTORY / "src" / "llamafactory"
    if LLAMA_FACTORY.exists() and lf_src.exists():
        checks.append({"name": "LLaMA-Factory", "status": "ok", "detail": str(LLAMA_FACTORY)})
    elif LLAMA_FACTORY.exists():
        checks.append({"name": "LLaMA-Factory", "status": "error",
                       "detail": f"Directory exists but src/llamafactory missing: {LLAMA_FACTORY}"})
    else:
        checks.append({"name": "LLaMA-Factory", "status": "error",
                       "detail": f"Directory not found: {LLAMA_FACTORY}"})

    # 9. GLM4 Model
    if GLM4_MODEL.exists():
        config_file = GLM4_MODEL / "config.json"
        if config_file.exists():
            checks.append({"name": "GLM4 Model", "status": "ok", "detail": str(GLM4_MODEL)})
        else:
            checks.append({"name": "GLM4 Model", "status": "error",
                           "detail": f"Dir exists but config.json missing: {GLM4_MODEL}"})
    else:
        checks.append({"name": "GLM4 Model", "status": "error",
                       "detail": f"Model not found: {GLM4_MODEL}"})

    # 10. MemoTrace
    if MEMO_TRACE.exists():
        size_mb = MEMO_TRACE.stat().st_size / (1024 * 1024)
        checks.append({"name": "MemoTrace", "status": "ok",
                       "detail": f"{MEMO_TRACE} ({size_mb:.1f} MB)"})
    else:
        checks.append({"name": "MemoTrace", "status": "error",
                       "detail": f"not found: {MEMO_TRACE}"})

    # 11. Dataset Registration
    ds_info = LLAMA_FACTORY / "data" / "dataset_info.json"
    if ds_info.exists():
        try:
            with open(ds_info, "r", encoding="utf-8") as f:
                info = json.load(f)
            if "custom_train" in info:
                checks.append({"name": "Dataset Reg", "status": "ok", "detail": "custom_train registered (sharegpt)"})
            else:
                checks.append({"name": "Dataset Reg", "status": "error", "detail": "custom_train not in dataset_info.json"})
        except Exception as e:
            checks.append({"name": "Dataset Reg", "status": "error", "detail": f"parse error: {e}"})
    else:
        checks.append({"name": "Dataset Reg", "status": "error",
                       "detail": f"dataset_info.json not found at {ds_info}"})

    return {"results": checks}

def api_config_full():
    log = []
    def log_fn(msg):
        log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        logger.info(msg)

    log_fn("=" * 40)
    log_fn("Starting full configuration...")

    # Create ai-env
    global AI_ENV_DIR, AI_ENV_PYTHON, AI_CHAT_PYTHON
    if not AI_ENV_PYTHON.exists():
        log_fn(f"Creating venv: {AI_ENV_DIR}")
        r = subprocess.run([str(SYSTEM_PYTHON) if SYSTEM_PYTHON.exists() else sys.executable, "-m", "venv", str(AI_ENV_DIR)],
            capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            log_fn("ai-env created successfully")
        else:
            log_fn(f"Failed: {r.stderr.strip()[:300]}")
    else:
        log_fn("ai-env already exists, skipping")

    # Copy LLaMA-Factory from USB to local drive
    usb_lf = USB_ROOT / "LLaMA-Factory"
    local_lf = LOCAL_DRIVE / "LLaMA-Factory"
    if local_lf.exists() and usb_lf.exists() and os.path.samefile(str(local_lf), str(usb_lf)):
        log_fn("LLaMA-Factory already on local drive, skipping")
    elif usb_lf.exists() and not local_lf.exists():
        log_fn(f"Copying LLaMA-Factory: {usb_lf} -> {local_lf}")
        try:
            import shutil
            shutil.copytree(str(usb_lf), str(local_lf))
            global LLAMA_FACTORY
            LLAMA_FACTORY = local_lf
            log_fn("LLaMA-Factory copied successfully")
        except Exception as e:
            log_fn(f"Copy failed: {e}")

    # Install deps
    if AI_ENV_PYTHON.exists():
        for label, pkgs in [("core", ["torch", "transformers", "accelerate", "huggingface_hub", "peft"]),
                             ("LLaMA-Factory", ["datasets", "einops", "omegaconf", "trl>=0.18.0,<=0.24.0",
                                               "bitsandbytes", "scipy", "sentencepiece", "protobuf"])]:
            log_fn(f"Installing {label} deps...")
            r = subprocess.run([str(AI_ENV_PYTHON), "-m", "pip", "install"] + pkgs,
                capture_output=True, text=True, timeout=1800,
                env={**os.environ, "HF_ENDPOINT": HF_MIRROR})
            log_fn(f"{label}: {'ok' if r.returncode == 0 else 'failed'}")

    # Register dataset
    data_dir = LLAMA_FACTORY / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ds_info_path = data_dir / "dataset_info.json"
    ds_info = {}
    if ds_info_path.exists():
        try:
            with open(ds_info_path, "r", encoding="utf-8") as f:
                ds_info = json.load(f)
        except Exception:
            pass
    ds_info["custom_train"] = {
        "file_name": "custom_train.json",
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {"role_tag": "role", "content_tag": "content",
                 "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"},
    }
    with open(ds_info_path, "w", encoding="utf-8") as f:
        json.dump(ds_info, f, ensure_ascii=False, indent=2)
    log_fn("Dataset registered: custom_train (sharegpt)")

    # Save config
    cfg = _load_config()
    cfg["llama_factory"] = str(LLAMA_FACTORY)
    cfg["models_dir"] = str(MODELS_DIR)
    _save_config(cfg)
    log_fn("Full configuration complete!")

    return {"log": log}

def api_config_copy():
    log = []
    def log_fn(msg):
        log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    import shutil
    global LLAMA_FACTORY
    usb_lf = USB_ROOT / "LLaMA-Factory"
    local_lf = LOCAL_DRIVE / "LLaMA-Factory"
    if local_lf.exists() and usb_lf.exists() and os.path.samefile(str(local_lf), str(usb_lf)):
        log_fn("LLaMA-Factory already on local drive, skipping")
    elif usb_lf.exists() and not local_lf.exists():
        log_fn(f"Copying LLaMA-Factory...")
        shutil.copytree(str(usb_lf), str(local_lf))
        LLAMA_FACTORY = local_lf
        log_fn("Done")
    cfg = _load_config()
    cfg["llama_factory"] = str(LLAMA_FACTORY)
    _save_config(cfg)
    return {"log": log}

def api_config_deps():
    log = []
    def log_fn(msg):
        log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    if not AI_ENV_PYTHON.exists():
        log_fn("ai-env not found")
        return {"log": log}
    for label, pkgs in [("core", ["torch", "transformers", "accelerate", "huggingface_hub", "peft"]),
                         ("LLaMA-Factory", ["datasets", "einops", "omegaconf", "trl>=0.18.0,<=0.24.0",
                                           "bitsandbytes", "scipy", "sentencepiece", "protobuf"])]:
        log_fn(f"Installing {label} deps...")
        r = subprocess.run([str(AI_ENV_PYTHON), "-m", "pip", "install"] + pkgs,
            capture_output=True, text=True, timeout=1800,
            env={**os.environ, "HF_ENDPOINT": HF_MIRROR})
        log_fn(f"{label}: {'ok' if r.returncode == 0 else 'failed'}")
    return {"log": log}

def api_wechat_status():
    """Check WeChat status comprehensively, matching original tkinter behavior."""
    # Check WeChat process
    wechat_running = False
    try:
        import psutil
        for proc in psutil.process_iter(["name"]):
            if proc.info["name"] and "WeChat.exe" in proc.info["name"]:
                wechat_running = True
                break
    except ImportError:
        r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq WeChat.exe"],
            capture_output=True, text=True, timeout=10)
        wechat_running = "WeChat.exe" in r.stdout

    # Check WeChat data directories
    wechat_data_found = False
    wechat_data_paths = []
    try:
        appdata = os.environ.get("APPDATA", "")
        localappdata = os.environ.get("LOCALAPPDATA", "")
        paths_to_check = [
            Path(appdata) / "Tencent" / "WeChat" if appdata else None,
            Path(localappdata) / "Tencent" / "WeChat" if localappdata else None,
        ]
        for p in paths_to_check:
            if p and p.exists():
                wechat_data_found = True
                wechat_data_paths.append(str(p))
    except Exception:
        pass

    # MemoTrace info
    mt_exists = MEMO_TRACE.exists()
    mt_size = MEMO_TRACE.stat().st_size if mt_exists else 0

    return {
        "wechat_running": wechat_running,
        "wechat_data": wechat_data_found,
        "wechat_data_paths": wechat_data_paths,
        "memotrace_exists": mt_exists,
        "memotrace_path": str(MEMO_TRACE),
        "memotrace_size": mt_size,
        "memotrace_size_mb": round(mt_size / (1024 * 1024), 1) if mt_exists else 0,
        "usb_root": str(USB_ROOT),
        "status_text": ("微信进程: " + ("运行中" if wechat_running else "未运行") +
                        " | 微信数据: " + ("已找到" if wechat_data_found else "未找到") +
                        " | MemoTrace: " + ("已找到" if mt_exists else "不存在")),
    }

def api_launch_memotrace():
    """Launch MemoTrace with admin privileges, same as original."""
    if not MEMO_TRACE.exists():
        return {"error": f"MemoTrace.exe not found: {MEMO_TRACE}\n\nUSB Root: {USB_ROOT}"}
    try:
        ret = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", str(MEMO_TRACE), None, str(MEMO_TRACE.parent), 1
        )
        if ret <= 32:
            # Admin launch failed, try normal
            subprocess.Popen([str(MEMO_TRACE)], cwd=str(MEMO_TRACE.parent))
            return {"success": True, "mode": "normal", "message": "MemoTrace started (normal mode)"}
        return {"success": True, "mode": "admin", "message": "MemoTrace started (admin mode)"}
    except Exception as e:
        try:
            subprocess.Popen([str(MEMO_TRACE)], cwd=str(MEMO_TRACE.parent))
            return {"success": True, "mode": "normal", "message": f"MemoTrace started (fallback): {e}"}
        except Exception as e2:
            return {"error": f"Launch failed: {e2}"}

def _get_chat_content(params):
    """Get chat file content from either filepath or base64-encoded file_content."""
    if params.get("file_content"):
        import base64
        return base64.b64decode(params["file_content"]).decode("utf-8", errors="ignore")
    filepath = params.get("filepath", "")
    if filepath and Path(filepath).exists():
        return Path(filepath).read_text(encoding="utf-8", errors="ignore")
    return None

def api_convert_preview(params):
    sender = params.get("sender", "")
    persona = params.get("persona", "你是一个友好的AI助手。")
    content = _get_chat_content(params)
    if not content:
        return {"error": "No file content. Upload a file or specify a valid filepath."}
    try:
        messages = _parse_chat_from_text(content)
        data = _convert_chat_to_sharegpt(messages, sender, persona)
        preview = []
        for i, item in enumerate(data[:3]):
            msgs = []
            for m in item["messages"][:6]:
                msgs.append({"role": m["role"], "content": m["content"][:200]})
            preview.append({"index": i + 1, "messages": msgs})
        return {"total_messages": len(messages), "total_conversations": len(data),
                "total_training_samples": len(data), "preview": preview}
    except Exception as e:
        return {"error": str(e)}

def api_convert_run(params):
    sender = params.get("sender", "")
    persona = params.get("persona", "你是一个友好的AI助手。")
    dataset = params.get("dataset", "custom_train").strip() or "custom_train"
    content = _get_chat_content(params)
    if not content:
        return {"error": "No file content. Upload a file or specify a valid filepath."}
    if not dataset.endswith(".json"):
        dataset = dataset + ".json"
    try:
        data = []
        if content:
            messages = _parse_chat_from_text(content)
            data = _convert_chat_to_sharegpt(messages, sender, persona)
        # Merge manual QA pairs
        manual_count = 0
        manual_raw = params.get("manual_pairs", "")
        if manual_raw:
            try:
                manual_pairs = json.loads(manual_raw)
                for pair in manual_pairs:
                    q = (pair.get("question") or "").strip()
                    a = (pair.get("answer") or "").strip()
                    if q and a:
                        data.append({"messages": [
                            {"role": "system", "content": persona},
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": a},
                        ]})
                        manual_count += 1
            except (json.JSONDecodeError, TypeError):
                pass
        if not data:
            return {"error": "No valid conversations or manual QA pairs found"}
        data_dir = LLAMA_FACTORY / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        output_path = data_dir / dataset
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        ds_name = output_path.stem
        _auto_register_dataset(ds_name)
        return {"success": True, "count": len(data), "output": str(output_path),
                "dataset_name": ds_name, "manual_count": manual_count}
    except Exception as e:
        return {"error": str(e)}

def api_convert_analyze_cutoff(params):
    """Analyze converted data token lengths and recommend optimal cutoff_len.
    Runs conversion in-memory, tokenizes all samples, returns distribution + recommendation."""
    sender = params.get("sender", "")
    persona = params.get("persona", "你是一个友好的AI助手。")
    content = _get_chat_content(params)
    if not content:
        return {"error": "No file content. Upload a file or specify a valid filepath."}

    try:
        # Run full conversion in-memory
        messages = _parse_chat_from_text(content)
        data = _convert_chat_to_sharegpt(messages, sender, persona)
        # Merge manual QA pairs
        manual_raw = params.get("manual_pairs", "")
        if manual_raw:
            try:
                for pair in json.loads(manual_raw):
                    q = (pair.get("question") or "").strip()
                    a = (pair.get("answer") or "").strip()
                    if q and a:
                        data.append({"messages": [
                            {"role": "system", "content": persona},
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": a},
                        ]})
            except (json.JSONDecodeError, TypeError):
                pass
        if not data:
            return {"error": "No valid training samples found"}

        # Tokenize all samples using the training tokenizer
        python_exe = str(_find_python_exe())
        # Write temp data for tokenizer script
        temp_data = json.dumps(data, ensure_ascii=False)
        temp_path = LLAMA_FACTORY / "_analyze_data.json"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(temp_data)

        code = f'''
import json, sys
from transformers import AutoTokenizer
data = json.load(open(r"{temp_path}", "r", encoding="utf-8"))
tokenizer = AutoTokenizer.from_pretrained(r"{GLM4_MODEL}", trust_remote_code=True)
lengths = []
for item in data:
    text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
    ids = tokenizer.encode(text)
    lengths.append(len(ids))
# Stats
lengths.sort()
n = len(lengths)
result = {{
    "samples": n,
    "min": lengths[0],
    "max": lengths[-1],
    "p50": lengths[n // 2],
    "p80": lengths[int(n * 0.8)],
    "p90": lengths[int(n * 0.9)],
    "p95": lengths[int(n * 0.95)],
    "p99": lengths[int(n * 0.99)],
}}
print(json.dumps(result, ensure_ascii=False))
'''
        script_path = LLAMA_FACTORY / "_analyze_cutoff.py"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        env = {**os.environ, "PYTHONPATH": str(LLAMA_FACTORY / "src"),
               "HF_ENDPOINT": HF_MIRROR, "PYTHONIOENCODING": "utf-8"}
        r = subprocess.run([str(python_exe), str(script_path)],
            capture_output=True, text=True, timeout=120, env=env)
        if r.returncode != 0:
            return {"error": f"Tokenizer failed: {r.stderr[:300]}"}

        stats = json.loads(r.stdout.strip())
        # Recommendation logic
        p95 = stats["p95"]
        if p95 <= 512:
            rec, reason = 512, "95%样本 ≤512 tokens，推荐 compact 模式"
        elif p95 <= 768:
            rec, reason = 768, "95%样本 ≤768 tokens，推荐平衡模式"
        elif p95 <= 1024:
            rec, reason = 1024, "95%样本 ≤1024 tokens，推荐标准模式"
        elif p95 <= 1536:
            rec, reason = 1536, "95%样本 ≤1536 tokens，需要较大上下文"
        else:
            rec, reason = 2048, "样本较长，推荐 2048（需确保显存充足）"

        stats["recommended"] = rec
        stats["reason"] = reason
        return {"success": True, "analysis": stats}
    except Exception as e:
        return {"error": str(e)}

def _auto_register_dataset(ds_name):
    """Register a dataset in dataset_info.json if not already present."""
    ds_info_path = LLAMA_FACTORY / "data" / "dataset_info.json"
    ds_info = {}
    if ds_info_path.exists():
        try:
            with open(ds_info_path, "r", encoding="utf-8") as f:
                ds_info = json.load(f)
        except Exception:
            ds_info = {}
    if ds_name not in ds_info:
        ds_info[ds_name] = {
            "file_name": f"{ds_name}.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"},
        }
        with open(ds_info_path, "w", encoding="utf-8") as f:
            json.dump(ds_info, f, ensure_ascii=False, indent=2)

def _parse_chat_from_text(text):
    """Parse chat text content into message list."""
    pattern = re.compile(r"^(\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2})\s+(.+)$")
    messages = []
    current_msg = None
    for line in text.splitlines():
        line = line.rstrip("\n\r")
        if not line:
            continue
        m = pattern.match(line)
        if m:
            if current_msg:
                messages.append(current_msg)
            current_msg = {"time": m.group(1), "content": m.group(2)}
        elif current_msg:
            current_msg["content"] += "\n" + line
    if current_msg:
        messages.append(current_msg)
    return messages

def _filter_message(content):
    content = content.strip()
    if not content:
        return True
    for pattern in FILTER_PATTERNS:
        if content == pattern or content.startswith(pattern):
            return True
    for kw in SYSTEM_KEYWORDS:
        if kw in content:
            return True
    return False

def _extract_sender_from_content(content, known_senders):
    """Extract sender name from the beginning of content.
    Format can be: 'SenderName: message' or 'SenderName\\nmessage'"""
    for sep in (': ', ':', '\n', ' '):
        idx = content.find(sep)
        if idx > 0:
            name = content[:idx].strip()
            if name and (not known_senders or name in known_senders):
                return name, content[idx + len(sep):].strip(), sep
    return None, content, None

def _convert_chat_to_sharegpt(messages, sender, persona):
    """Convert parsed WeChat messages into ShareGPT-format training data.

    Rules:
    - sender = user (使用者), non-sender = assistant (模型模仿对象)
    - Multi-turn segments split when same person sends consecutive messages (natural break)
    - Leading assistant messages preserved with synthetic user opener
    - System prompt prepended to every segment for consistent persona conditioning
    """
    # Step 1: Discover all sender names from chat content
    all_senders = set()
    for msg in messages:
        name, _, _ = _extract_sender_from_content(msg["content"].strip(), None)
        if name:
            all_senders.add(name)

    # Step 2: Classify, clean, and timestamp each message
    classified = []
    for msg in messages:
        content = msg["content"].strip()
        if _filter_message(content):
            continue
        name, body, _ = _extract_sender_from_content(content, all_senders)
        if name:
            role = "user" if sender and name == sender else "assistant"
            final_content = body if body else content
        else:
            role = "user"
            final_content = content
        final_content = _clean_content(final_content)
        if not final_content:
            continue
        classified.append({
            "role": role,
            "content": final_content,
            "time": msg.get("time", ""),
        })
    if not classified:
        return []

    # Step 3: Merge consecutive same-role messages
    merged = []
    for msg in classified:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append(msg)

    # Step 4: Split into conversation segments.
    # Natural break = same person sends again without reply (time gap in real life).
    raw_segments = []
    current = []
    last_role = None
    for msg in merged:
        if current and msg["role"] == last_role:
            raw_segments.append(current)
            current = []
        current.append(msg)
        last_role = msg["role"]
    if current:
        raw_segments.append(current)

    # Step 5: Build ShareGPT samples with sliding window
    # Long conversations are split into overlapping windows so every message
    # appears in at least one training sample.
    MAX_WINDOW = 12  # max messages per sample (6 turns user+assistant)
    SLIDE = 2        # slide by 1 turn (user+assistant pair)

    data = []
    for seg in raw_segments:
        # Normalize: ensure segment starts with user
        if seg[0]["role"] == "assistant":
            seg.insert(0, {"role": "user", "content": "在吗"})
        # Drop trailing user
        while seg and seg[-1]["role"] != "assistant":
            seg.pop()
        # Drop segments too short to form even one pair
        if len(seg) < 2:
            continue

        # Sliding window over long segments
        start = 0
        while start < len(seg) - 1:
            window = seg[start:start + MAX_WINDOW]
            # Ensure window starts with user
            while window and window[0]["role"] != "user":
                start += 1
                window = seg[start:start + MAX_WINDOW]
            if len(window) < 2:
                break
            # Ensure window ends with assistant
            if window[-1]["role"] != "assistant":
                window.pop()
            if len(window) >= 2:
                messages = [{"role": "system", "content": persona}]
                for m in window:
                    messages.append({"role": m["role"], "content": m["content"]})
                data.append({"messages": messages})
            start += SLIDE

    return data

def _clean_content(text):
    """Strip meaningless markers like [表情包] from message content
    so they don't pollute the training data."""
    for pattern in FILTER_PATTERNS:
        # Remove standalone or inline occurrences
        text = text.replace(pattern, "")
    # Collapse multiple spaces/newlines from marker removal
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def api_auto_detect_config():
    """Auto-detect hardware and return recommended training config."""
    gpu_info = {"count": 0, "vram_gb": 0, "name": "", "cuda_version": ""}
    cpu_ram_gb = 0
    try:
        import psutil
        cpu_ram_gb = psutil.virtual_memory().total // (1024**3)
    except ImportError:
        cpu_ram_gb = 0

    # Detect GPU via subprocess
    gpu_code = (
        "import json, torch\n"
        "d = {'count': 0, 'vram_gb': 0, 'name': '', 'cuda_version': ''}\n"
        "if torch.cuda.is_available():\n"
        "    d['count'] = torch.cuda.device_count()\n"
        "    d['cuda_version'] = str(torch.version.cuda)\n"
        "    d['name'] = torch.cuda.get_device_name(0)\n"
        "    p = torch.cuda.get_device_properties(0)\n"
        "    d['vram_gb'] = round(p.total_memory / (1024**3), 1)\n"
        "print(json.dumps(d, ensure_ascii=False))\n"
    )
    python_exe = _find_python_exe()
    try:
        r = subprocess.run([str(python_exe), "-c", gpu_code],
            capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            gpu_info = json.loads(r.stdout.strip())
    except Exception:
        pass

    vram = gpu_info["vram_gb"]
    gpu_name = gpu_info["name"]

    # ── Recommendation logic ──
    rec = {}

    # quantization_bit
    if vram >= 24:
        rec["quantization_bit"] = "none"
        rec["quantization_note"] = "显存充足，不需要量化"
    elif vram >= 12:
        rec["quantization_bit"] = "4"
        rec["quantization_note"] = "建议 4-bit 量化以容纳 7-9B 模型"
    elif vram >= 8:
        rec["quantization_bit"] = "4"
        rec["quantization_note"] = "必须 4-bit 量化，建议使用 ≤7B 模型"
    elif vram > 0:
        rec["quantization_bit"] = "8"
        rec["quantization_note"] = "显存较小，建议 8-bit 量化 + ≤2B 模型"
    else:
        rec["quantization_bit"] = "none"
        rec["quantization_note"] = "未检测到 GPU，将使用 CPU 训练（极慢）"

    # batch_size — conservative for 9B model
    if vram >= 24:
        rec["batch_size"] = 4
    elif vram >= 16:
        rec["batch_size"] = 2
    else:  # 12GB or less with 9B+4bit is tight
        rec["batch_size"] = 1

    # gradient_accumulation_steps (有效 batch = batch_size * grad_accum)
    if vram >= 24:
        rec["gradient_accumulation"] = 4
    elif vram >= 16:
        rec["gradient_accumulation"] = 4
    else:
        rec["gradient_accumulation"] = 8

    # cutoff_len (max sequence length)
    if vram >= 24:
        rec["cutoff_len"] = 2048
    elif vram >= 16:
        rec["cutoff_len"] = 1024
    else:
        rec["cutoff_len"] = 512

    # lora_rank (higher VRAM → can use higher rank)
    if vram >= 24:
        rec["lora_rank"] = 16
    elif vram >= 12:
        rec["lora_rank"] = 8
    elif vram >= 8:
        rec["lora_rank"] = 8
    else:
        rec["lora_rank"] = 4

    # dtype (bf16 for newer GPUs, fp16 for older)
    compute_cap = 0
    if gpu_info["count"] > 0:
        cc_code = ("import json, torch; p=torch.cuda.get_device_properties(0); "
                    "print(json.dumps({'major': p.major, 'minor': p.minor}))")
        try:
            r = subprocess.run([str(python_exe), "-c", cc_code],
                capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                cc = json.loads(r.stdout.strip())
                compute_cap = cc["major"]
        except Exception:
            pass
    if compute_cap >= 8:
        rec["dtype"] = "bfloat16"
        rec["dtype_note"] = "GPU 支持 bf16，训练速度更快"
    elif gpu_info["count"] > 0:
        rec["dtype"] = "float16"
        rec["dtype_note"] = "GPU 不支持 bf16，使用 fp16"
    else:
        rec["dtype"] = "float32"
        rec["dtype_note"] = "CPU 训练使用 fp32"

    # warmup_ratio
    rec["warmup_ratio"] = 0.05

    # lr_scheduler
    rec["lr_scheduler"] = "cosine"

    # validation_size
    rec["validation_size"] = 0.05

    return {
        "hardware": {
            "gpu_name": gpu_name,
            "gpu_count": gpu_info["count"],
            "vram_gb": vram,
            "cuda_version": gpu_info["cuda_version"],
            "cpu_ram_gb": cpu_ram_gb,
        },
        "recommended": rec,
    }

def api_finetune_datasets():
    data_dir = LLAMA_FACTORY / "data"
    datasets = []
    if data_dir.exists():
        for f in sorted(data_dir.glob("*.json")):
            if f.name != "dataset_info.json":
                datasets.append(f.stem)
    # Also list existing LoRA output dirs (only those with actual checkpoints)
    output_dir = LLAMA_FACTORY / "output"
    existing_outputs = []
    if output_dir.exists():
        for d in sorted(output_dir.iterdir()):
            if d.is_dir():
                # Check for adapter files at any depth, not just top-level checkpoint dirs
                has_adapter = False
                for root, dirs, files in os.walk(str(d)):
                    if any(f in files for f in ("adapter_model.safetensors", "adapter_model.bin", "adapter_config.json")):
                        has_adapter = True
                        break
                if has_adapter:
                    existing_outputs.append({"name": d.name, "has_checkpoint": True})
    return {
        "datasets": datasets,
        "model_path": str(GLM4_MODEL),
        "llama_factory": str(LLAMA_FACTORY),
        "existing_outputs": existing_outputs,
    }

def _detect_training_packages():
    """Detect which acceleration packages are available in the training Python env.
    Returns dict with bool keys: flash_attn, liger_kernel."""
    result = {"flash_attn": False, "liger_kernel": False}
    python_exe = _find_python_exe()
    code = "import importlib; import json; print(json.dumps({k: importlib.util.find_spec(k) is not None for k in ['flash_attn','liger_kernel']}))"
    try:
        r = subprocess.run([str(python_exe), "-c", code],
            capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            result.update(json.loads(r.stdout.strip()))
    except Exception:
        pass
    return result

def _detect_vram_gb():
    """Quick VRAM detection for default parameter tuning."""
    try:
        python_exe = _find_python_exe()
        code = "import torch; print(torch.cuda.get_device_properties(0).total_memory // (1024**3) if torch.cuda.is_available() else 0)"
        r = subprocess.run([str(python_exe), "-c", code],
            capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            return int(r.stdout.strip())
    except Exception:
        pass
    return 0

def _build_training_config(params):
    """Build training config dict from web UI params (supports all LLaMA-Factory args)."""
    import os as _os
    cpu_cores = _os.cpu_count() or 4
    nw = max(4, cpu_cores // 2)
    dl_workers = 0
    if sys.platform != "win32":
        dl_workers = min(4, nw)
    vram = _detect_vram_gb()
    if vram >= 24:
        cutoff, batch, grad_acc = 2048, 4, 4
    elif vram >= 16:
        cutoff, batch, grad_acc = 1024, 2, 4
    else:
        cutoff, batch, grad_acc = 512, 1, 8
    pkgs = _detect_training_packages()

    def _p(key, default=None):
        return params.get(key, default)
    def _pi(key, default=0):
        try: return int(_p(key, default))
        except: return default
    def _pf(key, default=0.0):
        try: return float(_p(key, default))
        except: return default
    def _pb(key, default=False):
        v = _p(key, default)
        if isinstance(v, bool): return v
        return str(v).lower() in ("true", "1", "yes")

    config = {
        "model_name_or_path": _p("model_path", str(GLM4_MODEL)),
        "stage": _p("stage", "sft"),
        "finetuning_type": _p("finetuning_type", "lora"),
        "do_train": True,
        "dataset": _p("dataset", "custom_train"),
        "output_dir": str(LLAMA_FACTORY / "output" / _p("output_name", "custom_lora")),
        "template": _p("template", "glm4"),
        "num_train_epochs": _pf("epochs", 3),
        "per_device_train_batch_size": _pi("batch_size", batch),
        "gradient_accumulation_steps": _pi("gradient_accumulation", grad_acc),
        "learning_rate": _pf("learning_rate", 2e-4),
        "cutoff_len": _pi("cutoff_len", cutoff),
        "lora_rank": _pi("lora_rank", 8),
        "lora_alpha": _pi("lora_alpha", 16),
        "lora_dropout": _pf("lora_dropout", 0.0),
        "lora_target": _p("lora_target", "all"),
        "use_dora": _pb("use_dora"),
        "use_rslora": _pb("use_rslora"),
        "logging_steps": _pi("logging_steps", 20),
        "save_steps": _pi("save_steps", 200),
        "save_total_limit": _pi("save_total_limit", 3),
        "overwrite_output_dir": _pb("overwrite_output_dir", True),
        "overwrite_cache": _pb("overwrite_cache"),
        "preprocessing_num_workers": _pi("preprocessing_num_workers", nw),
        "dataloader_num_workers": _pi("dataloader_num_workers", dl_workers),
        "flash_attn": _p("flash_attn", "fa2" if pkgs.get("flash_attn") else "auto"),
        "enable_liger_kernel": _pb("enable_liger_kernel", pkgs.get("liger_kernel", False)),
        "tokenized_path": str(LLAMA_FACTORY / "data" / "tokenized" / _p("dataset", "custom_train")),
        "bf16": _pb("bf16", True),
        "lr_scheduler_type": _p("lr_scheduler", "cosine"),
        "warmup_ratio": _pf("warmup_ratio", 0.05),
        "max_grad_norm": _pf("max_grad_norm", 1.0),
        "weight_decay": _pf("weight_decay", 0.0),
        "seed": _pi("seed", 42),
        "val_size": _pf("val_size", 0.0),
        "optim": _p("optim", "adamw_torch"),
    }
    qbit = _p("quantization_bit", "4")
    if qbit is not None and str(qbit).lower() != "none":
        config["quantization_bit"] = int(qbit)
    config["double_quantization"] = _pb("double_quantization", vram >= 16)
    ms = _pi("max_steps", -1)
    if ms > 0:
        config["max_steps"] = ms
    at = _p("additional_target", "")
    if at:
        config["additional_target"] = at
    rp = _p("resume_from_checkpoint", "")
    if rp:
        config["resume_from_checkpoint"] = rp
    return config

def api_finetune_gen_config(params):
    config = _build_training_config(params)
    config_path = LLAMA_FACTORY / "train_config.yaml"
    try:
        import yaml
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    except ImportError:
        config_path = LLAMA_FACTORY / "train_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    return {"success": True, "config": config, "config_path": str(config_path)}

# Regex to detect tqdm progress bar lines (noisy \r updates)
_TQDM_RE = re.compile(r"^\s*\d+%\|[█▏▎▍▌▋▊▉ ╶─═]+\|\s*\d+/\d+\s*\[")

def _is_tqdm_line(line):
    return bool(_TQDM_RE.match(line))

def api_finetune_start(params):
    global training_proc
    if _is_training_running():
        return {"error": "Training already running"}

    config = _build_training_config(params)

    config_path = LLAMA_FACTORY / "train_config.yaml"
    try:
        import yaml
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    except ImportError:
        config_path = LLAMA_FACTORY / "train_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    env = {**os.environ, "PYTHONPATH": str(LLAMA_FACTORY / "src"),
           "DISABLE_VERSION_CHECK": "1", "PYTHONIOENCODING": "utf-8", "HF_ENDPOINT": HF_MIRROR,
           "PYTHONUNBUFFERED": "1"}
    python_exe = str(_find_python_exe())
    cmd = [python_exe, "-m", "llamafactory.cli", "train", str(config_path)]

    def _run():
        global training_proc
        try:
            training_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", cwd=str(LLAMA_FACTORY), env=env)
            batch = []
            last_flush = time.time()
            last_progress = None  # only keep the latest tqdm line, not every \r update
            for line in training_proc.stdout:
                stripped = line.rstrip("\r\n")
                if not stripped:
                    continue
                if _is_tqdm_line(stripped):
                    last_progress = stripped  # keep latest, discard old
                    continue
                # Flush any pending progress line before real content
                if last_progress:
                    training_log_queue.put(last_progress)
                    batch.append(last_progress)
                    last_progress = None
                training_log_queue.put(stripped)
                batch.append(stripped)
                now = time.time()
                if len(batch) >= 50 or (now - last_flush) > 1.0:
                    if batch:
                        broadcast_event("training_log_batch", {"lines": batch})
                    batch = []
                    last_flush = now
            # Flush remaining
            if last_progress:
                batch.append(last_progress)
            if batch:
                broadcast_event("training_log_batch", {"lines": batch})
            training_proc.wait()
            rc = training_proc.returncode
            msg = "Training complete!" if rc == 0 else f"Training exited with code {rc}"
            training_log_queue.put(msg)
            broadcast_event("training_log", {"line": msg})
            broadcast_event("training_done", {"returncode": rc})
            # Auto-clean checkpoint dirs & copy system prompt
            if rc == 0:
                output_dir = config.get("output_dir", "")
                if output_dir and os.path.isdir(output_dir):
                    import shutil
                    for name in os.listdir(output_dir):
                        if name.startswith("checkpoint-"):
                            ckpt_path = os.path.join(output_dir, name)
                            try:
                                shutil.rmtree(ckpt_path)
                                logger.info(f"Cleaned checkpoint: {ckpt_path}")
                            except Exception:
                                pass
                    # Copy system prompt from dataset to output dir
                    dataset_name = config.get("dataset", "")
                    if dataset_name:
                        ds_file = LLAMA_FACTORY / "data" / f"{dataset_name}.json"
                        if ds_file.exists():
                            try:
                                ds_data = json.loads(ds_file.read_text(encoding="utf-8"))
                                if ds_data and ds_data[0]["messages"][0]["role"] == "system":
                                    sp = ds_data[0]["messages"][0]["content"]
                                    sp_file = os.path.join(output_dir, "system_prompt.txt")
                                    with open(sp_file, "w", encoding="utf-8") as f:
                                        f.write(sp)
                                    logger.info(f"Saved system prompt to {sp_file}")
                            except Exception:
                                pass
        except Exception as e:
            err = f"Training error: {e}"
            training_log_queue.put(err)
            broadcast_event("training_log", {"line": err})

    threading.Thread(target=_run, daemon=True).start()
    broadcast_event("training_started", {"config": config})
    return {"success": True}

def _is_training_running():
    """Check if training or preprocessing is currently running.
    Works with both Popen (training) and Thread (preprocessing) objects."""
    global training_proc
    if training_proc is None:
        return False
    if isinstance(training_proc, threading.Thread):
        return training_proc.is_alive()
    return training_proc.poll() is None

def api_finetune_stop():
    global training_proc
    if _is_training_running():
        try:
            if isinstance(training_proc, threading.Thread):
                # Can't force-kill a thread, but subprocess.run will finish
                training_proc = None
            else:
                training_proc.terminate()
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}
    return {"error": "No training running"}

def api_finetune_status():
    global training_proc
    running = _is_training_running()
    recent = []
    while not training_log_queue.empty():
        try:
            recent.append(training_log_queue.get_nowait())
        except Exception:
            break
    return {"running": running, "recent_logs": recent[-50:]}

def _build_preprocess_script(config_path, tokenized_path):
    """Build a Python script that preprocesses dataset to .arrow format via LLaMA-Factory."""
    lf_src = str(LLAMA_FACTORY / "src")
    lf_root = str(LLAMA_FACTORY)
    return f'''import sys, os, json, shutil
from multiprocessing import freeze_support

def main():
    sys.path.insert(0, r"{lf_src}")
    os.chdir(r"{lf_root}")

    from llamafactory.hparams import get_train_args
    from llamafactory.data import get_dataset
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.model.loader import load_tokenizer

    config_path = r"{config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith(".yaml"):
            import yaml
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    config["tokenized_path"] = r"{tokenized_path}"
    config["do_train"] = True
    config.setdefault("stage", "sft")
    config.setdefault("finetuning_type", "lora")
    config.setdefault("template", "glm4")
    config.setdefault("output_dir", os.path.join(r"{lf_root}", "output", "temp_preprocess"))
    config.setdefault("logging_steps", 1)
    config.setdefault("save_steps", 999999)
    config.setdefault("num_train_epochs", 0.0)
    config.setdefault("per_device_train_batch_size", 1)
    config.setdefault("learning_rate", 1e-4)
    config.setdefault("lora_rank", 8)
    config.setdefault("lora_alpha", 16)
    config.setdefault("lora_target", "all")
    config.setdefault("overwrite_output_dir", True)
    config["overwrite_cache"] = True
    # Use 1 worker on Windows to avoid multiprocessing spawn issues
    if sys.platform == "win32":
        config["preprocessing_num_workers"] = 1
    else:
        config.setdefault("preprocessing_num_workers", min(16, max(4, (os.cpu_count() or 4) // 2)))
    config.setdefault("dataloader_num_workers", 0)
    config.setdefault("flash_attn", "auto")
    config.setdefault("enable_liger_kernel", False)
    config.setdefault("bf16", True)
    config.setdefault("cutoff_len", 1024)
    config.setdefault("gradient_accumulation_steps", 1)
    config.setdefault("lr_scheduler_type", "cosine")
    config.setdefault("warmup_ratio", 0.0)
    config.setdefault("quantization_bit", 4)

    print("Parsing training arguments...", flush=True)
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(config)
    print(f"Model: {{model_args.model_name_or_path}}", flush=True)
    print(f"Dataset: {{data_args.dataset}}", flush=True)
    print(f"Tokenized path: {{data_args.tokenized_path}}", flush=True)

    # Remove existing tokenized data to force regeneration
    if os.path.exists(data_args.tokenized_path):
        print(f"Removing existing cache: {{data_args.tokenized_path}}", flush=True)
        shutil.rmtree(data_args.tokenized_path)

    print("Loading tokenizer...", flush=True)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    print("Tokenizer loaded.", flush=True)

    print("Loading template...", flush=True)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    print(f"Template loaded: {{data_args.template}}", flush=True)

    print("Loading and preprocessing dataset (this may take a while)...", flush=True)
    dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage=finetuning_args.stage or "sft",
        tokenizer=tokenizer,
        processor=tokenizer_module.get("processor"),
    )

    train_ds = dataset_module.get("train_dataset")
    eval_ds = dataset_module.get("eval_dataset")
    print(f"PREPROCESS_OK", flush=True)
    print(f"Train samples: {{len(train_ds) if train_ds else 0}}", flush=True)
    print(f"Eval samples: {{len(eval_ds) if eval_ds else 0}}", flush=True)

    # Count arrow files
    arrow_dir = data_args.tokenized_path
    arrow_count = 0
    total_mb = 0
    for root, dirs, files in os.walk(arrow_dir):
        for f in files:
            if f.endswith(".arrow"):
                arrow_count += 1
                total_mb += os.path.getsize(os.path.join(root, f)) / (1024 * 1024)
    print(f"Arrow files: {{arrow_count}} ({{total_mb:.1f}} MB)", flush=True)

if __name__ == "__main__":
    freeze_support()
    main()
'''

def api_finetune_preprocess(params):
    global training_proc
    if _is_training_running():
        return {"error": "Training already running — stop it before preprocessing"}

    config = _build_training_config(params)
    tokenized_path = config.get("tokenized_path", "")
    if not tokenized_path:
        return {"error": "Cannot determine tokenized_path"}

    config_path = LLAMA_FACTORY / "train_config_preprocess.yaml"
    try:
        import yaml
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    except ImportError:
        config_path = LLAMA_FACTORY / "train_config_preprocess.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    script = _build_preprocess_script(str(config_path), tokenized_path)
    script_path = LLAMA_FACTORY / "_preprocess_data.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)

    env = {**os.environ, "PYTHONPATH": str(LLAMA_FACTORY / "src"),
           "DISABLE_VERSION_CHECK": "1", "PYTHONIOENCODING": "utf-8", "HF_ENDPOINT": HF_MIRROR,
           "PYTHONUNBUFFERED": "1"}
    python_exe = str(_find_python_exe())
    cmd = [python_exe, str(script_path)]

    def _run_preprocess():
        global training_proc
        t = threading.current_thread()
        training_proc = t  # sentinel: running flag for other API checks
        try:
            result = subprocess.run(cmd, capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                cwd=str(LLAMA_FACTORY), env=env, timeout=1800)
            # Process output — filter tqdm noise, keep meaningful lines
            lines = []
            last_progress = None
            for raw_line in result.stdout.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if _is_tqdm_line(line):
                    last_progress = line
                    continue
                if last_progress:
                    lines.append(last_progress)
                    last_progress = None
                lines.append(line)
            if last_progress:
                lines.append(last_progress)
            if result.stderr:
                for raw_line in result.stderr.splitlines():
                    line = raw_line.strip()
                    if line and not _is_tqdm_line(line):
                        lines.append(line)
            # Send all output at once
            for line in lines:
                training_log_queue.put(line)
            if lines:
                broadcast_event("training_log_batch", {"lines": lines})
            rc = result.returncode
            msg = "Preprocessing complete!" if rc == 0 else f"Preprocessing exited with code {rc}"
            training_log_queue.put(msg)
            broadcast_event("training_log", {"line": msg})
            broadcast_event("training_done", {"returncode": rc})
        except subprocess.TimeoutExpired:
            err = "Preprocessing timed out (30 min)"
            training_log_queue.put(err)
            broadcast_event("training_log", {"line": err})
            broadcast_event("training_done", {"returncode": -1})
        except Exception as e:
            err = f"Preprocessing error: {e}"
            training_log_queue.put(err)
            broadcast_event("training_log", {"line": err})
        finally:
            training_proc = None

    broadcast_event("training_started", {"config": config, "mode": "preprocess"})
    threading.Thread(target=_run_preprocess, daemon=True).start()
    return {"success": True, "tokenized_path": tokenized_path}

def _scan_adapter_dirs(base_dir):
    """Recursively scan for LoRA adapter directories.
    Returns list of {name, path} dicts.
    Checks: adapter_model.safetensors, adapter_model.bin, or adapter_config.json"""
    results = []
    if not base_dir.exists():
        return results
    seen = set()
    for root, dirs, files in os.walk(str(base_dir)):
        has_adapter = any(f in files for f in (
            "adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"
        ))
        # Skip checkpoint subdirectories — only show final adapters
        if "checkpoint-" in str(root):
            continue
        if has_adapter:
            rel = Path(root).relative_to(base_dir)
            name = str(rel).replace("\\", "/")
            if name == ".":
                name = Path(root).name  # e.g. "4_lora"
            if name not in seen:
                seen.add(name)
                results.append({"name": name, "path": str(Path(root))})
    results.sort(key=lambda x: x["name"])
    return results

def api_chat_models():
    output_dir = LLAMA_FACTORY / "output"
    adapters = [{"name": "(基础模型)", "path": None}]
    adapters.extend(_scan_adapter_dirs(output_dir))
    loaded = list(loaded_models.keys())
    return {"models": adapters, "loaded": loaded, "active": active_model_name,
            "base_model": str(GLM4_MODEL)}

def _detect_gpu_via_nvidia_smi():
    """Use nvidia-smi as OS-level fallback — works regardless of Python/torch setup."""
    import shutil
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        # Also check the default install path
        for p in [r"C:\Windows\System32\nvidia-smi.exe",
                  r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"]:
            if Path(p).exists():
                nvidia_smi = p
                break
    if not nvidia_smi:
        return None
    try:
        r = subprocess.run([nvidia_smi, "--query-gpu=name,memory.total,driver_version",
                           "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            parts = [x.strip() for x in r.stdout.strip().split(",")]
            if len(parts) >= 2:
                return {
                    "count": 1,  # conservative
                    "name": parts[0],
                    "vram_gb": round(float(parts[1]) / 1024, 1),
                    "cuda_version": f"driver {parts[2]}" if len(parts) >= 3 else "",
                    "cc": 0,
                    "source": "nvidia-smi",
                }
    except Exception:
        pass
    return None

def _detect_gpu_via_torch():
    """Use torch (via best available Python) to detect GPU — most accurate."""
    python_exe = str(_find_python_exe())
    gpu_code = (
        "import json, torch\n"
        "d = {'count': 0, 'vram_gb': 0, 'name': '', 'cuda_version': '', 'cc': 0}\n"
        "if torch.cuda.is_available():\n"
        "    d['count'] = torch.cuda.device_count()\n"
        "    d['cuda_version'] = str(torch.version.cuda)\n"
        "    d['name'] = torch.cuda.get_device_name(0)\n"
        "    p = torch.cuda.get_device_properties(0)\n"
        "    d['vram_gb'] = round(p.total_memory / (1024**3), 1)\n"
        "    d['cc'] = p.major\n"
        "print(json.dumps(d, ensure_ascii=False))\n"
    )
    try:
        r = subprocess.run([python_exe, "-c", gpu_code],
            capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            info = json.loads(r.stdout.strip())
            if info.get("count", 0) > 0:
                info["source"] = "torch"
                return info
        # Log failure for debugging
        stderr_tail = (r.stderr or "")[-200:]
        logger.warning(f"Torch GPU detect via {python_exe}: rc={r.returncode} stderr={stderr_tail}")
    except json.JSONDecodeError:
        logger.warning(f"Torch GPU detect JSON parse failed, stdout={r.stdout[:200] if r else 'N/A'}")
    except Exception as e:
        logger.warning(f"Torch GPU detect exception: {e}")
    return None

def api_chat_auto_detect():
    """Auto-detect hardware and return recommended generation parameters.
    Tries torch first, falls back to nvidia-smi, then reports no GPU."""
    gpu_info = None

    # Method 1: torch (most accurate, gives compute capability)
    gpu_info = _detect_gpu_via_torch()

    # Method 2: nvidia-smi (OS-level, always works if driver installed)
    if not gpu_info:
        gpu_info = _detect_gpu_via_nvidia_smi()

    # Fallback: no GPU
    if not gpu_info:
        gpu_info = {"count": 0, "vram_gb": 0, "name": "", "cuda_version": "", "cc": 0, "source": "none"}

    cpu_ram_gb = 0
    try:
        import psutil
        cpu_ram_gb = psutil.virtual_memory().total // (1024**3)
    except ImportError:
        pass

    vram = gpu_info["vram_gb"]
    has_cuda = gpu_info["count"] > 0

    rec = {}
    if vram >= 24:
        rec["max_new_tokens"] = 1024
    elif vram >= 16:
        rec["max_new_tokens"] = 768
    elif vram >= 12:
        rec["max_new_tokens"] = 512
    elif vram >= 8:
        rec["max_new_tokens"] = 256
    elif has_cuda:
        rec["max_new_tokens"] = 128
    else:
        rec["max_new_tokens"] = 64

    rec["temperature"] = 0.9
    rec["top_p"] = 0.95
    rec["top_k"] = 50
    rec["repetition_penalty"] = 1.15
    rec["do_sample"] = True
    rec["num_beams"] = 1

    detect_source = gpu_info.get("source", "none")
    notes = []
    if has_cuda:
        notes.append(f"GPU: {gpu_info['name']} ({vram}GB VRAM) [via {detect_source}]")
        if vram >= 24:
            notes.append("显存充足，推荐 max_new_tokens=1024")
        elif vram >= 12:
            notes.append("显存适中，推荐 max_new_tokens=512")
        else:
            notes.append("显存有限，建议 max_new_tokens≤256 避免 OOM")
    else:
        if detect_source == "none":
            notes.append("未检测到 GPU（nvidia-smi 和 torch 均未找到），将使用 CPU 推理")
        else:
            notes.append("未检测到支持 CUDA 的 GPU，将使用 CPU 推理（较慢）")
        notes.append("建议 max_new_tokens≤64")
    if cpu_ram_gb > 0:
        notes.append(f"系统内存: {cpu_ram_gb}GB")

    return {
        "hardware": {
            "gpu_name": gpu_info["name"],
            "gpu_count": gpu_info["count"],
            "vram_gb": vram,
            "cuda_version": gpu_info.get("cuda_version", ""),
            "cpu_ram_gb": cpu_ram_gb,
            "compute_capability": gpu_info.get("cc", 0),
            "detect_source": detect_source,
        },
        "recommended": rec,
        "notes": notes,
    }

def api_chat_load(params):
    global infer_proc, active_model_name, chat_history
    model_path = params.get("model_path", str(GLM4_MODEL))
    adapter_path = params.get("adapter_path")
    cache_name = params.get("cache_name", Path(model_path).name)
    # Always kill existing models to free GPU memory (12GB only fits one 9B model)
    _kill_all_models()

    # Extract generation params from request
    gen_params = {k: v for k, v in params.items() if k in (
        "max_new_tokens", "temperature", "top_p", "top_k",
        "repetition_penalty", "do_sample", "num_beams", "system_prompt"
    )}
    script = _build_chat_server_script(model_path, adapter_path, gen_params)
    script_path = LLAMA_FACTORY / "_chat_server.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)

    python_exe = str(_find_python_exe())
    env = {**os.environ, "PYTHONPATH": str(LLAMA_FACTORY / "src"),
           "HF_ENDPOINT": HF_MIRROR, "PYTHONIOENCODING": "utf-8",
           "PYTHONUNBUFFERED": "1"}

    try:
        infer_proc = subprocess.Popen([python_exe, str(script_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env)

        # Python 3.7.0 bug: errors="replace" with text=True is ignored.
        # Decode manually to avoid UnicodeDecodeError on CJK paths.
        def _safe_readline(pipe):
            raw = pipe.readline()
            if not raw:
                return None
            return raw.decode("utf-8", errors="replace").strip()

        while True:
            line = _safe_readline(infer_proc.stdout)
            if line is None:
                break
            if line.startswith("READY:"):
                info = line[6:]
                loaded_models[cache_name] = {
                    "proc": infer_proc, "chat_history": [],
                    "model_path": model_path, "adapter_path": adapter_path,
                }
                active_model_name = cache_name
                chat_history = []
                broadcast_event("model_loaded", {"name": cache_name, "info": info})
                return {"success": True, "name": cache_name, "info": info}
            elif line.startswith("ERROR:"):
                return {"error": line[6:]}

        stderr_out = b""
        if infer_proc.stderr:
            stderr_out = infer_proc.stderr.read()
        err_text = stderr_out.decode("utf-8", errors="replace")[:500] if stderr_out else ""
        return {"error": f"Model process ended unexpectedly: {err_text}"}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Chat load error: {tb}")
        return {"error": str(e), "tb": tb[-400:]}

def api_chat_send(params):
    global chat_history
    if not infer_proc or infer_proc.poll() is not None:
        return {"error": "No model loaded"}
    user_msg = params.get("message", "").strip()
    if not user_msg:
        return {"error": "Empty message"}
    try:
        with infer_lock:
            req = json.dumps({"message": user_msg, "history": chat_history[-10:]}, ensure_ascii=False)
            infer_proc.stdin.write((req + "\n").encode("utf-8"))
            infer_proc.stdin.flush()
            raw = infer_proc.stdout.readline()
            if not raw:
                return {"error": "No response from model"}
            response_line = raw.decode("utf-8", errors="replace").strip()
            resp = json.loads(response_line)
            if "error" in resp:
                return {"error": resp["error"]}
            assistant_msg = resp.get("response", "")
            chat_history.append({"role": "user", "content": user_msg})
            chat_history.append({"role": "assistant", "content": assistant_msg})
            if active_model_name and active_model_name in loaded_models:
                loaded_models[active_model_name]["chat_history"] = list(chat_history)
            return {"success": True, "response": assistant_msg, "history": chat_history}
    except Exception as e:
        return {"error": str(e)}

def _kill_all_models():
    """Kill all loaded model subprocesses and clear state."""
    global infer_proc, active_model_name, chat_history, loaded_models
    for name, entry in list(loaded_models.items()):
        proc = entry["proc"]
        if proc.poll() is None:
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass
    loaded_models.clear()
    infer_proc = None
    active_model_name = None
    chat_history = []
    # Give GPU driver time to free memory before loading next model
    time.sleep(2)

def api_chat_unload():
    global active_model_name
    name = active_model_name
    _kill_all_models()
    broadcast_event("model_unloaded", {"name": name})
    return {"success": True}

def api_chat_switch(params):
    """Switch is not supported on limited VRAM — just unload and let user reload."""
    return {"error": "请先卸载当前模型，再加载新模型（12GB 显存只够放一个 9B 模型）"}

def _build_chat_server_script(model_path, adapter_path=None, gen_params=None):
    mp = model_path.replace("\\", "/")
    adapter_code = ""
    merge_code = ""
    if adapter_path:
        ap = adapter_path.replace("\\", "/")
        adapter_code = f'adapter_path = "{ap}"'
        merge_code = """
from peft import PeftModel
model = PeftModel.from_pretrained(model, adapter_path)
# Do NOT merge_and_unload with 4-bit — merging breaks quantized weights.
# Keep adapter loaded as separate LoRA layers.
print("LoRA adapter loaded", flush=True)
"""
    # Generation parameters
    gp = gen_params or {}
    def _gp(key, default):
        v = gp.get(key, default)
        if v is None or v == "": return default
        return v
    max_nt = int(_gp("max_new_tokens", 256))
    temp = float(_gp("temperature", 0.9))
    top_p = float(_gp("top_p", 0.95))
    top_k = int(_gp("top_k", 50))
    rep_pen = float(_gp("repetition_penalty", 1.15))
    do_samp = _gp("do_sample", True)
    if isinstance(do_samp, str): do_samp = do_samp.lower() in ("true", "1", "yes")
    n_beams = int(_gp("num_beams", 1))
    user_sys_prompt = json.dumps(_gp("system_prompt", ""))
    return f'''import sys, json, torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_path = "{mp}"
{adapter_code}
print("Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load system prompt from training data (if adapter has one)
system_prompt = "你是一个友好的AI助手。"
try:
    if adapter_path:
        sp_file = os.path.join(adapter_path, "system_prompt.txt")
        if os.path.exists(sp_file):
            with open(sp_file, "r", encoding="utf-8") as f:
                system_prompt = f.read().strip()
            print(f"Loaded training system prompt ({{len(system_prompt)}} chars)", flush=True)
except Exception:
    pass
# User-specified system prompt overrides training prompt
user_sp = {user_sys_prompt}
if user_sp: system_prompt = user_sp

# Use best available dtype and attention implementation
has_cuda = torch.cuda.is_available()
quant_kw = {{}}
if has_cuda:
    major = torch.cuda.get_device_capability()[0]
    use_bf16 = major >= 8
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    attn_kw = {{"attn_implementation": attn_impl}}
    # Use 4-bit quantization to fit 9B model in 12GB VRAM
    quant_kw = {{
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=False,
        )
    }}
    print(f"GPU cap={{major}}.x, bf16={{use_bf16}}, attn={{attn_impl}}, 4bit=True", flush=True)
else:
    dtype = torch.float32
    attn_kw = {{}}

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto" if has_cuda else None,
    trust_remote_code=True,
    **attn_kw,
    **quant_kw,
)
{merge_code}

print("READY:" + model_path, flush=True)

for line in sys.stdin:
    try:
        req = json.loads(line.strip())
        user_msg = req.get("message", "")
        history = req.get("history", [])
        messages = [{{"role": "system", "content": system_prompt}}]
        for h in history:
            messages.append({{"role": h["role"], "content": h["content"]}})
        messages.append({{"role": "user", "content": user_msg}})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        if has_cuda:
            inputs = {{k: v.cuda() for k, v in inputs.items()}}
        with torch.no_grad():
            gen_kwargs = {{k: v for k, v in [
                ("max_new_tokens", {max_nt}), ("do_sample", {do_samp}),
                ("temperature", {temp}), ("top_p", {top_p}),
                ("top_k", {top_k}), ("repetition_penalty", {rep_pen}),
                ("num_beams", {n_beams})
            ]}}
            outputs = model.generate(**inputs, **gen_kwargs)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(json.dumps({{"response": response}}, ensure_ascii=False), flush=True)
    except Exception as e:
        print(json.dumps({{"error": str(e)}}, ensure_ascii=False), flush=True)
'''

def api_system_stats():
    stats = {}
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        stats["cpu"] = {"percent": cpu, "cores": psutil.cpu_count()}
        stats["ram"] = {"percent": mem.percent, "used_gb": mem.used // (1024**3),
                        "total_gb": mem.total // (1024**3)}
        stats["disk"] = {"percent": disk.percent, "used_gb": disk.used // (1024**3),
                         "total_gb": disk.total // (1024**3)}
    except ImportError:
        stats["cpu"] = {"percent": 0, "error": "psutil not installed"}
        stats["ram"] = {"percent": 0, "error": "psutil not installed"}
        stats["disk"] = {"percent": 0, "error": "psutil not installed"}

    # Detect GPU via subprocess (works regardless of current Python version)
    gpu_info = {"cuda_available": False, "cuda_version": "", "devices": []}
    gpu_code = (
        "import json, torch\n"
        "d = {'ok': False}\n"
        "if torch.cuda.is_available():\n"
        "    d['ok'] = True\n"
        "    d['cuda_version'] = str(torch.version.cuda)\n"
        "    d['device_count'] = torch.cuda.device_count()\n"
        "    d['devices'] = []\n"
        "    for i in range(torch.cuda.device_count()):\n"
        "        p = torch.cuda.get_device_properties(i)\n"
        "        d['devices'].append({'id': i, 'name': torch.cuda.get_device_name(i),\n"
        "            'memory_total_mb': p.total_memory // (1024**2),\n"
        "            'memory_used_mb': torch.cuda.memory_allocated(i) // (1024**2),\n"
        "            'compute_capability': f'{p.major}.{p.minor}'})\n"
        "print(json.dumps(d, ensure_ascii=False))\n"
    )
    python_exe = _find_python_exe()
    try:
        r = subprocess.run([str(python_exe), "-c", gpu_code],
            capture_output=True, text=True, timeout=15,
            env={**os.environ, "HF_ENDPOINT": HF_MIRROR})
        if r.returncode == 0:
            d = json.loads(r.stdout.strip())
            if d.get("ok"):
                gpu_info = {"cuda_available": True, "cuda_version": d["cuda_version"],
                           "device_count": d["device_count"], "devices": d["devices"]}
    except Exception:
        pass
    stats["gpu"] = gpu_info
    stats["training_running"] = _is_training_running()
    stats["model_loaded"] = active_model_name is not None
    stats["active_model"] = active_model_name
    return stats

def api_logs_list():
    logs = []
    if LOG_DIR.exists():
        for f in sorted(LOG_DIR.glob("*.log"), reverse=True):
            logs.append({"name": f.name, "size": f.stat().st_size,
                         "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()})
    return {"logs": logs}

def api_logs_get(params):
    name = params.get("name", "")
    log_path = LOG_DIR / name
    if not log_path.exists():
        return {"error": "Log not found"}
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        content = "".join(lines[-300:])
        return {"name": name, "total_lines": len(lines), "content": content}
    except Exception as e:
        return {"error": str(e)}

def api_config_get():
    cfg = _load_config()
    return {
        "llama_factory": cfg.get("llama_factory", str(LLAMA_FACTORY)),
        "models_dir": cfg.get("models_dir", str(MODELS_DIR)),
        "ai_env_dir": cfg.get("ai_env_dir", str(AI_ENV_DIR)),
    }

def api_detect_sender(params):
    content = _get_chat_content(params)
    if not content:
        return {"error": "No file content. Upload a file or specify a valid filepath."}
    try:
        lines = content.splitlines()[:200]
        sender_counts = {}
        pattern = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2}\s+(.+)$")
        for line in lines:
            m = pattern.match(line.strip())
            if m:
                sender = m.group(1).strip()
                sender_counts[sender] = sender_counts.get(sender, 0) + 1
        if sender_counts:
            return {"sender": max(sender_counts, key=sender_counts.get)}
        return {"sender": ""}
    except Exception as e:
        return {"error": str(e)}

# ── HTTP Server ────────────────────────────────────────────────

ROUTES = {
    ("GET", "/api/status"): api_status,
    ("GET", "/api/env-check"): api_env_check,
    ("POST", "/api/config/full"): api_config_full,
    ("POST", "/api/config/copy"): api_config_copy,
    ("POST", "/api/config/deps"): api_config_deps,
    ("GET", "/api/config"): api_config_get,
    ("GET", "/api/wechat/status"): api_wechat_status,
    ("POST", "/api/wechat/launch-memotrace"): api_launch_memotrace,
    ("POST", "/api/convert/preview"): api_convert_preview,
    ("POST", "/api/convert/run"): api_convert_run,
    ("POST", "/api/convert/analyze-cutoff"): api_convert_analyze_cutoff,
    ("POST", "/api/convert/detect-sender"): api_detect_sender,
    ("GET", "/api/finetune/auto-detect"): api_auto_detect_config,
    ("GET", "/api/finetune/datasets"): api_finetune_datasets,
    ("POST", "/api/finetune/gen-config"): api_finetune_gen_config,
    ("POST", "/api/finetune/start"): api_finetune_start,
    ("POST", "/api/finetune/stop"): api_finetune_stop,
    ("POST", "/api/finetune/preprocess"): api_finetune_preprocess,
    ("GET", "/api/finetune/status"): api_finetune_status,
    ("GET", "/api/chat/models"): api_chat_models,
    ("GET", "/api/chat/auto-detect"): api_chat_auto_detect,
    ("POST", "/api/chat/load"): api_chat_load,
    ("POST", "/api/chat/send"): api_chat_send,
    ("POST", "/api/chat/unload"): api_chat_unload,
    ("POST", "/api/chat/switch"): api_chat_switch,
    ("GET", "/api/system/stats"): api_system_stats,
    ("GET", "/api/logs"): api_logs_list,
    ("GET", "/api/logs/get"): api_logs_get,
}

def _get_ui_html():
    return (Path(__file__).parent / "ui.html").read_text(encoding="utf-8")

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        # Try UTF-8 first (browser), fall back to GBK (some terminal clients)
        for enc in ("utf-8", "gbk"):
            try:
                body = raw.decode(enc)
                return json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        # Last resort: decode with replacement
        body = raw.decode("utf-8", errors="replace")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        # SSE for real-time events
        if path == "/api/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            q = queue.Queue()
            event_queues.append(q)
            try:
                while True:
                    try:
                        msg = q.get(timeout=15)
                        self.wfile.write(f"data: {msg}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(": keepalive\n\n".encode("utf-8"))
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                try:
                    event_queues.remove(q)
                except ValueError:
                    pass
            return

        # Serve UI
        if path == "/" or path == "/index.html":
            body = _get_ui_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(body)
            return

        # Route to API
        handler = ROUTES.get(("GET", path))
        if handler:
            flat_params = {k: v[0] if len(v) == 1 else v for k, v in params.items()}
            try:
                try:
                    result = handler(flat_params)
                except TypeError:
                    result = handler()
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return

        self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        handler = ROUTES.get(("POST", path))
        if handler:
            try:
                params = self._read_body()
                try:
                    result = handler(params)
                except TypeError:
                    result = handler()
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return

        self._send_json({"error": "Not found"}, 404)


def main():
    port = 8088
    host = "127.0.0.1"
    server = ThreadingHTTPServer((host, port), RequestHandler)
    print(f"AI Toolbox server running at http://{host}:{port}")
    logger.info(f"Server started on {host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        logger.info("Server stopped")


if __name__ == "__main__":
    main()
