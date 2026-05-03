[README.md](https://github.com/user-attachments/files/27314866/README.md)
#做这玩意的初衷是给盆友体验的，专门买了个u盘把模型，环境什么的都给他下载好，插电脑上就能用。所以放在这里的就比较简陋，大概率也不会有人真的拿过来用。不过做还是要做全套的，你说跟对象聊天聊什么，能让人这么上头，做这个的目的也就是为了让模型能暂时性的成为一个人，能记住所有小习惯，所有信息，至少能陪着聊聊天，让那个人走得再远一点吧。
为了提取微信的聊天记录，这里引用了另一个项目叫做memotrace，微信版本太高的用不了，找个3.9及以前的版本就能用了


## 功能
自动检测 Python / PyTorch / CUDA / 模型 / 框架是否就绪 
创建虚拟环境、安装依赖、注册数据集 
调用 MemoTrace 提取微信聊天记录
将微信 `.txt` 导出文件转为训练数据（ShareGPT 格式），支持人设定制、发送者自动检测、token 长度分析 
基于 LLaMA-Factory 做 QLoRA 微调，40+ 可调参数，自动根据显存推荐配置 
加载微调后的 LoRA 适配器对话，支持温度/top_p 等全参数调节 

### 环境要求

- Windows 10 / 11
- NVIDIA 显卡（推荐 8GB+ 显存，4-bit 量化最低 8GB 可微调 9B 模型）
- 或仅 CPU（仅对话，速度较慢）

### 1. 获取模型

下载 GLM-4-9B-Chat 模型到 `models/glm-4-9b-chat/`：

```bash
# 方式一：HuggingFace
git lfs install
git clone https://huggingface.co/THUDM/glm-4-9b-chat models/glm-4-9b-chat

# 方式二：ModelScope
git clone https://modelscope.cn/ZhipuAI/glm-4-9b-chat.git models/glm-4-9b-chat
```

### 2. 安装 Python 环境

项目自带 `pytorch-env/` 便携 Python 环境。如果没有，手动创建：

```bash
# 创建虚拟环境
python -m venv pytorch-env

# 激活并安装依赖
pytorch-env\Scripts\python.exe -m pip install torch transformers accelerate peft datasets \
    einops bitsandbytes scipy sentencepiece protobuf huggingface_hub
```

### 3. 获取 MemoTrace

从 [MemoTrace 项目](https://github.com/LC044/WeChatMsg/releases) 下载 `MemoTrace.exe`，放到根目录。

### 4. 启动

```bash
# 桌面版
启动AI工具箱.bat

# 或 Web 版
启动Web工具箱.bat
```

---

## 使用流程

### 完整微调流程

```
微信聊天 → MemoTrace 导出 .txt → 转换训练数据 → 微调模型 → 加载对话
```

#### 第一步：提取聊天记录

打开「微信工具」面板 → 启动 MemoTrace → 选择微信账号 → 导出聊天记录为 `.txt`

#### 第二步：转换训练数据

1. 打开「聊天记录转换」面板
2. 选择导出的 `.txt` 文件
3. 设置发送者名称（此人的消息作为 user，对方作为 assistant）
4. 可选：在「人设」框中输入 AI 的性格描述
5. 点击「检测序列长度」获取 token 分析
6. 点击「转换并保存」

#### 第三步：微调模型

1. 打开「模型微调」面板
2. 点击「自动检测配置」获取硬件推荐
3. 检查各个参数（见下方参数说明）
4. 可选：点击「生成配置」预览 YAML
5. 点击「开始训练」
6. 训练完成后，LoRA 适配器保存在 `LLaMA-Factory/output/` 中

#### 第四步：对话测试

1. 打开「模型对话」面板
2. 选择基础模型 + 你的 LoRA 适配器
3. 点击「自动检测」获取生成参数推荐
4. 点击「加载」
5. 开始对话

---

## 微调参数说明

<details>
<summary><b>基础配置</b></summary>

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 模型路径 | `models/glm-4-9b-chat` | 基座模型目录 |
| 数据集 | 下拉选择 | 在「聊天记录转换」中创建的数据集 |
| 输出目录 | `custom_lora` | LoRA 适配器保存位置 |
| 对话模板 | `glm4` | GLM-4 的 ChatML 模板 |
| 训练阶段 | `sft` | sft=监督微调, dpo=偏好对齐, pt=预训练 |
| 微调方式 | `lora` | lora=低秩适配, freeze=冻结微调, full=全参, oft=正交微调 |
| 验证集比例 | 0 | 0 为不验证，0.05 即 5% |

</details>

<details>
<summary><b>LoRA 配置</b></summary>

| 参数 | 默认值 | 说明 |
|------|--------|------|
| LoRA Rank | 8 | 秩越大越强，但更慢（8-64） |
| LoRA Alpha | 16 | 缩放系数，通常 = rank × 2 |
| LoRA Dropout | 0.0 | 0-0.5，防止过拟合 |
| LoRA Target | all | 目标模块，`all` 覆盖全部线性层 |
| 使用 DoRA | False | 权重分解 LoRA，效果更好 |
| 使用 RSLoRA | False | 秩稳定 LoRA |
| 额外训练模块 | 空 | 逗号分隔，如 `lm_head,embed_tokens` |

</details>

<details>
<summary><b>训练超参</b></summary>

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 训练轮数 | 3 | 聊天数据通常 3-5 轮足够 |
| 批次大小 | 自动 | 显存越大可设越大 |
| 梯度累积 | 自动 | 有效批次 = batch × 梯度累积 |
| 学习率 | 1e-4 | LoRA 推荐 1e-4 ~ 5e-4 |
| 序列长度 | 自动 | 超过此长度的文本被截断 |
| 学习率调度器 | cosine | cosine 和 constant_with_warmup 最常用 |
| 预热比例 | 0.05 | 前 5% 步数线性升温 |
| 最大梯度范数 | 1.0 | 梯度裁剪，0 为不裁剪 |
| 权重衰减 | 0.0 | L2 正则化 |
| 最大训练步数 | -1 | -1=由 epochs 决定 |
| 随机种子 | 42 | 固定种子保证可复现 |

</details>

<details>
<summary><b>优化与量化</b></summary>

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 量化位数 | 自动 | 8GB 显存建议 4-bit；≥24GB 可选 none |
| 双重量化 | 自动 | 进一步压缩，省约 0.5GB 显存 |
| Flash Attention | auto | fa2 可提速 + 省显存 |
| 使用 bf16 | True | RTX 30 系列及以上支持 |
| Liger Kernel | False | 开源训练加速 kernel |
| 优化器 | adamw_torch | adamw_8bit 可进一步省显存 |

</details>

<details>
<summary><b>高级设置</b></summary>

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 日志步数 | 20 | 每 N 步输出一次 loss |
| 保存步数 | 200 | 每 N 步保存一次检查点 |
| 最多保存数 | 3 | 只保留最近 N 个检查点 |
| 覆盖输出目录 | True | 覆盖同名输出 |
| 覆盖缓存 | False | 强制重新预处理数据 |
| 预处理进程数 | 自动 | Windows 建议 1-4 |
| 断点续训路径 | 空 | 填写 checkpoint 目录路径 |

</details>

---

## 生成参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| max_new_tokens | 256 | 单次回复最长 token 数 |
| temperature | 0.9 | 0=确定, 1=创意, >1=胡言乱语 |
| top_p | 0.95 | 核采样，累积概率阈值 |
| top_k | 50 | 仅从概率最高的 K 个 token 中采样 |
| repetition_penalty | 1.15 | >1 减少重复，太高会不自然 |
| do_sample | True | False=贪心解码（更稳定） |
| num_beams | 1 | 1=采样, >1=束搜索（更慢但质量可能更好） |
| system prompt | 空 | 自定义人设，留空则自动加载训练时的人设 |

---

## 文件结构

```
├── app/
│   ├── main.py              # 桌面 GUI（所有代码在一个文件）
│   ├── server.py            # Web 后端 API
│   └── ui.html              # Web 前端（单页面，所有 CSS/JS 内嵌）
├── LLaMA-Factory/           # 开源微调框架 v0.30.1
├── models/                  # 模型存放（不上传，需自行下载）
│   └── glm-4-9b-chat/
├── 启动AI工具箱.bat          # 启动桌面版
├── 启动Web工具箱.bat         # 启动 Web 版
├── .gitignore
└── README.md
```

## 技术栈

- **前端**: Tkinter (桌面) / 原生 HTML+CSS+JS (Web)
- **后端**: Python stdlib HTTP server (无框架依赖)
- **微调引擎**: LLaMA-Factory + HuggingFace Transformers + PEFT
- **量化**: BitsAndBytes 4-bit QLoRA
- **模型**: GLM-4-9B-Chat (智谱 AI)

## 常见问题

**Q: 插入另一台电脑后 Python 环境用不了？**
A: `pytorch-env/` 是便携 Python，换电脑后可能需要重建：删除该目录，用「自动配置」面板一键重建。

**Q: 训练时显存不足 (OOM)？**
A: 减小 `序列长度`、`批次大小`、`LoRA Rank`，或增大 `梯度累积`。

**Q: 聊天记录转换后训练效果不好？**
A: 检查几点：发送者是否正确识别、人设是否写清楚、聊天记录质量（至少 200 条）。

**Q: 为什么不上传模型文件？**
A: GLM-4-9B 约 18GB，远超 GitHub 100MB 限制。模型需从 HuggingFace/ModelScope 单独下载。

**Q: 支持其他模型吗？**
A: LLaMA-Factory 支持 GLM/Qwen/Llama/Baichuan/等
