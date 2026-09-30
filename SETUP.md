# 口语陪练 —— 搭建指南

## 怎么启动

**只有一个入口**：

```bash
cd kai
.venv/bin/python -m app                # 本机用，浏览器打开 http://127.0.0.1:8000
.venv/bin/python -m app --ssl          # 手机用，需要 HTTPS
.venv/bin/python -m app --port 8001    # 换端口
```

想省事就用 `.venv/bin/python run.py`（等价，会打印访问地址）。

> ⚠️ **不要运行 `python server.py`** —— 那个文件已经删掉了。
> 它是原型期的入口，正式版拆成了 `app/` 包。
> 如果你看到 `can't open file '.../server.py'`，就是这个原因，
> 换成上面的 `-m app` 即可。原型仍完整保留在 `prototype/` 里。


## 怎么说话：按住空格

进到练习界面后，**按住空格键说话，松开表示说完了**。

这样做是为了让对话两边都不会互相打断：

- 不按空格时**完全不采集**你的声音，所以背景音、咳嗽、键盘声都不会被当成"你在说话"
- 松开空格是明确的"我说完了"，AI 不需要靠猜（语音活动检测）来判断断句
- 于是不会出现"AI 半天不吭声，然后突然接上一句还把我打断"

按下空格时会立刻停下 AI 的播放（模拟真人被抢话），下方绿色提示条表示正在收音。


## ✅ 当前方案：Qwen Realtime（对话）+ 声纹过滤 + 文本纠错

```
对话层  →  阿里云百炼 qwen3.8-omni-flash-realtime   （个人可开通，国内直连）
纠错层  →  qwen-plus（OpenAI 兼容接口）
声纹层  →  3D-Speaker CAM++（纯本地 ONNX，不出本机）
```

### 目录结构

| 路径 | 用途 |
|---|---|
| `app/api/server.py` | HTTP + WebSocket 服务端（FastAPI） |
| `app/core/` | 协议、实时连接、声纹、录入（不依赖上层） |
| `app/services/` | 备课、纠错、会话编排、用户与画像 |
| `app/storage/` | SQLite、表结构、迁移 |
| `app/web/index.html` | 前端（单文件） |
| `models/campplus.onnx` | 声纹模型，**需自行下载**（见下文） |
| `scripts/` | 运维脚本（证书生成、连通性自检、公开性检查、压测） |
| `prototype/` | 原型期的旧代码，**已冻结**，仅供对照 |

### 三步跑起来

```bash
cd kai                              # 进到本项目目录
cp .env.example .env                # 编辑 .env，填 DASHSCOPE_API_KEY 和 QWEN_WORKSPACE_ID
mkdir -p models && curl -L -o models/campplus.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
.venv/bin/python scripts/check_qwen.py   # 自检：确认密钥/区域/模型都能用（几乎零成本）
.venv/bin/python -m app             # 启动，浏览器打开 http://127.0.0.1:8000
```

> **出问题先跑自检**。它只连一次、不发音频，能把"配置错"和"代码错"分开。
> 常见结果：`HTTP 401` = 密钥错；配置项没填 = 会明确指出来。

### ⚠️ 外放 vs 耳机：回声自问自答问题

**现象**：AI 只回应第一句，之后自己跟自己无限对话。

**原因**：你的 MacBook **麦克风和扬声器同时工作**，AI 的外放声音被麦克风拾取，
服务端 VAD 误判为"用户在说话"，于是 AI 回应自己 → 死循环。
（官方文档也确认 WebSocket 协议**不含回声消除/降噪**，需客户端自行处理。）

**已内置对策**：网页版通过浏览器的 `getUserMedia` 开启
**回声消除（echoCancellation）+ 降噪（noiseSuppression）+ 自动增益（autoGainControl）**，
并向服务端汇报「AI 正在说话」，这段时间不判定用户语音 —— 所以外放时也能随时插话。

这三项目前是写死的（见 `app/web/index.html` 的 `startMic`），不建议关：

| 关掉哪项 | 后果 |
|---|---|
| `echoCancellation` | AI 听见自己 → 自问自答 |
| `autoGainControl` | 音量过低 → VAD 不触发 → 感觉变慢 |
| `noiseSuppression` | 影响声纹判定（三项目中对声纹影响最大的一个） |

> 原型期的命令行版本（`realtime_qwen.py`）有 `--headphone` / `--barge-in` 等参数，
> 那些已随原型冻结，见 `prototype/`。**当前的网页版没有这些开关。**

### 🎙️ 声纹过滤：只认你的声音

**能解决什么**：旁人在旁边说话、电视声、家人聊天 —— 只要不是你的声音，
就不会被当成输入发给模型。这比单纯调音量阈值可靠得多，因为它是**认人**，不是认响度。

#### 1. 录入声纹（在网页里做，可反复重录）

打开网页 → 左上角用户名 → 新建或选择用户 → 自动进入**引导式录入**：

- 跟读 5 句话，每句读完点「读完了」
- **至少 3 句**才能保存（少于 3 句没法做多数表决）
- 每句当场校验：太短、太安静、和前面差别太大都会被拒并让你重读
- 录完显示质量分（好 / 一般 / 偏差）

**随时可以重录**：再点一次用户名 → 重新走一遍流程即可。
重录只覆盖声纹文件，画像、历史、纠错记录都不动。

声纹按用户分开存放：`data/voiceprints/<user_id>.json`。
每个用户一份，互不覆盖 —— 这正是多人共用一台电脑能各自练习的前提。

#### 1.5 单句多长才够？为什么是 2 秒

实测（CAM++，同一个人 vs 不同人各取多段算余弦）：

| 窗口 | 同人均值 | 异人均值 | 两类间隔 |
|---|---|---|---|
| 0.5s | 0.477 | 0.322 | 0.156 |
| 1.0s | 0.582 | 0.372 | 0.210 |
| **2.0s** | **0.672** | **0.422** | **0.249** |
| 3.0s | 0.745 | 0.458 | 0.287 |

1 秒窗口下同人分数低到 0.36、异人高到 0.51，**分布重叠严重**，
任何阈值都只能二选一：要么频繁误拒本人，要么频繁放进旁人。
2 秒以上才把两类真正分开。所以：

- `MIN_SPEECH_SEC = 2.0`（录入时每句的最短有效语音）
- `SpeakerVerifier(window_sec=2.0)`（判定时累积满 2 秒才出分）

实测标定（同人 45 对 / 异人 70 对）：
同人两两最低 0.485、均值 0.644；异人两两最高 0.360、均值 0.200。
中间 0.36~0.49 是真空带，判定阈值 `0.5` 和拒录阈值
`MIN_COHERENCE = 0.42` 都落在带内。
`tests/test_stage7_users.py` 里有回归测试钉住这几个数 ——
它们很容易在后续改动中被"顺手调一下"，所以连同来历一起写在测试里。

#### 2. 正常使用（自动启用）

在网页里勾选「只回应我的声音」即可。会话建立时会看到：

```
声纹已启用 (user=Alan good)
```

如果该用户还没录入声纹，会明确收到一条提示
（`voiceprint_missing`），而不是静默放行 ——
让你知道"这次没有过滤"，而不是困惑于"为什么外人说话 AI 也回"。

#### 2.5 声纹出问题时的排查顺序

1. **换麦克风了吗** —— 头号原因。骨传导（如 Shokz）与外置麦、
   内置麦之间特征差异很大。直接重录一次最省事。
2. **换环境了吗** —— 房间混响、背景噪声变化都会影响。
   重录，并且**在以后常用的位置录**。
3. **感冒/嗓子哑** —— 特征确实会变，好了再重录。
4. **只是想临时关掉** —— 取消勾选「只回应我的声音」即可，
   不用删声纹。

#### 2.6 声纹诊断脚本（原型期遗留，仍可用）

**如果"我正常说话却被拦"，不要靠猜阈值 —— 先测。**

```bash
.venv/bin/python diagnose_voiceprint.py                 # 录 3 段，测真实分数
.venv/bin/python diagnose_voiceprint.py --list-devices  # 看有哪些麦克风
.venv/bin/python diagnose_voiceprint.py --compare-devices
                                                        # 每台设备各录一段对比
```

它会告诉你：你真人说话到底得多少分、建议阈值是多少，
并写入 `voiceprint_diagnosis.json`。

**⚠️ 头号原因：换了麦克风**

声纹是在某个麦克风上录的，换设备后特征会显著变化。
骨传导耳机（如 Shokz）与外置麦、内置麦之间差异尤其大。

```bash
.venv/bin/python diagnose_voiceprint.py --compare-devices
```

会按相似度排序各设备，直接指出哪台最匹配。

**阈值现在不用改代码**：判定阈值固定在 `SpeakerVerifier` 的
`threshold=0.5`（实测标定，见上文 1.5 节）。要临时停用过滤，
就在网页上取消勾选「只回应我的声音」。

> 下面这些 `server.py` / `diagnose_voiceprint.py` / `enroll_voice.py`
> 都是**原型期的独立脚本**（`prototype/` 目录，已冻结）。
> 正式版的录入和过滤都在网页里做，不需要跑它们。
> 保留说明是因为排查思路（尤其是换设备）仍然适用。

> **重要：不要用 TTS（如 macOS `say`）测试声纹。**
> 实测 TTS 音频与真人语音不在同一特征分布 ——
> TTS 对你的真人档案只有 0.01~0.19 分，这个数据没有参考价值。

> **排查记录**：曾出现"三台设备全部采集不到声音"的情况，
> 实际原因是 `SpeakerVerifier` 的静音门槛按 int16 量级（±32768）编写，
> 而诊断脚本喂的是 float32（±1.0）音频 → RMS 恒小于门槛 → 永远出不了分。
> 已在 `core.py` 中按 dtype 归一化修复，float32 / int16 两种输入都支持。
> 详见 `问题诊断-声纹采集不到声音.md`。

### 浏览器 DSP（AEC / 降噪 / 自动增益）该不该关？

**结论：全部保持开启。** 实测关闭后声纹提升很小，但代价明确：

| 开关 | 建议 | 关掉会怎样 |
|---|---|---|
| `echoCancellation` | ★ 必须开 | **外放时 AI 听到自己 → 自言自语** |
| `autoGainControl` | ★ 应该开 | 音量偏小 → 服务端 VAD（阈值 0.5）不触发 → **感觉"变慢"** |
| `noiseSuppression` | 可开可关 | 唯一对声纹有影响的，但实测影响很小 |

其中「感觉变慢」不是错觉：服务端 `turn_detection.threshold=0.5`，
音量不够就一直不触发，表现为 AI 迟迟不回应。

排查用的 `?raw=1` 参数保留（关闭三个 DSP，标签会显示 `· 无DSP`），
仅用于对照实验，**不要日常使用**。

页面顶部会显示 `16000Hz · 麦克风名 · DSP状态` ——
换设备或环境变化时，一眼就能看出采集链路是否变了。
底部有麦克风选择器，可固定使用某一支麦克风
（声纹识别建议用**录声纹时那支**）。

#### 3. 调参

| 参数 | 用途 |
|---|---|
| `--no-voiceprint` | 关闭声纹过滤 |
| `--voiceprint-threshold F` | 默认 0.5。**旁人老被当成你 → 调大**（0.6）；**你自己老被忽略 → 调小**（0.4） |
| `--voiceprint FILE` | 指定档案路径 |

session 结束会显示：

```
声纹忽略    : 47 个音频块（约 1 秒非本人语音被过滤）
最近声纹相似度: 0.782（阈值 0.5）
```

#### 技术说明

- 模型：**3D-Speaker CAM++**（`models/campplus.onnx`，27MB），192 维声纹特征
- 推理：**纯 CPU，6~11ms/次**，完全不占用对话延迟
- 每积累 2 秒音频判一次，**3 次多数表决**，避免单次波动误判
- **同一台机器上运行**，声纹数据不出本机

#### 模型要自己下载（不随仓库分发）

27MB 的二进制不适合放进 git，需要手动放到位：

```bash
mkdir -p models
curl -L -o models/campplus.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
```

下载后确认大小和校验值：

```bash
ls -lh models/campplus.onnx        # 应为 28281138 字节
md5 models/campplus.onnx           # 应为 2ac7673f702e6e45ff45882a4dd55b1a
```

> 上面这条链接实测可用（2026-09 验证）：下载到的文件与开发时使用的
> 是**同一个文件**，MD5 完全一致，所以声纹判定阈值可以直接沿用。
>
> 不要从 ModelScope 的 `speech_campplus_sv_zh-cn_16k-common` 仓库取 ——
> 那里只有 PyTorch 的 `campplus_cn_common.bin`，**没有 ONNX**，
> 按那个路径下载会得到一个 145 字节的 JSON 报错页。

**没下载会怎样**：程序正常启动，但声纹录入会明确报错
"声纹模型不存在：…/models/campplus.onnx"，练习时则自动跳过声纹过滤
（并在页面上告知"本次不做声纹过滤"）。不会静默失效。
相关的两个测试会**跳过**而不是失败 —— 新克隆的人跑测试不该一见红就以为代码坏了。

**已知限制**（诚实说明）：

1. **冷启动有 1 次误判** —— 多数表决窗口填满前（约前 2 秒）可能放行一小段。
   影响很小，但确实存在。
2. **换麦克风或环境变化大时相似度会下降** —— 声纹对声道敏感。换了设备建议 `--test` 一下。
3. **感冒、明显变声时可能认不出你**，此时用 `--no-voiceprint` 临时关掉。
4. **不是安全级认证** —— 它用于"减少误触发"，不是身份验证，不要用于安全场景。

### 耳机模式：动态音量自适应（约 40ms 打断）


**问题**：戴耳机时仍要用很大声才能打断。

**原因（两层）**：

1. 耳机模式下 `echo_guard` 关闭 → 客户端**完全不参与打断**，全交给服务端 VAD
2. 服务端 VAD `threshold: 0.5` 偏高，且要等 `silence_duration_ms: 800` 才判定说完
3. **我设的固定阈值 700 本身就是错的** —— 实测你正常说话只有 ~400

**修法**：客户端本地抢先打断，且**阈值实时自适应**。

采样**只在模型没说话时进行**（避免把模型声音算进来），然后：

```
底噪      = 最近 4 秒音量的 20 分位
你的音量  = 最近 4 秒音量的 90 分位
打断阈值  = 底噪 + (你的音量 - 底噪) × 0.65
```

即**只要达到你正常音量的 65% 就能打断**。效果：

| 用户类型 | 底噪 | 说话音量 | 自动阈值 | 结果 |
|---|---|---|---|---|
| 小声说话 | 21 | 429 | 286 | ✅ 正常音量可打断 |
| 正常音量 | 21 | 1610 | 1054 | ✅ |
| 大声说话 | 21 | 5367 | 3496 | ✅ 且不会误触发 |
| 嘈杂+大声 | 148 | 3220 | 2145 | ✅ 底噪不被误判 |
| 很吵环境 | 445 | 4293 | 2946 | ✅ |

**所以你不用再调参数了** —— 大声的人阈值自动升高（不误触发），
小声的人阈值自动降低（正常音量也能打断）。

**若还要微调**：

| 参数 | 用途 |
|---|---|
| `--barge-floor RMS` | 动态阈值的**下限**（默认 120，冷启动兜底用 350） |
| `--vad-threshold F` | 服务端 VAD 灵敏度，默认 0.5。调小=更容易判定你在说话 |
| `--vad-silence MS` | 判你说完的静音时长，默认 800。调小=响应更快 |

**会话结束时**会打印实测值，便于判断：

```
实测底噪    : 28
实测说话音量: 1520
动态阈值    : 998（正常音量应高于此值）
```

若「动态阈值」接近甚至超过你的正常音量 → 说明采样异常，用 `--verbose` 排查。

### 自适应回声标定（外放模式）

外放误判的根源是**回声衰减系数靠猜**。程序会**自动标定**：

> 模型每次开始说话的头 6 个音频块（约 120ms），用户通常还没开口
> （他刚说完，模型才开始回应）。这段"静默期"麦克风收到的几乎纯是回声，
> 用它实测衰减系数。

- 首轮 0.2 → 约 5 轮内收敛到真值（误差 <5%）
- 取**中位数**抗异常值（万一你在标定期抢话）
- 标定期内不触发打断，避免噪声误判
- `--verbose` 可看到 `[回声标定] 实测=0.477 → 采用=0.461`

### 诊断信号（会话结束时）

| 输出 | 含义 |
|---|---|
| `回声抑制: 丢弃 N 个音频块` | N 大 = 回声多，正在拦截（正常） |
| `最终回声衰减: 0.xxx（已自动标定）` | 实测衰减系数，可据此手调 |
| `实测底噪 / 实测说话音量 / 动态阈值` | 耳机模式的自适应依据 |
| `打断次数: N` | **明显多于你实际抢话次数 → 阈值偏低**，按提示调参 |

### 获取两个配置值

1. **API Key**：https://bailian.console.aliyun.com/ → 注册（个人可开）→ **API-KEY** → 创建
2. **业务空间 ID**：控制台 → **业务空间管理** → 复制 ID

> ⚠️ **业务空间 ID 不是你的账号 ID**，这是最容易搞错的地方。填错会报
> `BadRequest.IllegalEndpoint: Workspace endpoint is invalid`。

### 协议要点（写代码时容易踩）

| 项 | 值 |
|---|---|
| 端点 | `wss://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime?model=xxx` |
| 鉴权 | `Authorization: Bearer $DASHSCOPE_API_KEY` |
| **输入音频** | **16 kHz** PCM 单声道 16-bit |
| **输出音频** | **24 kHz** PCM 单声道 16-bit ← **与输入不同！** |
| `modalities` | 必须是 `["text","audio"]`，**传 `["audio"]` 会报错** |
| 转写模型 | 由 `QWEN_ASR_MODEL` 配置（默认 `qwen3-asr-flash-realtime`） |

---

## 换模型（全部通过配置，不需要改代码）

启动时会打印实际生效的模型：

```
模型配置：
  语音对话  qwen3.8-omni-flash-realtime
  语音转写  qwen3-asr-flash-realtime
  文本（材料准备/纠错）  qwen-plus
  文本接口  https://dashscope.aliyuncs.com/compatible-mode/v1
```

改 `.env` 里对应的值、重启即可生效。

| 配置项 | 作用 | 默认 |
|---|---|---|
| `QWEN_MODEL` | **语音对话**：听懂并回话 | `qwen3.8-omni-flash-realtime` |
| `QWEN_ASR_MODEL` | **语音转写**：把你说的话变成文字 | `qwen3-asr-flash-realtime` |
| `QWEN_VOICE` | 音色（见下文，17 个可选） | `Tina` |
| `TEXT_MODEL` | **材料准备 / 语法纠错 / 学习画像** | `qwen-plus` |
| `TEXT_BASE_URL` | 文本接口地址（OpenAI 兼容） | 百炼 compatible-mode |
| `SESSION_MINUTES` | 一次练习时长（分钟） | `30` |
| `IDLE_NUDGE_SEC` | **冷场等待**：多久不说话 AI 主动找话题（秒，支持小数） | `15` |
| `IDLE_NUDGE_MAX` | 连续主动开口上限（`0` = 关闭冷场救场） | `6` |

### 语音模型和文本模型是两件事

这两个**互相独立**，但共用同一个 `DASHSCOPE_API_KEY` 和同一份额度，
不需要分别申请：

- **语音模型**决定能不能对话
- **文本模型**决定能不能准备材料、能不能纠错、能不能总结画像

也就是说，如果哪天文本模型不可用，你仍然可以和 AI 说话，
但"材料准备"会失败 —— 反过来也一样。

材料准备之所以单独用一个文本模型：它是**批处理式的一次性规划**
（分析主题、列出想问的问题），用便宜快的文本模型就够了；
让实时语音模型来做反而更贵，因为它按音频时长计费、上下文也更长。

### 想换成别的厂商

`TEXT_BASE_URL` + `TEXT_MODEL` + `DASHSCOPE_API_KEY` 三个一起改，
填任何 **OpenAI 兼容**的服务都能用（DeepSeek、Moonshot、本地 vLLM 等）。
语音那部分则依赖 Qwen Realtime 的 WebSocket 协议，换厂商需要改代码。


### 音色怎么选

**方式一：网页上直接选（推荐）**

点右上角 **「音色」** 按钮 → 列表里点 ▶ 试听 → 点一行选中，立即生效。
面板里也放了[官方音色列表](https://help.aliyun.com/zh/model-studio/omni-voice-list)
的链接，可以随时打开自己对比。

选择会记住（存在 `data/voice_override.txt`），重启后仍是你选的那个。
**优先级：网页选择 > `.env` 的 `QWEN_VOICE` > 内置默认值。**
想让配置文件说了算，删掉那个文件即可。

**方式二：命令行**

```bash
.venv/bin/python scripts/list_voices.py            # 看全部可选音色
.venv/bin/python scripts/list_voices.py --sample   # 生成样音试听
```

**注意有两套音色清单，这是实测踩的坑：**

- **对话音色**（17 个）—— `QWEN_VOICE` 能填的值，AI 说话用的
- **TTS 音色**（43 个）—— 只有这些能合成试听样音

两套**只有部分重叠**。`Tina` 只在对话那套里，用 TTS 合成
试听会报 `Invalid voice specified` —— 不是名字写错，是那个模型
根本不带这个音色。工具已经把两套分开列了。

对话音色全清单（`scripts/list_voices.py` 会打印同样的内容）：

`Tina`（默认）、`Cindy`、`Liora Mira`、`Raymond`、`Zane`、
`Katerina`、`Ryan`、`Mia`、`Cici`、`Theo Calm`、`Serena`、
`Maia`、`Evan`、`Qiao`、`Momo`、`Wil`、`Angel`

来源：百炼「[非实时（Qwen-Omni）和实时（Qwen-Omni-Realtime）支持的音色列表](https://help.aliyun.com/zh/model-studio/omni-voice-list)」。
不同模型支持的音色可能不同 —— 换模型后值得再跑一次 `list_voices.py`。

---

## 网页版（阶段 2）

### 为什么做成"网页前端 + Python 后端"

**不能做成纯前端** —— Qwen 需要 `DASHSCOPE_API_KEY`，放进浏览器任何人
F12 就能拿走。所以必须有后端持有密钥：

```
浏览器 ──WebSocket──> app/api/server.py ──(持key)──> Qwen Realtime
                          │
                          └─ 声纹过滤（判断是不是本人在说）
```

### 网页版的最大收益：浏览器白送回声消除

这是相比命令行版的**架构级优势**：

| 能力 | 命令行（WebSocket） | 网页（getUserMedia） |
|---|---|---|
| 回声消除 AEC | ❌ **我们手写了 ~200 行** | ✅ **浏览器内置** |
| 降噪 NS | ❌ 无 | ✅ 内置 |
| 自动增益 AGC | ❌ 无 | ✅ 内置 |

命令行版那套回声标定、动态阈值调参，在网页版**完全不需要** ——
是平台的免费能力。所以服务端里没有这些逻辑。

### 跑起来

```bash
# 电脑上（浏览器只在 localhost 下允许麦克风）
.venv/bin/python -m app
# 打开 http://127.0.0.1:8000
```

**手机/平板用**（需 HTTPS，否则浏览器拒绝开麦）：

```bash
.venv/bin/python scripts/make_cert.py           # 生成自签证书（只需一次）
.venv/bin/python -m app --host 0.0.0.0 --ssl
# 手机访问 https://<你的局域网IP>:8000
# 首次会提示证书不受信任 → 高级 → 继续前往
```

> ⚠️ **浏览器硬性限制**：`getUserMedia` 只在「安全上下文」可用。
> `http://localhost` ✅ / `https://任意` ✅ / `http://192.168.x.x` ❌
> 所以手机必须走 `--ssl`。

### 页面功能

- 实时音量条
- 对话记录（你说的话 + AI 回复，都有转写）
- **「打断」按钮** —— 立即停止播放
- 顶部显示声纹状态

### 已验证（2026-10-29）

| 测试项 | 结果 |
|---|---|
| 服务启动 / `/status` | ✅ 配置读取正常 |
| 页面加载 | ✅ HTTP 200，含 AudioWorklet/getUserMedia |
| 浏览器↔服务端↔Qwen 建连 | ✅ 收到 `ready` |
| **真实语音端到端** | ✅ 转写正确 → 模型回复 → 音频回传 24 块 |
| **HTTPS/WSS 手机路径** | ✅ 通过 `wss://<你的电脑IP>:8443` 全链路通 |
| 声纹拦截 | ✅ 不匹配语音上行块=0；关闭声纹后正常放行 |

### 技术要点（踩过的坑）

1. **不能用 `MediaRecorder`** —— 它给的是压缩格式（webm/opus），
   Qwen 要的是裸 PCM。用 **AudioWorklet** 拿原始采样。
2. **不能用 `ScriptProcessorNode`** —— 已废弃且延迟高。
3. **`AudioContext({sampleRate:16000})`** 让上行天然是 16k，
   不用手工重采样。
4. **下行 24k 要排队播放** —— 每块单独播会断续。用
   `playHead` 时间轴拼接，落后就追赶。
5. **声纹"未知期"不能放行** —— 判定要攒够 1 秒，这期间 verdict 是
   `None`。如果"未知就放行"，旁人只要说一句就足以让模型回应。
   必须用**待定缓冲**：未知时先扣住，判定为本人再补发。

---

## ⛔ 为什么不用 Azure OpenAI（重要）

**实测结论（2026-10，订阅 `de8e3cf2-...`）：**

```
ERROR: (CannotDeployDueToLocalRegulations)
Due to local regulatory requirements, in mainland China only enterprise
customers with a registered business license are eligible to subscribe
to the Azure OpenAI Service.
```

**中国大陆只有持营业执照的企业客户能订阅 Azure OpenAI。** 这是合规准入，不是配额问题：

- ❌ 申请配额 → 无效（走不到那一步）
- ❌ 升级订阅 → 无效（本来就是 Pay-As-You-Go）
- ❌ 换区域 → 无效（3 个资源、7 个模型全试过）
- ✅ 唯一途径是联系微软合作伙伴（企业流程）

**但 Azure Speech 的发音评测不受此限制，个人可用** —— 产品核心的评测能力仍然能走 Azure。

---

## 以下是原 Azure OpenAI 流程（留档）

### ⛔ 结论先行：中国大陆个人账号无法部署 Azure OpenAI


**实测结论（2026-10，订阅 `de8e3cf2-811b-4977-b223-009e3043ec57`）：**

```
ERROR: (CannotDeployDueToLocalRegulations)
Due to local regulatory requirements, in mainland China only enterprise
customers with a registered business license are eligible to subscribe
to the Azure OpenAI Service.
```

**这不是配额问题，是合规准入限制：**

> **中国大陆地区，只有持有营业执照的企业客户**才能订阅 Azure OpenAI 服务。

**三个资源、全部模型都试过，一律返回同一个错误。** 个人账号（Individual / MicrosoftCustomerAgreement）**无法通过任何配额申请绕开**——这不是"额度不够"，是"身份不符合准入条件"。

**官方给出的唯一正规途径**：
- 联系微软认证合作伙伴 https://azure.microsoft.com/zh-cn/partners/
- 邮件 mscnenq@microsoft.com
- 电话 400-082-0005（9:00–17:30 CST）

**⚠️ 这意味着 A 方案（Azure OpenAI Realtime）对个人用户不可行。**

**替代路径**：改用 **Qwen Realtime**（国内直连、即开即用、个人可注册）做对话层；**发音评测**改用 Azure Speech 的 Pronunciation Assessment（属于 Cognitive Services 的 Speech，**不走 Azure OpenAI 的准入限制**，个人可用）。

详见 `英语口语陪练-推荐方案.md` 的修订版。

---

## 以下是原 Azure OpenAI 流程（保留备用 / 供企业用户参考）

### 一、Azure 侧要做的三件事

#### 1. 创建资源

进 [Azure Portal](https://portal.azure.com) → **创建资源** → 搜索 **"Azure OpenAI"**（或 "Microsoft Foundry"）→ 创建。

**⚠️ 关键：区域必须支持 realtime 模型。** 推荐：
- `eastus2` / `swedencentral` / `westus3`

选错区域会出现"部署不了 realtime 模型"的问题，这是最常见的坑。

> **注意**：Azure OpenAI 需要**申请访问权限**（微软会审核用途，个人学习用途通常几天内批）。如果提示需要申请，按流程填表即可。

#### 2. 部署模型

进入资源 → **Foundry portal / AI Studio** → **Deployments** → **Deploy model**

选 `gpt-realtime-2.1`（或先省钱用 `gpt-realtime-2.1-mini`）。

**部署时你会给它起一个名字 —— 这个名字就是 `AZURE_OPENAI_DEPLOYMENT`。** 它和模型名是两回事，是最容易搞错的地方。建议就起 `gpt-realtime-2.1` 免得混淆。

#### 3. 拿 Endpoint 和 Key

资源 **Overview** 页 → Endpoint（形如 `https://xxx.openai.azure.com`）
**Keys and Endpoint** 页 → KEY 1

---

## 本订阅实测配额快照（2026-10，eastus2）

| 模型 | 配额 | 备注 |
|---|---|---|
| `gpt-realtime-mini` | **40 TPM** | 唯一非零的 realtime 模型 |
| `gpt-realtime` | 0 | |
| `gpt-realtime-1.5` / `-2` / `-2.1` / `-2.1-mini` | 0 | |
| `gpt-live-1` | 0 | |
| `gpt-4o-transcribe` | 400 | 非 realtime，可用 |
| `gpt-4o-mini-tts` | 50 | 非 realtime，可用 |

**注意**：配额是真实的，但**部署时会被 `CannotDeployDueToLocalRegulations` 拦在更前面**——配额够也没用。

---

## 二、本地配置

```bash
cd kai          # 进到本项目目录
cp .env.example .env
# 编辑 .env，填入三个值
```

### 模型选择（按配额可行性排序）

**⚠️ 重要前提**：如果你的 Azure 订阅是**新订阅 / 试用 / 学生 / Sponsorship** 类型，Azure OpenAI 的 TPM 配额**默认就是 0**，所有模型（不只 realtime）都部署不了。这不是模型选择问题，见下方「配额为 0 怎么办」。

| 部署模型 | Azure 生命周期 | 退役日期 | 建议 |
|---|---|---|---|
| `gpt-realtime-2.1` | GA | 2027-06-25 | 最新最好，但配额可能为 0 |
| `gpt-realtime-2.1-mini` | GA | 2027-06-25 | 2.x 小号，配额同样紧张 |
| **`gpt-realtime`** | **GA** | **2027-03-02** | ⭐ 首选，Tier 1 有 100k TPM |
| `gpt-realtime-mini` | GA | 2026-12-15 | 快退役了，别选 |
| `gpt-realtime-1.5` | GA | 2027-08-24 | 备选 |
| `gpt-4o-mini-realtime-preview` | — | 已退役 | ❌ 部署不了 |

---

## 配额为 0 怎么办

### ⛔ 第一步：确认订阅类型（最容易卡住的地方）

**如果你的配额申请被拒，且理由是这类措辞：**

> "At this time your quota increase request cannot be processed.
> For Free Trial customers, please go here: Trial Quota Request Upgrade to upgrade to Pay as You Go."

**这说明你的订阅是 Free Trial / 学生版，而不是即用即付。** 官方规则明确：

> **"Azure free trial / Student / Pass subscription are not eligible for a quota request."**
> （免费试用 / 学生 / Pass 订阅不具备申请配额的资格）

**这不是你的用途被拒，是订阅类型不合格。** 免费试用订阅**不能**申请配额提升——这是硬规则，反复提交多少次都一样。

**解法：先升级到即用即付（Pay-As-You-Go）**

1. 登录 [Azure Portal](https://portal.azure.com)
2. 搜索 **Subscriptions** → 选中你的订阅
3. 订阅概览页点 **Upgrade subscription**（如果没有这个按钮，点页面顶部的升级横幅）
4. 添加付款方式、验证手机号、起个订阅名、选支持计划
5. 点 **Upgrade**

**升级后你会保留**：
- 注册起 30 天内剩余的额度（例：11/1 注册、11/5 升级，未用额度可用到 11/30）
- 升级后 12 个月的免费服务

> ⚠️ **注意**：如果订阅曾因额度耗尽被禁用，且你有非免费资源在跑，升级后这些资源会重新启用并**开始计费**。

**升级完成后再重新提交配额申请**（https://aka.ms/oai/stuquotarequest），这次才会被受理。

---

### 确认配额状态

Foundry portal → **Management → Quota** → 看 Azure OpenAI 那几行是不是 `0 / 0`。

微软官方（Microsoft Q&A，微软工作人员回复）明确说明：

> "对于新订阅，Azure OpenAI 配额**不一定会自动配置**，每个区域/模型**合法地从 0 TPM 开始**，直到分配被授予。**它不会随时间自动配置（will not auto-provision over time）**。"
>
> "所有 Azure OpenAI 模型显示 0 TPM 是**预期行为（expected onboarding behavior）**，不是故障。"

**注意一个容易误判的现象**：非 OpenAI 模型（如 grok、Llama）能部署成功，而 Azure OpenAI 全部 0 —— 这**恰恰说明订阅、账单、账号都正常**，只是 OpenAI 的 TPM 容量没批。这不是你的配置问题。

### 申请路径

**表单**：https://aka.ms/oai/stuquotarequest

**填写要点**（官方建议）：
- Subscription ID
- Region（具体区域）
- Model（如 `gpt-realtime`）
- Deployment type：`Global Standard`
- **Requested TPM：填小值**，如 `10,000`
- 备注这是**小规模 POC 的初始启用**

> ⚠️ 官方明确警告：**申请过大的初始额度会触发系统自动拒绝。** 你的用量每月才 325 分钟，申请 10,000 TPM 绰绰有余。

### 三个能提高通过率的技巧

1. **换区域试试** —— 配额是**按区域**分配的。East US 容量最紧张（排队最久）。去 **Sweden Central / North Central US / Japan East** 看 Quota 页，可能某个区域还有余量。
2. **如果配额申请被拒**，说明拒绝邮件里的具体原因（有时会写），针对性修改后再提交**一次**。**不要同时挂多个申请**——这会让记录混乱反而更慢。
3. **付费订阅优于试用订阅** —— 审核对"零消费记录"的订阅天然不信任。即用即付（Pay-As-You-Go）即使月消费几美元也有帮助。

### 关于审核时长（要有心理预期）

微软**没有任何公开 SLA**。社区汇总的实际体验：配额申请**快则 3 天，慢则 2 个月**，热门区域尤其严重。

有第三方服务商声称能通过 CSP 通道"5 分钟接入""3 小时开通独立账号"。**我不推荐这条路**——那本质是把 API 凭证交给第三方，和前面说的"中转站"是同一类风险（数据经手第三方、上游随时可封、跑路）。**你的项目只是自己练口语，不值得为此交出凭证。**

### 如果不想等：这个原型有替代验证路径

**你其实不需要 API 就能回答最根本的问题。** 用 **ChatGPT 的语音模式（Advanced Voice）** 聊 30 分钟——它就是同一套 Realtime 技术的消费级形态。虽然拿不到费用数据、不能自定义 prompt，但"**这个语音体验够不够好、我想不想每周练 2 次**"这个决定性问题，30 分钟就有答案。

先确认值得投入，再去等配额。

---

## 三、运行

```bash
# 1. 自检（不产生语音费用）
.venv/bin/python check.py

# 2. 开始对话
.venv/bin/python realtime_cli.py --topic "travel plans"
```

**Ctrl-C 结束**，会打印时长、你的说话时长、轮次和费用估算。

---

## 四、这个原型能帮你验证什么

| 问题 | 怎么看 |
|---|---|
| 语音够不够自然 | 用耳朵听，这是最终标准 |
| 能不能自然打断 | AI 说话时故意插话 |
| 中文口音识别率 | 看 🧑 那行转写对不对 |
| 数值准不准 | 看结束时的费用估算 |
| 延迟高不高 | 体感（WebSocket 约 100–300ms） |

---

## 五、已知注意事项

1. **会话最长 60 分钟** —— 你每次 30 分钟没问题。官方文档建议监控 `session.created` 的 `expires_at`。
2. **WebSocket 是服务端方案**，延迟 ~100–300ms。官方推荐客户端用 **WebRTC**（~50–100ms）。你现在在本地跑，WebSocket 足够验证效果；将来做正式客户端再换 WebRTC。
3. **`api-version` 参数不需要** —— Azure 新版 GA 接口用的是 `/openai/v1` 路径，不带 `api-version`。网上很多老教程还在用 `?api-version=2024-10-01-preview`，那是旧格式。
4. **费用按音频时长计费**（不是 token）。脚本里的单价是参考值，**以你的实际账单为准**。
5. **认证用 `api-key` header**。脚本也支持 Entra ID（bearer token），需要 Azure CLI 登录并分配 `Cognitive Services OpenAI User` 角色——个人自用没必要。

---

## 六、下一步

跑通对话后，第二阶段接 **Azure Pronunciation Assessment** 做发音纠音。届时你会发现一个便利：**同一个 Azure 账号**，端点不同但账单和网络通道都统一。
