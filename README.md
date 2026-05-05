# clone-detection-agent

面向 C/C++ 模块的克隆检测 agent。当前版本已经打通本地可运行闭环，支持从目标项目目录发起扫描，完成候选克隆对检测、分层、大模型复核以及报告生成。

## 能做什么

本工具当前支持以下能力：

1. 解析目标模块中的 `.cpp` 文件，提取函数和结构信息
2. 执行 Type1-2 克隆检测
3. 可选执行 Type3-4 向量相似度检测
4. 对候选克隆对进行分层，区分 `high / medium / low`
5. 可选执行大模型复核，输出：
   - `judgement`
   - `score`
   - `explanation`
   - `refactor_worthiness`
   - `refactor_suggestion`
   - `risk_note`
6. 生成 Markdown 和 HTML 报告，便于项目负责人进行复核和治理决策

## 当前定位

当前版本定位是：

- 已完成核心流程原型
- 已支持在真实项目目录中本地运行
- 已支持模型配置和报告输出
- 已支持批量扫描脚本
- 尚未封装为标准 Python CLI 包

当前推荐的使用方式仍然是：

```powershell
python F:\clone-detection-agent\main.py --repo .
```

后续可以继续封装为标准 CLI，例如：

```powershell
clone-detect --repo .
```

## 目录说明

```text
clone-detection-agent/
├─ main.py                         # 单目标扫描入口
├─ config/
│  ├─ api-keys.json               # 本地 API 配置
│  └─ scan-targets.json           # 批量扫描目标列表
├─ detector/                      # 克隆检测主流程
├─ layering/                      # 分层逻辑
├─ model_eval/                    # 大模型评估
├─ models/                        # 数据模型
├─ reports/                       # Markdown / HTML 报告输出
├─ scripts/
│  └─ run_targets.py              # 批量扫描入口
└─ data/clone_detection/          # 默认输出目录
```

## 环境要求

建议先统一一套已验证环境，再给团队扩散。

推荐环境：

- Windows 10 / 11
- Python 3.10+
- LLVM / libclang
- 可访问的 embedding 接口
- 可访问的 OpenAI-compatible chat 接口

## 依赖项

### Python 依赖

当前代码显式依赖：

- `requests`
- `clang` Python 绑定

建议先创建虚拟环境，再安装依赖：

```powershell
cd F:\clone-detection-agent
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install requests clang
```

### 系统依赖

函数切片依赖 `libclang`。

你需要在本机安装 LLVM / libclang，并通过以下任一方式让程序找到它：

- 设置 `LIBCLANG_FILE`
- 或设置 `LIBCLANG_PATH`

示例：

```powershell
$env:LIBCLANG_PATH="C:\Program Files\LLVM\bin"
```

或者：

```powershell
$env:LIBCLANG_FILE="C:\Program Files\LLVM\bin\libclang.dll"
```

## 配置文件

默认配置文件路径：

```text
config/api-keys.json
```

注意：当前版本已经修正路径解析逻辑。即使你在“被检测项目目录”执行命令，默认也会读取 agent 仓库下的这份配置文件。

### 示例配置

```json
{
  "clone_detection": {
    "enabled": true,
    "api_url": "http://your-embedding-host:8003",
    "model_name": "your-embedding-model",
    "api_key": "your-embedding-key"
  },
  "model_evaluation": {
    "enabled": true,
    "mode": "openai",
    "model_name": "your-chat-model",
    "api_url": "https://your-openai-compatible-host/v1",
    "api_key": "your-chat-api-key"
  }
}
```

### `enabled` 开关说明

现在可以直接通过配置文件控制是否启用 Type3-4 检测和大模型评估，不必每次通过命令行传参。

#### 1. `clone_detection.enabled`

- `true`：启用 Type3-4 检测
- `false`：关闭 Type3-4 检测

#### 2. `model_evaluation.enabled`

- `true`：执行大模型评估
- `false`：跳过大模型评估

当 `model_evaluation.enabled=false` 时：

- 不调用模型接口
- 直接生成报告
- 报告中的模型相关字段显示为 `not_evaluated` / `null`

### 配置建议

#### 只做检测，不做模型评估

```json
{
  "clone_detection": {
    "enabled": true
  },
  "model_evaluation": {
    "enabled": false
  }
}
```

#### 只做 Type1-2，不做 Type3-4，也不做模型评估

```json
{
  "clone_detection": {
    "enabled": false
  },
  "model_evaluation": {
    "enabled": false
  }
}
```

#### 两者都启用

```json
{
  "clone_detection": {
    "enabled": true
  },
  "model_evaluation": {
    "enabled": true
  }
}
```

### 命令行与配置的关系

- 平时推荐通过 `config/api-keys.json` 里的 `enabled` 控制默认行为
- 如果命令行显式传了 `--enable-type34`，会强制打开 Type3-4
- 大模型评估当前没有单独的命令行关闭参数，是否评估主要由 `model_evaluation.enabled` 控制

## 使用方式

### 1. 单模块扫描

进入目标模块目录后执行：

```powershell
cd F:\project-gme\GME\module\acisadaptor
python F:\clone-detection-agent\main.py --repo .
```

这里的 `--repo .` 表示扫描当前目录，也就是当前模块目录。

### 2. 单模块启用 Type3-4

如果配置文件里已经把 `clone_detection.enabled=true`，直接执行：

```powershell
python F:\clone-detection-agent\main.py --repo .
```

如果你想临时强制启用 Type3-4，也可以执行：

```powershell
python F:\clone-detection-agent\main.py --repo . --enable-type34
```

### 3. 显式指定配置文件

```powershell
python F:\clone-detection-agent\main.py `
  --repo . `
  --api-config F:\clone-detection-agent\config\api-keys.json
```

### 4. 指定源码子目录

```powershell
python F:\clone-detection-agent\main.py `
  --repo F:\project-gme\GME\module\acisadaptor `
  --src-subdir src
```

### 5. 指定输出目录

```powershell
python F:\clone-detection-agent\main.py `
  --repo . `
  --work-dir F:\clone-detection-agent\data\clone_detection
```

## 批量扫描

### 1. 批量扫描的设计方式

`scripts/run_targets.py` 用来批量调用多个 `main.py`。

它当前的行为是：

- `main.py` 仍然从 agent 仓库调用
- `targets` 里的相对路径按“目标项目仓库根目录”解析
- 默认 `--repo-root .`
- 如果你在项目仓库根目录执行，`targets` 里的路径就直接相对于当前目录解析

### 2. 配置目标列表

编辑 [`config/scan-targets.json`](F:\clone-detection-agent\config\scan-targets.json:1)：

```json
{
  "targets": [
    "module/acisadaptor",
    "module/other_module"
  ]
}
```

这些路径表示相对于项目仓库根目录的模块路径。

### 3. 在项目仓库根目录执行

例如你当前在：

```powershell
cd F:\project-gme\GME
```

那么可以直接执行：

```powershell
python F:\clone-detection-agent\scripts\run_targets.py
```

如果配置文件中 `clone_detection.enabled=true`，批量扫描也会自动启用 Type3-4，不需要额外传 `--enable-type34`。

### 4. 显式指定项目仓库根目录

```powershell
python F:\clone-detection-agent\scripts\run_targets.py `
  --repo-root F:\project-gme\GME
```

### 5. 手动指定多个 target

```powershell
python F:\clone-detection-agent\scripts\run_targets.py `
  --repo-root F:\project-gme\GME `
  --target module/acisadaptor `
  --target module/other_module
```

### 6. 批量扫描和单模块扫描的区别

- `main.py --repo .`
  - 扫描当前目录这个模块
- `run_targets.py`
  - 从项目仓库根目录出发，按相对路径批量扫描多个模块

## 常用参数

### main.py 基础参数

- `--repo`
  - 目标仓库或模块路径，必填
- `--api-config`
  - API 配置文件路径
- `--work-dir`
  - 检测输出目录
- `--src-subdir`
  - 仅扫描目标路径下的某个子目录
- `--project-name`
  - 传给 Type1-2 检测的项目名，默认 `gme`

### main.py Type3-4 参数

- `--enable-type34`
- `--type34-api-url`
- `--type34-model-name`
- `--type34-api-key`
- `--type34-threshold`
- `--type34-batch-size`

### main.py 模型评估参数

- `--model-eval`
- `--model-eval-model-name`
- `--model-eval-api-url`
- `--model-eval-api-key`
- `--model-eval-temperature`
- `--model-eval-timeout`
- `--model-eval-max-body-chars`
- `--model-eval-max-retries`
- `--model-eval-retry-backoff`
- `--model-eval-inter-request-delay`

### run_targets.py 参数

- `--repo-root`
  - 目标项目仓库根目录，默认当前目录
- `--targets-file`
  - 批量扫描目标配置文件，默认 `config/scan-targets.json`
- `--target`
  - 手动追加单个目标，可重复传入
- `--work-dir`
  - 输出目录
- `--api-config`
  - 转发给 `main.py` 的配置文件
- `--enable-type34`
  - 临时强制启用 Type3-4

## 输出结果

默认输出目录：

```text
data/clone_detection/<module_name>/
```

例如：

```text
data/clone_detection/acisadaptor/
├─ functions.csv
├─ structs.csv
├─ func_clone_type12.csv
├─ func_clone_type34.csv
├─ func_clone_merged.csv
├─ acisadaptor_clone_report.md
└─ acisadaptor_clone_report.html
```

## 报告内容说明

HTML / Markdown 报告当前包含：

- 左右候选代码位置
- 检测来源
- Type1-2 / Type3-4 分数
- 分层结果
- 模型判定结果
- 模型打分
- 判定原因
- 重构价值评估
- 重构建议
- 风险说明
- 左右函数体对照

说明：

- `Similarity`
  - 检测器阶段得到的候选相似度
- `Model Score`
  - 大模型对该候选对的判定分数
- 当模型评估关闭时：
  - `Judgement` 显示为 `not_evaluated`
  - `Model Score` 显示为 `null`
  - `Explanation` 等字段显示为 `null`

## 推荐测试顺序

建议按下面顺序验证：

1. 先跑单模块基础流程

```powershell
cd F:\project-gme\GME\module\acisadaptor
python F:\clone-detection-agent\main.py --repo .
```

2. 在配置文件里分别测试：
   - `clone_detection.enabled=true / false`
   - `model_evaluation.enabled=true / false`

3. 再在项目仓库根目录测试批量扫描

```powershell
cd F:\project-gme\GME
python F:\clone-detection-agent\scripts\run_targets.py
```

4. 检查 HTML 报告内容和输出目录是否符合预期

## 常见问题

### 1. 报错：`Type3-4 detection enabled, but api_url/model_name is missing`

原因：

- 启用了 Type3-4
- 但没有提供 embedding 接口配置

处理方式：

- 检查 `config/api-keys.json` 中 `clone_detection` 配置
- 或显式传入：
  - `--type34-api-url`
  - `--type34-model-name`
  - `--type34-api-key`

### 2. 报错：`clang Python bindings are not available`

原因：

- 没有安装 `clang` Python 包
- 或系统中缺少 `libclang`

处理方式：

```powershell
pip install clang
```

并检查：

- `LIBCLANG_PATH`
- `LIBCLANG_FILE`

### 3. 报错：模型评估接口缺失

原因：

- `model_evaluation.enabled=true`
- 但 `model_evaluation.api_url` 或 `model_name` 未配置

处理方式：

- 检查 `config/api-keys.json`
- 或先把 `model_evaluation.enabled` 设为 `false`

### 4. 为什么在目标项目目录运行也能读取 agent 配置

因为当前版本已经把相对路径解析改为：

- 相对于 `main.py` 所在的 agent 根目录

这能避免从不同项目目录执行时找不到默认配置文件。

### 5. 批量扫描时 target 相对谁解析

在 `run_targets.py` 中：

- `targets` 里的相对路径相对于 `--repo-root` 解析
- 如果不传 `--repo-root`，默认相对于当前目录解析

因此推荐方式是：

```powershell
cd F:\project-gme\GME
python F:\clone-detection-agent\scripts\run_targets.py
```

## 当前限制

当前版本仍有这些限制：

- 还未封装为标准 `pip install` CLI
- 依赖本机 LLVM / libclang 环境
- 对目标项目的 include 环境仍有一定要求
- 模型返回新增字段时，质量依赖具体模型表现

## 下一步建议

如果准备给更多项目负责人使用，建议继续补齐：

1. `pyproject.toml`
2. 标准命令入口，例如 `clone-detect --repo .`
3. `requirements.txt`
4. `api-keys.example.json`
5. 环境检查命令，例如 `doctor`
6. 常见报错排查手册

