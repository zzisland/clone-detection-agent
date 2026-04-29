# clone-detection-agent

这是一个面向 C/C++ 模块的克隆检测工具。

它的主要流程是：

1. 解析源码并提取函数
2. 执行 Type1-2 / Type3-4 克隆检测
3. 对检测结果做分层
4. 调用大模型进行评估
5. 生成 Markdown 和 HTML 报告

## 当前目录说明

- `detector/`：克隆检测主流程
- `layering/`：结果分层
- `model_eval/`：大模型评估
- `models/`：共享数据结构
- `reports/`：报告生成
- `scripts/`：批量执行脚本
- `config/`：本地配置
- `data/clone_detection/`：检测中间结果和最终报告输出目录

## 使用方式

单模块执行：

```powershell
python main.py `
  --repo F:\你的目标模块路径 `
  --detector static `
  --enable-type34 `
  --model-eval openai
```

批量执行：

```powershell
python scripts\run_targets.py
```

## 输出位置

默认报告会输出到：

```text
data/clone_detection/<模块名>/
```

例如：

```text
data/clone_detection/acisadaptor/acisadaptor_clone_report.md
data/clone_detection/acisadaptor/acisadaptor_clone_report.html
```
