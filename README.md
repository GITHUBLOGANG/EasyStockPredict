
# PredictStock — 面向非计算机专业用户的股票预测与回测工具

这是一个简单、上手快速的股票价格预测与回测脚本，适合没有深厚编程背景的用户用来训练基础模型、生成预测并做一个入门级回测展示。项目以 `PredictStock.py` 为主，封装了数据获取、模型（RNN/LSTM/可选 LightGBM）、回测与可视化功能。

**目标用户**：金融分析师、学生或对量化感兴趣但不是计算机专业的个人。目标是用最少的编程步骤完成从数据获取到训练、预测与回测的全流程。

核心设计原则：易用性、最小配置、可重复运行。

**重要说明（请先阅读）**
- 你需要在本机设置 `TUSHARE_TOKEN` 环境变量，用于访问 Tushare API（本项目不会包含或上传 token）。
- 若需要使用 LightGBM 的功能，请在系统上安装 `lightgbm`（某些 Windows 环境须安装编译工具或使用 pip wheel）。

**主要文件**
- `PredictStock.py` — 主脚本，包含训练、预测、回测、分析函数。
- `requirements.txt` — 推荐依赖列表。
- `saved_models/` — 训练后模型默认保存目录。
- `outputs/` — 预测图、回测结果、导出 CSV 放在此目录（脚本会创建）。

快速指南（面向非编程用户）
------------------

1) 准备环境（PowerShell）

```powershell
# 创建并激活虚拟环境
python -m venv venv
venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

2) 设置 Tushare Token（一次性操作）

```powershell
$env:TUSHARE_TOKEN = "你的_tushare_token_字符串"
# 若希望永久设置，请在系统环境变量中添加同名变量
```

3) 最简单运行示例（交互式方式）

在 PowerShell 中运行 Python 交互或创建一个短脚本：

```powershell
python -c "from PredictStock import train_and_predict; train_and_predict('000001.SZ', time_range='全部', forecast_days=30)"
```

说明：
- 第一个参数为股票代码（Tushare 格式，如 `000001.SZ`）。
- `time_range` 可选：`全部`, `一年`, `三年`, `五年` 等，脚本会自动把时间区间转为数据范围。
- `forecast_days`：要预测的天数（例如 30）。

4) 更友好的方式：在 Python 脚本中调用（便于修改参数）

创建文件 `run_example.py`：

```python
from PredictStock import train_and_predict

train_and_predict(
	stock_code='000001.SZ',
	time_range='三年',
	forecast_days=30,
	model_choice='LSTM',
	do_backtest=True,
	show_states=True,
	save_png=True
)
```

然后在 PowerShell 执行：

```powershell
python run_example.py
```

5) Gradio（图形界面）

脚本中已导入 `gradio`，如果脚本包含 Gradio 入口（某些版本会有），运行脚本会弹出本地网页界面，便于不写代码直接输入股票代码并点击运行。如果需要，我可以帮你把 Gradio 的快速启动片段添加到仓库。

输出说明（如何查看结果）
------------------
- 模型文件（如果保存）位于 `saved_models/`，文件名包含股票代码与时间区间。
- 图表与导出文件会写入 `outputs/`，例如预测折线图、回测的净值曲线以及 CSV 格式的预测结果。
- 回测结果会打印或以字典形式返回：年化收益、Sharpe、最大回撤、最终权益等。

常见参数说明（非必须掌握）
------------------
- `model_choice`: `'SimpleRNN'`, `'LSTM'`, （若安装 LightGBM 则有 `'LightGBM'`）。
- `do_backtest`: 是否使用预测结果进行简单回测（True/False）。
- `threshold`：预测收益阈值，超过该阈值会触发买入信号（回测使用）。
- `lstm_epochs`：训练轮数（对初学者可保留默认或设置为较小值如 10-30）。

常见问题与排查
------------------
- 无法获取数据 / 异常提示缺少 token：请确认 `TUSHARE_TOKEN` 已正确设置且有效。你可以在 PowerShell 输出 `$env:TUSHARE_TOKEN` 进行验证。
- LightGBM 安装失败：Windows 上直接 `pip install lightgbm` 有时会失败，请参考 LightGBM 官方文档或使用 conda 安装（`conda install -c conda-forge lightgbm`），或在不需要时忽略 LightGBM（脚本会检测是否可用）。
- TensorFlow 相关错误：若遇到 GPU/驱动问题，可先安装 CPU 版本或将 `tensorflow` 指定为兼容的版本，或在 `pip` 安装时使用 `tensorflow-cpu`（取决于你的硬件）。

安全与隐私
------------------
- 切勿将你的 `TUSHARE_TOKEN` 提交到公开仓库。建议使用系统环境变量或 CI/CD secret 管理。

下一步建议（我可以帮你做）
------------------
- 我可以把一个简单的 `run_example.py` 添加到仓库，作为一键运行示例。
- 我可以为你把仓库初始化并提交到 GitHub（你提供远程仓库地址即可），并展示将要运行的 PowerShell 命令供你确认。
- 如果你希望非编程用户用 GUI，我可以把 Gradio 的界面入口实现并写好说明。

许可证
------------------
本项目使用 MIT 许可证（见 `LICENSE`）。如果需要其他许可证，请告诉我。

联系方式与贡献
------------------
欢迎提交 issue 或 PR。若希望我继续改进（添加 GUI、自动化训练脚本或更友好的安装器），回复我你想要的功能即可。

