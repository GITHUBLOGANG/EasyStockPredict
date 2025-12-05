import os
import tushare as ts
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Input, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号
from PIL import Image
import gradio as gr
import io
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False
from sklearn.cluster import KMeans

# 导入增强的模型分析工具
try:
    from model_analysis_tools import (
        enhanced_compute_metrics,
        plot_bias_analysis,
        analyze_lgb_feature_importance,
        generate_feature_importance_report,
        diagnose_bias_problem,
        generate_bias_analysis_report
    )
    ANALYSIS_TOOLS_AVAILABLE = True
except ImportError:
    ANALYSIS_TOOLS_AVAILABLE = False

# 创建保存模型的文件夹
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 获取股票数据的函数（前复权）
# 注意：原始代码中包含硬编码的 tushare token，已注释以避免将私密信息提交到仓库。
# 如果你想直接在脚本中使用 token，可以在本行取消注释并替换为你的 token（不推荐把 token 提交到 Git）：
# pro = ts.pro_api('your_token_here')

# 推荐做法：在本地通过环境变量提供你的 TUSHARE token，或在 CI/CD 中使用 secret。
# 下面的逻辑会尝试从环境变量 `TUSHARE_TOKEN` 读取 token，若未设置则提示用户。
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
if TUSHARE_TOKEN and TUSHARE_TOKEN.strip():
    pro = ts.pro_api(TUSHARE_TOKEN.strip())
else:
    # 如果没有环境变量，我们仍然保留一个带注释的占位，用户可以选择自行填入或设置环境变量。
    raise RuntimeError(
        "未检测到环境变量 TUSHARE_TOKEN。请设置你的 tushare token（推荐），或在本文件中取消注释原始 token 并替换。"
    )

def get_stock_data(stock_code, start_date=None, end_date=None):
    def _norm_date(d, default=None):
        if d is None:
            return default
        try:
            # accept int/str/timestamp
            s = str(d)
            # remove non-digits
            s = ''.join(ch for ch in s if ch.isdigit())
            if len(s) >= 8:
                return s[:8]
            # fallback parse
            return pd.to_datetime(d).strftime('%Y%m%d')
        except Exception:
            return default
    today_str = pd.Timestamp.today().strftime('%Y%m%d')
    start_str = _norm_date(start_date, '19900101')
    end_str = _norm_date(end_date, today_str)
    df = pro.daily(
        ts_code=stock_code,
        start_date=start_str,
        end_date=end_str
    )
    if df.empty:
        raise ValueError(f"{stock_code} 没有获取到数据")
    df = df[['trade_date', 'close']]
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    df = df.sort_index()
    return df

# 数据预处理：归一化
def preprocess_data(df):
    # 保留占位：不再在全量数据上 fit，实际在 train_and_predict 中仅用训练集拟合
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = None
    return scaled_data, scaler

# 创建时间序列数据集
def create_dataset(data, time_step):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# 构建 RNN 模型
def build_rnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SimpleRNN(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# LightGBM 特征工程（滞后与滚动统计）
def build_lgb_features(df):
    d = df.copy()
    for i in range(1, 6):
        d[f'lag_{i}'] = d['close'].shift(i)
    d['roll_mean_5'] = d['close'].shift(1).rolling(window=5, min_periods=5).mean()
    d['roll_std_5'] = d['close'].shift(1).rolling(window=5, min_periods=5).std()
    d = d.dropna()
    feature_cols = [c for c in d.columns if c != 'close']
    X = d[feature_cols].values
    y = d['close'].values
    return d, X, y, feature_cols

# 市场状态识别（KMeans，基于收益波动等特征）
def compute_market_states(df, n_states=3, window=20, random_state=42):
    d = df.copy()
    d['ret'] = d['close'].pct_change()
    d['vol20'] = d['ret'].rolling(window=window, min_periods=window).std()
    d['vol5'] = d['ret'].rolling(window=5, min_periods=5).std()
    d['mom20'] = d['close'] / d['close'].shift(window) - 1
    feat = d[['vol20', 'vol5', 'mom20']].dropna()
    if len(feat) < n_states:
        d['state'] = np.nan
        return d['state']
    km = KMeans(n_clusters=n_states, n_init=10, random_state=random_state)
    labels = km.fit_predict(feat.values)
    state_series = pd.Series(index=feat.index, data=labels)
    d['state'] = state_series
    return d['state']

def summarize_state_labels(df, states, window=20):
    # 基于波动(vol20)与动量(mom20)给出直观标签
    d = df.copy()
    d['ret'] = d['close'].pct_change()
    d['vol20'] = d['ret'].rolling(window=window, min_periods=window).std()
    d['mom20'] = d['close'] / d['close'].shift(window) - 1
    stat = pd.DataFrame({'state': states}).join(d[['vol20','mom20']])
    stat = stat.dropna()
    labels = {}
    if len(stat) == 0:
        return labels
    grp = stat.groupby('state').agg({'vol20':'mean','mom20':'mean'})
    vol_med = grp['vol20'].median()
    for s, row in grp.iterrows():
        vol_tag = '高波动' if row['vol20'] > vol_med else '低波动'
        if row['mom20'] > 0.01:
            mom_tag = '上行趋势'
        elif row['mom20'] < -0.01:
            mom_tag = '下行趋势'
        else:
            mom_tag = '震荡'
        labels[int(s)] = f"{vol_tag}/{mom_tag}"
    return labels

# 简单基础回测（单标的，阈值买入，含手续费/滑点）
def simple_backtest(close_series, pred_series, threshold=0.0, fee=0.0003, slippage=0.0002, capital=100000.0, position_size=1.0):
    # 对齐索引，pred_series 对应目标日的收盘预测
    idx = close_series.index.intersection(pred_series.index)
    close = close_series.loc[idx]
    pred = pred_series.loc[idx]
    prev_close = close.shift(1)
    pred_ret = (pred - prev_close) / prev_close
    signal = (pred_ret > threshold).astype(int)
    # 交易在当日开盘执行，用前一日的预测，因此信号右移一日
    position = signal.shift(1).fillna(0)
    ret = close.pct_change().fillna(0)
    pos_change = position.diff().abs().fillna(position)
    # 成本：仅在换手时收取一次性成本（买卖各一次近似为总和）
    cost = pos_change * (fee + slippage)
    # 仓位比例
    strategy_ret = position * (position_size * ret) - cost
    idx_ret = (1 + strategy_ret).cumprod()
    equity = capital * idx_ret
    peak = equity.cummax()
    drawdown = (equity / peak - 1).fillna(0)
    ann_factor = 252
    # 年化收益应基于收益指数而非资金值
    ann_return = idx_ret.iloc[-1] ** (ann_factor / max(len(idx_ret), 1)) - 1 if len(idx_ret) > 0 else 0.0
    vol = strategy_ret.std() * np.sqrt(ann_factor) if strategy_ret.std() > 0 else 0.0
    sharpe = (strategy_ret.mean() * ann_factor) / vol if vol > 0 else 0.0
    max_dd = drawdown.min()
    return {
        'ann_return': float(ann_return),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'final_equity': float(equity.iloc[-1]) if len(equity) > 0 else capital,
        'equity_series': equity,
        'position_series': position
    }

def _segments_from_binary_series(binary_series):
    # 将0/1持仓序列划分为连续片段
    segs = []
    in_seg = False
    start = None
    for idx, val in binary_series.items():
        if val and not in_seg:
            in_seg = True
            start = idx
        elif not val and in_seg:
            segs.append((start, prev_idx))
            in_seg = False
        prev_idx = idx
    if in_seg:
        segs.append((start, prev_idx))
    return segs

# 评价指标与工具
def smape(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = 1e-8
    return np.mean(2.0 * np.abs(y_pred - y_true) / denom)

def compute_metrics(y_true, y_pred):
    # 使用基础指标计算
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    s_mape = smape(y_true, y_pred)
    
    base_metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2),
        'smape': float(s_mape)
    }
    
    # 如果增强分析工具可用，添加偏差分析指标
    if ANALYSIS_TOOLS_AVAILABLE:
        try:
            bias_metrics = enhanced_compute_metrics(y_true, y_pred)
            # 合并基础指标和偏差指标
            base_metrics.update(bias_metrics)
        except Exception as e:
            print(f"偏差分析计算失败: {e}")
    
    return base_metrics

# 滚动验证骨架（后续可扩展为多模型与动态选模）
def rolling_validate_rnn(scaled_series, time_step=120, train_span=240, val_span=20, epochs=5, batch_size=32):
    metrics_list = []
    X_total, y_total = create_dataset(scaled_series, time_step)
    # 起点对齐 time_step
    # 滚动窗口：每次用 train_span 训练，预测接下来的 val_span
    start_idx = 0
    while True:
        train_end = start_idx + train_span
        val_end = train_end + val_span
        if val_end > len(X_total):
            break
        X_train = X_total[start_idx:train_end]
        y_train = y_total[start_idx:train_end]
        X_val = X_total[train_end:val_end]
        y_val = y_total[train_end:val_end]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        model = build_rnn_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        y_pred = model.predict(X_val, verbose=0)
        metrics_list.append(compute_metrics(y_val, y_pred))
        start_idx += val_span
    if not metrics_list:
        return None
    # 平均指标
    avg = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0].keys()}
    return avg

# 训练并预测函数
def train_and_predict(stock_code, time_range='全部', start_date=None, end_date=None, forecast_days=30, model_choice='SimpleRNN',
                     dynamic_select=False, do_backtest=False, threshold=0.0, fee=0.0003, slippage=0.0002, n_states=3,
                     show_states=False, show_positions=False, export_csv=False, save_png=False,
                     capital=100000.0, position_size=1.0, position_alpha=0.35,
                     lstm_epochs=30, use_early_stopping=True, es_patience=5,
                     retrain=False, enable_bias_analysis=True, enable_feature_importance=True):
    # 获取股票数据
    if start_date and end_date:
        stock_data = get_stock_data(stock_code, start_date, end_date)
    else:
        stock_data = get_stock_data(stock_code)
    # 根据时间范围筛选数据
    if time_range not in ['全部', '自定义输入']:
        years = {'一年': 1, '三年': 3, '五年': 5, '十年': 10}[time_range]
        start_date = stock_data.index.max() - pd.DateOffset(years=years)
        end_date = stock_data.index.max()
        stock_data = stock_data[stock_data.index >= start_date]
    elif time_range == '全部':
        start_date = stock_data.index.min()
        end_date = stock_data.index.max()
    # 统一格式化日期
    start_str = pd.to_datetime(start_date).strftime('%Y%m%d')
    end_str = pd.to_datetime(end_date).strftime('%Y%m%d')
    model_path = os.path.join(MODEL_DIR, f"{stock_code}_{start_str}_{end_str}_model_{model_choice}.h5")

    # 数据预处理 — 设置时间划分边界
    close_values = stock_data['close'].values.reshape(-1, 1)
    preliminary_train_end = int(len(close_values) * 0.8)
    preliminary_train_end = max(preliminary_train_end, 1)

    selected_model = model_choice
    if dynamic_select:
        print(f"启用动态选模，候选模型包括: {['SimpleRNN', 'LSTM'] + (['LightGBM'] if LGB_AVAILABLE else [])}")
        # 近窗动态选模：在训练结束前留出最近20样本作为验证，比较多模型在验证集上的 MAE
        candidate_models = ['SimpleRNN', 'LSTM'] + (['LightGBM'] if LGB_AVAILABLE else [])
        # 统一以价格作为预测目标做对比
        # 划定验证窗口长度
        val_win = 20
        # 基于 close 序列准备 RNN/LSTM 的 scaler
        _, scaler_dyn = preprocess_data(stock_data)
        preliminary_train_end = int(len(stock_data) * 0.8)
        scaler_dyn.fit(stock_data['close'].values.reshape(-1, 1)[:preliminary_train_end])
        # 分别训练并计算验证集 MAE
        val_scores = {}
        for m in candidate_models:
            if m in ['SimpleRNN', 'LSTM']:
                scaled_data_dyn = scaler_dyn.transform(stock_data['close'].values.reshape(-1, 1))
                time_step = 120
                X_all, y_all = create_dataset(scaled_data_dyn, time_step)
                train_size_dyn = max(preliminary_train_end - time_step, int(len(X_all) * 0.8))
                # 留出最后 val_win 作为验证
                if train_size_dyn <= val_win:
                    continue
                X_train_dyn = X_all[:train_size_dyn - val_win]
                y_train_dyn = y_all[:train_size_dyn - val_win]
                X_val_dyn = X_all[train_size_dyn - val_win:train_size_dyn]
                y_val_dyn = y_all[train_size_dyn - val_win:train_size_dyn]
                X_train_dyn = X_train_dyn.reshape(X_train_dyn.shape[0], X_train_dyn.shape[1], 1)
                X_val_dyn = X_val_dyn.reshape(X_val_dyn.shape[0], X_val_dyn.shape[1], 1)
                if m == 'LSTM':
                    mdl = build_lstm_model((X_train_dyn.shape[1], 1))
                else:
                    mdl = build_rnn_model((X_train_dyn.shape[1], 1))
                callbacks = [EarlyStopping(monitor='loss', patience=es_patience, restore_best_weights=True)] if use_early_stopping else []
                mdl.fit(X_train_dyn, y_train_dyn, epochs=int(max(lstm_epochs//2, 5)), batch_size=32, verbose=0, callbacks=callbacks)
                y_pred_dyn = mdl.predict(X_val_dyn, verbose=0)
                y_true_dyn = y_val_dyn.reshape(-1, 1)
                y_pred_dyn_inv = scaler_dyn.inverse_transform(y_pred_dyn)
                y_true_dyn_inv = scaler_dyn.inverse_transform(y_true_dyn)
                val_scores[m] = mean_absolute_error(y_true_dyn_inv, y_pred_dyn_inv)
            elif m == 'LightGBM' and LGB_AVAILABLE:
                feat_df, X_all, y_all, _ = build_lgb_features(stock_data)
                boundary_date = stock_data.index[min(preliminary_train_end, len(stock_data) - 1)]
                mask = feat_df.index < boundary_date
                X_tr_all, y_tr_all = X_all[mask], y_all[mask]
                if len(X_tr_all) <= val_win:
                    continue
                X_train_dyn = X_tr_all[:-val_win]
                y_train_dyn = y_tr_all[:-val_win]
                X_val_dyn = X_tr_all[-val_win:]
                y_val_dyn = y_tr_all[-val_win:]
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'learning_rate': 0.05,
                    'num_leaves': 31,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 1,
                    'verbose': -1
                }
                dtrain = lgb.Dataset(X_train_dyn, label=y_train_dyn)
                mdl = lgb.train(params, dtrain, num_boost_round=200)
                y_pred_dyn = mdl.predict(X_val_dyn)
                val_scores[m] = mean_absolute_error(y_val_dyn, y_pred_dyn)
        if len(val_scores) > 0:
            # 选择 MAE 最小的模型
            print(f"动态选模结果 - 各模型MAE: {val_scores}")
            selected_model = sorted(val_scores.items(), key=lambda x: x[1])[0][0]
            print(f"动态选模选择了: {selected_model}")

    if selected_model in ['SimpleRNN', 'LSTM']:
        # 仅用训练集拟合归一化
        _, scaler = preprocess_data(stock_data)
        scaler.fit(close_values[:preliminary_train_end])
        scaled_data = scaler.transform(close_values)

        # 创建时间序列数据集
        time_step = 120  # 使用过去120天的数据来预测
        X, y = create_dataset(scaled_data, time_step)

        # 划分训练集和测试集
        train_size = max(preliminary_train_end - time_step, int(len(X) * 0.8))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Reshape 输入数据为 RNN/LSTM 格式
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # 检查是否已有保存的模型
        if os.path.exists(model_path) and not retrain:
            model = load_model(model_path)
        else:
            if selected_model == 'LSTM':
                model = build_lstm_model((X_train.shape[1], 1))
            else:
                model = build_rnn_model((X_train.shape[1], 1))
            callbacks = [EarlyStopping(monitor='loss', patience=es_patience, restore_best_weights=True)] if use_early_stopping else []
            model.fit(X_train, y_train, epochs=int(lstm_epochs), batch_size=32, verbose=0, callbacks=callbacks)
            model.save(model_path)

        # 进行预测并反归一化
        predicted_stock_price = model.predict(X_test, verbose=0)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
        real_stock_price = scaler.inverse_transform(y_test.reshape(-1, 1))

        # 未来股价预测（迭代）
        last_seq = scaler.transform(stock_data['close'].values.reshape(-1, 1))[-time_step:].reshape(1, time_step, 1)
        future_predictions = []
        for _ in range(forecast_days):
            next_pred = model.predict(last_seq, verbose=0)
            future_predictions.append(next_pred[0, 0])
            last_seq = np.append(last_seq[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    elif selected_model == 'LightGBM':
        print(f"尝试使用LightGBM模型，LGB_AVAILABLE={LGB_AVAILABLE}")
        if not LGB_AVAILABLE:
            # 不再回退，而是抛出错误
            raise ImportError("错误: LightGBM库不可用，请先安装LightGBM库: pip install lightgbm")
        # 构造特征并按日期划分
        feat_df, X_all, y_all, feature_cols = build_lgb_features(stock_data)
        boundary_date = stock_data.index[min(preliminary_train_end, len(stock_data) - 1)]
        train_mask = feat_df.index < boundary_date
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[~train_mask], y_all[~train_mask]
        if len(X_train) < 20 or len(X_test) == 0:
            # 不再回退，而是抛出错误
            raise ValueError(f"错误: 数据不足，X_train长度={len(X_train)}，X_test长度={len(X_test)}，无法训练LightGBM模型")
        # 训练
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbose': -1
        }
        try:
            print(f"开始训练LightGBM模型，训练集大小={len(X_train)}，测试集大小={len(X_test)}")
            dtrain = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(params, dtrain, num_boost_round=300)
            # 预测
            y_pred = model.predict(X_test)
            print(f"LightGBM模型训练和预测成功")
        except Exception as e:
            # 不再回退，而是抛出错误
            raise RuntimeError(f"错误: LightGBM模型训练失败，错误信息: {str(e)}")

        predicted_stock_price = y_pred.reshape(-1, 1)
        real_stock_price = y_test.reshape(-1, 1)
        # 未来预测（基于最近窗口迭代更新滞后）
        future_predictions = []
        hist_close = list(stock_data['close'].values)
        for _ in range(forecast_days):
            # 构造一行特征：使用最近的真实/预测 close 作为滞后
            lag_1 = hist_close[-1]
            lag_2 = hist_close[-2] if len(hist_close) >= 2 else lag_1
            lag_3 = hist_close[-3] if len(hist_close) >= 3 else lag_2
            lag_4 = hist_close[-4] if len(hist_close) >= 4 else lag_3
            lag_5 = hist_close[-5] if len(hist_close) >= 5 else lag_4
            arr = np.array([lag_1, lag_2, lag_3, lag_4, lag_5])
            roll_mean_5 = np.mean(arr)
            roll_std_5 = np.std(arr)
            feature_vec = np.array([lag_1, lag_2, lag_3, lag_4, lag_5, roll_mean_5, roll_std_5]).reshape(1, -1)
            pred = model.predict(feature_vec)[0]
            future_predictions.append(pred)
            hist_close.append(float(pred))
        future_predictions = np.array(future_predictions)
    else:
        raise ValueError('未知的模型选择')

    # 计算评估指标
    metrics = compute_metrics(real_stock_price, predicted_stock_price)
    # 指标简要解读
    try:
        price_scale = float(np.median(real_stock_price)) if len(real_stock_price) > 0 else 1.0
        mae_rel = float(metrics['mae'] / max(price_scale, 1e-8))
        smape_v = float(metrics['smape'])
        def tag(value, low, high):
            return '偏小' if value < low else ('偏大' if value > high else '正常')
        metrics['mae_comment'] = tag(mae_rel, 0.002, 0.02)  # 0.2%-2%价格尺度
        metrics['smape_comment'] = tag(smape_v, 0.02, 0.15)
    except Exception:
        metrics['mae_comment'] = ''
        metrics['smape_comment'] = ''
    
    # 偏差分析与可视化
    bias_analysis_image = None
    bias_report = None
    if ANALYSIS_TOOLS_AVAILABLE and enable_bias_analysis:
        try:
            # 生成偏差分析可视化
            bias_analysis_image = plot_bias_analysis(real_stock_price, predicted_stock_price, test_dates)
            
            # 诊断偏差问题
            bias_diagnosis = diagnose_bias_problem(metrics)
            
            # 生成偏差分析报告
            bias_report = generate_bias_analysis_report(metrics, bias_diagnosis)
            
            # 保存偏差分析图像
            if save_png:
                ts_suffix = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                bias_img_path = os.path.join(OUTPUT_DIR, f"{stock_code}_{start_str}_{end_str}_{selected_model}_{ts_suffix}_bias_analysis.png")
                bias_analysis_image.save(bias_img_path)
                
            # 保存偏差分析报告
            if export_csv:
                report_path = os.path.join(OUTPUT_DIR, f"{stock_code}_{start_str}_{end_str}_{selected_model}_{ts_suffix}_bias_report.txt")
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(bias_report)
                    
        except Exception as e:
            print(f"偏差分析执行失败: {e}")
    
    # LightGBM特征重要性分析
    feature_importance_image = None
    feature_report = None
    if ANALYSIS_TOOLS_AVAILABLE and enable_feature_importance and selected_model == 'LightGBM':
        try:
            # 分析特征重要性
            importance_results = analyze_lgb_feature_importance(model, feature_cols)
            feature_importance_image = importance_results['visualization']
            
            # 生成特征重要性报告
            feature_report = generate_feature_importance_report(importance_results)
            
            # 保存特征重要性图像
            if save_png:
                ts_suffix = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                feat_img_path = os.path.join(OUTPUT_DIR, f"{stock_code}_{start_str}_{end_str}_{selected_model}_{ts_suffix}_feature_importance.png")
                feature_importance_image.save(feat_img_path)
                
            # 保存特征重要性报告
            if export_csv:
                report_path = os.path.join(OUTPUT_DIR, f"{stock_code}_{start_str}_{end_str}_{selected_model}_{ts_suffix}_feature_report.txt")
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(feature_report)
                    
            # 将重要特征信息添加到指标中
            metrics['top_features'] = importance_results['feature_importance_df']['feature'].head(5).tolist()
            metrics['n_features_95'] = importance_results['n_features_95']
            
        except Exception as e:
            print(f"特征重要性分析执行失败: {e}")
    
    # 存储分析报告供Gradio界面使用
    metrics['bias_report'] = bias_report
    metrics['feature_report'] = feature_report

    # 获取测试集对应的日期索引
    test_dates = stock_data.index[-len(real_stock_price):]
    # 获取未来预测日期
    future_dates = pd.bdate_range(start=test_dates[-1] + pd.Timedelta(days=1), periods=forecast_days)

    # 绘图
    plt.figure(figsize=(14, 7))
    ax = plt.gca()
    ax.set_axisbelow(True)
    # 先画背景，再画线，确保线在最上层
    state_legend_handles = []

    # 状态背景色（可选）
    if show_states:
        states = compute_market_states(stock_data, n_states=n_states)
        if (states is not None) and (getattr(states, 'dropna', lambda: [])().size if hasattr(states, 'dropna') else True):
            # 使用更鲜明的颜色配置，与测试环境保持一致
            palette = ["#f5f5f5", "#4a86e8", "#ff9900", "#6aa84f"]  # 更鲜明的颜色
            last_state = None
            seg_start = None
            prev_dt = None
            for dt, st in states.items():
                if pd.isna(st):
                    continue
                if last_state is None:
                    last_state = st
                    seg_start = dt
                elif st != last_state:
                    # 使用更高的透明度以提高可见性
                    ax.axvspan(seg_start, dt, color=palette[int(last_state) % len(palette)], alpha=0.4, linewidth=0, zorder=0)
                    seg_start = dt
                    last_state = st
                prev_dt = dt
            if last_state is not None and seg_start is not None and prev_dt is not None:
                ax.axvspan(seg_start, prev_dt, color=palette[int(last_state) % len(palette)], alpha=0.4, linewidth=0, zorder=0)
            # 为状态创建图例项
            try:
                import matplotlib.patches as mpatches
                label_map = summarize_state_labels(stock_data, states)
                unique_states = sorted(int(s) for s in states.dropna().unique())
                
                
                
                
                
                for s in unique_states:
                    desc = label_map.get(int(s), '')
                    text = f"状态 {s}" + (f"（{desc}）" if desc else '')
                    patch = mpatches.Patch(color=palette[int(s) % len(palette)], label=text, alpha=0.3)
                    state_legend_handles.append(patch)
            except Exception:
                pass

    # 再画折线，置于更高zorder
    real_line, = ax.plot(stock_data.index, stock_data['close'], color='red', label='All Real Stock Price', zorder=3)
    pred_line, = ax.plot(test_dates, predicted_stock_price, color='blue', label='Predicted Stock Price', zorder=3)
    future_line, = ax.plot(future_dates, future_predictions, color='green', label='Future Predictions', zorder=3)

    if len(future_predictions) > 0:
        # 最高、最低、最后一天
        max_idx = np.argmax(future_predictions)
        min_idx = np.argmin(future_predictions)
        max_date = future_dates[max_idx]
        min_date = future_dates[min_idx]
        max_price = future_predictions[max_idx]
        min_price = future_predictions[min_idx]
        last_date = future_dates[-1]
        last_price = future_predictions[-1]

        plt.annotate(f"最高: {max_price:.2f}",
                     xy=(max_date, max_price),
                     xytext=(0, 20),
                     textcoords="offset points",
                     xycoords='data',
                     fontsize=12, color='green',
                     bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.5))
        plt.annotate(f"最低: {min_price:.2f}",
                     xy=(min_date, min_price),
                     xytext=(0, -30),
                     textcoords="offset points",
                     xycoords='data',
                     fontsize=12, color='green',
                     bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.5))
        plt.annotate(f"{last_price:.2f}", 
                     xy=(last_date, last_price), 
                     xytext=(0, 10),
                     textcoords="offset points",
                     xycoords='data',
                     fontsize=12, color='green',
                     bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.5))

    title_suffix = ''
    try:
        title_suffix = f" | 模型: {metrics.get('selected_model','')} | 数据最后日期: {pd.to_datetime(stock_data.index.max()).strftime('%Y-%m-%d')}"
    except Exception:
        pass
    plt.title(f'{stock_code} Stock Price Prediction ({time_range})' + title_suffix)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend(loc='upper left', ncol=1)
    plt.tight_layout()

    # 可选：基础回测
    if do_backtest:
        pred_series = pd.Series(index=test_dates, data=predicted_stock_price.reshape(-1))
        close_series = stock_data['close']
        bt = simple_backtest(close_series, pred_series, threshold=threshold, fee=fee, slippage=slippage)
        metrics.update({
            'bt_ann_return': bt['ann_return'],
            'bt_sharpe': bt['sharpe'],
            'bt_max_drawdown': bt['max_drawdown'],
            'bt_final_equity': bt['final_equity']
        })
        # 持仓区间（可选叠加）
        if show_positions and bt.get('position_series') is not None:
            pos = bt['position_series'].reindex(test_dates).fillna(0).astype(bool)
            for s, e in _segments_from_binary_series(pos):
                ax.axvspan(s, e, color='#d2f8d2', alpha=float(position_alpha), linewidth=0, zorder=1)

        # 导出CSV（可选）
        if export_csv:
            ts_suffix = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            base = f"{stock_code}_{start_str}_{end_str}_{model_choice}_{ts_suffix}"
            # 预测对比
            df_pred = pd.DataFrame({
                'date': test_dates,
                'real_close': real_stock_price.reshape(-1),
                'pred_close': predicted_stock_price.reshape(-1)
            })
            df_pred.to_csv(os.path.join(OUTPUT_DIR, base + '_pred.csv'), index=False)
            # 未来预测
            df_future = pd.DataFrame({'date': future_dates, 'future_pred_close': future_predictions})
            df_future.to_csv(os.path.join(OUTPUT_DIR, base + '_future.csv'), index=False)
            # 回测曲线与持仓
            try:
                bt['equity_series'].rename('equity').to_csv(os.path.join(OUTPUT_DIR, base + '_equity.csv'))
                bt['position_series'].astype(int).rename('position').to_csv(os.path.join(OUTPUT_DIR, base + '_position.csv'))
            except Exception:
                pass
            # 指标
            pd.DataFrame([metrics]).to_csv(os.path.join(OUTPUT_DIR, base + '_metrics.csv'), index=False)

    # 合并图例：状态图例 + 线条图例
    try:
        if state_legend_handles:
            handles, labels = ax.get_legend_handles_labels()
            handles = handles + state_legend_handles
            labels = labels + [h.get_label() for h in state_legend_handles]
            ax.legend(handles, labels)
    except Exception:
        pass

    # 可选：保存PNG（在所有覆盖层处理后执行）
    if save_png:
        ts_suffix = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        base = f"{stock_code}_{start_str}_{end_str}_{selected_model}_{ts_suffix}"
        try:
            plt.savefig(os.path.join(OUTPUT_DIR, base + '_plot.png'), dpi=150)
        except Exception:
            pass

    # 最终导出到内存，生成展示图片
    img_io = io.BytesIO()
    plt.savefig(img_io, format='PNG')
    plt.close()
    img_io.seek(0)
    img = Image.open(img_io)

    metrics['selected_model'] = selected_model
    metrics['last_data_date'] = pd.to_datetime(stock_data.index.max()).strftime('%Y-%m-%d')
    return metrics, img

# Gradio 界面函数
def gradio_interface(stock_code, time_range, custom_code, custom_time, model_choice,
                     dynamic_select, do_backtest, threshold, fee, slippage,
                     show_states, show_positions, export_csv, save_png,
                     capital, position_size, position_alpha,
                     lstm_epochs, use_early_stopping, es_patience,
                     retrain, enable_bias_analysis=True, enable_feature_importance=True):
    code = custom_code if stock_code == "自定义输入" else stock_code
    # 判断时间范围
    if time_range == "自定义输入" and custom_time:
        try:
            start_date, end_date = [s.strip() for s in custom_time.split(",")]
        except Exception:
            return "自定义时间格式错误，应为: 20200101,20231231", None
    else:
        start_date, end_date = None, None
    metrics, img = train_and_predict(
        code, time_range, start_date, end_date,
        model_choice=model_choice,
        dynamic_select=dynamic_select,
        do_backtest=do_backtest,
        threshold=threshold,
        fee=fee,
        slippage=slippage,
        show_states=show_states,
        show_positions=show_positions,
        export_csv=export_csv,
        save_png=save_png,
        capital=capital,
        position_size=position_size,
        position_alpha=position_alpha,
        lstm_epochs=lstm_epochs,
        use_early_stopping=use_early_stopping,
        es_patience=es_patience,
        retrain=retrain,
        enable_bias_analysis=enable_bias_analysis,
        enable_feature_importance=enable_feature_importance
    )
    text = (
        f"MSE: {metrics['mse']:.4f}\n"
        f"MAE: {metrics['mae']:.4f}\n"
        f"R²: {metrics['r2']:.4f}\n"
        f"SMAPE: {metrics['smape']:.4f}"
    )
    
    # 添加偏差分析指标（如果可用）
    if 'bias' in metrics:
        text += (
            f"\n\n======= 偏差分析 ======="
            f"\n整体偏差: {metrics['bias']:.4f}"
            f"\n相对偏差: {metrics['rel_bias']*100:.2f}%"
            f"\n偏差标准差: {metrics['bias_std']:.4f}"
            f"\n正偏差比例: {metrics['pos_deviation_ratio']*100:.1f}%"
            f"\n负偏差比例: {metrics['neg_deviation_ratio']*100:.1f}%"
        )
    
    # 模型信息
    if 'selected_model' in metrics:
        text = f"模型: {metrics['selected_model']}\n" + text
        # 如果是LightGBM且有特征重要性信息
        if metrics['selected_model'] == 'LightGBM' and 'top_features' in metrics:
            text += f"\n\n重要特征（前5个）: {', '.join(metrics['top_features'][:5])}"
            if 'n_features_95' in metrics:
                text += f"\n贡献95%重要性的特征数: {metrics['n_features_95']}"
    
    # 数据最后日期
    if 'last_data_date' in metrics:
        text += f"\n\n数据最后日期: {metrics['last_data_date']}"
    
    # 回测指标追加显示（如果有）
    if 'bt_ann_return' in metrics:
        # 年化与夏普的粗略解读
        def tag(v, low, high):
            return '偏小' if v < low else ('偏大' if v > high else '正常')
        ann_tag = tag(metrics['bt_ann_return'], -0.1, 0.3)
        shp_tag = tag(metrics['bt_sharpe'], 0.5, 2.0)
        dd_tag = tag(metrics['bt_max_drawdown'], -0.5, -0.1)  # 越接近0越好，负值
        text += (
            f"\n\n======= 回测结果 ======="
            f"\n年化: {metrics['bt_ann_return']:.4f}（{ann_tag}）"
            f"\n夏普: {metrics['bt_sharpe']:.4f}（{shp_tag}）"
            f"\n最大回撤: {metrics['bt_max_drawdown']:.4f}（{dd_tag}）"
            f"\n期末净值: {metrics['bt_final_equity']:.4f}"
        )
    
    # 误差解读
    if metrics.get('mae_comment') or metrics.get('smape_comment'):
        text += (
            f"\n\n======= 误差解读 ======="
            f"\nMAE解读: {metrics.get('mae_comment','')}"
            f"\nSMAPE解读: {metrics.get('smape_comment','')}"
        )
    return text, img

def show_custom_input(stock_code):
    return gr.update(visible=(stock_code == "自定义输入"))

def show_custom_time(time_range):
    return gr.update(visible=(time_range == "自定义输入"))

fixed_stocks = ['002027.SZ', '601288.SH', '601318.SH', '600900.SH', '601398.SH']

with gr.Blocks() as interface:
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("预测与模型"):
                    stock_dropdown = gr.Dropdown(
                        choices=fixed_stocks + ['自定义输入'],
                        label="选择股票",
                        value='002027.SZ'
                    )
                    custom_code = gr.Textbox(label="输入股票代码", visible=False)
                    time_range = gr.Dropdown(
                        choices=['一年', '三年', '五年', '十年', '全部', '自定义输入'],
                        label="时间范围",
                        value='全部'
                    )
                    custom_time = gr.Textbox(label="输入起止日期（格式: 20200101,20231231）", visible=False)
                    model_choice = gr.Dropdown(
                        choices=['SimpleRNN', 'LSTM', 'LightGBM'],
                        label="模型选择",
                        value='SimpleRNN'
                    )
                    dynamic_select = gr.Checkbox(label="近窗动态选模", value=False)
                    lstm_epochs = gr.Slider(label="LSTM/简单RNN训练轮数", minimum=5, maximum=100, step=5, value=30)
                    use_early_stopping = gr.Checkbox(label="启用早停(EarlyStopping)", value=True)
                    es_patience = gr.Slider(label="早停耐心(patience)", minimum=2, maximum=20, step=1, value=5)
                    retrain = gr.Checkbox(label="强制重新训练(忽略缓存)", value=False)
                with gr.TabItem("回测与可视化"):
                    do_backtest = gr.Checkbox(label="启用基础回测", value=False)
                    threshold = gr.Number(label="买入阈值(预测收益)", value=0.0)
                    fee = gr.Number(label="手续费", value=0.0003)
                    slippage = gr.Number(label="滑点", value=0.0002)
                    capital = gr.Number(label="初始资金", value=100000)
                    position_size = gr.Number(label="仓位比例(0~1)", value=1.0)
                    show_states = gr.Checkbox(label="叠加市场状态背景", value=False)
                    show_positions = gr.Checkbox(label="标记持仓区间(需启用回测)", value=False)
                    export_csv = gr.Checkbox(label="导出CSV结果到outputs/", value=False)
                    save_png = gr.Checkbox(label="保存图像PNG到outputs/", value=False)
                    position_alpha = gr.Slider(label="持仓标记透明度", minimum=0.1, maximum=0.6, step=0.05, value=0.35)
                    # 新增分析选项
                    enable_bias_analysis = gr.Checkbox(label="启用预测偏差分析", value=True)
                    enable_feature_importance = gr.Checkbox(label="启用特征重要性分析(LightGBM)", value=True)
            predict_btn = gr.Button("预测")
        with gr.Column(scale=2):
            output_text = gr.Textbox(label="评估/回测指标", lines=8)
            output_img = gr.Image(label="预测图", interactive=False)

    stock_dropdown.change(show_custom_input, stock_dropdown, custom_code)
    time_range.change(show_custom_time, time_range, custom_time)
    predict_btn.click(
        gradio_interface,
        inputs=[stock_dropdown, time_range, custom_code, custom_time, model_choice,
                dynamic_select, do_backtest, threshold, fee, slippage,
                show_states, show_positions, export_csv, save_png,
                capital, position_size, position_alpha,
                lstm_epochs, use_early_stopping, es_patience,
                retrain, enable_bias_analysis, enable_feature_importance],
        outputs=[output_text, output_img]
    )

interface.launch(share=True)