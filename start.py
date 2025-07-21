#!/usr/bin/env python3

import os
import datetime
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import ccxt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time
import mplfinance as mpf

# --- Configuratie ---
DATA_DIR        = "./data"
CACHE_EXPIRY_DAYS = 1
LIMIT_MAP       = {"15m": 3000, "1h": 2000, "4h": 1500}
COINS           = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "SUI/USDT", "ADA/USDT", "XRP/USDT"]
TIMEFRAMES      = ["15m", "1h", "4h"]
SEQ_LEN         = 60
WINDOW_SIZE     = 300
EPOCHS          = 50
LEARNING_RATE   = 0.001
MODEL_DIR       = "./models"
LOG_DIR         = "./logs"

# --- LSTM Model ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers,
                               dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out     = self.dropout(out[:, -1, :])
        return self.linear(out).squeeze()

# --- Helpers for data ---
def merge_and_cache_df(existing_df, new_df, timeframe):
    merged = pd.concat([existing_df, new_df]).sort_index().drop_duplicates()
    limit  = LIMIT_MAP.get(timeframe, SEQ_LEN + WINDOW_SIZE)
    return merged.iloc[-limit:]

def load_or_fetch_data(exchange, coin, timeframe):
    limit  = LIMIT_MAP[timeframe]
    folder = os.path.join(DATA_DIR, coin.replace('/','_'))
    os.makedirs(folder, exist_ok=True)
    fname  = os.path.join(folder, f"{timeframe}.csv")

    if os.path.exists(fname) and (time.time()-os.path.getmtime(fname))/86400 < CACHE_EXPIRY_DAYS:
        df_old  = pd.read_csv(fname, parse_dates=['timestamp'], index_col='timestamp')
        since   = int(df_old.index[-1].timestamp() * 1000)
        raw     = exchange.fetch_ohlcv(coin, timeframe=timeframe, since=since, limit=limit)
        df_new  = pd.DataFrame(raw, columns=['timestamp','open','high','low','close','volume'])
        df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
        df_new.set_index('timestamp', inplace=True)
        df = merge_and_cache_df(df_old, df_new, timeframe)
    else:
        raw = exchange.fetch_ohlcv(coin, timeframe=timeframe, limit=limit)
        df  = pd.DataFrame(raw, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

    df.to_csv(fname)
    return df

# --- Indicators & preprocessing ---
def add_indicators(df):
    df['ema']       = df['close'].ewm(span=14, adjust=False).mean()
    delta           = df['close'].diff()
    up              = delta.clip(lower=0)
    down            = -delta.clip(upper=0)
    rs              = up.rolling(14).mean() / down.rolling(14).mean()
    df['rsi']       = 100 - (100/(1+rs))
    df['bb_mid']    = df['close'].rolling(20).mean()
    df['bb_std']    = df['close'].rolling(20).std()
    df['bb_upper']  = df['bb_mid'] + 2*df['bb_std']
    df['bb_lower']  = df['bb_mid'] - 2*df['bb_std']
    # MACD
    ema12           = df['close'].ewm(span=12, adjust=False).mean()
    ema26           = df['close'].ewm(span=26, adjust=False).mean()
    df['macd']      = ema12 - ema26
    df['macd_signal']= df['macd'].ewm(span=9, adjust=False).mean()
    # OBV
    df['obv']       = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    # ATR
    high_low        = df['high'] - df['low']
    high_close      = (df['high'] - df['close'].shift()).abs()
    low_close       = (df['low'] - df['close'].shift()).abs()
    tr              = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr']       = tr.rolling(14).mean()
    # StochRSI
    df['rsi_min']   = df['rsi'].rolling(14).min()
    df['rsi_max']   = df['rsi'].rolling(14).max()
    df['stoch_rsi'] = (df['rsi'] - df['rsi_min'])/(df['rsi_max']-df['rsi_min'])
    # Volume SMA
    df['vol_sma']   = df['volume'].rolling(20).mean()
    df.dropna(inplace=True)
    # Outlier filter
    df = df[(df['close'] < df['ema']*1.3) & (df['close'] > df['ema']*0.7)]
    return df

# --- Sequences ---
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len - 2):
        xs.append(data[i:i+seq_len])
        ys.append(np.mean(data[i+seq_len:i+seq_len+3, 0]))
    return np.array(xs), np.array(ys)

# --- Metrics ---
def evaluate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    return mse, mae, r2

# --- Evaluation plots ---
def plot_evaluation(X, y, preds_full, train_losses, val_losses, coin, tf):
    corr = np.corrcoef(preds_full, y[:len(preds_full)])[0,1]
    plt.scatter(y[:300], preds_full[:300], s=5)
    plt.title(f"Scatter {coin}[{tf}] R={corr:.2f}"); plt.show()
    plt.plot(y[:200], label='Actual')
    plt.plot(preds_full[:200], label='Predicted')
    plt.title(f"Overlay {coin}[{tf}]"); plt.legend(); plt.show()

# --- Save/Load model ---
def save_model(model, coin, timeframe, folder="models"):
    os.makedirs(folder, exist_ok=True)
    fn = f"{folder}/{coin.replace('/', '_')}_{timeframe}_model.pth"
    torch.save(model.state_dict(), fn)
    print(f"✅ Model opgeslagen: {fn}")

def load_model(model, coin, timeframe, folder="models"):
    fn = f"{folder}/{coin.replace('/', '_')}_{timeframe}_model.pth"
    if os.path.exists(fn):
        model.load_state_dict(torch.load(fn))
        model.eval()
        print(f"✅ Model geladen: {fn}")
    else:
        print(f"⚠️ Geen modelbestand gevonden: {fn}")

# --- Main training pipeline ---
def run_pipeline():
    exchange = ccxt.mexc()
    start_all = time.time()
    for coin in COINS:
        for tf in TIMEFRAMES:
            print(f"🚀 {coin}[{tf}] 5-fold CV met early stopping")
            # --- Load and preprocess data with caching ---
            df = load_or_fetch_data(exchange, coin, tf)
            # Save preprocessed cache
            preproc_folder = os.path.join(DATA_DIR, coin.replace('/','_'))
            proc_file = os.path.join(preproc_folder, f"preprocessed_{tf}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv")
            df = add_indicators(df)
            df.to_csv(proc_file)
            # --- Pre-train validation & automatic drop ---
            min_rows = SEQ_LEN * 2 + WINDOW_SIZE + 3
            if len(df) < min_rows:
                print(f"Dataset te klein na indicators ({len(df)} < {min_rows}), overslaan...")
                continue
            if df['close'].std() < 1e-3:
                print(f"Dataset te vlak (std={df['close'].std():.6f}), overslaan...")
                continue
            # Prepare sequences
            arr = df[['close','ema','rsi','bb_upper','bb_lower','macd','obv','atr','stoch_rsi','vol_sma']].values
            scaled = MinMaxScaler().fit_transform(arr)
            X, y = create_sequences(scaled, SEQ_LEN)
            if len(X) < 5:
                print(f"Te weinig samples voor CV: {len(X)} < 5, overslaan...")
                continue
            # ... rest of CV and training loop unchanged ...
        for tf in TIMEFRAMES:
            print(f"\n🚀 {coin}[{tf}] 5-fold CV met early stopping")
            df = load_or_fetch_data(exchange, coin, tf)
            df = add_indicators(df)
            arr = df[['close','ema','rsi','bb_upper','bb_lower','macd','obv','atr','stoch_rsi','vol_sma']].values
            if len(arr)<SEQ_LEN*2:
                print("Te weinig data voor CV"); continue
            scaled=MinMaxScaler().fit_transform(arr)
            X, y = create_sequences(scaled, SEQ_LEN)
            if len(X)<5:
                print("Te weinig samples voor 5-fold"); continue
            kf=KFold(n_splits=5, shuffle=False)
            best_state, best_val=float('inf'),None
            train_losses, val_losses=[],[]
            for fold,(tr,vl) in enumerate(kf.split(X),1):
                Xtr,Xvl=X[tr],X[vl]; ytr,yvl=y[tr],y[vl]
                Xt, Yt = torch.tensor(Xtr).float(), torch.tensor(ytr).float()
                Xv, Yv = torch.tensor(Xvl).float(), torch.tensor(yvl).float()
                model=LSTMModel(input_size=X.shape[2])
                opt=optim.Adam(model.parameters(),lr=LEARNING_RATE); crit=nn.MSELoss()
                patience,wait=5,0; tl,vlosses=[],[]
                for ep in range(EPOCHS):
                    model.train()
                    opt.zero_grad()
                    l = crit(model(Xt), Yt)
                    l.backward()
                    opt.step()
                    tl.append(l.item())
                    # Validation step
                    model.eval()
                    v = crit(model(Xv), Yv).item()
                    vlosses.append(v)
                    # Check for best model
                    if best_val is None or v < best_val:
                        best_val = v
                        best_state = model.state_dict().copy()
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            print(f"Early stop fold {fold} epoch {ep+1}")
                            break
                train_losses.append(tl)
                val_losses.append(vlosses)
            # finalize
            model.load_state_dict(best_state); model.eval()
            preds_full = model(torch.tensor(X).float()).detach().numpy().flatten()
            # plots
            plot_evaluation(X,y,preds_full,train_losses,val_losses,coin,tf)
            # save
            save_model(model, coin, tf)
            # tekst stats per coin/timeframe
            mse,mae,r2=evaluate_metrics(y[:len(preds_full)],preds_full)
            corr=np.corrcoef(y[:len(preds_full)],preds_full)[0,1]
            train_gap=np.mean([abs(tl[-1]-vl[-1]) for tl,vl in zip(train_losses,val_losses)])
            prices=X[:len(preds_full), -1,0]
            returns=[(prices[i+1]-prices[i])/prices[i] for i in range(len(prices)-1) if preds_full[i+1]>preds_full[i]]
            equity=np.cumsum(returns)
            sharpe=(np.mean(returns)/ (np.std(returns)+1e-8)) if returns else 0
            winrate=(sum(r>0 for r in returns)/len(returns)) if returns else 0
            max_dd=np.max(np.maximum.accumulate(equity)-equity) if equity.size else 0
            bh=(prices[-1]-prices[0])/prices[0]
            print(f"📊 Overzicht {coin}[{tf}]:")
            # --- Append run metrics to CSV ---
            row = {
                'timestamp': datetime.datetime.now().isoformat(),
                'coin': coin,
                'timeframe': tf,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'corr': corr,
                'train_val_gap': train_gap,
                'winrate': winrate,
                'sharpe': sharpe,
                'max_dd': max_dd,
                'pnl_usd': float(equity[-1])
            }
            csv_path = os.path.join(LOG_DIR, 'metrics.csv')
            df_row = pd.DataFrame([row])
            if os.path.exists(csv_path):
                df_old = pd.read_csv(csv_path)
                df_new = pd.concat([df_old, df_row], ignore_index=True)
            else:
                df_new = df_row
            df_new.to_csv(csv_path, index=False)
            print(f" MSE: {mse:.6f} | MAE: {mae:.6f} | R²: {r2:.3f} | Corr: {corr:.3f}")
            print(f" Train–Val gap: {train_gap:.6f}")
            print(f" Winrate: {winrate*100:.1f}% | Sharpe: {sharpe:.2f} | Max DD: {max_dd*100:.1f}%")
            print(f" Model Equity vs B&H: {equity[-1]:.4f} vs {bh:.4f}\n")
    print(f"⏱️ Done in {time.time()-start_all:.2f}s")


# --- Continuous trading functie ---
def continuous_trading(interval_minutes=15):
    exchange = ccxt.mexc()
    while True:
        for coin in COINS:
            for tf in TIMEFRAMES:
                print(f"\n🔄 Voorspelling voor {coin}[{tf}]...")
                df = load_or_fetch_data(exchange, coin, tf)
                df = add_indicators(df)
                if len(df) < SEQ_LEN + 5:
                    print("⚠️ Te weinig data voor voorspelling.")
                    continue
                scaled = MinMaxScaler().fit_transform(
                    df[['close','ema','rsi','bb_upper','bb_lower','macd','macd_signal','obv','atr','stoch_rsi','vol_sma']]
                )
                X, _ = create_sequences(scaled, SEQ_LEN)
                model = LSTMModel(input_size=X.shape[2])
                load_model(model, coin, tf)
                with torch.no_grad():
                    x = torch.tensor(X[-1:]).float()
                    pred = model(x).item()
                    current = scaled[-1, 0]
                    change_pct = (pred - current) / current * 100
                    print(f"📈 Verwachte verandering: {change_pct:.2f}%")
        print(f"\n⏳ Wachten {interval_minutes} minuten...\n")
        time.sleep(interval_minutes * 60)

# --- CLI Menu ---
if __name__ == "__main__":
    while True:
        print("\nSelectie:\n1. Eénmalige run\n2. Continue trading\n3. Exit")
        choice = input("Kies (1/2/3): ")
        if choice == '1':
            run_pipeline()
        elif choice == '2':
            continuous_trading()
        else:
            print("Einde programma.")
            break
