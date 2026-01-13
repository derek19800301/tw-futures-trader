import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# 設定頁面標題與佈局
st.set_page_config(page_title="台指期交易指引系統", page_icon="📈")

class TaiwanFuturesTrader:
    """台指期交易決策系統 (Streamlit 版)"""
    
    def __init__(self):
        # --- 策略參數 ---
        self.MA_TREND_PERIOD = 20
        self.MA_BIAS_PERIOD = 5
        self.ADX_PERIOD = 7
        self.ADX_THRESHOLD = 25
        self.BIAS_THRESHOLD = 0.025
        self.STOP_LOSS_PCT = 0.015
        self.ATR_PERIOD = 14
        
        # 合約規格
        self.TX_POINT_VALUE = 200
        self.MTX_POINT_VALUE = 50
        self.TX_MARGIN = 167000
        self.MTX_MARGIN = 42000

    @st.cache_data(ttl=3600) # 快取資料1小時
    def download_data(_self, days_back=100):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        try:
            # 下載資料
            df = yf.download('^TWII', start=start_date, end=end_date, progress=False)
            
            # 檢查是否下載到空資料
            if df is None or df.empty:
                return None

            # 處理多重索引 (yfinance 新版修正)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # 確保索引格式正確
            df.index = pd.to_datetime(df.index).normalize()
            
            # 再次檢查必要欄位
            if 'Close' not in df.columns:
                return None
                
            return df
        except Exception as e:
            return None

    def calculate_indicators(self, df):
        if df is None or df.empty:
            return pd.DataFrame() # 回傳空表

        df = df.copy()
        try:
            df['MA20'] = df['Close'].rolling(window=self.MA_TREND_PERIOD).mean()
            df['MA5'] = df['Close'].rolling(window=self.MA_BIAS_PERIOD).mean()
            df['MA20_Slope'] = df['MA20'].diff()
            df['Bias_MA5'] = (df['Close'] - df['MA5']) / df['MA5']
            
            # ATR
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = true_range.rolling(self.ATR_PERIOD).mean()
            
            # ADX
            high, low, close = df['High'], df['Low'], df['Close']
            tr = true_range
            pos_dm = np.where((high - high.shift()) > (low.shift() - low), high - high.shift(), 0)
            pos_dm = np.where(pos_dm < 0, 0, pos_dm)
            neg_dm = np.where((low.shift() - low) > (high - high.shift()), low.shift() - low, 0)
            neg_dm = np.where(neg_dm < 0, 0, neg_dm)
            
            alpha = 1 / self.ADX_PERIOD
            tr_smooth = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()
            pos_dm_smooth = pd.Series(pos_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
            neg_dm_smooth = pd.Series(neg_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
            
            pos_di = 100 * (pos_dm_smooth / tr_smooth)
            neg_di = 100 * (neg_dm_smooth / tr_smooth)
            dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
            df['ADX'] = dx.ewm(alpha=alpha, adjust=False).mean()
            
            return df.dropna()
        except Exception:
            return pd.DataFrame()

    def get_analysis(self, df):
        # 防呆：如果傳入空表，直接回傳預設值
        if df is None or df.empty:
            return None, "資料不足", {'action': 'WAIT', 'reason': '歷史資料不足無法計算'}, None

        latest = df.iloc[-1]
        close, ma20, ma20_slope = latest['Close'], latest['MA20'], latest['MA20_Slope']
        adx, bias, atr = latest['ADX'], latest['Bias_MA5'], latest['ATR']
        
        # 市場狀態
        trend = '盤整'
        if close > ma20 and ma20_slope > 0: trend = '多頭'
        elif close < ma20 and ma20_slope < 0: trend = '空頭'
        
        tradable = False
        if adx > self.ADX_THRESHOLD and abs(bias) < self.BIAS_THRESHOLD:
            tradable = True
            
        # 訊號
        signal = {'action': 'WAIT', 'direction': None, 'reason': ''}
        if not tradable:
            if adx <= self.ADX_THRESHOLD: signal['reason'] = f'動能不足 (ADX={adx:.1f})'
            else: signal['reason'] = f'乖離過大 ({bias*100:.2f}%)'
        elif trend == '多頭':
            signal.update({'action': 'BUY', 'direction': 'LONG', 'reason': '多頭趨勢確立，動能充足'})
        elif trend == '空頭':
            signal.update({'action': 'SELL', 'direction': 'SHORT', 'reason': '空頭趨勢確立，動能充足'})
            
        # 停損
        stop_loss = None
        if signal['action'] in ['BUY', 'SELL']:
            stop_distance = max(close * self.STOP_LOSS_PCT, atr * 2)
            sl_price = close - stop_distance if signal['direction'] == 'LONG' else close + stop_distance
            stop_loss = {'price': sl_price, 'distance': stop_distance}
            
        return latest, trend, signal, stop_loss

# --- 主介面 ---
st.title("🇹🇼 台指期交易決策助手")
st.caption("基於 MA20 + ADX + ATR 的量化策略")

# 側邊欄設定
with st.sidebar:
    st.header("⚙️ 參數與資金")
    capital = st.number_input("操作本金 (TWD)", value=500000, step=10000)
    risk_per_trade = st.slider("單筆風險上限 (%)", 1.0, 5.0, 2.0) / 100
    
    st.markdown("---")
    if st.button("🔄 更新今日數據"):
        st.cache_data.clear()
        st.rerun()

# 執行邏輯
trader = TaiwanFuturesTrader()
df_raw = trader.download_data()

# 嚴格檢查資料是否可用
if df_raw is not None and not df_raw.empty:
    df = trader.calculate_indicators(df_raw)
    
    # 檢查計算後是否變為空值 (例如資料筆數太少被 dropna 刪光)
    if df.empty:
        st.warning("⚠️ 取得的資料筆數不足以計算技術指標 (MA20/ADX)，請稍後再試。")
    else:
        latest_data, trend, signal, stop_loss = trader.get_analysis(df)
        
        if latest_data is not None:
            # 1. 顯示主要訊號
            col1, col2, col3 = st.columns(3)
            # 安全取得前一日收盤價
            prev_close = df.iloc[-2]['Close'] if len(df) > 1 else latest_data['Close']
            
            col1.metric("收盤價", f"{latest_data['Close']:.0f}", f"{latest_data['Close'] - prev_close:.0f}")
            col2.metric("市場趨勢", trend, delta_color="normal" if trend=="盤整" else "inverse")
            
            action_color = "gray"
            if signal['action'] == 'BUY': action_color = "red"
            elif signal['action'] == 'SELL': action_color = "green"
            
            col3.markdown(f"### 訊號: :{action_color}[{signal['action']}]")

            # 2. 詳細資訊
            st.info(f"💡 策略理由: {signal['reason']}")
            
            with st.expander("📊 查看技術指標詳情", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("ADX 動能", f"{latest_data['ADX']:.1f}", help="需大於 25")
                c2.metric("乖離率", f"{latest_data['Bias_MA5']*100:.2f}%", help="絕對值需小於 2.5%")
                c3.metric("ATR 波動", f"{latest_data['ATR']:.0f}")

            # 3. 資金管理建議
            if stop_loss:
                st.markdown("### 💰 資金管理建議")
                sl_dist = stop_loss['distance']
                max_risk_amt = capital * risk_per_trade
                
                # 計算口數
                tx_risk_lots = int(max_risk_amt / (sl_dist * trader.TX_POINT_VALUE))
                mtx_risk_lots = int(max_risk_amt / (sl_dist * trader.MTX_POINT_VALUE))
                
                st.write(f"**停損點位**: {stop_loss['price']:.0f} (距離 {sl_dist:.0f} 點)")
                st.write(f"**單筆最大虧損限制**: ${max_risk_amt:,.0f}")
                
                w1, w2 = st.columns(2)
                w1.success(f"大台建議口數: **{tx_risk_lots}** 口")
                w2.success(f"小台建議口數: **{mtx_risk_lots}** 口")
            
            # 4. 圖表
            st.markdown("### 📈 近期走勢")
            st.line_chart(df[['Close', 'MA20']].tail(100))
        else:
            st.error("❌ 分析失敗：無法計算當前訊號")

else:
    st.error("⚠️ 無法下載台股資料。")
    st.markdown("""
    **可能原因：**
    1. Yahoo Finance 暫時阻擋連線 (稍後再按「更新」試試)
    2. 目前非開盤時間或剛開盤，資料源尚未更新
    """)