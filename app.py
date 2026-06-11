import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Quant Portfolio Builder Pro", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Quant Portfolio Builder Pro")
st.markdown("*Multi-Factor Scoring + Fractional Kelly Optimization + Risk-Adjusted Allocation*")

# ============================================================
# 1. CONFIGURAZIONE SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Parametri Strategia")

total_capital = st.sidebar.number_input("Capitale Totale (€)", value=10000, step=1000, min_value=1000)

st.sidebar.subheader("📊 Risk-Adjusted Allocation")
risk_propensity = st.sidebar.slider(
    "Propensione al Rischio", 
    min_value=1, 
    max_value=100, 
    value=50, 
    help="1% = Conservativo (priorità a Quality e stabilità), 100% = Aggressivo (priorità al Momentum)"
)

st.sidebar.subheader("Parametri Kelly")
col_k1, col_k2 = st.sidebar.columns(2)
with col_k1:
    p_win = st.number_input("Probabilità p", min_value=0.1, max_value=0.95, value=0.55, step=0.01)
with col_k2:
    payoff_ratio = st.number_input("Payoff Ratio b", min_value=0.5, max_value=10.0, value=1.5, step=0.1)

kelly_fraction = st.sidebar.slider("Frazionamento Kelly", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

st.sidebar.subheader("Pesi Fattori (Base)")
w_momentum = st.sidebar.slider("Peso Momentum", 0.0, 1.0, 0.4, 0.05)
w_quality = st.sidebar.slider("Peso Quality (Sharpe)", 0.0, 1.0, 0.35, 0.05)
w_volatility = st.sidebar.slider("Peso Volatilità (invertita)", 0.0, 1.0, 0.25, 0.05)

lookback_days = st.sidebar.selectbox("Periodo Analisi", [90, 180, 252, 500], index=1)
top_pct = st.sidebar.slider("Selezione Top %", 1, 20, 10, 1)
min_stocks = st.sidebar.number_input("Minimo azioni portfolio", min_value=1, max_value=50, value=5)

st.sidebar.subheader("🌍 Universo di Investimento")
input_method = st.sidebar.radio("Metodo", ["📝 Lista Ticker", "📁 Carica CSV", "🎲 Demo Mode (dati fittizi)"])

tickers_list = []
if input_method == "📝 Lista Ticker":
    ticker_input = st.sidebar.text_area(
        "Ticker separati da virgola",
        value="AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META,JPM,V,MA,UNH,JNJ,XOM,PG,WMT,HD,BAC,ABBV,PFE,KO,PEP,COST,AVGO,TMO,DIS,ABT,ADBE,CSCO,CRM,VZ,ACN,TXN,NEE,PM,RTX,HON,AMGN,IBM,QCOM,LOW,SPGI,INTU,BMY,CAT,GE,BA,AMAT,SBUX,GILD,MDT",
        height=120
    )
    tickers_list = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
elif input_method == "📁 Carica CSV":
    uploaded_file = st.sidebar.file_uploader("CSV con colonna 'Ticker'", type=['csv'])
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        if 'Ticker' in df_upload.columns:
            tickers_list = df_upload['Ticker'].dropna().astype(str).str.upper().tolist()
        else:
            st.sidebar.error("Colonna 'Ticker' non trovata!")
else:
    tickers_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "V", "MA",
                   "UNH", "JNJ", "XOM", "PG", "WMT", "HD", "BAC", "ABBV", "PFE", "KO",
                   "PEP", "COST", "AVGO", "TMO", "DIS", "ABT", "ADBE", "CSCO", "CRM", "VZ"]
    st.sidebar.info("🎲 Modalità Demo attiva - dati simulati")

# ============================================================
# 2. FUNZIONI CORE
# ============================================================

def generate_demo_data(tickers, lookback_days):
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=lookback_days, freq='B')
    data = {}
    for ticker in tickers:
        drift = np.random.normal(0.0005, 0.001)
        volatility = np.random.uniform(0.15, 0.45)
        returns = np.random.normal(drift, volatility/np.sqrt(252), len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        data[ticker] = pd.Series(prices, index=dates)
    return pd.DataFrame(data)

def fetch_data_batch(tickers, lookback_days, demo_mode=False):
    if demo_mode:
        st.info("🎲 Generazione dati demo in corso...")
        return generate_demo_data(tickers, lookback_days)
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_days + 30)
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        progress_text.text("📡 Tentativo 1/3: Connessione diretta...")
        
        batch_size = 50
        all_data = []
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            progress_text.text(f"📡 Scaricando batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}...")
            
            try:
                data = yf.download(
                    tickers=batch,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                    threads=True,
                    ignore_tz=True
                )
                
                if not data.empty:
                    all_data.append(data)
                time.sleep(0.5)
                
            except Exception as e:
                st.warning(f"Batch {batch} fallito: {str(e)[:50]}")
                continue
            
            progress_bar.progress(min(0.7, (i + len(batch)) / len(tickers)))
        
        if all_data:
            progress_text.text("📊 Unione dati...")
            close_data = []
            for data_chunk in all_data:
                if len(tickers) == 1:
                    if 'Close' in data_chunk.columns:
                        close_data.append(data_chunk['Close'].to_frame(tickers[0]))
                else:
                    if isinstance(data_chunk.columns, pd.MultiIndex):
                        if 'Close' in data_chunk.columns.get_level_values(0):
                            close_data.append(data_chunk['Close'])
                        elif 'Adj Close' in data_chunk.columns.get_level_values(0):
                            close_data.append(data_chunk['Adj Close'])
            
            if close_data:
                close_prices = pd.concat(close_data, axis=1)
                close_prices = close_prices.loc[:, ~close_prices.columns.duplicated()]
                threshold = max(30, len(close_prices) * 0.5)
                close_prices = close_prices.dropna(axis=1, thresh=threshold)
                
                if len(close_prices.columns) >= min(len(tickers) * 0.3, 5):
                    progress_bar.progress(1.0)
                    progress_text.empty()
                    progress_bar.empty()
                    st.success(f"✅ Scaricati {len(close_prices.columns)} ticker via batch download")
                    return close_prices
        
    except Exception as e:
        st.warning(f"Download diretto fallito: {str(e)[:100]}")
    
    progress_text.text("📡 Tentativo 2/3: Download singoli ticker...")
    
    close_prices = pd.DataFrame()
    successful = []
    
    for i, ticker in enumerate(tickers[:100]):
        try:
            time.sleep(0.3)
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)
            
            if not hist.empty and 'Close' in hist.columns:
                close_prices[ticker] = hist['Close']
                successful.append(ticker)
                
            progress_bar.progress(min(0.9, (i + 1) / min(len(tickers), 100)))
            
        except Exception as e:
            continue
    
    if len(successful) >= 5:
        progress_text.empty()
        progress_bar.empty()
        st.success(f"✅ Scaricati {len(successful)} ticker via download singolo")
        return close_prices
    
    progress_text.empty()
    progress_bar.empty()
    st.warning("⚠️ Yahoo Finance blocca gli IP cloud. Attivazione automatica Demo Mode.")
    return generate_demo_data(tickers[:min(30, len(tickers))], lookback_days)

def calculate_factors(close_prices):
    factors = pd.DataFrame(index=close_prices.columns)
    returns = close_prices.pct_change().dropna()
    
    factors['Momentum'] = (close_prices.iloc[-1] / close_prices.iloc[0]) - 1
    
    mean_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    factors['Quality'] = mean_return / volatility.replace(0, np.nan)
    
    factors['Volatility'] = volatility
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    factors['Max_Drawdown'] = drawdown.min()
    
    return factors.dropna()

def zscore_normalize(series):
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0, index=series.index)
    return (series - mean) / std

def score_stocks(factors, w_mom, w_qual, w_vol):
    df = factors.copy()
    df['Z_Momentum'] = zscore_normalize(df['Momentum'])
    df['Z_Quality'] = zscore_normalize(df['Quality'])
    df['Z_Volatility'] = zscore_normalize(df['Volatility'])
    
    df['Base_Score'] = (w_mom * df['Z_Momentum'] + 
                        w_qual * df['Z_Quality'] - 
                        w_vol * df['Z_Volatility'])
    
    return df

def calculate_risk_adjusted_allocations(factors, risk_propensity):
    """
    Calcola pesi dinamici basati sulla propensione al rischio.
    R = 0 (conservativo) -> priorità a Quality, penalità a Volatility e Drawdown
    R = 1 (aggressivo) -> priorità a Momentum
    """
    df = factors.copy()
    R = risk_propensity / 100.0
    
    # Allocation Score personalizzato
    df['Alloc_Score'] = (df['Momentum'] * R) + \
                        (df['Quality'] * (1 - R)) - \
                        (df['Volatility'] * (1 - R) * 2) - \
                        (abs(df['Max_Drawdown']) * (1 - R) * 2)
    
    # Evita pesi negativi
    df['Raw_Weight'] = np.where(df['Alloc_Score'] > 0, df['Alloc_Score'], 0.01)
    
    return df

def calculate_kelly(p, b, fraction):
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0, kelly * fraction)

def backtest_portfolio(close_prices, selected_tickers, lookback_days):
    returns = close_prices[selected_tickers].pct_change().dropna()
    port_returns = returns.mean(axis=1)
    
    total_return = (1 + port_returns).prod() - 1
    positive_days = (port_returns > 0).sum()
    total_days = len(port_returns)
    
    p_empirical = positive_days / total_days if total_days > 0 else 0.5
    
    avg_win = port_returns[port_returns > 0].mean() if positive_days > 0 else 0.001
    avg_loss = abs(port_returns[port_returns < 0].mean()) if (port_returns < 0).sum() > 0 else 0.001
    b_empirical = avg_win / avg_loss if avg_loss > 0 else 1.0
    
    return {
        'p_empirical': p_empirical,
        'b_empirical': b_empirical,
        'total_return': total_return,
        'sharpe': port_returns.mean() / port_returns.std() * np.sqrt(252) if port_returns.std() > 0 else 0,
        'max_dd': ((1 + port_returns).cumprod() - (1 + port_returns).cumprod().cummax()).min()
    }

# ============================================================
# 3. ESECUZIONE PRINCIPALE
# ============================================================

if st.sidebar.button("🚀 Analizza e Costruisci Portafoglio", type="primary", use_container_width=True):
    
    if not tickers_list:
        st.warning("⚠️ Inserisci dei ticker o attiva la Demo Mode")
        st.stop()
    
    if len(tickers_list) > 200:
        st.warning(f"⚠️ Limitato a 200 ticker (hai inserito {len(tickers_list)})")
        tickers_list = tickers_list[:200]
    
    st.header("🔍 Fase 1: Download Dati")
    
    demo_mode = (input_method == "🎲 Demo Mode (dati fittizi)")
    close_prices = fetch_data_batch(tickers_list, lookback_days, demo_mode)
    
    if close_prices is None or close_prices.empty:
        st.error("❌ Impossibile procedere senza dati. Attivazione Modalità Demo automatica.")
        close_prices = generate_demo_data(tickers_list[:min(30, len(tickers_list))], lookback_days)
    
    valid_tickers = close_prices.columns.tolist()
    st.success(f"✅ Dati validi per {len(valid_tickers)} ticker")
    
    if len(valid_tickers) < min_stocks:
        st.warning(f"⚠️ Solo {len(valid_tickers)} ticker validi. Continuo con demo mode...")
        if len(valid_tickers) < 2:
            st.error("Dati insufficienti. Usa esclusivamente la Demo Mode.")
            st.stop()
    
    st.header("📊 Fase 2: Calcolo Fattori Multipli")
    
    with st.spinner("Calcolo Momentum, Quality e Volatilità..."):
        factors = calculate_factors(close_prices)
        factors_with_scores = score_stocks(factors, w_momentum, w_quality, w_volatility)
        
        # Applica Risk-Adjusted Allocation
        factors_with_weights = calculate_risk_adjusted_allocations(factors, risk_propensity)
        
        # Unisci i due dataframe
        for col in ['Base_Score', 'Alloc_Score', 'Raw_Weight']:
            if col in factors_with_weights.columns:
                factors_with_scores[col] = factors_with_weights[col]
    
    # Seleziona top ticker basati su Alloc_Score (Risk-Adjusted)
    n_select = max(min_stocks, min(len(factors_with_scores), int(len(factors_with_scores) * top_pct / 100)))
    selected_df = factors_with_scores.nlargest(n_select, 'Alloc_Score').copy()
    
    # Calcola percentuali di allocazione
    selected_df['Allocation_Pct'] = (selected_df['Raw_Weight'] / selected_df['Raw_Weight'].sum()) * 100
    
    st.subheader(f"🏆 Top {n_select} Titoli Selezionati (Risk-Adjusted)")
    st.dataframe(
        selected_df[['Momentum', 'Quality', 'Volatility', 'Max_Drawdown', 'Base_Score', 'Alloc_Score', 'Allocation_Pct']].style.format({
            'Momentum': '{:.2%}',
            'Quality': '{:.2f}',
            'Volatility': '{:.2%}',
            'Max_Drawdown': '{:.2%}',
            'Base_Score': '{:.2f}',
            'Alloc_Score': '{:.2f}',
            'Allocation_Pct': '{:.2f}%'
        }),
        use_container_width=True
    )
    
    st.header("🎯 Fase 3: Backtest e Calcolo Kelly")
    
    bt = backtest_portfolio(close_prices, selected_df.index.tolist(), lookback_days)
    
    col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)
    with col_bt1:
        st.metric("Rendimento Periodo", f"{bt['total_return']:.2%}")
    with col_bt2:
        st.metric("Sharpe Ratio", f"{bt['sharpe']:.2f}")
    with col_bt3:
        st.metric("Max Drawdown", f"{bt['max_dd']:.2%}")
    with col_bt4:
        st.metric("Win Rate (giorni)", f"{bt['p_empirical']:.2%}")
    
    st.subheader("📐 Parametri Kelly")
    
    use_empirical = st.checkbox("Usa parametri empirici dal backtest", value=False, 
                               help="Se attivo, usa p e b calcolati dal backtest. Se disattivo, usa i valori manuali della sidebar.")
    
    if use_empirical:
        p_used = bt['p_empirical']
        b_used = bt['b_empirical']
        st.info(f"📊 Parametri empirici: p={p_used:.3f}, b={b_used:.2f}")
    else:
        p_used = p_win
        b_used = payoff_ratio
        st.info(f"📊 Parametri manuali: p={p_used:.3f}, b={b_used:.2f}")
    
    kelly_pct = calculate_kelly(p_used, b_used, kelly_fraction)
    
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.metric("Kelly Puro", f"{calculate_kelly(p_used, b_used, 1.0):.2%}")
    with col_k2:
        st.metric(f"Kelly Frazionario ({kelly_fraction:.0%})", f"{kelly_pct:.2%}")
    with col_k3:
        st.metric("Capitale da Investire", f"€{total_capital * kelly_pct:,.2f}")
    
    if kelly_pct <= 0:
        st.warning("⚠️ Kelly ≤ 0! Parametri troppo conservativi. Uso fallback 10%.")
        kelly_pct = 0.10
    
    st.header("💰 Fase 4: Allocazione Portafoglio")
    
    capital_to_invest = total_capital * kelly_pct
    capital_reserve = total_capital - capital_to_invest
    
    # Allocazione basata su pesi Risk-Adjusted (NON uguale per tutti!)
    portfolio = selected_df.copy()
    portfolio['Allocazione (€)'] = (portfolio['Allocation_Pct'] / 100) * capital_to_invest
    portfolio['Peso % su Totale'] = (portfolio['Allocazione (€)'] / total_capital) * 100
    
    # Prezzi attuali
    try:
        portfolio['Prezzo Attuale'] = close_prices[portfolio.index].iloc[-1]
        portfolio['Azioni (stimato)'] = portfolio['Allocazione (€)'] / portfolio['Prezzo Attuale']
    except:
        portfolio['Azioni (stimato)'] = 0
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.metric("💵 Capitale Investito", f"€{capital_to_invest:,.2f}")
    with col_p2:
        st.metric("💰 Riserva/Liquidità", f"€{capital_reserve:,.2f}")
    with col_p3:
        st.metric("📈 N° Azioni", f"{len(portfolio)}")
    
    st.subheader("📋 Portfolio Finale (Con Pesi Differenziati)")
    st.dataframe(
        portfolio[['Momentum', 'Quality', 'Volatility', 'Alloc_Score', 'Allocation_Pct', 'Allocazione (€)', 'Peso % su Totale']].style.format({
            'Momentum': '{:.2%}',
            'Quality': '{:.2f}',
            'Volatility': '{:.2%}',
            'Alloc_Score': '{:.2f}',
            'Allocation_Pct': '{:.2f}%',
            'Allocazione (€)': '{:,.2f} €',
            'Peso % su Totale': '{:.2f}%'
        }),
        use_container_width=True
    )
    
    # Grafico a barre dell'allocazione
    st.subheader("📊 Distribuzione Allocazione (Risk-Adjusted)")
    chart_data = portfolio['Peso % su Totale'].sort_values(ascending=True)
    st.bar_chart(chart_data, use_container_width=True)
    
    # Profilo di rischio basato sulla propensione
    st.markdown("### 💡 Analisi del Profilo di Rischio Scelto")
    if risk_propensity <= 30:
        st.info("🛡️ **Profilo Conservativo:** Il portafoglio privilegia la stabilità (Quality) e protegge dai drawdown. Crescita più lenta ma con minori oscillazioni.")
    elif risk_propensity <= 70:
        st.warning("⚖️ **Profilo Bilanciato:** Il portafoglio cerca un compromesso tra Momentum e Quality per ammortizzare i ribassi.")
    else:
        st.error("🚀 **Profilo Aggressivo:** Il portafoglio insegue il Momentum puro. Massima esposizione a titoli volatili. Potenziale di alto rendimento, ma con rischio di forti oscillazioni.")
    
    st.header("📋 Riepilogo Strategia")
    
    with st.expander("📝 Dettagli completi della strategia", expanded=True):
        st.markdown(f"""
        **Parametri Utilizzati:**
        - **Universo:** {len(valid_tickers)} azioni analizzate
        - **Periodo Lookback:** {lookback_days} giorni
        - **Propensione al Rischio:** {risk_propensity}% ({'Conservativo' if risk_propensity <= 30 else 'Bilanciato' if risk_propensity <= 70 else 'Aggressivo'})
        - **Selezione:** Top {top_pct}% ({len(portfolio)} azioni)
        
        **Kelly Criterion:**
        - Probabilità p: {p_used:.3f}
        - Payoff ratio b: {b_used:.2f}
        - Frazionamento: {kelly_fraction:.0%}
        - **Allocazione ottimale: {kelly_pct:.2%}** del capitale
        
        **Gestione del Rischio:**
        - Capitale investito: €{capital_to_invest:,.2f}
        - Capitale in riserva: €{capital_reserve:,.2f}
        
        **Metodo Allocazione:** Risk-Adjusted con pesi proporzionali allo score di allocazione
        """)
    
    st.divider()
    st.caption("""
    ⚠️ **Disclaimer:** Questo strumento è a scopo educativo. Le performance passate non garantiscono risultati futuri. 
    Il Criterio di Kelly è teorico e richiede stime accurate di p e b. Non costituisce consulenza finanziaria.
    """)

else:
    st.info("⚙️ Configura i parametri nella sidebar e clicca **Analizza e Costruisci Portafoglio**")
    
    st.markdown("""
    ### Come funziona questa app:
    
    1. **📥 Input:** Inserisci ticker o carica CSV
    2. **📊 Fattori:** Calcola Momentum, Quality e Volatilità
    3. **🎯 Risk-Adjusted Allocation:** Pesi dinamici basati sulla tua propensione al rischio
    4. **🎯 Kelly:** Calcola frazione ottimale del capitale da investire
    5. **💰 Allocazione:** Distribuzione pesata per score (non uguale per tutti!)
    
    ### 🎛️ Nuovo Parametro: Propensione al Rischio
    - **1-30% (Conservativo):** Priorità a Quality, penalizza volatilità e drawdown
    - **31-70% (Bilanciato):** Compromesso tra Momentum e Quality
    - **71-100% (Aggressivo):** Priorità al Momentum, ignora volatilità
    """)
