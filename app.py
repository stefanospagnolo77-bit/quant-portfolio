import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Quant Portfolio Builder Pro", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Quant Portfolio Builder Pro")
st.markdown("*Multi-Factor Scoring + Fractional Kelly Optimization*")

# ============================================================
# 1. CONFIGURAZIONE SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Parametri Strategia")

# Capitale
total_capital = st.sidebar.number_input("Capitale Totale (€)", value=10000, step=1000, min_value=1000)

# Kelly Parameters
st.sidebar.subheader("Parametri Kelly")
col_k1, col_k2 = st.sidebar.columns(2)
with col_k1:
    p_win = st.number_input("Probabilità p", min_value=0.1, max_value=0.95, value=0.55, step=0.01, 
                           help="Probabilità empirica di successo del portafoglio")
with col_k2:
    payoff_ratio = st.number_input("Payoff Ratio b", min_value=0.5, max_value=10.0, value=1.5, step=0.1,
                                  help="Guadagno medio / Perdita media")

kelly_fraction = st.sidebar.slider("Frazionamento Kelly", min_value=0.1, max_value=1.0, value=0.25, step=0.05,
                                  help="0.25 = Quarter Kelly (conservativo), 0.5 = Half Kelly")

# Fattori
st.sidebar.subheader("Pesi Fattori")
w_momentum = st.sidebar.slider("Peso Momentum", 0.0, 1.0, 0.4, 0.05)
w_quality = st.sidebar.slider("Peso Quality (Sharpe)", 0.0, 1.0, 0.35, 0.05)
w_volatility = st.sidebar.slider("Peso Volatilità (invertita)", 0.0, 1.0, 0.25, 0.05)

# Lookback period
lookback_days = st.sidebar.selectbox("Periodo Analisi", [90, 180, 252, 500], index=1,
                                    help="Giorni di dati storici da analizzare")

# Top selection
top_pct = st.sidebar.slider("Selezione Top %", 1, 20, 10, 1,
                           help="Percentuale di titoli da selezionare dal totale")
min_stocks = st.sidebar.number_input("Minimo azioni portfolio", min_value=1, max_value=50, value=5)

# Input metodo
st.sidebar.subheader("📥 Universo di Investimento")
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
    # Demo mode - dati fittizi per testare l'app
    tickers_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "V", "MA",
                   "UNH", "JNJ", "XOM", "PG", "WMT", "HD", "BAC", "ABBV", "PFE", "KO",
                   "PEP", "COST", "AVGO", "TMO", "DIS", "ABT", "ADBE", "CSCO", "CRM", "VZ"]
    st.sidebar.info("🎲 Modalità Demo attiva - dati simulati")

# ============================================================
# 2. FUNZIONI CORE
# ============================================================

def generate_demo_data(tickers, lookback_days):
    """Genera dati di mercato simulati per la modalità demo."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=lookback_days, freq='B')
    
    data = {}
    for ticker in tickers:
        # Simulazione random walk con drift positivo per alcuni titoli
        drift = np.random.normal(0.0005, 0.001)
        volatility = np.random.uniform(0.15, 0.45)
        returns = np.random.normal(drift, volatility/np.sqrt(252), len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        data[ticker] = pd.Series(prices, index=dates)
    
    return pd.DataFrame(data)

def fetch_data_batch(tickers, lookback_days, demo_mode=False):
    """
    Scarica dati in batch per tutti i ticker.
    Ritorna DataFrame con prezzi di chiusura.
    """
    if demo_mode:
        st.info("🎲 Generazione dati demo in corso...")
        return generate_demo_data(tickers, lookback_days)
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_days + 30)
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        progress_text.text("📡 Connessione a Yahoo Finance...")
        
        # Download batch
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=False,
            group_by='ticker'
        )
        
        progress_bar.progress(0.5)
        progress_text.text("📊 Elaborazione dati scaricati...")
        
        # Gestione robusta della struttura dati
        close_prices = pd.DataFrame()
        
        if len(tickers) == 1:
            # Singolo ticker
            ticker = tickers[0]
            if isinstance(data.columns, pd.MultiIndex):
                if ('Close', ticker) in data.columns:
                    close_prices = data[('Close', ticker)].to_frame(ticker)
                elif 'Close' in data.columns.get_level_values(0):
                    close_prices = data['Close'].to_frame(ticker)
            else:
                if 'Close' in data.columns:
                    close_prices = data['Close'].to_frame(ticker)
                elif 'Adj Close' in data.columns:
                    close_prices = data['Adj Close'].to_frame(ticker)
        else:
            # Multipli ticker
            if isinstance(data.columns, pd.MultiIndex):
                # yfinance nuovo formato con MultiIndex
                if 'Close' in data.columns.get_level_values(0):
                    close_prices = data['Close']
                elif 'Adj Close' in data.columns.get_level_values(0):
                    close_prices = data['Adj Close']
            else:
                # Formato vecchio o singolo livello
                if 'Close' in data.columns:
                    close_prices = data['Close']
                    if len(close_prices.shape) == 1:
                        close_prices = close_prices.to_frame()
                elif 'Adj Close' in data.columns:
                    close_prices = data['Adj Close']
                    if len(close_prices.shape) == 1:
                        close_prices = close_prices.to_frame()
        
        # Rimuovi colonne con troppi NaN
        if not close_prices.empty:
            threshold = max(30, len(close_prices) * 0.7)
            close_prices = close_prices.dropna(axis=1, thresh=threshold)
        
        progress_bar.progress(1.0)
        progress_text.empty()
        progress_bar.empty()
        
        if close_prices.empty:
            st.error("❌ Nessun dato valido scaricato da Yahoo Finance")
            st.info("💡 Prova a usare la Demo Mode per testare l'app")
            return None
        
        # Verifica che abbiamo abbastanza dati
        if len(close_prices) < 30:
            st.error(f"❌ Solo {len(close_prices)} giorni di dati disponibili (servono almeno 30)")
            return None
            
        return close_prices
        
    except Exception as e:
        progress_text.empty()
        progress_bar.empty()
        st.error(f"❌ Errore download: {str(e)}")
        st.info("💡 Prova la Demo Mode per testare l'app senza connessione Yahoo")
        return None

def calculate_factors(close_prices):
    """
    Calcola i fattori per ogni ticker:
    - Momentum: rendimento totale nel periodo
    - Quality: Sharpe Ratio (rendimento/volatilità)
    - Volatility: volatilità annualizzata (da minimizzare)
    """
    factors = pd.DataFrame(index=close_prices.columns)
    
    # Returns giornalieri
    returns = close_prices.pct_change().dropna()
    
    # 1. MOMENTUM (rendimento totale)
    factors['Momentum'] = (close_prices.iloc[-1] / close_prices.iloc[0]) - 1
    
    # 2. QUALITY (Sharpe Ratio annualizzato)
    mean_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    factors['Quality'] = mean_return / volatility.replace(0, np.nan)
    
    # 3. VOLATILITY (da invertire - preferiamo bassa volatilità)
    factors['Volatility'] = volatility
    
    # 4. MAX DRAWDOWN (rischio)
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    factors['Max_Drawdown'] = drawdown.min()
    
    return factors.dropna()

def zscore_normalize(series):
    """Z-score normalization robusta."""
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0, index=series.index)
    return (series - mean) / std

def score_stocks(factors, w_mom, w_qual, w_vol):
    """
    Calcola lo score combinato con pesi personalizzati.
    """
    df = factors.copy()
    
    # Z-score per ogni fattore
    df['Z_Momentum'] = zscore_normalize(df['Momentum'])
    df['Z_Quality'] = zscore_normalize(df['Quality'])
    df['Z_Volatility'] = zscore_normalize(df['Volatility'])
    
    # Score combinato (volatilità ha peso negativo - preferiamo bassa vol)
    df['Score'] = (w_mom * df['Z_Momentum'] + 
                   w_qual * df['Z_Quality'] - 
                   w_vol * df['Z_Volatility'])
    
    return df.sort_values('Score', ascending=False)

def calculate_kelly(p, b, fraction):
    """
    Calcola la frazione ottimale di Kelly.
    f* = (bp - q) / b dove q = 1-p
    """
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0, kelly * fraction)

def backtest_portfolio(close_prices, selected_tickers, lookback_days):
    """
    Esegue un semplice backtest per stimare p e b empirici.
    """
    returns = close_prices[selected_tickers].pct_change().dropna()
    
    # Portfolio returns (equal weight)
    port_returns = returns.mean(axis=1)
    
    # Metriche
    total_return = (1 + port_returns).prod() - 1
    positive_days = (port_returns > 0).sum()
    total_days = len(port_returns)
    
    # Stima p (probabilità giorno positivo)
    p_empirical = positive_days / total_days if total_days > 0 else 0.5
    
    # Stima b (payoff ratio)
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
    
    # Limite ticker per performance
    if len(tickers_list) > 500:
        st.warning(f"⚠️ Limitato a 500 ticker (hai inserito {len(tickers_list)})")
        tickers_list = tickers_list[:500]
    
    st.header("🔍 Fase 1: Download Dati")
    
    demo_mode = (input_method == "🎲 Demo Mode (dati fittizi)")
    
    # Download dati
    close_prices = fetch_data_batch(tickers_list, lookback_days, demo_mode)
    
    if close_prices is None:
        st.error("❌ Impossibile procedere senza dati. Prova la Demo Mode.")
        st.stop()
    
    # Info dati
    valid_tickers = close_prices.columns.tolist()
    st.success(f"✅ Dati validi per {len(valid_tickers)} ticker su {len(tickers_list)} richiesti")
    
    if len(valid_tickers) < min_stocks:
        st.error(f"❌ Troppo pochi dati validi ({len(valid_tickers)}). Servono almeno {min_stocks}.")
        st.stop()
    
    # ============================================================
    # FASE 2: CALCOLO FATTORI
    # ============================================================
    st.header("📊 Fase 2: Calcolo Fattori Multipli")
    
    with st.spinner("Calcolo Momentum, Quality e Volatilità..."):
        factors = calculate_factors(close_prices)
        scored = score_stocks(factors, w_momentum, w_quality, w_volatility)
    
    # Mostra distribuzione fattori
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.metric("Media Momentum", f"{factors['Momentum'].mean():.2%}")
    with col_f2:
        st.metric("Media Sharpe", f"{factors['Quality'].mean():.2f}")
    with col_f3:
        st.metric("Media Volatilità", f"{factors['Volatility'].mean():.2%}")
    
    # Selezione top
    n_select = max(min_stocks, int(len(scored) * top_pct / 100))
    selected = scored.head(n_select)
    
    st.subheader(f"🏆 Top {n_select} Titoli Selezionati")
    st.dataframe(
        selected[['Momentum', 'Quality', 'Volatility', 'Max_Drawdown', 'Score']].style.format({
            'Momentum': '{:.2%}',
            'Quality': '{:.2f}',
            'Volatility': '{:.2%}',
            'Max_Drawdown': '{:.2%}',
            'Score': '{:.2f}'
        }).background_gradient(subset=['Score'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    # ============================================================
    # FASE 3: BACKTEST E KELLY
    # ============================================================
    st.header("🎯 Fase 3: Backtest e Calcolo Kelly")
    
    # Backtest
    bt = backtest_portfolio(close_prices, selected.index.tolist(), lookback_days)
    
    col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)
    with col_bt1:
        st.metric("Rendimento Periodo", f"{bt['total_return']:.2%}")
    with col_bt2:
        st.metric("Sharpe Ratio", f"{bt['sharpe']:.2f}")
    with col_bt3:
        st.metric("Max Drawdown", f"{bt['max_dd']:.2%}")
    with col_bt4:
        st.metric("Win Rate (giorni)", f"{bt['p_empirical']:.2%}")
    
    # Calcolo Kelly
    st.subheader("📐 Parametri Kelly")
    
    # Usa valori empirici o input utente
    use_empirical = st.checkbox("Usa parametri empirici dal backtest", value=False)
    
    if use_empirical:
        p_used = bt['p_empirical']
        b_used = bt['b_empirical']
        st.info(f"Parametri empirici: p={p_used:.3f}, b={b_used:.2f}")
    else:
        p_used = p_win
        b_used = payoff_ratio
    
    kelly_pct = calculate_kelly(p_used, b_used, kelly_fraction)
    
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.metric("Kelly Puro", f"{calculate_kelly(p_used, b_used, 1.0):.2%}")
    with col_k2:
        st.metric(f"Kelly Frazionario ({kelly_fraction:.0%})", f"{kelly_pct:.2%}", 
                 delta=f"Riduzione rischio {((1-kelly_fraction)*100):.0f}%")
    with col_k3:
        st.metric("Capitale da Investire", f"€{total_capital * kelly_pct:,.2f}")
    
    if kelly_pct <= 0:
        st.error("⚠️ Kelly ≤ 0! Parametri troppo conservativi. Uso fallback 10%.")
        kelly_pct = 0.10
    
    # ============================================================
    # FASE 4: ALLOCAZIONE
    # ============================================================
    st.header("💰 Fase 4: Allocazione Portafoglio")
    
    capital_to_invest = total_capital * kelly_pct
    capital_reserve = total_capital - capital_to_invest
    n_stocks = len(selected)
    allocation_per_stock = capital_to_invest / n_stocks
    
    # Creazione portfolio finale
    portfolio = selected.copy()
    portfolio['Allocazione (€)'] = allocation_per_stock
    portfolio['Peso %'] = (allocation_per_stock / total_capital) * 100
    portfolio['Azioni (stimato)'] = allocation_per_stock / close_prices[portfolio.index].iloc[-1]
    
    # Visualizzazione
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.metric("💵 Capitale Investito", f"€{capital_to_invest:,.2f}")
    with col_p2:
        st.metric("🏦 Riserva/Liquidità", f"€{capital_reserve:,.2f}")
    with col_p3:
        st.metric("📈 N° Azioni", f"{n_stocks}")
    
    st.subheader("📋 Portfolio Finale")
    st.dataframe(
        portfolio[['Momentum', 'Quality', 'Volatility', 'Score', 'Allocazione (€)', 'Peso %']].style.format({
            'Momentum': '{:.2%}',
            'Quality': '{:.2f}',
            'Volatility': '{:.2%}',
            'Score': '{:.2f}',
            'Allocazione (€)': '{:,.2f} €',
            'Peso %': '{:.2f}%'
        }).background_gradient(subset=['Score', 'Quality'], cmap='RdYlGn')
        .background_gradient(subset=['Volatility'], cmap='RdYlGn_r'),
        use_container_width=True
    )
    
    # Grafico allocazione
    st.subheader("📊 Distribuzione Allocazione")
    
    chart_data = portfolio['Peso %'].sort_values(ascending=True)
    st.bar_chart(chart_data, use_container_width=True)
    
    # ============================================================
    # RIEPILOGO STRATEGIA
    # ============================================================
    st.header("📋 Riepilogo Strategia")
    
    with st.expander("📝 Dettagli completi della strategia", expanded=True):
        st.markdown(f"""
        **Parametri Utilizzati:**
        - **Universo:** {len(valid_tickers)} azioni analizzate
        - **Periodo Lookback:** {lookback_days} giorni
        - **Fattori:** Momentum ({w_momentum:.0%}), Quality ({w_quality:.0%}), Low-Vol ({w_volatility:.0%})
        - **Selezione:** Top {top_pct}% ({n_stocks} azioni)
        
        **Kelly Criterion:**
        - Probabilità p: {p_used:.3f}
        - Payoff ratio b: {b_used:.2f}
        - Frazionamento: {kelly_fraction:.0%}
        - **Allocazione ottimale: {kelly_pct:.2%}** del capitale
        
        **Gestione del Rischio:**
        - Capitale investito: €{capital_to_invest:,.2f}
        - Capitale in riserva: €{capital_reserve:,.2f}
        - Cuscinetto anti-drawdown: {((1-kelly_pct)*100):.1f}%
        
        **Rebalancing:** Ogni 3-6 mesi ricalcolare fattori e dimensioni posizioni
        """)
    
    # Disclaimer
    st.divider()
    st.caption("""
    ⚠️ **Disclaimer:** Questo strumento è a scopo educativo. Le performance passate non garantiscono risultati futuri. 
    Il Criterio di Kelly è teorico e richiede stime accurate di p e b. Non costituisce consulenza finanziaria.
    """)

else:
    # Stato iniziale
    st.info("👈 Configura i parametri nella sidebar e clicca **Analizza e Costruisci Portafoglio**")
    
    st.markdown("""
    ### Come funziona questa app:
    
    1. **📥 Input:** Inserisci fino a 500 ticker o usa la Demo Mode
    2. **📊 Fattori:** Calcola Momentum, Quality (Sharpe) e Low-Volatility
    3. **🏆 Scoring:** Z-score normalization e ranking combinato
    4. **🎯 Kelly:** Stima p e b, calcola frazione ottimale con conservativismo
    5. **💰 Allocazione:** Distribuisce il capitale sul top decile
    
    ### Suggerimenti ticker per testare:
    - **USA:** AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, V, MA
    - **EU:** ENI.MI, ISP.MI, SAP.DE, ASML.AS, TOTF.PA, SAN.PA
    - **ETF:** SPY, QQQ, IWM, VTI, VOO
    """)
