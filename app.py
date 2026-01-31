import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Propfirm Trading Analyzer", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    div[data-testid="stExpander"] { border: none !important; box-shadow: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    file_path = "tradeJournal.csv"
    if not os.path.exists(file_path):
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(file_path)
        df.columns = [c.strip().upper() for c in df.columns]
        
        # Robust date parsing
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df = df.dropna(subset=['DATE'])
        
        # Standardize Results for the UI
        if 'RESULT' in df.columns:
            df['RESULT'] = df['RESULT'].fillna('BREAKEVEN').astype(str).str.strip().str.upper()
            # Catch any lingering variants
            df['RESULT'] = df['RESULT'].replace({'WIN': 'WON', 'LOSS': 'LOST', 'BE': 'BREAKEVEN', 'NAN': 'BREAKEVEN', '': 'BREAKEVEN'})
        
        # Handle PAIR and TIME columns to prevent sorting errors (TypeError: float vs str)
        if 'PAIR' in df.columns:
            df['PAIR'] = df['PAIR'].fillna('UNKNOWN').astype(str).str.strip()
        if 'TIME' in df.columns:
            df['TIME'] = df['TIME'].fillna('N/A').astype(str).str.strip()

        if 'GROWTH %' in df.columns:
            df['GROWTH %'] = pd.to_numeric(df['GROWTH %'], errors='coerce').fillna(0)
        
        # Feature Engineering
        df = df.sort_values('DATE')
        df['YEAR'] = df['DATE'].dt.year
        df['MONTH'] = df['DATE'].dt.strftime('%b')
        df['DAY'] = df['DATE'].dt.day_name()
        
        return df
    except Exception as e:
        st.error(f"Error loading tradeJournal.csv: {e}")
        return pd.DataFrame()

# --- SIMULATION ENGINE ---
def run_simulation(data, start_cap, risk_pct, compound=False):
    balance = start_cap
    history = []
    
    for _, row in data.iterrows():
        # Basis for calculation (current balance if compounding, or initial capital)
        basis = balance if compound else start_cap
        
        # Calculation Logic per User Requirement:
        # The 'GROWTH %' column in the journal represents the result multiplier.
        # Example: 0.02 means 2% of the capital ($2,000 on a $100k account).
        # We scale this based on the 'Risk per Trade' slider.
        # If Risk Slider = 1.0 (1%), then 0.01 in CSV = 1% risk unit (1R).
        
        r_multiple = row['GROWTH %'] / 0.01  # Converts 0.02 to 2R, -0.01 to -1R
        risk_amount = basis * (risk_pct / 100) # Value of 1R in this simulation
        
        pnl = risk_amount * r_multiple
        balance += pnl
        
        history.append({
            'DATE': row['DATE'],
            'PAIR': row['PAIR'] if 'PAIR' in row else 'N/A',
            'TIME': row['TIME'] if 'TIME' in row else 'N/A',
            'RESULT': row['RESULT'],
            'GROWTH': row['GROWTH %'],
            'PNL': pnl,
            'BALANCE': balance,
            'DAY': row['DAY'],
            'MONTH': row['MONTH'],
            'YEAR': row['YEAR'],
            'COMMENTS': row['COMMENTS'] if 'COMMENTS' in row and pd.notna(row['COMMENTS']) else '',
            'TRADE IMAGE': row['TRADE IMAGE'] if 'TRADE IMAGE' in row and pd.notna(row['TRADE IMAGE']) else ''
        })
        
    return pd.DataFrame(history)

# --- APP LAYOUT ---
df = load_data()

if df.empty:
    st.error("tradeJournal.csv not found or empty. Please run aggregator.py first to generate the master file.")
else:
    # Initialize session state for simulation results to persist through interactions
    if 'sim_results' not in st.session_state:
        st.session_state['sim_results'] = None
    if 'last_cap' not in st.session_state:
        st.session_state['last_cap'] = 100000

    st.sidebar.header("🎯 Strategy Filters")
    
    # 1. Date Range Filters
    all_years = sorted(df['YEAR'].unique())
    sel_years = st.sidebar.multiselect("Select Years", all_years, default=all_years)
    
    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    avail_months = [m for m in months_order if m in df['MONTH'].unique()]
    sel_months = st.sidebar.multiselect("Select Months", avail_months, default=avail_months)
    
    # 2. Asset & Time Filters
    df_filtered = df[(df['YEAR'].isin(sel_years)) & (df['MONTH'].isin(sel_months))]
    
    # Sorting now safe because NaNs are handled and types are forced to string
    pairs = sorted(df_filtered['PAIR'].unique())
    sel_pairs = st.sidebar.multiselect("Select Assets", pairs, default=pairs)
    
    times = sorted(df_filtered[df_filtered['PAIR'].isin(sel_pairs)]['TIME'].unique())
    # Fixed circular reference by defining 'times' before using it in the default argument
    sel_times = st.sidebar.multiselect("Select Time Slots", times, default=times)
    
    st.sidebar.divider()
    
    # 3. Capital & Risk Configuration
    st.sidebar.header("💰 Risk Parameters")
    init_cap = st.sidebar.number_input("Starting Capital ($)", value=100000, step=10000)
    risk_pct = st.sidebar.slider("Simulation Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1, help="Scales your journal results. 1.0% risk means 0.02 growth = $2,000 profit.")
    compounding = st.sidebar.toggle("Compound Results", value=False)
    
    # Filtered Data for Sim
    sim_input = df_filtered[
        (df_filtered['PAIR'].isin(sel_pairs)) & 
        (df_filtered['TIME'].isin(sel_times))
    ].copy()
    
    st.title("🛡️ FTMO PTA Strategy Simulator")
    
    if st.button("🚀 Run Simulation", type="primary"):
        if sim_input.empty:
            st.warning("No trades found for this selection. Try adjusting your filters.")
            st.session_state['sim_results'] = None
        else:
            st.session_state['sim_results'] = run_simulation(sim_input, init_cap, risk_pct, compounding)
            st.session_state['last_cap'] = init_cap

    # --- DISPLAY RESULTS ---
    if st.session_state['sim_results'] is not None:
        res = st.session_state['sim_results']
        
        # KPI Dashboard
        m1, m2, m3, m4 = st.columns(4)
        final_bal = res['BALANCE'].iloc[-1]
        won = len(res[res['RESULT'] == 'WON'])
        lost = len(res[res['RESULT'] == 'LOST'])
        be = len(res) - won - lost
        
        m1.metric("Final Balance", f"${final_bal:,.2f}")
        m2.metric("Win Rate", f"{(won/len(res))*100:.1f}%")
        m3.metric("W / L / BE", f"{won} / {lost} / {be}")
        
        # Max Drawdown
        res['PEAK'] = res['BALANCE'].cummax()
        res['DD'] = (res['PEAK'] - res['BALANCE']) / res['PEAK'] * 100
        m4.metric("Max Drawdown", f"{res['DD'].max():.2f}%")
        
        tab1, tab2, tab3 = st.tabs(["📈 Equity Growth", "📊 Monthly Analysis", "📋 Trade Log"])
        
        with tab1:
            st.plotly_chart(px.area(res, x='DATE', y='BALANCE', title="Cumulative Portfolio Growth"), use_container_width=True)
        
        with tab2:
            # Heatmap logic
            pivot = res.groupby(['YEAR', 'MONTH'])['PNL'].sum().reset_index()
            pivot['MONTH'] = pd.Categorical(pivot['MONTH'], categories=months_order, ordered=True)
            heat = pivot.pivot(index='YEAR', columns='MONTH', values='PNL').fillna(0)
            
            # Ensure columns are in Jan-Dec order regardless of data presence
            cols_to_show = [m for m in months_order if m in heat.columns]
            heat = heat.reindex(columns=cols_to_show)
            
            st.plotly_chart(px.imshow(heat, text_auto=True, color_continuous_scale='RdYlGn', title="Monthly PnL Heatmap ($)"), use_container_width=True)
            
            st.divider()
            
            # Monthly Drill-down
            st.subheader("🕵️ Monthly Inspect")
            available_periods = res[['YEAR', 'MONTH', 'DATE']].copy()
            available_periods = available_periods.drop_duplicates(subset=['YEAR', 'MONTH']).sort_values('DATE')
            available_periods['LABEL'] = available_periods['MONTH'] + " " + available_periods['YEAR'].astype(str)
            
            selected_label = st.selectbox("Select Month to Inspect", options=available_periods['LABEL'])
            
            if selected_label:
                parts = selected_label.split(" ")
                sel_m, sel_y = parts[0], int(parts[1])
                month_data = res[(res['MONTH'] == sel_m) & (res['YEAR'] == sel_y)].sort_values('DATE')
                
                if not month_data.empty:
                    st.plotly_chart(px.line(month_data, x='DATE', y='BALANCE', title=f"Equity Path: {selected_label}", markers=True), use_container_width=True)
                    
                    st.write(f"#### Trades for {selected_label}")
                    st.dataframe(
                        month_data[['DATE', 'PAIR', 'TIME', 'RESULT', 'GROWTH', 'PNL', 'BALANCE', 'COMMENTS', 'TRADE IMAGE']],
                        column_config={
                            "TRADE IMAGE": st.column_config.LinkColumn("View Chart"),
                            "GROWTH": st.column_config.NumberColumn("Journal Growth", format="%.2%"),
                            "PNL": st.column_config.NumberColumn("Sim PnL", format="$%.2f"),
                            "BALANCE": st.column_config.NumberColumn("Sim Balance", format="$%.2f")
                        },
                        use_container_width=True, hide_index=True
                    )
            
        with tab3:
            st.write("### Full Simulation Strategy Log")
            st.download_button("📥 Export Simulation Results", res.to_csv(index=False), "sim_results.csv")
            st.dataframe(
                res[['DATE', 'PAIR', 'TIME', 'RESULT', 'GROWTH', 'PNL', 'BALANCE', 'COMMENTS', 'TRADE IMAGE']],
                column_config={
                    "TRADE IMAGE": st.column_config.LinkColumn("View Chart"),
                    "GROWTH": st.column_config.NumberColumn("Journal Growth", format="%.2%"),
                    "PNL": st.column_config.NumberColumn("Sim PnL", format="$%.2f"),
                    "BALANCE": st.column_config.NumberColumn("Sim Balance", format="$%.2f")
                },
                use_container_width=True, hide_index=True
            )
    else:
        st.info("👈 Adjust your strategy filters and click **Run Simulation** to see the performance data.")