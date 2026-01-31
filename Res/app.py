import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="FTMO Pro Simulator", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
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
        df = df.dropna(subset=['DATE', 'PAIR', 'TIME'])
        
        # Standardize Results
        df['RESULT'] = df['RESULT'].fillna('BREAKEVEN').astype(str).str.strip().str.upper()
        # If result is empty but growth is 0, it's a breakeven
        df.loc[(df['RESULT'] == 'NAN') & (df['GROWTH %'] == 0), 'RESULT'] = 'BREAKEVEN'
        
        df['GROWTH %'] = pd.to_numeric(df['GROWTH %'], errors='coerce').fillna(0)
        
        # Sort and Features
        df = df.sort_values('DATE')
        df['YEAR'] = df['DATE'].dt.year
        df['MONTH'] = df['DATE'].dt.strftime('%b')
        df['DAY'] = df['DATE'].dt.day_name()
        return df
    except Exception as e:
        st.error(f"Load Error: {e}")
        return pd.DataFrame()

# --- SIM ENGINE ---
def run_simulation(data, start_cap, risk_pct, compound=True):
    balance = start_cap
    history = []
    for _, row in data.iterrows():
        risk_val = balance * (risk_pct/100) if compound else start_cap * (risk_pct/100)
        # Using 0.01 (1%) as the R-multiplier base
        pnl = risk_val * (row['GROWTH %'] / 0.01)
        balance += pnl
        history.append({
            'DATE': row['DATE'],
            'PAIR': row['PAIR'],
            'TIME': row['TIME'],
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

# --- APP UI ---
df = load_data()

if df.empty:
    st.error("tradeJournal.csv not found or empty. Run aggregator.py first.")
else:
    # Initialize session state for simulation results
    if 'sim_results' not in st.session_state:
        st.session_state['sim_results'] = None
    if 'sim_init_cap' not in st.session_state:
        st.session_state['sim_init_cap'] = 0

    st.sidebar.header("🕹️ Controls")
    
    years = sorted(df['YEAR'].unique())
    sel_years = st.sidebar.multiselect("Years", years, default=years)
    
    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    avail_months = [m for m in months_order if m in df['MONTH'].unique()]
    sel_months = st.sidebar.multiselect("Months", avail_months, default=avail_months)
    
    df_filtered = df[(df['YEAR'].isin(sel_years)) & (df['MONTH'].isin(sel_months))]
    
    pairs = sorted(df_filtered['PAIR'].unique())
    sel_pairs = st.sidebar.multiselect("Assets", pairs, default=pairs)
    
    times = sorted(df_filtered[df_filtered['PAIR'].isin(sel_pairs)]['TIME'].unique())
    sel_times = st.sidebar.multiselect("Time Slots", times, default=times)
    
    cap = st.sidebar.number_input("Capital ($)", value=100000)
    risk = st.sidebar.slider("Risk (%)", 0.1, 5.0, 1.0)
    comp = st.sidebar.toggle("Compounding", value=True)
    
    sim_input = df_filtered[(df_filtered['PAIR'].isin(sel_pairs)) & (df_filtered['TIME'].isin(sel_times))]
    
    st.title("📈 FTMO Strategy Simulator")
    
    if st.button("Run Simulation", type="primary"):
        if sim_input.empty:
            st.warning("No data for this selection.")
            st.session_state['sim_results'] = None
        else:
            st.session_state['sim_results'] = run_simulation(sim_input, cap, risk, comp)
            st.session_state['sim_init_cap'] = cap

    # Display results if they exist in session state
    if st.session_state['sim_results'] is not None:
        res = st.session_state['sim_results']
        init_cap = st.session_state['sim_init_cap']
        
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        final_bal = res['BALANCE'].iloc[-1]
        win_count = len(res[res['RESULT'] == 'WON'])
        loss_count = len(res[res['RESULT'] == 'LOST'])
        be_count = len(res) - win_count - loss_count
        
        m1.metric("Final Balance", f"${final_bal:,.2f}")
        m2.metric("Win Rate", f"{(win_count/len(res))*100:.1f}%")
        m3.metric("W / L / BE", f"{win_count} / {loss_count} / {be_count}")
        
        # Drawdown
        res['PEAK'] = res['BALANCE'].cummax()
        res['DD'] = (res['PEAK'] - res['BALANCE']) / res['PEAK'] * 100
        m4.metric("Max Drawdown", f"{res['DD'].max():.2f}%")
        
        t1, t2, t3 = st.tabs(["Equity Chart", "Analytics", "Trade History"])
        
        with t1:
            st.plotly_chart(px.area(res, x='DATE', y='BALANCE', title="Cumulative Portfolio Growth"), use_container_width=True)
        
        with t2:
            # Heatmap Logic
            pivot = res.groupby(['YEAR', 'MONTH'])['PNL'].sum().reset_index()
            pivot['MONTH'] = pd.Categorical(pivot['MONTH'], categories=months_order, ordered=True)
            heat = pivot.pivot(index='YEAR', columns='MONTH', values='PNL').fillna(0)
            
            # Reindex columns to ensure chronological order regardless of alphabetical sorting
            cols_to_show = [m for m in months_order if m in heat.columns]
            heat = heat.reindex(columns=cols_to_show)
            
            st.plotly_chart(px.imshow(heat, text_auto=True, color_continuous_scale='RdYlGn', title="Monthly PnL Heatmap"), use_container_width=True)
            
            st.divider()
            
            # Monthly Drill-down Selection
            st.subheader("🔍 Monthly Drill-down")
            # Get unique month-year strings for the selector, sorted chronologically
            available_monthly_periods = res[['YEAR', 'MONTH', 'DATE']].copy()
            available_monthly_periods = available_monthly_periods.drop_duplicates(subset=['YEAR', 'MONTH']).sort_values('DATE')
            available_monthly_periods['LABEL'] = available_monthly_periods['MONTH'] + " " + available_monthly_periods['YEAR'].astype(str)
            
            selected_label = st.selectbox("Select Month to Inspect", options=available_monthly_periods['LABEL'])
            
            if selected_label:
                parts = selected_label.split(" ")
                sel_m = parts[0]
                sel_y = int(parts[1])
                
                # Filter data for this specific month
                month_data = res[(res['MONTH'] == sel_m) & (res['YEAR'] == sel_y)].sort_values('DATE')
                
                if not month_data.empty:
                    # Monthly Metrics
                    mc1, mc2, mc3 = st.columns(3)
                    month_profit = month_data['PNL'].sum()
                    mc1.metric(f"Profit for {selected_label}", f"${month_profit:,.2f}")
                    mc2.metric("Trades", len(month_data))
                    mc3.metric("Win Rate", f"{(len(month_data[month_data['RESULT']=='WON'])/len(month_data))*100:.1f}%")

                    # Monthly Equity Curve
                    fig_month = px.line(month_data, x='DATE', y='BALANCE', title=f"Equity Curve: {selected_label}", markers=True)
                    fig_month.update_layout(yaxis_title="Account Balance ($)", xaxis_title="Trade Date")
                    st.plotly_chart(fig_month, use_container_width=True)
                    
                    # Monthly Trade Table
                    st.write(f"#### Trades for {selected_label}")
                    st.dataframe(
                        month_data[['DATE', 'PAIR', 'TIME', 'RESULT', 'GROWTH', 'PNL', 'BALANCE', 'COMMENTS', 'TRADE IMAGE']],
                        column_config={
                            "TRADE IMAGE": st.column_config.LinkColumn("Trade Image"),
                            "GROWTH": st.column_config.NumberColumn("Growth", format="%.2%"),
                            "PNL": st.column_config.NumberColumn("PnL", format="$%.2f"),
                            "BALANCE": st.column_config.NumberColumn("Balance", format="$%.2f")
                        },
                        use_container_width=True,
                        hide_index=True
                    )
            
        with t3:
            st.write("### Full Simulation Trade Log")
            st.download_button("Export Results", res.to_csv(index=False), "simulation.csv")
            st.dataframe(
                res[['DATE', 'PAIR', 'TIME', 'RESULT', 'GROWTH', 'PNL', 'BALANCE', 'COMMENTS', 'TRADE IMAGE']],
                column_config={
                    "TRADE IMAGE": st.column_config.LinkColumn("Trade Image"),
                    "GROWTH": st.column_config.NumberColumn("Growth", format="%.2%"),
                    "PNL": st.column_config.NumberColumn("PnL", format="$%.2f"),
                    "BALANCE": st.column_config.NumberColumn("Balance", format="$%.2f")
                },
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("Adjust the sidebar and click Run to see results.")