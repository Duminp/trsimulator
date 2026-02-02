import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="PTA Analyzer", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    div[data-testid="stExpander"] { border: none !important; box-shadow: none !important; }
    .stDataFrame { background-color: #ffffff; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING & CLEANING ---
@st.cache_data
def load_and_clean_data():
    file_options = ["trade_journal.csv", "tradeJournal.csv"]
    file_path = None
    for opt in file_options:
        if os.path.exists(opt):
            file_path = opt
            break
            
    if not file_path:
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(file_path)
        # Normalize column names to uppercase
        df.columns = [str(c).strip().upper() for c in df.columns]
        
        # Robust Date Parsing for UK format (Day first)
        df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')
        
        # Remove records with missing core data
        df = df.dropna(subset=['DATE', 'PAIR', 'TIME'])
        
        # Filter for the specific range
        df = df[(df['DATE'].dt.year >= 2024) & (df['DATE'].dt.year <= 2026)]
        
        # Clean GROWTH %
        df['GROWTH %'] = pd.to_numeric(df['GROWTH %'], errors='coerce').fillna(0)
        
        # String standardization
        str_cols = ['RESULT', 'PAIR', 'TYPE', 'LIVEORDEMO', 'STATUS', 'COMMENTS']
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].fillna('N/A').astype(str).str.strip().str.upper()
        
        # Standardize results
        if 'RESULT' in df.columns:
            df['RESULT'] = df['RESULT'].replace({
                'WIN': 'WON', 'LOSS': 'LOST', 'BE': 'BREAKEVEN', 
                'NAN': 'BREAKEVEN', '': 'BREAKEVEN', 'N/A': 'BREAKEVEN'
            })
        
        # Status mapping
        if 'LIVEORDEMO' in df.columns:
            df['STATUS'] = df['LIVEORDEMO']
        elif 'STATUS' not in df.columns:
            df['STATUS'] = 'LIVE'
        
        df = df.sort_values('DATE')
        df['DAY_NAME'] = df['DATE'].dt.day_name()
        df['MONTH_NAME'] = df['DATE'].dt.strftime('%b')
        df['YEAR'] = df['DATE'].dt.year
        return df
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return pd.DataFrame()

# --- SIMULATION ENGINE ---
def run_simulation(data, start_capital, risk_per_trade_pct, compounding=False):
    current_balance = start_capital
    history = []
    total_withdrawn = 0
    
    if data.empty:
        return pd.DataFrame()
        
    last_period = None
    month_start_balance = start_capital
    
    for _, row in data.iterrows():
        trade_date = row['DATE']
        current_period = (trade_date.year, trade_date.month)
        
        # --- MONTHLY RESET LOGIC ---
        # Logic: Withdraw profits if above capital, carry over loss if below capital.
        if last_period is not None and current_period != last_period:
            if current_balance > start_capital:
                profit_withdrawn = current_balance - start_capital
                total_withdrawn += profit_withdrawn
                current_balance = start_capital
            
            # Record the balance used to start the new month
            month_start_balance = current_balance
        
        last_period = current_period
        
        # Risk amount calculation
        # Per requirement: compounding off by default. Risk 1% of the 'basis'.
        basis = current_balance if compounding else start_capital
        risk_amount = basis * (risk_per_trade_pct / 100)
        
        # Growth Application
        growth_val = row['GROWTH %']
        r_multiple = growth_val / 0.01 
        trade_pnl = risk_amount * r_multiple
        current_balance += trade_pnl
        
        # DD logic: Assumption that DD is calculated based on defined starting capital
        current_drawdown_pct = 0
        current_drawdown_amt = 0
        if current_balance < start_capital:
            current_drawdown_amt = start_capital - current_balance
            current_drawdown_pct = (current_drawdown_amt / start_capital) * 100
        
        realized_wealth = current_balance + total_withdrawn
        monthly_growth_wealth = (current_balance - month_start_balance) + start_capital
        
        history.append({
            'Date': row['DATE'],
            'Day': row['DAY_NAME'],
            'Month': row['MONTH_NAME'],
            'Year': row['YEAR'],
            'Pair': row['PAIR'],
            'Time': row['TIME'],
            'Type': row['TYPE'],
            'Result': row['RESULT'],
            'Status': row['STATUS'],
            'PnL': trade_pnl,
            'Balance': current_balance,
            'Realized_Wealth': realized_wealth,
            'Monthly_Reset_Wealth': monthly_growth_wealth,
            'Month_Start_Ref': month_start_balance,
            'Drawdown_Pct': current_drawdown_pct,
            'Drawdown_Amt': current_drawdown_amt,
            'Growth_Value': growth_val,
            'Chart': row['TRADE IMAGE'] if 'TRADE IMAGE' in row else '',
            'Comments': row['COMMENTS']
        })
        
    return pd.DataFrame(history)

# --- APP LAYOUT ---
df = load_and_clean_data()

st.sidebar.title("🛠️ Settings")

if not df.empty:
    with st.sidebar.expander("1. Global Filters", expanded=True):
        status_filter = st.radio("Account Mode", ["LIVE", "DEMO", "BOTH"], index=0)
        
        available_years = sorted(df['YEAR'].unique())
        sel_years = st.multiselect("Select Years", options=available_years, default=available_years)
        
        available_assets = sorted(df['PAIR'].unique())
        sel_assets = st.multiselect("Select Assets", options=available_assets, default=available_assets)
        
        filtered_df = df.copy()
        if status_filter != "BOTH":
            filtered_df = filtered_df[filtered_df['STATUS'] == status_filter]
        filtered_df = filtered_df[filtered_df['YEAR'].isin(sel_years) & filtered_df['PAIR'].isin(sel_assets)]
        
        available_times = sorted(filtered_df['TIME'].unique())
        times = st.multiselect("Select Time Slots", options=available_times, default=available_times)

    with st.sidebar.expander("2. Risk & Capital", expanded=True):
        capital = st.number_input("Starting Capital ($)", value=100000, step=10000)
        risk = st.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1)
        compounding = st.toggle("Compound Results", value=False, help="Calculate risk based on current balance instead of initial capital.")

    sim_data = filtered_df[filtered_df['TIME'].isin(times)].copy()
    
    st.title("📈 Surgical Pro Simulator")
    st.info(f"💡 **Ruleset**: Profits withdrawn monthly. Losses carry over. Risk: {risk}% of {'Balance' if compounding else 'Initial Capital'}.")
    
    results = run_simulation(sim_data, capital, risk, compounding)
    
    if not results.empty:
        # --- TOP METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        final_wealth = results['Realized_Wealth'].iloc[-1]
        roi = ((final_wealth - capital) / capital) * 100
        win_rate = (len(results[results['Result'] == 'WON']) / len(results)) * 100
        max_dd_pct = results['Drawdown_Pct'].max()

        m1.metric("Realized Wealth", f"${final_wealth:,.0f}", f"{roi:+.2f}% ROI")
        m2.metric("Win Rate", f"{win_rate:.1f}%")
        m3.metric("Total Trades", len(results))
        m4.metric("Max Drawdown", f"{max_dd_pct:.2f}%", delta_color="inverse")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "🔍 Analytics", "📜 Trade Log", "📉 Drawdown"])

        with tab1:
            st.plotly_chart(px.area(results, x='Date', y='Realized_Wealth', title='Total Realized Wealth Trend', color_discrete_sequence=['#00CC96']), use_container_width=True)
            
            # Monthly Heatmap
            monthly_pnl = results.groupby(['Year', 'Month'])['PnL'].sum().reset_index()
            m_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_pnl['Month'] = pd.Categorical(monthly_pnl['Month'], categories=m_order, ordered=True)
            heatmap_data = monthly_pnl.pivot(index="Year", columns="Month", values="PnL").fillna(0)
            st.plotly_chart(px.imshow(heatmap_data, text_auto='.0f', title="Monthly PnL Breakdown ($)", color_continuous_scale='RdYlGn'), use_container_width=True)

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                day_stats = results.groupby('Day')['PnL'].sum().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']).fillna(0)
                st.plotly_chart(px.bar(day_stats, title="PnL by Day", color=day_stats.values, color_continuous_scale='Viridis'), use_container_width=True)
            with c2:
                pair_stats = results.groupby('Pair')['PnL'].sum()
                st.plotly_chart(px.pie(values=pair_stats.values, names=pair_stats.index, title="Profit by Asset"), use_container_width=True)

        with tab3:
            st.subheader("📜 Historical Trade Log")
            
            display_mode = st.radio("Wealth Tracking Mode", ["Cumulative (Realized)", "Monthly Reset"], horizontal=True)
            wealth_col = "Realized_Wealth" if "Cumulative" in display_mode else "Monthly_Reset_Wealth"
            wealth_lbl = "Total Wealth ($)" if "Cumulative" in display_mode else "Monthly Progress ($)"

            l_col1, l_col2, l_col3 = st.columns([1, 1, 2])
            show_all = l_col1.checkbox("View All History", value=True)
            
            log_display = results.copy()
            if not show_all:
                sel_y = l_col2.selectbox("Year", options=sorted(log_display['Year'].unique(), reverse=True))
                log_display = log_display[log_display['Year'] == sel_y]
                sel_m = l_col3.multiselect("Months", options=m_order, default=log_display['Month'].unique())
                if sel_m: log_display = log_display[log_display['Month'].isin(sel_m)]

            st.dataframe(
                log_display[['Date', 'Pair', 'Time', 'Type', 'Result', 'PnL', 'Balance', wealth_col, 'Growth_Value', 'Chart', 'Comments']],
                column_config={
                    "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                    "PnL": st.column_config.NumberColumn("PnL ($)", format="$%.2f"),
                    "Balance": st.column_config.NumberColumn("Account Bal ($)", format="$%.2f"),
                    wealth_col: st.column_config.NumberColumn(wealth_lbl, format="$%.2f"),
                    "Growth_Value": st.column_config.NumberColumn("Journal Growth", format="%.4f"),
                    "Chart": st.column_config.LinkColumn("View Chart", display_text="Open")
                },
                use_container_width=True, hide_index=True
            )

        with tab4:
            st.subheader("📉 Drawdown & Risk Stats")
            
            # Month selection for drawdown drill-down
            dd_months = results[['Year', 'Month']].drop_duplicates().sort_values(['Year', 'Month'])
            dd_months['Label'] = dd_months['Month'].astype(str) + " " + dd_months['Year'].astype(str)
            selected_dd_period = st.selectbox("Select Period for Daily Drawdown Detail", options=["Entire Filtered History"] + list(dd_months['Label']))

            if selected_dd_period != "Entire Filtered History":
                sel_m_str, sel_y_val = selected_dd_period.split()
                dd_subset = results[(results['Month'] == sel_m_str) & (results['Year'] == int(sel_y_val))].copy()
            else:
                dd_subset = results.copy()

            # Loss Streaks (calculated on global results)
            results['is_loss'] = results['Result'] == 'LOST'
            results['streak_id'] = (results['is_loss'] != results['is_loss'].shift()).cumsum()
            results['streak_count'] = results.groupby('streak_id')['is_loss'].cumsum()
            max_consecutive_losses = results[results['is_loss']]['streak_count'].max() if any(results['is_loss']) else 0
            
            # Daily performance
            daily_stats = dd_subset.groupby('Date')[['PnL', 'Drawdown_Amt', 'Drawdown_Pct']].agg({'PnL': 'sum', 'Drawdown_Amt': 'max', 'Drawdown_Pct': 'max'}).reset_index()
            
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Max Loss Streak", f"{max_consecutive_losses} Trades")
            d2.metric("Max Drawdown (%)", f"{dd_subset['Drawdown_Pct'].max():.2f}%")
            d3.metric("Max Drawdown ($)", f"${dd_subset['Drawdown_Amt'].max():,.2f}")
            
            # Daily performance aggregation
            daily_sum_pnl = results.groupby('Date')['PnL'].sum().reset_index()
            daily_sum_pnl['Peak'] = daily_sum_pnl['PnL'].cumsum().cummax()
            daily_sum_pnl['Daily_Drawdown'] = daily_sum_pnl['Peak'] - daily_sum_pnl['PnL'].cumsum()
            d4.metric("Worst Daily Drop ($)", f"${daily_sum_pnl['Daily_Drawdown'].max():,.2f}")
            
            # Enhanced Chart with Threshold lines
            fig_dd_curve = px.line(dd_subset, x='Date', y='Drawdown_Pct', 
                                   title=f'Drawdown Trend: {selected_dd_period} (%)', 
                                   color_discrete_sequence=['#EF553B'],
                                   hover_data={'Drawdown_Amt': ':$,.2f', 'Drawdown_Pct': ':.2f%'})
            
            # Add horizontal lines for 5% and 10%
            fig_dd_curve.add_hline(y=5.0, line_dash="dash", line_color="orange", annotation_text="5% Threshold", annotation_position="top left")
            fig_dd_curve.add_hline(y=10.0, line_dash="dash", line_color="red", annotation_text="10% Threshold", annotation_position="top left")
            
            fig_dd_curve.update_layout(yaxis_title="Drawdown (%)", yaxis_autorange="reversed", yaxis_range=[max(12, dd_subset['Drawdown_Pct'].max() + 2), 0])
            st.plotly_chart(fig_dd_curve, use_container_width=True)
            
            st.write("### 📅 Daily Performance Breakdown")
            st.dataframe(
                daily_stats.sort_values('Date', ascending=False),
                column_config={
                    "Date": st.column_config.DateColumn("Date"),
                    "PnL": st.column_config.NumberColumn("Net Daily Profit ($)", format="$%.2f"),
                    "Drawdown_Amt": st.column_config.NumberColumn("Peak Drawdown ($)", format="$%.2f"),
                    "Drawdown_Pct": st.column_config.NumberColumn("Peak Drawdown (%)", format="%.2f%%")
                },
                use_container_width=True, hide_index=True
            )

    else:
        st.warning("No trades match your current filter criteria.")
else:
    st.error("trade_journal.csv not found. Please aggregate your data first.")