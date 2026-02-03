import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="PTA analyzer", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    div[data-testid="stExpander"] { border: none !important; box-shadow: none !important; }
    .stDataFrame { background-color: #ffffff; border-radius: 10px; }
    .plotly-graph-div { width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTS ---
MONTH_ORDER = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

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
        df.columns = [str(c).strip().upper() for c in df.columns]
        
        df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['DATE', 'PAIR', 'TIME'])
        df = df[(df['DATE'].dt.year >= 2024) & (df['DATE'].dt.year <= 2026)]
        
        df['GROWTH %'] = pd.to_numeric(df['GROWTH %'], errors='coerce').fillna(0)
        
        str_cols = ['RESULT', 'PAIR', 'TYPE', 'LIVEORDEMO', 'STATUS', 'COMMENTS']
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].fillna('N/A').astype(str).str.strip().str.upper()
        
        if 'RESULT' in df.columns:
            df['RESULT'] = df['RESULT'].replace({
                'WIN': 'WON', 'LOSS': 'LOST', 'BE': 'BREAKEVEN', 
                'NAN': 'BREAKEVEN', '': 'BREAKEVEN', 'N/A': 'BREAKEVEN'
            })
        
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
        
        if last_period is not None and current_period != last_period:
            if current_balance > start_capital:
                profit = current_balance - start_capital
                total_withdrawn += profit
                current_balance = start_capital
            month_start_balance = current_balance
        
        last_period = current_period
        
        basis = current_balance if compounding else start_capital
        risk_amount = basis * (risk_per_trade_pct / 100)
        
        growth_val = row['GROWTH %']
        r_multiple = growth_val / 0.01 
        trade_pnl = risk_amount * r_multiple
        current_balance += trade_pnl
        
        current_drawdown_amt = max(0, start_capital - current_balance)
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
            'Drawdown_Pct': current_drawdown_pct,
            'Drawdown_Amt': current_drawdown_amt,
            'Growth_Value': growth_val,
            'Chart': row['TRADE IMAGE'] if 'TRADE IMAGE' in row else '',
            'Comments': row['COMMENTS']
        })
        
    return pd.DataFrame(history)

# --- APP LAYOUT ---
df = load_and_clean_data()

# Initialize Session State for Simulator
if 'sim_history' not in st.session_state:
    st.session_state.sim_history = []
if 'sim_current_capital' not in st.session_state:
    st.session_state.sim_current_capital = 100000.0
if 'last_cycle_details' not in st.session_state:
    st.session_state.last_cycle_details = None

st.sidebar.title("🛠️ Settings")

# Simulator Focus Toggle to "freeze" settings
simulator_mode = st.sidebar.toggle("🎯 Simulator Mode", value=False, help="Enable this to freeze global filters and use the Cycle Simulator.")

if not df.empty:
    with st.sidebar.expander("1. Global Filters", expanded=not simulator_mode):
        status_filter = st.radio("Account Mode", ["LIVE", "DEMO", "BOTH"], index=0, disabled=simulator_mode)
        available_years = sorted(df['YEAR'].unique())
        sel_years = st.multiselect("Select Years", options=available_years, default=available_years, disabled=simulator_mode)
        available_assets = sorted(df['PAIR'].unique())
        sel_assets = st.multiselect("Select Assets", options=available_assets, default=available_assets, disabled=simulator_mode)
        
        filtered_df = df.copy()
        if status_filter != "BOTH":
            filtered_df = filtered_df[filtered_df['STATUS'] == status_filter]
        filtered_df = filtered_df[filtered_df['YEAR'].isin(sel_years) & filtered_df['PAIR'].isin(sel_assets)]
        
        available_times = sorted(filtered_df['TIME'].unique())
        times = st.multiselect("Select Time Slots", options=available_times, default=available_times, disabled=simulator_mode)

    with st.sidebar.expander("2. Risk & Capital", expanded=True):
        capital = st.number_input("Starting Capital ($)", value=100000, step=10000, disabled=simulator_mode)
        risk = st.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1, disabled=simulator_mode)
        compounding = st.toggle("Compound Results", value=False, disabled=simulator_mode)

    sim_data = filtered_df[filtered_df['TIME'].isin(times)].copy()
    
    st.title("📈 PTA analyzer")
    
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

        # --- ASSET BREAKDOWN METRICS (Condensed One-Liner) ---
        pair_stats = results.groupby('Pair').agg(
            Earning=('Growth_Value', lambda x: x.sum() * 100),
            TradeCount=('Pair', 'count'),
            WinRate=('Result', lambda x: (x == 'WON').sum() / len(x) * 100)
        ).reset_index()

        stats_html = []
        for _, row in pair_stats.iterrows():
            stats_html.append(
                f"<span style='color:#666;'>{row['Pair']}:</span> "
                f"<b style='color:#00CC96;'>{row['Earning']:+.2f}%</b> "
                f"<span style='font-size:0.8rem; opacity:0.8;'>({row['TradeCount']} trades | {row['WinRate']:.1f}% WR)</span>"
            )
        
        st.markdown(
            f"<div style='font-size: 0.9rem; padding: 12px; background: white; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #00CC96; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>"
            f"{' &nbsp;&nbsp; | &nbsp;&nbsp; '.join(stats_html)}</div>", 
            unsafe_allow_html=True
        )

        # Tab Order -> 📊 Performance | 🔍 PnL Analysis | 📉 Draw Down Analysis | 📜 Trade Log | Simulator
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Performance", "🔍 PnL Analysis", "📉 Draw Down Analysis", "📜 Trade Log", "🚀 Simulator"])

        with tab1:
            st.plotly_chart(px.area(results, x='Date', y='Realized_Wealth', title='Total Realized Wealth Trend', color_discrete_sequence=['#00CC96']), use_container_width=True)
            
            # --- CHRONOLOGICAL MONTHLY HEATMAP ---
            monthly_group = results.groupby(['Year', 'Month'])
            monthly_pnl = monthly_group['PnL'].sum().reset_index()
            monthly_pnl['Month'] = pd.Categorical(monthly_pnl['Month'], categories=MONTH_ORDER, ordered=True)
            pnl_pivot = monthly_pnl.pivot(index="Year", columns="Month", values="PnL").reindex(columns=MONTH_ORDER).fillna(0)
            
            def get_month_details(group):
                tpnl = group['PnL'].sum()
                tgrowth = group['Growth_Value'].sum() * 100
                counts = group.groupby('Pair').size()
                count_str = "<br>".join([f"{p}: {c}" for p, c in counts.items()])
                return f"${tpnl:,.0f}<br>{tgrowth:+.1f}%<br><span style='font-size: 0.7rem;'>{count_str}</span>"

            text_details_list = []
            for name, group in monthly_group:
                text_details_list.append({
                    'Year': name[0],
                    'Month': name[1],
                    'Text': get_month_details(group)
                })
            
            text_details = pd.DataFrame(text_details_list)
            text_details['Month'] = pd.Categorical(text_details['Month'], categories=MONTH_ORDER, ordered=True)
            text_pivot = text_details.pivot(index="Year", columns="Month", values="Text").reindex(columns=MONTH_ORDER).fillna("")

            fig_heat = px.imshow(
                pnl_pivot,
                labels=dict(x="Month", y="Year", color="PnL ($)"),
                x=MONTH_ORDER,
                y=pnl_pivot.index,
                color_continuous_scale='RdYlGn',
                title="Monthly Performance Breakdown (% and Trades)",
                aspect="auto"
            )
            
            for i, year in enumerate(pnl_pivot.index):
                for j, month in enumerate(MONTH_ORDER):
                    val = text_pivot.iloc[i, j]
                    if val:
                        fig_heat.add_annotation(x=month, y=year, text=val, showarrow=False, font=dict(size=10, color="black"))
            
            st.plotly_chart(fig_heat, use_container_width=True)

        with tab2:
            st.subheader("🧪 Detailed Asset Statistical Table")
            summary_table = results.groupby('Pair').agg({
                'Growth_Value': lambda x: f"{x.sum()*100:+.2f}%",
                'PnL': lambda x: f"${x.sum():,.2f}",
                'Result': [('Total Trades', 'count'), 
                           ('Wins', lambda x: (x == 'WON').sum()),
                           ('Losses', lambda x: (x == 'LOST').sum())]
            })
            summary_table.columns = ['Total % Earning', 'Realized PnL ($)', 'Total Trades', 'Wins', 'Losses']
            summary_table['Win Rate'] = ((summary_table['Wins'] / summary_table['Total Trades']) * 100).round(1).astype(str) + '%'
            st.table(summary_table[['Total Trades', 'Total % Earning', 'Realized PnL ($)', 'Win Rate', 'Wins', 'Losses']])

            st.divider()

            # --- ROW 1: MONTH-WISE EARNINGS (FULL WIDTH) ---
            month_pair_stats = results.groupby(['Month', 'Pair'])['PnL'].sum().reset_index()
            month_pair_stats['Month'] = pd.Categorical(month_pair_stats['Month'], categories=MONTH_ORDER, ordered=True)
            month_pair_stats = month_pair_stats.sort_values('Month')
            
            fig_month = px.bar(month_pair_stats, 
                             x='Month', y='PnL', color='Pair', barmode='group',
                             title="Month-wise Earnings Breakdown (Aggregated)", 
                             text_auto='.2s',
                             color_discrete_sequence=px.colors.qualitative.Bold)
            fig_month.update_traces(textposition='outside')
            st.plotly_chart(fig_month, use_container_width=True)

            st.divider()

            # --- ROW 2: DAY ANALYSIS & PIE CHART (SHARED ROW) ---
            col_bar, col_pie = st.columns(2)

            with col_bar:
                day_pair_stats = results.groupby(['Day', 'Pair'])['PnL'].sum().reset_index()
                day_pair_stats['Day'] = pd.Categorical(day_pair_stats['Day'], categories=DAY_ORDER, ordered=True)
                day_pair_stats = day_pair_stats.sort_values('Day')
                
                fig_day = px.bar(day_pair_stats, x='Day', y='PnL', color='Pair', barmode='group',
                               title="PnL by Day (Asset Comparison)", 
                               text_auto='.2s',
                               color_discrete_sequence=px.colors.qualitative.Bold)
                fig_day.update_traces(textposition='outside')
                st.plotly_chart(fig_day, use_container_width=True)
                
            with col_pie:
                fig_pie = px.pie(results.groupby('Pair')['PnL'].sum().reset_index(), 
                               values='PnL', names='Pair', title="Aggregate Profit Distribution by Asset")
                st.plotly_chart(fig_pie, use_container_width=True)

        with tab4:
            st.subheader("📜 Historical Trade Log")
            display_mode = st.radio("Wealth Tracking Mode", ["Cumulative (Realized)", "Monthly Reset"], horizontal=True)
            wealth_col = "Realized_Wealth" if "Cumulative" in display_mode else "Monthly_Reset_Wealth"
            wealth_lbl = "Total Wealth ($)" if "Cumulative" in display_mode else "Monthly Progress ($)"

            l_col1, l_col2, l_col3 = st.columns([1, 1, 2])
            show_all = l_col1.checkbox("View All History", value=True)
            
            log_display = results.copy()
            if not show_all:
                sel_y = l_col2.selectbox("Select Year", options=sorted(log_display['Year'].unique(), reverse=True))
                log_display = log_display[log_display['Year'] == sel_y]
                sel_m = l_col3.multiselect("Select Months", options=MONTH_ORDER, default=log_display['Month'].unique())
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

        with tab3:
            st.subheader("📉 Draw Down Analysis")
            
            # --- MONTHLY PEAK DRAWDOWN HEATMAP ---
            monthly_dd = results.groupby(['Year', 'Month'])['Drawdown_Pct'].max().reset_index()
            monthly_dd['Month'] = pd.Categorical(monthly_dd['Month'], categories=MONTH_ORDER, ordered=True)
            dd_heat_pivot = monthly_dd.pivot(index="Year", columns="Month", values="Drawdown_Pct").reindex(columns=MONTH_ORDER).fillna(0)
            
            fig_dd_heat = px.imshow(
                dd_heat_pivot,
                labels=dict(x="Month", y="Year", color="Max DD %"),
                x=MONTH_ORDER,
                y=dd_heat_pivot.index,
                color_continuous_scale='YlOrRd',
                text_auto='.1f',
                title="Peak Monthly Drawdown Heatmap (%)",
                aspect="auto"
            )
            st.plotly_chart(fig_dd_heat, use_container_width=True)
            
            st.divider()
            
            dd_months = results[['Year', 'Month']].drop_duplicates().sort_values(['Year', 'Month'])
            dd_months['Label'] = dd_months['Month'].astype(str) + " " + dd_months['Year'].astype(str)
            selected_dd_period = st.selectbox("Select Period for Daily Drawdown Detail", options=["Entire Filtered History"] + list(dd_months['Label']))

            if selected_dd_period != "Entire Filtered History":
                sel_m_str, sel_y_val = selected_dd_period.split()
                dd_subset = results[(results['Month'] == sel_m_str) & (results['Year'] == int(sel_y_val))].copy()
            else:
                dd_subset = results.copy()

            fig_dd_curve = px.line(dd_subset, x='Date', y='Drawdown_Pct', title=f'Drawdown Trend: {selected_dd_period} (%)', color_discrete_sequence=['#EF553B'])
            fig_dd_curve.add_hline(y=5.0, line_dash="dash", line_color="orange", annotation_text="5% Threshold")
            fig_dd_curve.add_hline(y=10.0, line_dash="dash", line_color="red", annotation_text="10% Threshold")
            fig_dd_curve.update_layout(yaxis_title="Drawdown (%)", yaxis_autorange="reversed", yaxis_range=[max(12, dd_subset['Drawdown_Pct'].max() + 2), 0])
            st.plotly_chart(fig_dd_curve, use_container_width=True)
            
            daily_stats = dd_subset.groupby('Date')[['PnL', 'Drawdown_Amt', 'Drawdown_Pct']].agg({'PnL': 'sum', 'Drawdown_Amt': 'max', 'Drawdown_Pct': 'max'}).reset_index()
            st.dataframe(daily_stats.sort_values('Date', ascending=False), use_container_width=True, hide_index=True)

        with tab5:
            st.header("🚀 Advanced Cycle Simulator (With 2025 Data)")
            if not simulator_mode:
                st.warning("⚠️ Enable 'Simulator Mode' in the sidebar to use this feature.")
            else:
                # 2025 Data Only
                sim_data_2025 = df[df['YEAR'] == 2025].copy()
                
                sc1, sc2 = st.columns([1, 2])
                
                with sc1:
                    st.write("### Cycle Setup")
                    s_cap = st.number_input("Simulator Start Capital ($)", value=st.session_state.sim_current_capital, step=1000.0)
                    s_risk = st.slider("Risk per Trade (%) ", 0.1, 5.0, 1.0, 0.1)
                    s_comp = st.toggle("Compound within Cycle", value=False)
                    
                    st.divider()
                    available_m = [m for m in MONTH_ORDER if m in sim_data_2025['MONTH_NAME'].unique()]
                    sel_months_sim = st.multiselect("Select Months for this Cycle", options=available_m)
                    
                    # Preferred Time Slot selection for Simulator
                    available_t = sorted(sim_data_2025['TIME'].unique())
                    sel_times_sim = st.multiselect("Select preferred Time Slots for this Cycle", options=available_t, default=available_t)
                    
                    if st.button("Calculate Profit for Cycle", type="primary"):
                        if not sel_months_sim:
                            st.error("Please select at least one month.")
                        elif not sel_times_sim:
                            st.error("Please select at least one time slot.")
                        else:
                            cycle_data = sim_data_2025[
                                (sim_data_2025['MONTH_NAME'].isin(sel_months_sim)) & 
                                (sim_data_2025['TIME'].isin(sel_times_sim))
                            ]
                            
                            c_balance = s_cap
                            c_peak = s_cap
                            c_trades = []
                            
                            for _, row in cycle_data.iterrows():
                                b_ref = c_balance if s_comp else s_cap
                                r_amt = b_ref * (s_risk / 100)
                                trade_growth = row['GROWTH %']
                                pnl = r_amt * (trade_growth / 0.01)
                                c_balance += pnl
                                
                                if c_balance > c_peak: c_peak = c_balance
                                dd_pct = ((c_peak - c_balance) / c_peak * 100) if c_peak > 0 else 0
                                
                                c_trades.append({
                                    'Date': row['DATE'],
                                    'Pair': row['PAIR'],
                                    'Time': row['TIME'],
                                    'Result': row['RESULT'],
                                    'Trade_Growth': trade_growth,
                                    'PnL': pnl,
                                    'Balance': c_balance,
                                    'Drawdown': dd_pct,
                                    'Comments': row['COMMENTS']
                                })
                            
                            cycle_profit = sum([t['PnL'] for t in c_trades])
                            
                            st.session_state.last_cycle = {
                                'Months': ", ".join(sel_months_sim),
                                'Times': ", ".join(sel_times_sim),
                                'Initial': s_cap,
                                'Profit': cycle_profit,
                                'Gross_Balance': c_balance
                            }
                            st.session_state.last_cycle_details = pd.DataFrame(c_trades)
                            st.rerun()

                with sc2:
                    st.write("### Current Cycle Result")
                    if 'last_cycle' in st.session_state:
                        lc = st.session_state.last_cycle
                        # FIXED: Safe dictionary access with fallback to avoid KeyError: 'Times'
                        months_lbl = lc.get('Months', 'N/A')
                        times_lbl = lc.get('Times', 'N/A')
                        st.success(f"**Cycle Results:** {months_lbl} | Time slots: {times_lbl}")
                        
                        st.metric("Cycle Profit", f"${lc['Profit']:,.2f}", f"{ (lc['Profit']/lc['Initial'])*100 :.2f}%")
                        
                        withdrawal = st.number_input("Withdrawal Amount to record ($)", value=max(0.0, lc['Profit']), step=100.0)
                        
                        if st.button("Complete Cycle & Update Balance"):
                            final_bal = lc['Gross_Balance'] - withdrawal
                            st.session_state.sim_history.append({
                                'Period': lc['Months'],
                                'Opening': lc['Initial'],
                                'Profit': lc['Profit'],
                                'Withdrawal': withdrawal,
                                'Closing': final_bal
                            })
                            st.session_state.sim_current_capital = final_bal
                            del st.session_state.last_cycle
                            st.session_state.last_cycle_details = None
                            st.rerun()
                    
                    st.write("### Simulation History")
                    if st.session_state.sim_history:
                        h_df = pd.DataFrame(st.session_state.sim_history)
                        st.table(h_df)
                        if st.button("Reset Simulator State"):
                            st.session_state.sim_history = []
                            st.session_state.sim_current_capital = 100000.0
                            st.session_state.last_cycle_details = None
                            if 'last_cycle' in st.session_state: del st.session_state.last_cycle
                            st.rerun()
                    else:
                        st.info("No cycles completed yet. Run a cycle on the left to start.")

                if st.session_state.last_cycle_details is not None:
                    st.divider()
                    st.subheader(f"📊 Trade-by-Trade Details for current Cycle")
                    cd_data = st.session_state.last_cycle_details
                    
                    cv1, cv2 = st.columns(2)
                    with cv1:
                        fig_c_equity = px.line(cd_data, x=cd_data.index, y='Balance', markers=True, title="Cycle Equity Curve ($)", color_discrete_sequence=['#00CC96'])
                        fig_c_equity.update_layout(xaxis_title="Trade Index", yaxis_title="Balance")
                        st.plotly_chart(fig_c_equity, use_container_width=True)
                    
                    with cv2:
                        fig_c_dd = px.area(cd_data, x=cd_data.index, y='Drawdown', title="Cycle Drawdown (%)", color_discrete_sequence=['#EF553B'])
                        fig_c_dd.update_layout(xaxis_title="Trade Index", yaxis_title="Drawdown %", yaxis_autorange="reversed")
                        st.plotly_chart(fig_c_dd, use_container_width=True)
                    
                    st.write("#### Execution Log")
                    st.dataframe(
                        cd_data[['Date', 'Pair', 'Time', 'Result', 'PnL', 'Balance', 'Drawdown', 'Comments']],
                        column_config={
                            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                            "PnL": st.column_config.NumberColumn("Trade PnL", format="$%.2f"),
                            "Balance": st.column_config.NumberColumn("Running Bal", format="$%.2f"),
                            "Drawdown": st.column_config.NumberColumn("DD %", format="%.2f%%")
                        },
                        use_container_width=True, hide_index=True
                    )

    else:
        st.warning("No trades match your current filter criteria.")
else:
    st.error("trade_journal.csv not found. Please aggregate your data first.")