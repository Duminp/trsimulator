import streamlit as st
import pandas as pd
import random
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Strategy Simulator v5", layout="wide", page_icon="📉")

# --- CSS STYLING ---
st.markdown("""
<style>
    .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 15px; text-align: center; }
    .stButton>button { width: 100%; font-weight: bold; }
    .success-text { color: green; font-weight: bold; }
    .fail-text { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if 'manual_started' not in st.session_state:
    st.session_state.manual_started = False
if 'current_capital' not in st.session_state:
    st.session_state.current_capital = 0
if 'month_counter' not in st.session_state:
    st.session_state.month_counter = 0
if 'history_trades' not in st.session_state:
    st.session_state.history_trades = []
if 'history_months' not in st.session_state:
    st.session_state.history_months = []
if 'global_trade_counter' not in st.session_state:
    st.session_state.global_trade_counter = 0

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.markdown("---")
    st.header("⚙️ Settings")
    st.markdown("---")
    # 1. MODE SELECTION
    st.subheader("1. Execution Mode")
    exec_mode = st.radio("Choose Mode:", ["⚡ Run All (Batch)", "👣 Manual Step-by-Step"])

    st.markdown("---")
    # 2. CAPITAL & RISK
    st.subheader("2. Capital & Risk")
    # We use a key here to ensure we can access it reliably in session state if needed
    initial_capital = st.number_input("Starting Capital ($)", value=10000.0, step=100.0, key="widget_initial_capital")
    
    risk_percent_input = st.slider("Risk per Trade (%)", min_value=1.0, max_value=20.0, value=10.0, step=0.5)
    risk_percent = risk_percent_input / 100.0
    
    st.markdown("---")
    # 3. STRATEGY SPECS
    st.subheader("3. Strategy Specs")
    win_rate_input = st.slider("Win Rate (%)", min_value=10, max_value=99, value=80, step=1)
    win_rate = win_rate_input / 100.0
    trades_per_month = st.number_input("Trades per Month", min_value=1, value=16)
    
    st.markdown("---")
    # 4. DURATION
    st.subheader("4. Duration")
    months_to_simulate = st.slider("Total Duration (Months)", min_value=1, max_value=24, value=6)

    # 5. RESET BUTTON (For Manual Mode)
    if exec_mode == "👣 Manual Step-by-Step":
        st.markdown("---")
        if st.button("🔄 Reset / New Simulation"):
            # RESET LOGIC: Explicitly grab the current widget value
            st.session_state.manual_started = False
            st.session_state.history_trades = []
            st.session_state.history_months = []
            st.session_state.month_counter = 0
            st.session_state.global_trade_counter = 0
            # Force update capital from the sidebar input
            st.session_state.current_capital = initial_capital 
            st.rerun()

# --- CORE SIMULATION FUNCTION ---
def run_single_month(start_cap, month_num, trade_count_start, trades_qty, win_prob, risk_pct, withdrawal_amt):
    curr_cap = start_cap
    month_trades = []
    wins = 0
    losses = 0
    consecutive_losses = 0
    failed_trades = []

    # Instant fail check
    if curr_cap <= 0:
        return curr_cap, [], {
            "Month": month_num, "Start Balance": 0, "End Balance": 0, "Withdrawn": 0, 
            "Remaining": 0, "Failed Trade #s": "BLOWN", "Win Rate": "0%"
        }

    for t in range(1, trades_qty + 1):
        trade_id = trade_count_start + t
        
        if consecutive_losses >= 2:
            is_win = True
            note = "Streak Breaker"
        else:
            is_win = random.random() < win_prob
            note = "Normal"

        risk_amt = curr_cap * risk_pct
        
        if is_win:
            profit = risk_amt
            curr_cap += profit
            wins += 1
            consecutive_losses = 0
            res = "WIN"
        else:
            loss = risk_amt
            curr_cap -= loss
            losses += 1
            consecutive_losses += 1
            failed_trades.append(str(t))
            res = "LOSS"
            profit = -loss

        month_trades.append({
            "Global Trade": trade_id,
            "Month": month_num,
            "Trade #": t,
            "Result": res,
            "PnL": profit,
            "Balance": curr_cap,
            "Note": note
        })

    pre_withdraw_cap = curr_cap
    actual_withdraw = 0
    if curr_cap > withdrawal_amt:
        curr_cap -= withdrawal_amt
        actual_withdraw = withdrawal_amt
    elif curr_cap > 0:
        actual_withdraw = 0 
    
    month_summary = {
        "Month": month_num,
        "Start Balance": start_cap,
        "End Balance": pre_withdraw_cap,
        "Withdrawn": actual_withdraw,
        "Remaining": curr_cap,
        "Failed Trade #s": ", ".join(failed_trades) if failed_trades else "None",
        "Win Rate": f"{(wins/trades_qty)*100:.0f}%"
    }
    
    return curr_cap, month_trades, month_summary

# --- APP LAYOUT ---

st.title("📉 Trading Simulator v5")
st.markdown(f"**Config:** {win_rate*100:.0f}% WR | {trades_per_month} Trades/Mo | **{risk_percent*100:.1f}% Risk**")

# ==========================================
# MODE 1: BATCH RUN
# ==========================================
if exec_mode == "⚡ Run All (Batch)":
    
    st.info("👇 **Pre-configure your monthly withdrawals:**")
    cols = st.columns(min(months_to_simulate, 4))
    monthly_withdrawals = []
    for i in range(months_to_simulate):
        with cols[i % 4]:
            w = st.number_input(f"Month {i+1} ($)", value=3000.0, step=500.0, key=f"batch_w_{i}")
            monthly_withdrawals.append(w)

    if st.button("🚀 EXECUTE FULL SIMULATION", type="primary"):
        curr = initial_capital
        glob_count = 0
        all_trades = []
        all_months = []
        
        for i in range(months_to_simulate):
            w_amt = monthly_withdrawals[i]
            curr, m_trades, m_sum = run_single_month(curr, i+1, glob_count, trades_per_month, win_rate, risk_percent, w_amt)
            all_trades.extend(m_trades)
            all_months.append(m_sum)
            glob_count += len(m_trades)
            if curr <= 0: break

        # Results Display
        df_m = pd.DataFrame(all_months)
        df_t = pd.DataFrame(all_trades)
        
        final_cap = df_m.iloc[-1]['Remaining'] if not df_m.empty else 0
        tot_wd = df_m['Withdrawn'].sum() if not df_m.empty else 0
        net = (final_cap + tot_wd) - initial_capital
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Final Capital", f"${final_cap:,.2f}")
        c2.metric("Total Withdrawn", f"${tot_wd:,.2f}")
        c3.metric("Net Profit", f"${net:,.2f}", delta=f"{(net/initial_capital)*100:.1f}%")
        
        st.divider()
        st.subheader("Monthly Breakdown")
        st.dataframe(df_m.style.format({"Start Balance":"${:,.2f}", "End Balance":"${:,.2f}", "Withdrawn":"${:,.2f}", "Remaining":"${:,.2f}"}), use_container_width=True)
        
        if not df_t.empty:
            st.subheader("Equity Curve")
            st.plotly_chart(px.line(df_t, x="Global Trade", y="Balance"), use_container_width=True)

# ==========================================
# MODE 2: MANUAL STEP-BY-STEP
# ==========================================
else:
    # 1. Start Button
    if not st.session_state.manual_started:
        st.info("👋 Ready to start? Confirm Capital in sidebar and click below.")
        if st.button("🏁 Start Manual Simulation"):
            st.session_state.manual_started = True
            # FIX: Explicitly grab sidebar value here
            st.session_state.current_capital = initial_capital 
            st.session_state.month_counter = 0
            st.session_state.global_trade_counter = 0
            st.session_state.history_trades = []
            st.session_state.history_months = []
            st.rerun()
    
    # 2. Game Loop
    else:
        # Check Game Over / Victory Conditions
        game_over = st.session_state.current_capital <= 0
        victory = st.session_state.month_counter >= months_to_simulate
        
        if game_over:
            st.error("💀 Account Blown! Game Over.")
        elif victory:
            st.success("✅ Simulation Period Complete! Well done.")
            st.balloons()
        
        # 3. Active Controls (Only show if game is NOT over)
        if not game_over and not victory:
            next_month = st.session_state.month_counter + 1
            
            # Layout: Controls on Top
            st.markdown(f"### 🗓️ Execute Month {next_month}")
            
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.metric("Current Capital", f"${st.session_state.current_capital:,.2f}")
            with c2:
                step_withdraw = st.number_input(f"Withdrawal for Month {next_month} ($)", value=3000.0, step=100.0, key=f"step_w_{next_month}")
            with c3:
                st.write("") # Spacer
                st.write("") # Spacer
                if st.button(f"▶️ Run Month {next_month}", type="primary"):
                    new_cap, m_trades, m_sum = run_single_month(
                        st.session_state.current_capital, 
                        next_month, 
                        st.session_state.global_trade_counter, 
                        trades_per_month, 
                        win_rate, 
                        risk_percent, 
                        step_withdraw
                    )
                    
                    st.session_state.current_capital = new_cap
                    st.session_state.month_counter += 1
                    st.session_state.global_trade_counter += len(m_trades)
                    st.session_state.history_trades.extend(m_trades)
                    st.session_state.history_months.append(m_sum)
                    st.rerun()

        # 4. PERSISTENT DATA TABLE (Shows regardless of game state)
        st.divider()
        st.subheader("📊 Simulation History")
        
        if st.session_state.history_months:
            df_m_live = pd.DataFrame(st.session_state.history_months)
            
            # Calc Totals
            tot_wd = df_m_live['Withdrawn'].sum()
            curr_bal = st.session_state.current_capital
            
            # Mini Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Remaining Capital", f"${curr_bal:,.2f}")
            m2.metric("Total Withdrawn", f"${tot_wd:,.2f}")
            m3.metric("Net Profit", f"${(curr_bal + tot_wd) - initial_capital:,.2f}")

            st.dataframe(
                df_m_live.style.format({"Start Balance":"${:,.2f}", "End Balance":"${:,.2f}", "Withdrawn":"${:,.2f}", "Remaining":"${:,.2f}"}), 
                use_container_width=True
            )
        else:
            st.caption("No months simulated yet. Click Start above.")

        # 5. PERSISTENT CHART
        if st.session_state.history_trades:
            st.divider()
            df_t_live = pd.DataFrame(st.session_state.history_trades)
            st.plotly_chart(px.line(df_t_live, x="Global Trade", y="Balance", title="Live Equity Curve"), use_container_width=True)