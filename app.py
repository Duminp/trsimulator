import streamlit as st
import pandas as pd
import random
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Pro Strategy Simulator v4", layout="wide", page_icon="📉")

# --- CSS STYLING ---
st.markdown("""
<style>
    .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 15px; text-align: center; }
    .stButton>button { width: 100%; font-weight: bold; }
    .success-text { color: green; font-weight: bold; }
    .fail-text { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE (For Manual Mode) ---
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
    st.header("⚙️ Settings")
    
    # 1. MODE SELECTION
    st.subheader("1. Execution Mode")
    exec_mode = st.radio("Choose Mode:", ["⚡ Run All (Batch)", "👣 Manual Step-by-Step"])

    # 2. CAPITAL & RISK
    st.subheader("2. Capital & Risk")
    initial_capital = st.number_input("Starting Capital ($)", value=10000.0, step=100.0)
    
    # NEW: Configurable Risk
    risk_percent_input = st.slider("Risk per Trade (%)", min_value=1.0, max_value=20.0, value=10.0, step=0.5)
    risk_percent = risk_percent_input / 100.0
    
    # 3. STRATEGY SPECS
    st.subheader("3. Strategy Specs")
    win_rate_input = st.slider("Win Rate (%)", min_value=10, max_value=99, value=80, step=1)
    win_rate = win_rate_input / 100.0
    trades_per_month = st.number_input("Trades per Month", min_value=1, value=16)
    
    # 4. DURATION
    st.subheader("4. Duration")
    months_to_simulate = st.slider("Total Duration (Months)", min_value=1, max_value=24, value=6)

    # 5. RESET BUTTON (Only for Manual)
    if exec_mode == "👣 Manual Step-by-Step":
        st.markdown("---")
        if st.button("🔄 Reset / New Simulation"):
            st.session_state.manual_started = False
            st.session_state.history_trades = []
            st.session_state.history_months = []
            st.session_state.month_counter = 0
            st.session_state.current_capital = initial_capital
            st.rerun()

# --- CORE SIMULATION FUNCTION (Runs 1 Month) ---
def run_single_month(start_cap, month_num, trade_count_start, trades_qty, win_prob, risk_pct, withdrawal_amt):
    curr_cap = start_cap
    month_trades = []
    wins = 0
    losses = 0
    consecutive_losses = 0
    failed_trades = []

    if curr_cap <= 0:
        return curr_cap, [], 0, 0, "BLOWN"

    for t in range(1, trades_qty + 1):
        trade_id = trade_count_start + t
        
        # Streak Breaker Logic
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

    # Withdrawal Logic
    pre_withdraw_cap = curr_cap
    actual_withdraw = 0
    if curr_cap > withdrawal_amt:
        curr_cap -= withdrawal_amt
        actual_withdraw = withdrawal_amt
    elif curr_cap > 0:
        actual_withdraw = 0 # Keep capital if low
    
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

# --- APP LOGIC ---

st.title("📉 Pro Trading Simulator v4")
st.markdown(f"**Config:** {win_rate*100:.0f}% WR | {trades_per_month} Trades/Mo | **{risk_percent*100:.1f}% Risk**")

# ==========================================
# MODE 1: BATCH RUN (Original Logic)
# ==========================================
if exec_mode == "⚡ Run All (Batch)":
    
    # Input Withdrawals Upfront
    st.info("👇 **Pre-configure your monthly withdrawals:**")
    cols = st.columns(min(months_to_simulate, 4))
    monthly_withdrawals = []
    for i in range(months_to_simulate):
        with cols[i % 4]:
            w = st.number_input(f"Month {i+1} ($)", value=3000.0, step=500.0, key=f"batch_w_{i}")
            monthly_withdrawals.append(w)

    if st.button("🚀 EXECUTE FULL SIMULATION", type="primary"):
        # Reset temp tracking
        curr = initial_capital
        glob_count = 0
        all_trades = []
        all_months = []
        
        for i in range(months_to_simulate):
            w_amt = monthly_withdrawals[i]
            # Run Logic
            curr, m_trades, m_sum = run_single_month(curr, i+1, glob_count, trades_per_month, win_rate, risk_percent, w_amt)
            
            all_trades.extend(m_trades)
            all_months.append(m_sum)
            glob_count += len(m_trades)
            
            if m_sum == "BLOWN": break

        # Show Results
        df_t = pd.DataFrame(all_trades)
        df_m = pd.DataFrame(all_months)
        
        # Metrics
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
    # 1. Initialization
    if not st.session_state.manual_started:
        st.info("👋 Ready to start? Click below to initialize the simulation.")
        if st.button("🏁 Start Manual Simulation"):
            st.session_state.manual_started = True
            st.session_state.current_capital = initial_capital
            st.session_state.month_counter = 0
            st.session_state.global_trade_counter = 0
            st.session_state.history_trades = []
            st.session_state.history_months = []
            st.rerun()
    
    # 2. Execution Loop
    else:
        # Check if done
        if st.session_state.month_counter >= months_to_simulate:
            st.success("✅ Simulation Period Complete!")
            st.balloons()
        elif st.session_state.current_capital <= 0:
            st.error("💀 Account Blown! Game Over.")
        else:
            # Active Simulation State
            next_month = st.session_state.month_counter + 1
            
            col_dash, col_action = st.columns([2, 1])
            
            with col_action:
                st.markdown(f"### 🗓️ Execute Month {next_month}")
                st.write(f"**Current Capital:** ${st.session_state.current_capital:,.2f}")
                
                # Dynamic Withdrawal Input for THIS specific step
                step_withdraw = st.number_input(f"Planned Withdrawal for Month {next_month} ($)", value=3000.0, step=100.0, key=f"step_w_{next_month}")
                
                if st.button(f"▶️ Run Month {next_month}"):
                    # Run logic
                    new_cap, m_trades, m_sum = run_single_month(
                        st.session_state.current_capital, 
                        next_month, 
                        st.session_state.global_trade_counter, 
                        trades_per_month, 
                        win_rate, 
                        risk_percent, 
                        step_withdraw
                    )
                    
                    # Update State
                    st.session_state.current_capital = new_cap
                    st.session_state.month_counter += 1
                    st.session_state.global_trade_counter += len(m_trades)
                    st.session_state.history_trades.extend(m_trades)
                    st.session_state.history_months.append(m_sum)
                    st.rerun()

            with col_dash:
                st.subheader("📊 Live Results")
                if st.session_state.history_months:
                    df_m_live = pd.DataFrame(st.session_state.history_months)
                    st.dataframe(
                        df_m_live.style.format({"Start Balance":"${:,.2f}", "End Balance":"${:,.2f}", "Withdrawn":"${:,.2f}", "Remaining":"${:,.2f}"}), 
                        use_container_width=True
                    )
                    
                    # Mini Stats
                    tot_wd = df_m_live['Withdrawn'].sum()
                    st.info(f"💰 Total Banked So Far: **${tot_wd:,.2f}**")
                else:
                    st.write("Waiting for first month results...")

        # Always show charts at bottom if data exists
        if st.session_state.history_trades:
            st.divider()
            df_t_live = pd.DataFrame(st.session_state.history_trades)
            st.plotly_chart(px.line(df_t_live, x="Global Trade", y="Balance", title="Live Equity Curve"), use_container_width=True)