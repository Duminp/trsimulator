import streamlit as st
import pandas as pd
import random
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Strategy Simulator", layout="wide", page_icon="📉")

# --- CSS STYLING ---
st.markdown("""
<style>
    .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 15px; text-align: center; }
    /* Primary Button Style */
    .stButton>button[kind="primary"] {
        background-color: #FF4B4B;
        color: white;
        border: none;
    }
    .stButton>button { width: 100%; font-weight: bold; }
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

# --- CORE SIMULATION FUNCTION (Defined at Top) ---
def run_single_month(start_cap, month_num, trade_count_start, trades_qty, win_prob, risk_pct, withdrawal_amt):
    curr_cap = start_cap
    month_trades = []
    wins = 0
    losses = 0
    consecutive_losses = 0
    failed_trades = []

    # Immediate fail check
    if curr_cap <= 0:
        return curr_cap, [], {
            "Month": month_num, "Start Balance": 0, "End Balance": 0, "Withdrawn": 0, 
            "Remaining": 0, "Failed Trade #s": "BLOWN", "Win Rate": "0%"
        }

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

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("⚙️ Simulator Controls")
    
    # 1. MODE SELECTION
    # We use simple indices to avoid string matching errors
    MODE_BATCH = "⚡ Run All (Batch)"
    MODE_MANUAL = "👣 Manual Step-by-Step"
    
    exec_mode = st.radio("Execution Mode:", [MODE_BATCH, MODE_MANUAL])
    
    # 2. RESET BUTTON (Top of Sidebar)
    if exec_mode == MODE_MANUAL:
        # Only active if game started
        btn_type = "primary" if st.session_state.manual_started else "secondary"
        btn_disabled = not st.session_state.manual_started
        
        # Use primary red button if active, otherwise disabled grey
        if st.session_state.manual_started:
            if st.button("🔄 RESET SIMULATION", type="primary"):
                st.session_state.manual_started = False
                st.session_state.history_trades = []
                st.session_state.history_months = []
                st.session_state.month_counter = 0
                st.session_state.global_trade_counter = 0
                st.rerun()
        else:
            st.button("🔄 RESET SIMULATION", disabled=True)
            
        st.divider()

    # Determine if inputs should be disabled (Locked during manual play)
    inputs_disabled = st.session_state.manual_started

    # 3. SETTINGS INPUTS
    with st.expander("💰 Capital & Risk", expanded=True):
        initial_capital = st.number_input(
            "Starting Capital ($)", 
            value=10000.0, step=100.0, key="widget_initial_capital", disabled=inputs_disabled
        )
        risk_percent_input = st.slider(
            "Risk per Trade (%)", 
            min_value=1.0, max_value=20.0, value=10.0, step=0.5, disabled=inputs_disabled
        )
        risk_percent = risk_percent_input / 100.0

    with st.expander("📊 Strategy Specs", expanded=False):
        win_rate_input = st.slider(
            "Win Rate (%)", 
            min_value=10, max_value=99, value=80, step=1, disabled=inputs_disabled
        )
        win_rate = win_rate_input / 100.0
        
        trades_per_month = st.number_input(
            "Trades per Month", 
            min_value=1, value=16, disabled=inputs_disabled
        )

    with st.expander("⏳ Duration", expanded=False):
        months_to_simulate = st.slider(
            "Total Months", 
            min_value=1, max_value=24, value=6, disabled=inputs_disabled
        )

    # Force reset if user switches modes while playing
    if exec_mode == MODE_BATCH and st.session_state.manual_started:
        st.session_state.manual_started = False
        st.rerun()

# --- MAIN PAGE LAYOUT ---
# IMPORTANT: This indentation must be FLUSH LEFT (Not inside sidebar)

st.title("📉 Trading Simulator")
st.markdown(f"**Config:** {win_rate*100:.0f}% WR | {trades_per_month} Trades/Mo | **{risk_percent*100:.1f}% Risk**")

# ==========================================
# LOGIC FOR BATCH RUN
# ==========================================
if exec_mode == MODE_BATCH:
    
    st.info("👇 **Pre-configure your monthly withdrawals:**")
    
    # Creates columns for inputs
    cols = st.columns(min(months_to_simulate, 4))
    monthly_withdrawals = []
    
    # Safe loop for inputs
    for i in range(months_to_simulate):
        col_idx = i % 4
        with cols[col_idx]:
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

        # Show Results
        df_m = pd.DataFrame(all_months)
        df_t = pd.DataFrame(all_trades)
        
        if not df_m.empty:
            final_cap = df_m.iloc[-1]['Remaining']
            tot_wd = df_m['Withdrawn'].sum()
            net = (final_cap + tot_wd) - initial_capital
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Final Capital", f"${final_cap:,.2f}")
            c2.metric("Total Withdrawn", f"${tot_wd:,.2f}")
            c3.metric("Net Profit", f"${net:,.2f}", delta=f"{(net/initial_capital)*100:.1f}%")
            
            st.divider()
            st.subheader("Monthly Breakdown")
            st.dataframe(df_m.style.format({"Start Balance":"${:,.2f}", "End Balance":"${:,.2f}", "Withdrawn":"${:,.2f}", "Remaining":"${:,.2f}"}), use_container_width=True)
            
            st.subheader("Equity Curve")
            st.plotly_chart(px.line(df_t, x="Global Trade", y="Balance"), use_container_width=True)

# ==========================================
# LOGIC FOR MANUAL RUN
# ==========================================
else:
    # 1. Start Screen
    if not st.session_state.manual_started:
        st.info("👋 Ready to start? Confirm Capital in sidebar and click below.")
        
        if st.button("🏁 Start Manual Simulation", type="primary"):
            st.session_state.manual_started = True 
            st.session_state.current_capital = initial_capital 
            st.session_state.month_counter = 0
            st.session_state.global_trade_counter = 0
            st.session_state.history_trades = []
            st.session_state.history_months = []
            st.rerun()
    
    # 2. Active Game Screen
    else:
        game_over = st.session_state.current_capital <= 0
        victory = st.session_state.month_counter >= months_to_simulate
        
        if game_over:
            st.error("💀 Account Blown! Game Over.")
        elif victory:
            st.success("✅ Simulation Period Complete! Well done.")
            st.balloons()
        
        # Action Panel (Only if game active)
        if not game_over and not victory:
            next_month = st.session_state.month_counter + 1
            
            st.markdown(f"### 🗓️ Execute Month {next_month}")
            
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.metric("Current Capital", f"${st.session_state.current_capital:,.2f}")
            with c2:
                step_withdraw = st.number_input(f"Withdrawal for Month {next_month} ($)", value=3000.0, step=100.0, key=f"step_w_{next_month}")
            with c3:
                st.write("") 
                st.write("") 
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

        # History Table (Always Visible)
        st.divider()
        st.subheader("📊 Simulation History")
        
        if st.session_state.history_months:
            df_m_live = pd.DataFrame(st.session_state.history_months)
            
            tot_wd = df_m_live['Withdrawn'].sum()
            curr_bal = st.session_state.current_capital
            
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

        if st.session_state.history_trades:
            st.divider()
            df_t_live = pd.DataFrame(st.session_state.history_trades)
            st.plotly_chart(px.line(df_t_live, x="Global Trade", y="Balance", title="Live Equity Curve"), use_container_width=True)