import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px;
    margin: 0 auto;
    border-radius: 10px;
}
.metric-call {
    background-color: #90ee90;
    color: black;
    margin-right: 10px;
    padding: 8px 12px;
}
.metric-put {
    background-color: #ffcccb;
    color: black;
    padding: 8px 12px;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: bold;
    margin: 0;
}
.metric-label {
    font-size: 1rem;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# Black-Scholes Model class
class BlackScholes:
    def __init__(self, time_to_maturity, strike, current_price, volatility, interest_rate):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_prices(self):
        # guard for zero time to maturity or zero volatility
        t = max(self.time_to_maturity, 1e-9)
        v = max(self.volatility, 1e-9)

        d1 = (log(self.current_price / self.strike) +
              (self.interest_rate + 0.5 * v**2) * t) / (v * sqrt(t))
        d2 = d1 - v * sqrt(t)

        call_price = self.current_price * norm.cdf(d1) - (
            self.strike * exp(-(self.interest_rate * t)) * norm.cdf(d2)
        )
        put_price = (
            self.strike * exp(-(self.interest_rate * t)) * norm.cdf(-d2)
        ) - self.current_price * norm.cdf(-d1)

        self.call_price = float(call_price)
        self.put_price = float(put_price)
        return self.call_price, self.put_price

# Sidebar Inputs
with st.sidebar:
    st.title("📊 Black-Scholes Model")
    st.write("`Created by:`")
    st.markdown(
        '<a href="https://www.linkedin.com/in/navjotdhah" target="_blank">'
        '<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" '
        'width="22" height="22" style="vertical-align: middle; margin-right: 8px;">'
        '`Navjot Dhah`</a>',
        unsafe_allow_html=True
    )

    current_price = st.number_input("Current Asset Price", value=100.0, format="%.2f")
    strike = st.number_input("Strike Price", value=100.0, format="%.2f")
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0, format="%.4f")
    volatility = st.number_input("Volatility (σ)", value=0.2, format="%.4f")
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05, format="%.4f")

    st.markdown("---")
    st.subheader("Heatmap Parameters")
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=max(0.01, current_price*0.8), step=0.01, format="%.2f")
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=max(spot_min+0.01, current_price*1.2), step=0.01, format="%.2f")
    vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=max(0.01, volatility*0.5), step=0.01)
    vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=min(1.0, volatility*1.5), step=0.01)

    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)

# Heatmap function
def plot_heatmap(bs_model, spot_range, vol_range, strike):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))

    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            bs_temp.calculate_prices()
            call_prices[i, j] = bs_temp.call_price
            put_prices[i, j] = bs_temp.put_price

    fig_call, ax_call = plt.subplots(figsize=(8, 6))
    sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2),
                yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f",
                cmap="viridis", ax=ax_call)
    ax_call.set_title('CALL Prices')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')

    fig_put, ax_put = plt.subplots(figsize=(8, 6))
    sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2),
                yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f",
                cmap="viridis", ax=ax_put)
    ax_put.set_title('PUT Prices')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')

    return fig_call, fig_put

# Main page content
st.title("Black-Scholes Pricing Model")

input_data = {
    "Current Asset Price": [current_price],
    "Strike Price": [strike],
    "Time to Maturity (Years)": [time_to_maturity],
    "Volatility (σ)": [volatility],
    "Risk-Free Interest Rate": [interest_rate],
}
st.table(pd.DataFrame(input_data))

bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price = bs_model.calculate_prices()

col1, col2 = st.columns([1,1])
with col1:
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.title("Options Price Heatmaps")
st.info("Explore how option prices fluctuate with varying Spot Prices and Volatility levels.")

col1, col2 = st.columns([1,1])
with col1:
    st.subheader("Call Price Heatmap")
    fig_call, _ = plot_heatmap(bs_model, spot_range, vol_range, strike)
    st.pyplot(fig_call)
with col2:
    st.subheader("Put Price Heatmap")
    _, fig_put = plot_heatmap(bs_model, spot_range, vol_range, strike)
    st.pyplot(fig_put)
