import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from gtda.homology import VietorisRipsPersistence
import persim
import tensorflow as tf
from tensorflow.keras import layers

# Quantum Computing Imports (Updated)
from qiskit import QuantumCircuit
from qiskit_aer import Aer  # Works with these versions
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.utils import QuantumInstance

# Rest of your imports...
from arch import arch_model
import pandas_ta as ta
import pandas_market_calendars as mcal
from abc import ABC, abstractmethod
from collections import deque
import asyncio
import platform
from datetime import datetime, timedelta
import mlflow
import mlflow.tensorflow
import shap
import joblib
from sklearn.model_selection import TimeSeriesSplit
import logging
import os
from alpaca_data_service import data_store, execute_order, get_account_info
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app configuration
st.set_page_config(page_title="QTRL-Pro Trading Platform", layout="wide")
st.title("QTRL-Pro: Quantum-Topological RL Trading Platform")
st.markdown("A production-ready trading platform using quantum kernels, TDA, and Alpaca real-time trading.")

# 1. Abstract Model Class
class TradingModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict_signals(self, data, features):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path):
        pass

# 2. QTRL Agent with Quantum Kernel
class QTRLAgent(TradingModel):
    def __init__(self, n_actions=3, feature_dim=8, fast_mode=False):
        super().__init__()
        self.fast_mode = fast_mode
        self.model = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(n_actions, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 0.1
        if not fast_mode:
            quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'), shots=1000)
            self.quantum_circuit = ZZFeatureMap(feature_dimension=feature_dim, reps=2)
            self.quantum_kernel = QuantumKernel(feature_map=self.quantum_circuit, quantum_instance=quantum_instance)
        self.target_model = tf.keras.models.clone_model(self.model)
        self.energy_penalty = 0.02

    def fit(self, X, y):
        if len(self.memory) > 1000:
            batch = np.random.choice(len(self.memory), 32)
            states, actions, rewards, next_states = zip(*[self.memory[i] for i in batch])
            states = np.array(states)
            next_states = np.array(next_states)
            if not self.fast_mode:
                states = self.quantum_kernel.evaluate(states)
                next_states = self.quantum_kernel.evaluate(next_states)
            targets = self.model.predict(states, verbose=0)
            next_q = self.target_model.predict(next_states, verbose=0)
            for i, action in enumerate(actions):
                targets[i, action] = rewards[i] + self.gamma * np.max(next_q[i])
            self.model.fit(states, targets, epochs=1, verbose=0)
            self.target_model.set_weights(self.model.get_weights())
        mlflow.tensorflow.log_model(self.model, "qtrl_model")

    def predict_signals(self, data, features):
        if not self.fast_mode:
            features = self.quantum_kernel.evaluate(features)
        predictions = self.model.predict(features, verbose=0)
        actions = np.argmax(predictions, axis=1)
        signals = ['Buy' if a == 0 else 'Sell' if a == 1 else 'Hold' for a in actions]
        predicted_prices = data['close'].values * (1 + 0.01 * np.max(predictions, axis=1))
        return signals, predicted_prices

    def save(self, path):
        self.model.save(path)
        mlflow.log_artifact(path)

    @classmethod
    def load(cls, path):
        instance = cls(fast_mode=True)
        instance.model = tf.keras.models.load_model(path)
        instance.target_model = tf.keras.models.clone_model(instance.model)
        return instance

# 3. LSTM Model
class LSTMModel(TradingModel):
    def __init__(self, time_steps):
        super().__init__()
        self.time_steps = time_steps
        self.model = tf.keras.Sequential([
            layers.LSTM(100, return_sequences=True, input_shape=(time_steps, 1)),
            layers.LSTM(50, return_sequences=False),
            layers.Dense(25),
            layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        self.model.fit(X_scaled, y, epochs=20, batch_size=32, validation_split=0.2, verbose=0, callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5)
        ])
        mlflow.tensorflow.log_model(self.model, "lstm_model")

    def predict_signals(self, data, features):
        X_scaled = self.scaler.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
        predictions = self.model.predict(X_scaled, verbose=0)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        signals = ['Buy' if p > data['close'].iloc[i+self.time_steps-1] else 'Sell' for i, p in enumerate(predictions)]
        return signals[:len(data)-self.time_steps], predictions[:len(data)-self.time_steps]

    def save(self, path):
        self.model.save(path)
        mlflow.log_artifact(path)

    @classmethod
    def load(cls, path):
        instance = cls(time_steps=60)
        instance.model = tf.keras.models.load_model(path)
        return instance

# 4. Random Forest Model
class RFModel(TradingModel):
    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        joblib.dump(self.model, "rf_model.pkl")
        mlflow.log_artifact("rf_model.pkl")

    def predict_signals(self, data, features):
        X_scaled = self.scaler.transform(features)
        predictions = self.model.predict(X_scaled)
        signals = ['Buy' if p > data['close'].iloc[i+1] else 'Sell' for i, p in enumerate(predictions)]
        return signals, predictions

    def save(self, path):
        joblib.dump(self.model, path)
        mlflow.log_artifact(path)

    @classmethod
    def load(cls, path):
        instance = cls()
        instance.model = joblib.load(path)
        return instance

# 5. GARCH-AI Model
class GARCHAIModel(TradingModel):
    def __init__(self):
        super().__init__()
        self.rf = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        garch = arch_model(X[:, 0] * 100, vol='Garch', p=1, q=1)
        garch_fit = garch.fit(disp='off')
        garch_vol = garch_fit.conditional_volatility / 100
        X_combined = np.column_stack((X, garch_vol[-len(X):]))
        X_scaled = self.scaler.fit_transform(X_combined)
        self.rf.fit(X_scaled, y)
        joblib.dump(self.rf, "garch_ai_model.pkl")
        mlflow.log_artifact("garch_ai_model.pkl")

    def predict_signals(self, data, features):
        garch = arch_model(data['Returns'].values * 100, vol='Garch', p=1, q=1)
        garch_fit = garch.fit(disp='off')
        garch_vol = garch_fit.conditional_volatility / 100
        X_combined = np.column_stack((features, garch_vol[-len(features):]))
        X_scaled = self.scaler.transform(X_combined)
        predictions = self.rf.predict(X_scaled)
        signals = ['Buy' if p > data['close'].iloc[i+1] else 'Sell' for i, p in enumerate(predictions)]
        return signals, predictions

    def save(self, path):
        joblib.dump(self.rf, path)
        mlflow.log_artifact(path)

    @classmethod
    def load(cls, path):
        instance = cls()
        instance.rf = joblib.load(path)
        return instance

# 6. Fetch Data from Alpaca Data Store
def prepare_data(data, time_steps=60):
    scaler = StandardScaler()
    prices = scaler.fit_transform(data['close'].values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(prices) - time_steps):
        X.append(prices[i:i+time_steps])
        y.append(prices[i+time_steps])
    return np.array(X), np.array(y), scaler

# 7. Jump-Diffusion Model
def jump_diffusion_price(data, mu=0.1, sigma=0.2, lambda_jump=0.1, jump_size=0.05):
    prices = data['close'].values
    n = len(prices)
    dt = 1 / 252
    dW = np.random.normal(0, np.sqrt(dt), n)
    jumps = np.random.poisson(lambda_jump * dt, n) * np.random.normal(0, jump_size, n)
    simulated = np.zeros(n)
    simulated[0] = prices[0]
    for i in range(1, n):
        simulated[i] = simulated[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[i-1] + jumps[i-1])
    return simulated

# 8. Topological Feature Extraction
def topological_features(data, max_dim=2):
    vr = VietorisRipsPersistence(homology_dimensions=[0, 1, 2], alpha=0.1)
    data_scaled = StandardScaler().fit_transform(data[['close', 'Returns', 'Volatility', 'Sentiment', 'RSI']])
    diagrams = vr.fit_transform(data_scaled[None, :, :])[0]
    persistence = persim.sliced_wasserstein_distance(diagrams[diagrams[:, 2] == 1], diagrams[diagrams[:, 2] == 1])
    curvature = np.mean(np.abs(np.gradient(data_scaled, axis=0)))
    return persistence, curvature, diagrams

# 9. Chart Pattern Detection
def detect_chart_patterns(data):
    if len(data) < 150:
        return []
    patterns = []
    prices = data['close'].values
    window = 50
    for i in range(window, len(prices) - window):
        left_shoulder = np.max(prices[i-window:i-10])
        head = np.max(prices[i-10:i+10])
        right_shoulder = np.max(prices[i+10:i+window])
        if head > left_shoulder and head > right_shoulder and abs(left_shoulder - right_shoulder) < 0.1 * head and head > 1.1 * min(left_shoulder, right_shoulder):
            patterns.append((data['timestamp'].iloc[i], 'Head and Shoulders'))
        if i > 2 * window:
            peak1 = np.max(prices[i-2*window:i-window])
            peak2 = np.max(prices[i-window:i])
            if abs(peak1 - peak2) < 0.05 * peak1 and peak1 > prices[i] * 1.1:
                patterns.append((data['timestamp'].iloc[i], 'Double Top'))
            trough1 = np.min(prices[i-2*window:i-window])
            trough2 = np.min(prices[i-window:i])
            if abs(trough1 - trough2) < 0.05 * trough1 and trough1 < prices[i] * 0.9:
                patterns.append((data['timestamp'].iloc[i], 'Double Bottom'))
    return patterns

# 10. Risk Management
def compute_var_cvar(returns, alpha=0.05):
    var = np.percentile(returns, alpha * 100)
    cvar = returns[returns <= var].mean()
    return var, cvar

def kelly_criterion(returns, cov_matrix):
    if cov_matrix.shape[0] == 1:
        return returns / cov_matrix[0, 0] if cov_matrix[0, 0] != 0 else 0.1
    inv_cov = np.linalg.pinv(cov_matrix)
    return inv_cov @ returns

# 11. Portfolio Performance
def compute_portfolio(data, signals, initial_cash=10000, commission=0.001, execute_trades=False):
    cash = initial_cash
    shares = 0
    portfolio = []
    returns = []
    for i in range(1, len(data)):
        price = data['close'].iloc[i]
        signal = signals[i]
        returns_i = data['Returns'].iloc[i]
        cov_matrix = np.cov(data['Returns'].iloc[max(0, i-20):i].values)
        kelly = kelly_criterion(np.array([returns_i]), cov_matrix if cov_matrix.ndim == 2 else np.array([[cov_matrix]]))
        position_size = min(kelly[0], 1.0) * cash / price
        if signal == 'Buy' and cash >= price:
            shares_to_buy = position_size
            cost = shares_to_buy * price * (1 + commission)
            if cash >= cost:
                shares += shares_to_buy
                cash -= cost
                if execute_trades:
                    execute_order(data['symbol'].iloc[0], shares_to_buy, "buy")
        elif signal == 'Sell' and shares > 0:
            cash += shares * price * (1 - commission)
            if execute_trades:
                execute_order(data['symbol'].iloc[0], shares, "sell")
            shares = 0
        portfolio_value = cash + shares * price
        portfolio.append(portfolio_value)
        returns.append((portfolio_value - portfolio[-1]) / portfolio[-1] if i > 1 else 0)
    var, cvar = compute_var_cvar(np.array(returns))
    return portfolio, var, cvar

# 12. Advanced Financial Metrics
def compute_metrics(actual, predicted, portfolio, benchmark_returns):
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    mae = np.mean(np.abs(actual - predicted))
    returns = np.diff(portfolio) / portfolio[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
    downside_returns = returns[returns < 0]
    sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252) if np.std(downside_returns) != 0 else 0
    positive_returns = np.sum(returns[returns > 0])
    negative_returns = np.sum(np.abs(returns[returns < 0]))
    omega = positive_returns / negative_returns if negative_returns != 0 else float('inf')
    max_drawdown = np.max(np.maximum.accumulate(portfolio) - portfolio) / np.max(portfolio) if np.max(portfolio) != 0 else 0
    calmar = np.mean(returns) * 252 / max_drawdown if max_drawdown != 0 else float('inf')
    information_ratio = (np.mean(returns) - np.mean(benchmark_returns)) / np.std(returns - benchmark_returns) * np.sqrt(252) if np.std(returns - benchmark_returns) != 0 else 0
    return {
        'RMSE': rmse, 'MAE': mae, 'Sharpe Ratio': sharpe, 'Sortino Ratio': sortino,
        'Omega Ratio': omega, 'Calmar Ratio': calmar, 'Information Ratio': information_ratio
    }

# 13. Explainability with SHAP
def compute_shap_values(model, features, data):
    explainer = shap.KernelExplainer(model.model.predict, features)
    shap_values = explainer.shap_values(features[:100], nsamples=100)
    return shap_values

# 14. Correlation Heatmap
def compute_correlation_heatmap(data_dict):
    closes = pd.DataFrame({ticker: data['close'] for ticker, data in data_dict.items()})
    return closes.corr()

# 15. Main App Logic
async def main():
    st.header("Multi-Asset Analysis with Alpaca Real-Time Trading")
    tickers = ["AAPL", "MSFT", "TSLA", "SPY"]
    fast_mode = st.checkbox("Fast Mode (Disable Quantum/TDA)", value=False)
    execute_trades = st.checkbox("Execute Real-Time Trades (Paper Trading)", value=False)

    # Display account information
    st.subheader("Alpaca Account Information")
    account_info = get_account_info()
    if account_info:
        st.write(f"Account Number: {account_info['account_number']}")
        st.write(f"Buying Power: ${account_info['buying_power']:.2f}")
        st.write(f"Portfolio Value: ${account_info['portfolio_value']:.2f}")
        st.write(f"Status: {account_info['status']}")
    else:
        st.error("Failed to fetch account information. Check API keys.")

    # Wait for data_store to populate
    for ticker in tickers:
        if ticker not in data_store or len(data_store[ticker]) < 100:
            st.warning(f"Waiting for sufficient data for {ticker}...")
            return
        data_store[ticker]['symbol'] = ticker  # Add symbol for order execution

    # Correlation Heatmap
    st.subheader("Asset Correlation Heatmap")
    corr_matrix = compute_correlation_heatmap(data_store)
    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=tickers, y=tickers, colorscale='Viridis'))
    st.plotly_chart(fig)

    for ticker in tickers:
        st.header(f"Analysis for {ticker}")
        data = data_store[ticker]

        # Candlestick Chart with Patterns
        st.subheader("Candlestick Chart with Patterns")
        patterns = detect_chart_patterns(data)
        fig = go.Figure(data=[
            go.Candlestick(
                x=data['timestamp'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Candlestick'
            )
        ])
        for date, pattern in patterns:
            idx = data.index[data['timestamp'] == date].tolist()
            if idx:
                fig.add_annotation(x=data['timestamp'].iloc[idx[0]], y=data['close'].iloc[idx[0]],
                                   text=pattern, showarrow=True, arrowhead=1)
        st.plotly_chart(fig)

        # Jump-Diffusion Simulation
        st.subheader("Jump-Diffusion Price Simulation")
        simulated_prices = jump_diffusion_price(data)
        st.line_chart(pd.DataFrame({
            'Actual Price': data['close'],
            'Simulated Price': simulated_prices
        }, index=data['timestamp']))

        # Topological Features
        st.subheader("Topological Feature Analysis")
        persistence, curvature, diagrams = topological_features(data)
        st.write(f"Persistence Score: {persistence:.4f}, Market Curvature: {curvature:.4f}")
        fig = go.Figure(data=[
            go.Scatter(x=diagrams[diagrams[:, 2] == 1][:, 0], y=diagrams[diagrams[:, 2] == 1][:, 1],
                       mode='markers', name='H1 Persistence')
        ])
        st.plotly_chart(fig)

        # Model Comparison
        st.subheader("Model Comparison and Trading Signals")
        time_steps = 60
        X, y, scaler = prepare_data(data, time_steps)

        # QTRL Model
        qtrl_agent = QTRLAgent(fast_mode=fast_mode)
        features = np.column_stack((
            data['Returns'].values,
            data['Volatility'].values,
            data['Sentiment'].values,
            data['RSI'].values,
            np.ones(len(data)) * persistence,
            np.ones(len(data)) * curvature
        ))
        qtrl_signals, qtrl_predicted_prices = qtrl_agent.predict_signals(data, features)
        qtrl_portfolio, qtrl_var, qtrl_cvar = compute_portfolio(data, qtrl_signals, execute_trades=execute_trades)
        qtrl_metrics = compute_metrics(data['close'].values[time_steps:], scaler.inverse_transform(qtrl_predicted_prices[time_steps:].reshape(-1, 1)).flatten(), qtrl_portfolio, data_store['SPY']['Returns'].values[time_steps:])

        # LSTM Model
        lstm_model = LSTMModel(time_steps)
        lstm_model.fit(X, y)
        lstm_signals, lstm_predictions = lstm_model.predict_signals(data, X)
        lstm_portfolio, lstm_var, lstm_cvar = compute_portfolio(data.iloc[time_steps:], lstm_signals, execute_trades=execute_trades)
        lstm_metrics = compute_metrics(data['close'].values[time_steps:], lstm_predictions, lstm_portfolio, data_store['SPY']['Returns'].values[time_steps:])

        # Random Forest Model
        rf_model = RFModel()
        rf_features = np.column_stack((data['Returns'].values[:-1], data['Volatility'].values[:-1], data['Sentiment'].values[:-1], data['RSI'].values[:-1]))
        rf_model.fit(rf_features[time_steps:], data['close'].values[time_steps+1:])
        rf_signals, rf_predictions = rf_model.predict_signals(data, rf_features[time_steps:])
        rf_portfolio, rf_var, rf_cvar = compute_portfolio(data.iloc[time_steps:], rf_signals, execute_trades=execute_trades)
        rf_metrics = compute_metrics(data['close'].values[time_steps:], rf_predictions, rf_portfolio, data_store['SPY']['Returns'].values[time_steps:])

        # GARCH-AI Model
        garch_model = GARCHAIModel()
        garch_model.fit(rf_features[time_steps:], data['close'].values[time_steps+1:])
        garch_signals, garch_predictions = garch_model.predict_signals(data, rf_features[time_steps:])
        garch_portfolio, garch_var, garch_cvar = compute_portfolio(data.iloc[time_steps:], garch_signals, execute_trades=execute_trades)
        garch_metrics = compute_metrics(data['close'].values[time_steps:], garch_predictions, garch_portfolio, data_store['SPY']['Returns'].values[time_steps:])

        # SPY Benchmark
        spy_data = data_store['SPY']
        spy_portfolio = [10000 * spy_data['close'].iloc[i] / spy_data['close'].iloc[0] for i in range(len(spy_data))]
        spy_metrics = compute_metrics(spy_data['close'].values[time_steps:], spy_data['close'].values[time_steps:], spy_portfolio[time_steps:], spy_data['Returns'].values[time_steps:])

        # Display Metrics
        st.subheader("Model Performance Comparison")
        metrics_df = pd.DataFrame({
            'Model': ['QTRL-Pro', 'LSTM', 'Random Forest', 'GARCH-AI', 'SPY Benchmark'],
            'RMSE': [qtrl_metrics['RMSE'], lstm_metrics['RMSE'], rf_metrics['RMSE'], garch_metrics['RMSE'], spy_metrics['RMSE']],
            'MAE': [qtrl_metrics['MAE'], lstm_metrics['MAE'], rf_metrics['MAE'], garch_metrics['MAE'], spy_metrics['MAE']],
            'Sharpe Ratio': [qtrl_metrics['Sharpe Ratio'], lstm_metrics['Sharpe Ratio'], rf_metrics['Sharpe Ratio'], garch_metrics['Sharpe Ratio'], spy_metrics['Sharpe Ratio']],
            'Sortino Ratio': [qtrl_metrics['Sortino Ratio'], lstm_metrics['Sortino Ratio'], rf_metrics['Sortino Ratio'], garch_metrics['Sortino Ratio'], spy_metrics['Sortino Ratio']],
            'Omega Ratio': [qtrl_metrics['Omega Ratio'], lstm_metrics['Omega Ratio'], rf_metrics['Omega Ratio'], garch_metrics['Omega Ratio'], spy_metrics['Omega Ratio']],
            'Calmar Ratio': [qtrl_metrics['Calmar Ratio'], lstm_metrics['Calmar Ratio'], rf_metrics['Calmar Ratio'], garch_metrics['Calmar Ratio'], spy_metrics['Calmar Ratio']],
            'Information Ratio': [qtrl_metrics['Information Ratio'], lstm_metrics['Information Ratio'], rf_metrics['Information Ratio'], garch_metrics['Information Ratio'], 0]
        })
        st.table(metrics_df)

        # SHAP Explainability
        st.subheader("SHAP Explainability (QTRL-Pro)")
        shap_values = compute_shap_values(qtrl_agent, features[:100], data)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=['Returns', 'Volatility', 'Sentiment', 'RSI', 'Persistence', 'Curvature'], y=np.mean(shap_values, axis=0)))
        st.plotly_chart(fig)

        # Visualize Predictions
        st.subheader("Price Predictions")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['timestamp'][time_steps:], y=data['close'][time_steps:], mode='lines', name='Actual Price'))
        fig.add_trace(go.Scatter(x=data['timestamp'][time_steps:], y=qtrl_predicted_prices[time_steps:], mode='lines', name='QTRL-Pro Predicted', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=data['timestamp'][time_steps:], y=lstm_predictions, mode='lines', name='LSTM Predicted', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=data['timestamp'][time_steps:], y=rf_predictions, mode='lines', name='RF Predicted', line=dict(dash='dashdot')))
        fig.add_trace(go.Scatter(x=data['timestamp'][time_steps:], y=garch_predictions, mode='lines', name='GARCH-AI Predicted', line=dict(dash='solid')))
        buy_points = data.iloc[time_steps:][[s == 'Buy' for s in qtrl_signals[time_steps:]]]
        sell_points = data.iloc[time_steps:][[s == 'Sell' for s in qtrl_signals[time_steps:]]]
        fig.add_trace(go.Scatter(x=buy_points['timestamp'], y=buy_points['close'], mode='markers', name='QTRL-Pro Buy', marker=dict(size=10, color='green')))
        fig.add_trace(go.Scatter(x=sell_points['timestamp'], y=sell_points['close'], mode='markers', name='QTRL-Pro Sell', marker=dict(size=10, color='red')))
        st.plotly_chart(fig)

        # Visualize Portfolio Performance
        st.subheader("Portfolio Performance")
        portfolio_df = pd.DataFrame({
            'QTRL-Pro Portfolio': qtrl_portfolio,
            'LSTM Portfolio': lstm_portfolio + [lstm_portfolio[-1]] * (len(qtrl_portfolio) - len(lstm_portfolio)),
            'RF Portfolio': rf_portfolio + [rf_portfolio[-1]] * (len(qtrl_portfolio) - len(rf_portfolio)),
            'GARCH-AI Portfolio': garch_portfolio + [garch_portfolio[-1]] * (len(qtrl_portfolio) - len(garch_portfolio)),
            'SPY Portfolio': spy_portfolio
        }, index=data['timestamp'][1:])
        st.line_chart(portfolio_df)

        # Risk Analysis
        st.subheader("Risk Analysis")
        st.write(f"QTRL-Pro VaR (5%): {qtrl_var:.4f}, CVaR (5%): {qtrl_cvar:.4f}")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=np.diff(qtrl_portfolio) / qtrl_portfolio[:-1], name='QTRL-Pro Returns'))
        fig.add_vline(x=qtrl_var, line_dash='dash', line_color='red', annotation_text='VaR')
        st.plotly_chart(fig)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        mlflow.set_tracking_uri("file://./mlruns")
        with mlflow.start_run():
            asyncio.run(main())