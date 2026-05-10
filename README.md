## Stock Trend Prediction using LSTM

A deep learning project that predicts whether a stock's price will go up or down the next day using an LSTM (Long Short-Term Memory) neural network.


## Overview

This project uses 5+ years of historical stock data (AAPL) along with technical indicators to train a binary classification model. Instead of predicting the exact price (regression), we predict the direction — making it more practical and easier to evaluate.

Model: Stacked LSTM neural network

Task: Binary classification (Up / Down)

Stock: Apple Inc. (AAPL)

Data: 2018 – 2024 (via yfinance)

Platform: Google Colab




## Results

Accuracy : 53.45%

Up Recall : 0.83

Down Recall : 0.17

Baseline (random) : 50.00%

Beating 50% on unseen stock data is considered a genuine result. Even professional algorithmic trading systems targeting 55–60% accuracy are considered strong.



## Features Used
Raw OHLCV data:

Open, High, Low, Close, Volume


## Technical indicators:

SMA (10-day, 50-day) — Simple Moving Average

EMA (20-day) — Exponential Moving Average

RSI — Relative Strength Index

MACD + MACD Signal Line

Bollinger Band Width

Daily Return

Volume Change

Total: 14 input features per time step


## Model Architecture

Input: (60 time steps × 14 features)

LSTM(64 units, return_sequences=True)

Dropout(0.2)

LSTM(32 units)

Dropout(0.2)

Dense(16, ReLU)

Dense(1, Sigmoid)

probability of price going UP


## How It Works

1. Download 6 years of AAPL stock data using yfinance

2. Engineer 14 technical features from raw OHLCV data

3. Scale features using MinMaxScaler (fit on train only — no data leakage)
   
5. Create sliding window sequences of 60 days each
   
7. Train a stacked LSTM with Dropout and BatchNormalization
   
9. Evaluate using accuracy, confusion matrix, and ROC-AUC
    
11. Predict tomorrow's trend on live data

## Libraries Used
1. yfinance - Download stock data

2. pandas / numpy - Data manipulation
 
3. scikit-learn - Preprocessing & evaluation
   
4. TensorFlow / Keras - Build & train LSTM model
   
5. matplotlib / seaborn - Visualization

## Limitations

No sentiment or news data included

Trained only on AAPL — may not generalize to all stocks

Static model — needs periodic retraining as markets evolve

Not intended for real trading use


## Future Improvements

Add news sentiment as a feature using NLP

Try Transformer/Attention-based models

Train on multiple stocks for better generalization

Implement walk-forward validation for more realistic evaluation
