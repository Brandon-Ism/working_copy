"""
# Converted to Python Script

# Data Preprocessing for Runner Injury Prediction

This script preprocesses both daily and weekly time-series data for training GRU/LSTM models to predict runner injuries. The preprocessing steps include:

1. Data loading and initial exploration
2. Data cleaning and handling missing values
3. Feature normalization/standardization
4. Sequence creation for time-series modeling (athlete-aware)
5. Temporal train/validation split
6. Saving preprocessed datasets
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import pickle

np.random.seed(42)

def create_sequences_by_athlete(df, sequence_length, features, scaler=None):
    sequences = []
    labels = []

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df[features])

    for athlete_id in df['Athlete ID'].unique():
        athlete_data = df[df['Athlete ID'] == athlete_id].sort_values('Date')
        scaled_features = scaler.transform(athlete_data[features])

        for i in range(len(scaled_features) - sequence_length):
            seq = scaled_features[i:(i + sequence_length)]
            label = athlete_data['injury'].iloc[i + sequence_length]

            sequences.append(seq)
            labels.append(label)

    return np.array(sequences), np.array(labels), scaler

def preprocess_timeseries(df, sequence_length=7, is_daily=True, train_end_date=None):
    feature_cols = df.columns.drop(['injury', 'Athlete ID', 'Date']).tolist()
    df[feature_cols] = df[feature_cols].replace(-0.01, np.nan)
    df[feature_cols] = df.groupby('Athlete ID')[feature_cols].fillna(method='ffill')
    df[feature_cols] = df[feature_cols].fillna(0)

    if train_end_date is None:
        date_range = df['Date'].max() - df['Date'].min()
        train_end_date = df['Date'].min() + date_range * 0.8

    train_df = df[df['Date'] <= train_end_date].copy()
    val_df = df[df['Date'] > train_end_date].copy()

    X_train, y_train, scaler = create_sequences_by_athlete(
        train_df, sequence_length, feature_cols
    )

    X_val, y_val, _ = create_sequences_by_athlete(
        val_df, sequence_length, feature_cols, scaler
    )

    return X_train, X_val, y_train, y_val, scaler, feature_cols

if __name__ == '__main__':
    day_df = pd.read_csv('../data/day_approach_maskedID_timeseries.csv')
    week_df = pd.read_csv('../data/week_approach_maskedID_timeseries.csv')

    day_df['Date'] = pd.to_datetime(day_df['Date'])
    week_df['Date'] = pd.to_datetime(week_df['Date'])

    print("Daily approach dataset shape:", day_df.shape)
    print("\nWeekly approach dataset shape:", week_df.shape)

    print("\nDaily approach columns:")
    print(day_df.columns.tolist())
    print("\nWeekly approach columns:")
    print(week_df.columns.tolist())

    print("\nDaily approach date range:")
    print(f"Start: {day_df['Date'].min()}")
    print(f"End: {day_df['Date'].max()}")

    X_daily_train, X_daily_val, y_daily_train, y_daily_val, scaler_daily, daily_features = preprocess_timeseries(
        day_df,
        sequence_length=7,
        is_daily=True
    )

    X_weekly_train, X_weekly_val, y_weekly_train, y_weekly_val, scaler_weekly, weekly_features = preprocess_timeseries(
        week_df,
        sequence_length=4,
        is_daily=False
    )

    print("Daily approach preprocessed data shapes:")
    print("X_daily_train shape:", X_daily_train.shape)
    print("X_daily_val shape:", X_daily_val.shape)
    print("y_daily_train shape:", y_daily_train.shape)
    print("y_daily_val shape:", y_daily_val.shape)

    print("\nWeekly approach preprocessed data shapes:")
    print("X_weekly_train shape:", X_weekly_train.shape)
    print("X_weekly_val shape:", X_weekly_val.shape)
    print("y_weekly_train shape:", y_weekly_train.shape)
    print("y_weekly_val shape:", y_weekly_val.shape)

    print("\nDaily approach class distribution:")
    print("Train - Injury rate: {:.2f}%".format(y_daily_train.mean() * 100))
    print("Val - Injury rate: {:.2f}%".format(y_daily_val.mean() * 100))

    print("\nWeekly approach class distribution:")
    print("Train - Injury rate: {:.2f}%".format(y_weekly_train.mean() * 100))
    print("Val - Injury rate: {:.2f}%".format(y_weekly_val.mean() * 100))

    os.makedirs('preprocessed_data', exist_ok=True)

    np.save('preprocessed_data/X_daily_train.npy', X_daily_train)
    np.save('preprocessed_data/X_daily_val.npy', X_daily_val)
    np.save('preprocessed_data/y_daily_train.npy', y_daily_train)
    np.save('preprocessed_data/y_daily_val.npy', y_daily_val)

    np.save('preprocessed_data/X_weekly_train.npy', X_weekly_train)
    np.save('preprocessed_data/X_weekly_val.npy', X_weekly_val)
    np.save('preprocessed_data/y_weekly_train.npy', y_weekly_train)
    np.save('preprocessed_data/y_weekly_val.npy', y_weekly_val)

    with open('preprocessed_data/daily_features.pkl', 'wb') as f:
        pickle.dump(daily_features, f)

    with open('preprocessed_data/weekly_features.pkl', 'wb') as f:
        pickle.dump(weekly_features, f)

    with open('preprocessed_data/scaler_daily.pkl', 'wb') as f:
        pickle.dump(scaler_daily, f)

    with open('preprocessed_data/scaler_weekly.pkl', 'wb') as f:
        pickle.dump(scaler_weekly, f)

    print("Preprocessed data saved to 'preprocessed_data' directory")
    print("\nFiles saved:")
    print(os.listdir('preprocessed_data'))
