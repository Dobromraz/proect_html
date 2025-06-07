import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

def train_weather_model():
    df_weather = pd.read_csv('processed_csv/proces_weather.csv')
    df_weather.fillna(df_weather.median(numeric_only=True), inplace=True)
    X = df_weather[['Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']]
    y = df_weather['Temperature (C)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def make_weather_prediction(model, input_data):
    input_data_list = [float(i) for i in input_data.split(',')]
    input_df = pd.DataFrame([input_data_list], columns=['Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)'])
    prediction = model.predict(input_df)
    return round(prediction[0], 2)

def train_goal_model():
    df_goal = pd.read_csv('processed_csv/proces_goal.csv')
    df_goal.fillna(df_goal.median(numeric_only=True), inplace=True)
    X = df_goal[['own_goal']]
    y = df_goal['penalty']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def make_goal_prediction(model, input_data):
    input_data_list = [float(i) for i in input_data.split(',')]
    input_df = pd.DataFrame([input_data_list], columns=['own_goal'])
    prediction = model.predict(input_df)
    return "Да" if prediction[0] == 1 else "Нет"

def train_students_model():
    df_students = pd.read_csv('processed_csv/proces_stud.csv')
    df_students.fillna(df_students.median(numeric_only=True), inplace=True)
    X = df_students[['reading score', 'writing score']]
    y = df_students['math score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def make_students_prediction(model, input_data):
    input_data_list = [float(i) for i in input_data.split(',')]
    input_df = pd.DataFrame([input_data_list], columns=['reading score', 'writing score'])
    prediction = model.predict(input_df)
    return int(prediction[0])

def train_credit_model():
    df_credit = pd.read_csv('processed_csv/proces_credit.csv')
    
    # Конвертируем 'Y'/'N' в 1/0
    if 'FLAG_OWN_CAR' in df_credit.columns:
        df_credit['FLAG_OWN_CAR'] = df_credit['FLAG_OWN_CAR'].replace({'Y': 1, 'N': 0})
    if 'FLAG_OWN_REALTY' in df_credit.columns:
        df_credit['FLAG_OWN_REALTY'] = df_credit['FLAG_OWN_REALTY'].replace({'Y': 1, 'N': 0})

    features = ['DAYS_BIRTH', 'CNT_CHILDREN', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
    target = 'AMT_INCOME_TOTAL'
    
    valid_cols = [col for col in features + [target] if col in df_credit.columns]
    df_credit_clean = df_credit[valid_cols].dropna()

    final_features = [col for col in features if col in df_credit_clean.columns]
    
    X = df_credit_clean[final_features]
    y = df_credit_clean[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def make_credit_prediction(model, input_data):
    input_data_list = [float(i) for i in input_data.split(',')]
    input_df = pd.DataFrame([input_data_list], columns=['DAYS_BIRTH', 'CNT_CHILDREN', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY'])
    input_df['DAYS_BIRTH'] = -input_df['DAYS_BIRTH'] * 365
    prediction = model.predict(input_df)
    return f"{prediction[0]:.2f}"

def train_winner_model():
    df_match = pd.read_csv('processed_csv/proces_match.csv')
    df_match.dropna(subset=['home_team', 'away_team', 'home_score', 'away_score'], inplace=True)
    le = LabelEncoder()
    all_teams = pd.concat([df_match['home_team'], df_match['away_team']]).unique()
    le.fit(all_teams)
    df_match['home_team_encoded'] = le.transform(df_match['home_team'])
    df_match['away_team_encoded'] = le.transform(df_match['away_team'])
    X = df_match[['home_team_encoded', 'away_team_encoded']]
    y = (df_match['home_score'] > df_match['away_score']).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def make_winner_prediction(model, input_data):
    input_data_list = [float(i) for i in input_data.split(',')]
    input_df = pd.DataFrame([input_data_list], columns=['home_team_encoded', 'away_team_encoded'])
    prediction = model.predict(input_df)
    return "Победа хозяев" if prediction[0] == 1 else "Победа гостей или ничья"