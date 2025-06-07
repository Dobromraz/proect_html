import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
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
    df_goal = pd.read_csv('processed_csv/goal_model_data_v2.csv')
    df_goal.fillna(0, inplace=True)

    X = df_goal[['goals_scored', 'goals_conceded']]
    y = df_goal['goal_success']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, df_goal

def make_goal_prediction(model, df_goal, player_name, enemy_team):
    row = df_goal[(df_goal['player_name'] == player_name) & (df_goal['enemy_team'] == enemy_team)]

    if row.empty:
        return "Недостаточно данных для прогноза"

    input_df = row[['goals_scored', 'goals_conceded']]
    proba = model.predict_proba(input_df)[0]

    if len(proba) == 1:
        probability = proba[0] if model.classes_[0] == 1 else 0.0
    else:
        probability = proba[1]

    # 🎯 Реалистичное затухание уверенности
    adjusted = (probability ** 0.5) * 0.6
    adjusted = max(0.03, min(adjusted, 0.6))  # максимум 60%, минимум 3%

    return f"{round(adjusted * 100, 2)}%"

def train_students_model():
    df = pd.read_csv('processed_csv/education_career_success.csv')

    # Допустим, ты пока используешь GPA как общий балл — этого недостаточно.
    # Давайте сделаем "заглушку", симулируя 3 предмета (если их нет в CSV):
    df['Russian_Score'] = df['High_School_GPA'] * 20 + 5
    df['Math_Score'] = df['SAT_Score'] % 100
    df['English_Score'] = df['University_GPA'] * 20

    # Используем все три оценки
    X = df[['Russian_Score', 'Math_Score', 'English_Score']]
    y = df['Field_of_Study']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

def make_students_prediction(model, russian, math, english):
    input_df = pd.DataFrame([[russian, math, english]], columns=['Russian_Score', 'Math_Score', 'English_Score'])
    predicted_field = model.predict(input_df)[0]
    avg_score = round((russian + math + english) / 3, 2)

    tips = []
    if russian < 60:
        tips.append("📖 Улучшить знания по русскому языку.")
    if math < 60:
        tips.append("🧮 Повысить уровень математики.")
    if english < 60:
        tips.append("🗣️ Усилить английский язык.")

    result = f"""
🎓 Прогноз профессии:
- Вероятная специальность: **{predicted_field}**
- Средний балл: {avg_score}

📊 Оценки:
- Русский: {russian}
- Математика: {math}
- Английский: {english}

🧠 Рекомендации:
{chr(10).join(tips) if tips else '✅ Отличные показатели — вы на верном пути к успешной карьере!'}
    """.strip()

    return result

def train_credit_model():
    df = pd.read_csv('processed_csv/proces_credit.csv')

    df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].replace({'Y': 1, 'N': 0})
    df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].replace({'Y': 1, 'N': 0})
    df['AGE'] = (-df['DAYS_BIRTH']) // 365

    # Генерация случайных значений для обучения (если нет реальных)
    np.random.seed(42)
    df['CREDIT_TERM'] = np.random.randint(6, 36, len(df))
    df['CREDIT_AMOUNT'] = np.random.randint(50000, 500000, len(df))

    def get_percent(term):
        if term < 12:
            return 0.10
        elif term <= 24:
            return 0.15
        else:
            return 0.20

    df['INTEREST'] = df['CREDIT_TERM'].apply(get_percent)
    df['MONTHLY_PAYMENT'] = (df['CREDIT_AMOUNT'] * (1 + df['INTEREST'])) / df['CREDIT_TERM']

    df['EXPENSES'] = 10000 + df['CNT_CHILDREN'] * 4000 + df['FLAG_OWN_CAR'] * 3000 + df['FLAG_OWN_REALTY'] * 6000
    df['INCOME_MONTHLY'] = df['AMT_INCOME_TOTAL'] / 12
    df['REMAINING'] = df['INCOME_MONTHLY'] - df['MONTHLY_PAYMENT'] - df['EXPENSES']
    df['CREDIT_APPROVED'] = df['REMAINING'] >= 3000

    features = ['AGE', 'CNT_CHILDREN', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                'CREDIT_AMOUNT', 'CREDIT_TERM', 'AMT_INCOME_TOTAL']
    target = 'CREDIT_APPROVED'

    X = df[features]
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

def make_credit_prediction(model, age, children, own_car, own_realty, income_monthly, credit_amount, credit_term):
    own_car_flag = 1 if own_car.upper() == 'Y' else 0
    own_realty_flag = 1 if own_realty.upper() == 'Y' else 0

    annual_income = income_monthly * 12
    percent = 0.10 if credit_term < 12 else (0.15 if credit_term <= 24 else 0.20)
    monthly_payment = (credit_amount * (1 + percent)) / credit_term

    base_expense = 10000
    child_expense = children * 4000
    car_expense = 3000 if own_car_flag else 0
    realty_expense = 6000 if own_realty_flag else 0
    total_expenses = base_expense + child_expense + car_expense + realty_expense

    remaining = income_monthly - monthly_payment - total_expenses
    approved_by_calc = remaining >= 3000

    input_df = pd.DataFrame([{
        'AGE': int(age),
        'CNT_CHILDREN': int(children),
        'FLAG_OWN_CAR': own_car_flag,
        'FLAG_OWN_REALTY': own_realty_flag,
        'CREDIT_AMOUNT': credit_amount,
        'CREDIT_TERM': credit_term,
        'AMT_INCOME_TOTAL': annual_income
    }])
    predicted_label = model.predict(input_df)[0]

    decision = "✅ Кредит одобрен" if approved_by_calc and predicted_label == 1 else "❌ Кредит не одобрен"

    result = f"""
📊 Расчёты:
- 💵 Месячный доход: {income_monthly:,.0f} сом
- 💳 Запрошено: {credit_amount:,.0f} сом на {credit_term} мес
- 📈 Платёж с {int(percent * 100)}%: {monthly_payment:,.0f} сом

🧾 Расходы:
- Базовые нужды: {base_expense}
- Детей: {child_expense}
- Машина: {car_expense}
- Недвижимость: {realty_expense}
- 👉 Всего: {total_expenses:,.0f}

💡 Остаток: {remaining:,.0f} сом

📋 Данные:
- Возраст: {age} | Детей: {children}
- Авто: {'Да' if own_car_flag else 'Нет'} | Недвижимость: {'Да' if own_realty_flag else 'Нет'}

🧠 Вердикт: {decision}
    """.strip()

    return result

def train_winner_model():
    df_match = pd.read_csv('processed_csv/proces_match.csv')
    df_match.dropna(subset=['home_team', 'away_team', 'home_score', 'away_score'], inplace=True)

    # Целевая переменная: победа хозяев
    df_match['home_win'] = (df_match['home_score'] > df_match['away_score']).astype(int)

    # Обучение модели: по названиям, закодированным через LabelEncoder
    le = LabelEncoder()
    all_teams = pd.concat([df_match['home_team'], df_match['away_team']]).unique()
    le.fit(all_teams)
    df_match['home_encoded'] = le.transform(df_match['home_team'])
    df_match['away_encoded'] = le.transform(df_match['away_team'])

    X = df_match[['home_encoded', 'away_encoded']]
    y = df_match['home_win']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, le, df_match

def make_winner_prediction(model, label_encoder, df_match, home_team, away_team):
    try:
        home_encoded = label_encoder.transform([home_team])[0]
        away_encoded = label_encoder.transform([away_team])[0]
    except:
        return "Одна из команд не найдена в данных."

    # Подготовка входных данных
    input_df = pd.DataFrame([[home_encoded, away_encoded]], columns=['home_encoded', 'away_encoded'])
    proba = model.predict_proba(input_df)[0][1]  # шанс победы хозяев

    # Уточнённое понижение для реализма:
    realistic = round(proba * 100 - proba * 100 * 0.25, 2)  # понижаем на 25%
    realistic = max(0, min(realistic, 99))  # ограничим результат

    return f"Шанс победы хозяев ({home_team}) над {away_team}: {realistic}%"
