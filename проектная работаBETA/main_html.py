from flask import Flask, render_template, request, redirect, url_for, session, flash
import hashlib
from database import login_db, regist_db, check_user_exists, add_prediction_to_history, init_db
from main_pd import make_weather_prediction, make_goal_prediction, make_students_prediction, make_credit_prediction, make_winner_prediction
from main_pd import train_weather_model, train_goal_model, train_students_model, train_credit_model, train_winner_model

app = Flask(__name__)
app.secret_key = 'super_app_prognos'

weather_model = train_weather_model()
goal_model = train_goal_model()
students_model = train_students_model()
credit_model = train_credit_model()
winner_model = train_winner_model()

@app.route('/logout',methods=['GET'])
def logout():
    session.clear()
    flash("Выход успешно совершён!")
    return redirect(url_for('login'))


@app.route('/')
def home():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = login_db(username, password)
        if user:
            session['user_id'] = user['id']
            return redirect(url_for('home'))
        else:
            flash('Неверные данные для входа', 'error')
    return render_template('login.html')

@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if check_user_exists(username, email):
            flash('Такой логин или email уже существует')
            return render_template('register.html')
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        regist_db(username, email, password_hash)
        flash('Регистрация успешно завершена')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/weather', methods=['POST', 'GET'])
def weather():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    if request.method == 'POST':
        humidity = request.form['humidity']
        wind_speed = request.form['wind_speed']
        pressure = request.form['pressure']
        input_data = f"{humidity},{wind_speed},{pressure}"
        prediction = make_weather_prediction(weather_model, input_data)
        result_text = f"Предсказанная температура: {prediction}°C"
        add_prediction_to_history(session['user_id'], 'Погода', result_text)
        return render_template('result.html', prediction=result_text)
    return render_template('weather.html')


@app.route('/goal', methods=['POST', 'GET'])
def goal():
    if not session.get('user_id'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        player_name = request.form['player_name']
        enemy_team = request.form['enemy_team']
        own_goal = int(request.form['own_goal'])
        penalty = int(request.form['penalty'])

        # Загружаем модель v2 и данные
        model, df_goal = train_goal_model()

        # Вызов новой модели
        result_text = make_goal_prediction(model, df_goal, player_name, enemy_team)

        # Добавление в историю
        formatted = f"Игрок: {player_name}, Против: {enemy_team}, Автогол: {own_goal}, Пенальти: {penalty} — шанс: {result_text}"
        add_prediction_to_history(session['user_id'], 'Гол', formatted)

        return render_template('result.html', prediction=formatted)

    return render_template('goal.html')

@app.route('/winner', methods=['POST', 'GET'])
def winner():
    if not session.get('user_id'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        home_team = request.form['home_team']
        away_team = request.form['away_team']

        # Загружаем модель, энкодер и данные (лучше один раз глобально, но для простоты — тут)
        model, label_encoder, df_match = train_winner_model()

        # Получаем прогноз
        result_text = make_winner_prediction(model, label_encoder, df_match, home_team, away_team)

        # Сохраняем в историю
        history_text = f"{home_team} vs {away_team} — {result_text}"
        add_prediction_to_history(session['user_id'], 'Победитель', history_text)

        return render_template('result.html', prediction=history_text)

    return render_template('winner.html')

@app.route('/students', methods=['POST', 'GET'])
def students():
    if not session.get('user_id'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            russian_score = float(request.form['russian_score'])
            math_score = float(request.form['math_score'])
            english_score = float(request.form['english_score'])
        except ValueError:
            return render_template('result.html', prediction="❌ Ошибка: введите корректные числовые значения баллов.")

        # Обучение или загрузка модели
        model = train_students_model()

        # Получение прогноза и рекомендаций
        result_text = make_students_prediction(model, russian_score, math_score, english_score)

        # Сохраняем в историю
        add_prediction_to_history(session['user_id'], 'Профориентация', result_text)

        return render_template('result.html', prediction=result_text)

    return render_template('students.html')

# В файле main_html.py замените старую функцию credit() на эту:

@app.route('/credit', methods=['POST', 'GET'])
def credit():
    if not session.get('user_id'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            children = int(request.form['children'])
            car = request.form['car']  # 'Y' or 'N'
            realty = request.form['realty']  # 'Y' or 'N'
            income_monthly = float(request.form['income'])  # ежемесячная зарплата
            credit_amount = float(request.form['credit_amount'])  # сумма кредита
            credit_term = int(request.form['credit_term'])  # срок кредита в месяцах

        except (ValueError, KeyError):
            return render_template('result.html', prediction="❌ Ошибка: проверьте правильность введённых данных.")

        # Получаем прогноз с объяснением
        result_text = make_credit_prediction(
            credit_model, age, children, car, realty, income_monthly, credit_amount, credit_term
        )

        # Добавим в историю пользователя
        add_prediction_to_history(session['user_id'], 'Одобрение кредита', result_text)

        return render_template('result.html', prediction=result_text)

    return render_template('credit.html')

@app.route('/result', methods=['POST'])
def result():
    prediction = request.form.get('prediction')
    
    if not prediction:
        prediction = "Нет переданного результата."
    
    return render_template('result.html', prediction=prediction)

init_db()
app.run(debug=True)