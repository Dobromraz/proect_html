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

@app.route('/logout')
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
        own_goal = request.form['own_goal']
        input_data = f"{own_goal}"
        
        prediction = make_goal_prediction(goal_model, input_data)
        result_text = f"Будет ли пенальти: {prediction}"
        add_prediction_to_history(session['user_id'], 'Пенальти', result_text)
        
        return render_template('result.html', prediction=result_text)
    return render_template('goal.html')

@app.route('/winner', methods=['POST', 'GET'])
def winner():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    if request.method == 'POST':
        home_team = request.form['home_team']
        away_team = request.form['away_team']
        input_data = f"{home_team},{away_team}"
        prediction = make_winner_prediction(winner_model, input_data)
        result_text = f"Предсказание победы: {prediction}"
        add_prediction_to_history(session['user_id'], 'Победитель', result_text)
        return render_template('result.html', prediction=result_text)
    return render_template('winner.html')

@app.route('/students', methods=['POST', 'GET'])
def students():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    if request.method == 'POST':
        reading_score = request.form['reading_score']
        writing_score = request.form['writing_score']
        input_data = f"{reading_score},{writing_score}"
        prediction = make_students_prediction(students_model, input_data)
        result_text = f"Предсказанный балл по математике: {prediction}"
        add_prediction_to_history(session['user_id'], 'Успеваемость', result_text)
        return render_template('result.html', prediction=result_text)
    return render_template('students.html')

# В файле main_html.py замените старую функцию credit() на эту:

@app.route('/credit', methods=['POST', 'GET'])
def credit():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    if request.method == 'POST':
        age = request.form['age']
        children = request.form['children']
        car = request.form['car']
        realty = request.form['realty']
        
        input_data = f"{age},{children},{car},{realty}"
        
        prediction = make_credit_prediction(credit_model, input_data)
        result_text = f"Предсказанный годовой доход: {prediction}"
        add_prediction_to_history(session['user_id'], 'Прогноз дохода', result_text)
        
        return render_template('result.html', prediction=result_text)
    return render_template('credit.html')

init_db()
app.run(debug=True)