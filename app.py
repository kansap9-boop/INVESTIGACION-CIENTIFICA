from flask import Flask, render_template_string, request, jsonify, flash, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_aqui'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuración para archivos subidos
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Crear carpeta de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargar el modelo entrenado
import pickle

try:
    with open('CKD_LR_hp.pkl', 'rb') as f:
        modelo = pickle.load(f)
    print("✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    modelo = None


# Cargar dataset de referencia
try:
    dataset_ref = pd.read_csv('kidney_disease.csv')
    print("Dataset de referencia cargado exitosamente")
except Exception as e:
    print(f"Error al cargar dataset de referencia: {e}")
    dataset_ref = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validar_datos(datos):
    """Valida que los datos estén en rangos apropiados basados en el dataset de entrenamiento"""
    validaciones = {
        'age': (1, 120),
        'sg': (1.005, 1.025),
        'al': (0, 5),
        'su': (0, 5),
        'sc': (0.1, 20.0),
        'bu': (1.0, 200.0),
        'bgr': (50, 500),
        'hemo': (3.0, 20.0),
        'pcv': (10, 60),
        'rc': (2.0, 8.0),
        'wc': (2000, 30000)
    }
    
    errores = []
    
    for campo, (min_val, max_val) in validaciones.items():
        if campo in datos:
            valor = float(datos[campo])
            if valor < min_val or valor > max_val:
                errores.append(f"{campo}: valor {valor} fuera del rango válido ({min_val}-{max_val})")
    
    return errores

# Template HTML principal
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diagnostico DE LA ENFERMEDAD RENAL CRONICA</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
        }
        
        .navbar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1000;
        }
        
        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            color: white;
        }
        
        .navbar ul {
            display: flex;
            list-style: none;
            gap: 1.5rem;
        }
        
        .navbar ul li a {
            color: white;
            text-decoration: none;
            font-weight: 500;
        }
        
        .navbar ul li a:hover {
            color: #ffd700;
        }
        
        .burguer {
            display: none;
            flex-direction: column;
            cursor: pointer;
            background: none;
            border: none;
        }
        
        .burguer span {
            width: 25px;
            height: 3px;
            background: white;
            margin: 3px 0;
        }
        
        .hero-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            padding: 120px 20px 80px;
            margin-top: 60px;
        }
        
        .hero-section h1 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        
        .hero-section p {
            font-size: 1.2rem;
            margin-bottom: 2rem;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .btn-primary {
            display: inline-block;
            background: #ffd700;
            color: #333;
            padding: 12px 30px;
            text-decoration: none;
            border-radius: 25px;
            font-weight: bold;
        }
        
        .btn-primary:hover {
            background: #ffed4e;
        }
        
        .cards-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 60px 20px;
        }
        
        .cards-container h2 {
            text-align: center;
            font-size: 2.2rem;
            margin-bottom: 2rem;
            color: #333;
        }
        
        .cards-container p {
            font-size: 1.1rem;
            text-align: justify;
            margin-bottom: 1.5rem;
            color: #555;
        }
        
        .about-section {
            background: white;
            padding: 60px 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .about-section ul {
            max-width: 600px;
            margin: 0 auto;
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 10px;
        }
        
        .about-section ul li {
            padding: 10px 0;
            border-bottom: 1px solid #ddd;
        }
        
        footer {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            padding: 40px 20px;
        }
        
        footer img {
            max-width: 300px;
            border-radius: 10px;
            margin-top: 20px;
        }
        
        @media (max-width: 768px) {
            .navbar ul {
                display: none;
            }
            .burguer {
                display: flex;
            }
            .hero-section h1 {
                font-size: 1.8rem;
            }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="logo"> GRUPO 3 </div>
        <ul>
            <li><a href="#INICIO">INICIO</a></li>
            <li><a href="#PRESENTACION">PRESENTACION</a></li>
            <li><a href="#MACHINELEARNING">MACHINELEARNING</a></li>
            <li><a href="#GRUPO">GRUPO</a></li>
            <li><a href="/dataset-info">DATASET</a></li>
            <li><a href="/subir-csv">EVALUAR CSV</a></li>
        </ul>
        <button class="burguer">
            <span></span>
            <span></span>
            <span></span>
        </button>
    </nav>
    
    <header id="INICIO" class="hero-section">
        <h1>DIAGNOSTICO DE LA ENFERMEDAD RENAL CRONICA UTILIZANDO MACHINE LEARNING</h1>
        <p>Un proyecto de investigación para la detección temprana de la enfermedad renal crónica realizado 
            por estudiantes de la Universidad Nacional del Santa.</p>
        <a class="btn-primary" href="/evaluar">EVALUAR</a>
    </header>   
    
    <main>
        <section id="PRESENTACION" class="cards-container">
            <h2>PRESENTACION</h2>
            <p>Esta aplicación utiliza un modelo de Stacking que fusiona dos modelos (Tab-Transformer y Long Short Term Memory) para predecir la probabilidad de
                que una persona tenga enfermedad renal crónica y garantizar su diagnóstico de manera temprana, en función de sus respuestas a una serie de preguntas.
                Los usuarios responden preguntas sobre datos clínicos, y el modelo evalúa la probabilidad de tener ERC.

                El modelo procesará los datos proporcionados y generará un resultado basado en un análisis estadístico y patrones detectados previamente. 
                **Advertencia:** Esta herramienta es solo informativa y no sustituye un diagnóstico profesional.
                Se recomienda realizar una consulta profesional para obtener resultados oficiales y completos.
            </p>
        </section>
        
        <section id="MACHINELEARNING" class="cards-container">
            <h2>MACHINE LEARNING</h2>
            <p>El modelo de Stacking combina dos modelos: Tab-Transformer y Long Short Term Memory (LSTM). 
                El Tab-Transformer es un modelo de aprendizaje profundo diseñado para trabajar con datos tabulares, 
                mientras que LSTM es una red neuronal recurrente que maneja secuencias de datos. 
                Juntos, estos modelos permiten una predicción más precisa de la enfermedad renal crónica.</p>
            <p>El modelo se entrena con un conjunto de datos que incluye información clínica y demográfica de pacientes con y sin enfermedad renal crónica.
                Durante el entrenamiento, el modelo aprende a identificar patrones y relaciones entre las características de los pacientes y la presencia de la enfermedad.
                Una vez entrenado, el modelo puede predecir la probabilidad de que un nuevo paciente tenga enfermedad renal crónica basándose en sus respuestas a las preguntas del cuestionario.</p>
            <p>El modelo de Stacking se implementa utilizando la biblioteca de aprendizaje automático Scikit-learn, que proporciona herramientas para la creación y evaluación de modelos de machine learning.</p>
        </section>
        
        <section id="GRUPO" class="about-section">
            <h2>GRUPO</h2>
            <p>Este proyecto fue realizado por estudiantes de la Universidad Nacional del Santa, 
                quienes se unieron para aplicar sus conocimientos en machine learning y contribuir a la detección temprana de la enfermedad renal crónica.</p>
            <p>El equipo está compuesto por:</p>
            <ul>
                <li>Norabuena Melgarejo Joshua</li>
                <li>Robles Cueva Maddox</li>
                <li>Olortigue Sander</li>
                <li>Santa Cruz Maria</li>
            </ul>
        </section>
    </main>
    
    <footer>
        <img src="/static/images/grupo.jpg" alt="GRUPO DE INVESTIGACION">
        <p>Grupo de investigacion de la promocion 2024, Ingeniería de Sistemas e informática
        con el profesor Daza Vergaray</p>
        <p>&copy; 2025 Grupo 3. Todos los derechos reservados.</p>
    </footer>
</body>
</html>
"""

# Template HTML para la página de evaluación
evaluacion_template = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluación - Diagnóstico ERC</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            color: #333;
        }
        
        .navbar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1000;
        }
        
        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            color: white;
        }
        
        .navbar ul {
            display: flex;
            list-style: none;
            gap: 1.5rem;
        }
        
        .navbar ul li a {
            color: white;
            text-decoration: none;
            font-weight: 500;
        }
        
        .container {
            max-width: 800px;
            margin: 100px auto 20px;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        
        .main-title {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2rem;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        
        .form-group input, .form-group select {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        
        .form-group input:focus, .form-group select:focus {
            border-color: #667eea;
            outline: none;
        }
        
        .btn-evaluar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            width: 100%;
            margin-top: 20px;
        }
        
        .btn-evaluar:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .result-container {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 5px solid #667eea;
        }
        
        .result-success {
            border-left-color: #28a745;
            background-color: #d4edda;
        }
        
        .result-warning {
            border-left-color: #ffc107;
            background-color: #fff3cd;
        }
        
        .result-danger {
            border-left-color: #dc3545;
            background-color: #f8d7da;
        }
        
        .btn-back {
            display: inline-block;
            background: #6c757d;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        .warning-box {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
            border-left: 5px solid #f39c12;
        }
        
        .image-container {
            text-align: center;
            margin: 50px 0;
        }
        
        .image-container img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="logo">GRUPO 3</div>
        <ul>
            <li><a href="/">INICIO</a></li>
            <li><a href="/evaluar">EVALUACIÓN</a></li>
            <li><a href="/#MACHINELEARNING">MACHINE LEARNING</a></li>
            <li><a href="/#GRUPO">GRUPO</a></li>
            <li><a href="/dataset-info">DATASET</a></li>
            <li><a href="/subir-csv">EVALUAR CSV</a></li>
        </ul>
    </nav>
    
    <div class="container">
        <a href="/" class="btn-back">← Volver al Inicio</a>
        
        <h1 class="main-title">Predicción de la Enfermedad Renal Crónica basada en medidas diagnósticas</h1>
        
        <div class="image-container">
            <img src="https://img.freepik.com/vector-premium/concepto-tratamiento-enfermedades-renales-medico-medicina_227564-186.jpg" width="400" alt="Diagnóstico Renal">
        </div>
        
        <div class="warning-box">
            <p><strong>⚠️ Advertencia Importante:</strong> Esta herramienta es solo informativa y no sustituye un diagnóstico médico profesional. 
            Consulte siempre con un profesional de la salud para obtener un diagnóstico oficial.</p>
        </div>
        
        <form id="ckdForm" method="POST" action="/procesar_evaluacion">
            <div class="form-group">
                <label for="age">¿Cuál es su edad?</label>
                <input type="number" id="age" name="age" min="0" max="120" value="30" required>
            </div>
            
            <div class="form-group">
                <label for="sg">¿Cuál es su gravedad específica urinaria?</label>
                <select id="sg" name="sg" required>
                    <option value="1.005">1.005</option>
                    <option value="1.010" selected>1.010</option>
                    <option value="1.015">1.015</option>
                    <option value="1.020">1.020</option>
                    <option value="1.025">1.025</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="al">¿Cuál es su nivel de albúmina en orina?</label>
                <select id="al" name="al" required>
                    <option value="0" selected>0</option>
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="3">3</option>
                    <option value="4">4</option>
                    <option value="5">5</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="su">¿Cuál es su nivel de azúcar en orina?</label>
                <select id="su" name="su" required>
                    <option value="0" selected>0</option>
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="3">3</option>
                    <option value="4">4</option>
                    <option value="5">5</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="sc">¿Cuál es su nivel de creatinina sérica? (mg/dl)</label>
                <input type="number" id="sc" name="sc" min="0" max="20" step="0.1" value="1.0" required>
            </div>
            
            <div class="form-group">
                <label for="bu">¿Cuál es su nivel de urea? (mg/dl)</label>
                <input type="number" id="bu" name="bu" min="0" max="200" step="0.1" value="30" required>
            </div>
            
            <div class="form-group">
                <label for="bgr">¿Cuál es su nivel de glucosa en sangre? (mg/dl)</label>
                <input type="number" id="bgr" name="bgr" min="50" max="500" value="100" required>
            </div>
            
            <div class="form-group">
                <label for="hemo">¿Cuál es su nivel de hemoglobina? (g/dl)</label>
                <input type="number" id="hemo" name="hemo" min="0" max="20" step="0.1" value="12" required>
            </div>
            
            <div class="form-group">
                <label for="pcv">¿Cuál es su volumen celular empaquetado (PCV)? (%)</label>
                <input type="number" id="pcv" name="pcv" min="0" max="100" value="40" required>
            </div>
            
            <div class="form-group">
                <label for="rc">¿Cuál es su conteo de glóbulos rojos? (millones/cmm)</label>
                <input type="number" id="rc" name="rc" min="0" max="10" step="0.1" value="4.5" required>
            </div>
            
            <div class="form-group">
                <label for="wc">¿Cuál es su conteo de glóbulos blancos? (células/cmm)</label>
                <input type="number" id="wc" name="wc" min="1000" max="30000" value="7000" required>
            </div>
            
            <div class="form-group">
                <label for="dm">¿Tiene antecedentes de diabetes?</label>
                <select id="dm" name="dm" required>
                    <option value="No" selected>No</option>
                    <option value="Sí">Sí</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="htn">¿Tiene antecedentes de hipertensión?</label>
                <select id="htn" name="htn" required>
                    <option value="No" selected>No</option>
                    <option value="Sí">Sí</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="ane">¿Presenta anemia diagnosticada?</label>
                <select id="ane" name="ane" required>
                    <option value="No" selected>No</option>
                    <option value="Sí">Sí</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="appet">¿Cómo describiría su apetito?</label>
                <select id="appet" name="appet" required>
                    <option value="bueno" selected>Bueno</option>
                    <option value="pobre">Pobre</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="rbc">¿Qué forma tienen sus glóbulos rojos?</label>
                <select id="rbc" name="rbc" required>
                    <option value="normal" selected>Normal</option>
                    <option value="anormal">Anormal</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="pc">¿Cuál es su condición de células del sedimento urinario?</label>
                <select id="pc" name="pc" required>
                    <option value="normal" selected>Normal</option>
                    <option value="anormal">Anormal</option>
                </select>
            </div>
            
            <button type="submit" class="btn-evaluar">EVALUAR RIESGO</button>
        </form>
        
        {% if resultado %}
        <div class="result-container {{ resultado.clase }}">
            <h3>Resultado de la Evaluación</h3>
            <p><strong>{{ resultado.texto }}</strong></p>
            <p><strong>Exactitud del modelo:</strong> {{ resultado.probabilidad }}%</p>
            <br>
            <p><em>Nota: Este resultado no constituye un diagnóstico médico. Consulte a un profesional de la salud.</em></p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/evaluar')
def evaluar():
    return render_template_string(evaluacion_template)

@app.route('/procesar_evaluacion', methods=['POST'])
def procesar_evaluacion():
    try:
        if modelo is None:
            return render_template_string(evaluacion_template, resultado={
                'texto': 'Error: Modelo no disponible',
                'probabilidad': 0,
                'clase': 'result-danger'
            })
        
        # Obtener datos del formulario
        datos = {
            'sg': float(request.form['sg']),
            'al': int(request.form['al']),
            'su': int(request.form['su']),
            'sc': float(request.form['sc']),
            'bu': float(request.form['bu']),
            'bgr': int(request.form['bgr']),
            'hemo': float(request.form['hemo']),
            'pcv': int(request.form['pcv']),
            'rc': float(request.form['rc']),
            'wc': int(request.form['wc']),
            'dm': 1 if request.form['dm'] == 'Sí' else 0,
            'htn': 1 if request.form['htn'] == 'Sí' else 0,
            'ane': 1 if request.form['ane'] == 'Sí' else 0,
            'appet': 0 if request.form['appet'] == 'bueno' else 1,
            'rbc': 0 if request.form['rbc'] == 'normal' else 1,
            'pc': 0 if request.form['pc'] == 'normal' else 1,
            'age': int(request.form['age'])
        }
        
        # Validar datos
        errores = validar_datos(datos)
        if errores:
            return render_template_string(evaluacion_template, resultado={
                'texto': f'Datos fuera de rango: {", ".join(errores)}',
                'probabilidad': 0,
                'clase': 'result-warning'
            })
        
        # Crear DataFrame
        user_input = pd.DataFrame(datos, index=[0])
        
        # Realizar predicción
        prediction = modelo.predict(user_input)
        probability = modelo.predict_proba(user_input)[0]
        
        # Preparar resultado
        if prediction[0] == 1:
            resultado = {
                'texto': 'Alto riesgo de ERC',
                'probabilidad': int(probability[1] * 100),
                'clase': 'result-danger'
            }
        else:
            resultado = {
                'texto': 'Sin indicios de ERC',
                'probabilidad': int(probability[0] * 100),
                'clase': 'result-success'
            }
        
        return render_template_string(evaluacion_template, resultado=resultado)
        
    except Exception as e:
        return render_template_string(evaluacion_template, resultado={
            'texto': f'Error al procesar la evaluación: {str(e)}',
            'probabilidad': 0,
            'clase': 'result-danger'
        })

@app.route('/dataset-info')
def dataset_info():
    """Página de información del dataset"""
    if dataset_ref is None:
        return "<h1>Error: Dataset no disponible</h1><a href='/'>Volver al inicio</a>"
    
    # Calcular estadísticas del dataset
    stats = {
        'total_filas': len(dataset_ref),
        'total_columnas': len(dataset_ref.columns),
        'columnas': list(dataset_ref.columns),
        'tipos_datos': dataset_ref.dtypes.to_dict(),
        'valores_nulos': dataset_ref.isnull().sum().to_dict(),
        'estadisticas_numericas': dataset_ref.describe().to_dict() if len(dataset_ref.select_dtypes(include=[np.number]).columns) > 0 else {}
    }
    
    # Contar casos positivos y negativos si existe columna de target
    if 'classification' in dataset_ref.columns:
        stats['distribucion_clases'] = dataset_ref['classification'].value_counts().to_dict()
    elif 'class' in dataset_ref.columns:
        stats['distribucion_clases'] = dataset_ref['class'].value_counts().to_dict()
    
    dataset_info_template = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Información del Dataset - ERC</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: Arial, sans-serif; background-color: #f5f5f5; color: #333; }
            .navbar {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1rem 2rem; display: flex; justify-content: space-between;
                align-items: center; position: fixed; top: 0; width: 100%; z-index: 1000;
            }
            .logo { font-size: 1.5rem; font-weight: bold; color: white; }
            .navbar ul { display: flex; list-style: none; gap: 1.5rem; }
            .navbar ul li a { color: white; text-decoration: none; font-weight: 500; }
            .container {
                max-width: 1200px; margin: 100px auto 20px; padding: 20px;
                background: white; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }
            .main-title { text-align: center; color: #333; margin-bottom: 30px; font-size: 2rem; }
            .stat-card {
                background: #f8f9fa; padding: 20px; margin: 15px 0;
                border-radius: 8px; border-left: 4px solid #667eea;
            }
            .stat-title { font-size: 1.2rem; font-weight: bold; margin-bottom: 10px; color: #333; }
            .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }
            .stat-item { background: white; padding: 15px; border-radius: 5px; border: 1px solid #ddd; }
            .btn-back {
                display: inline-block; background: #6c757d; color: white;
                padding: 10px 20px; text-decoration: none; border-radius: 5px; margin-bottom: 20px;
            }
            table { width: 100%; border-collapse: collapse; margin-top: 15px; }
            th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f8f9fa; font-weight: bold; }
        </style>
    </head>
    <body>
        <nav class="navbar">
            <div class="logo">GRUPO 3</div>
            <ul>
                <li><a href="/">INICIO</a></li>
                <li><a href="/evaluar">EVALUACIÓN</a></li>
                <li><a href="/dataset-info">DATASET</a></li>
                <li><a href="/subir-csv">EVALUAR CSV</a></li>
            </ul>
        </nav>
        
        <div class="container">
            <a href="/" class="btn-back">← Volver al Inicio</a>
            <h1 class="main-title">Información del Dataset: kidney_disease.csv</h1>
            
            <div class="stat-card">
                <div class="stat-title">📊 Estadísticas Generales</div>
                <div class="stat-grid">
                    <div class="stat-item">
                        <strong>Total de Filas:</strong><br>{{ stats.total_filas }}
                    </div>
                    <div class="stat-item">
                        <strong>Total de Columnas:</strong><br>{{ stats.total_columnas }}
                    </div>
                    {% if stats.distribucion_clases %}
                    <div class="stat-item">
                        <strong>Distribución de Clases:</strong><br>
                        {% for clase, cantidad in stats.distribucion_clases.items() %}
                            {{ clase }}: {{ cantidad }}<br>
                        {% endfor %}
                    </div>
                    {% endif %}
                </div>
            </div>
            
            <div class="stat-card">
                <div class="stat-title">📋 Columnas del Dataset</div>
                <table>
                    <thead>
                        <tr>
                            <th>Columna</th>
                            <th>Tipo de Dato</th>
                            <th>Valores Nulos</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for col in stats.columnas %}
                        <tr>
                            <td>{{ col }}</td>
                            <td>{{ stats.tipos_datos[col] }}</td>
                            <td>{{ stats.valores_nulos[col] }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            {% if stats.estadisticas_numericas %}
            <div class="stat-card">
                <div class="stat-title">📈 Estadísticas de Variables Numéricas</div>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Variable</th>
                                <th>Media</th>
                                <th>Desv. Estándar</th>
                                <th>Mínimo</th>
                                <th>Máximo</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for col, stats_col in stats.estadisticas_numericas.items() %}
                            <tr>
                                <td>{{ col }}</td>
                                <td>{{ "%.2f"|format(stats_col.mean) }}</td>
                                <td>{{ "%.2f"|format(stats_col.std) }}</td>
                                <td>{{ "%.2f"|format(stats_col.min) }}</td>
                                <td>{{ "%.2f"|format(stats_col.max) }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
            {% endif %}
            
            <div class="stat-card">
                <div class="stat-title">ℹ️ Descripción del Dataset</div>
                <p>Este dataset contiene información médica de pacientes para la detección de enfermedad renal crónica (ERC). 
                Incluye variables clínicas y de laboratorio que son utilizadas por el modelo de Machine Learning para realizar predicciones.</p>
                <br>
                <p><strong>Uso:</strong> Los datos se utilizan para entrenar el modelo de Stacking que combina Tab-Transformer y LSTM para la predicción de ERC.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return render_template_string(dataset_info_template, stats=stats)

@app.route('/subir-csv')
def subir_csv():
    """Página para subir archivos CSV"""
    upload_template = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Evaluar CSV - ERC</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: Arial, sans-serif; background-color: #f5f5f5; color: #333; }
            .navbar {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1rem 2rem; display: flex; justify-content: space-between;
                align-items: center; position: fixed; top: 0; width: 100%; z-index: 1000;
            }
            .logo { font-size: 1.5rem; font-weight: bold; color: white; }
            .navbar ul { display: flex; list-style: none; gap: 1.5rem; }
            .navbar ul li a { color: white; text-decoration: none; font-weight: 500; }
            .container {
                max-width: 800px; margin: 100px auto 20px; padding: 20px;
                background: white; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }
            .main-title { text-align: center; color: #333; margin-bottom: 30px; font-size: 2rem; }
            .upload-area {
                border: 2px dashed #667eea; border-radius: 10px; padding: 40px;
                text-align: center; background: #f8f9fa; margin: 20px 0;
            }
            .upload-area:hover { background: #e3f2fd; }
            .btn-upload {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 12px 30px; border: none; border-radius: 25px;
                font-size: 16px; font-weight: bold; cursor: pointer; margin: 10px;
            }
            .btn-back {
                display: inline-block; background: #6c757d; color: white;
                padding: 10px 20px; text-decoration: none; border-radius: 5px; margin-bottom: 20px;
            }
            .warning-box {
                background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px;
                padding: 15px; margin: 20px 0; border-left: 5px solid #f39c12;
            }
            .info-box {
                background: #e8f4ff; border: 1px solid #bee5eb; border-radius: 5px;
                padding: 15px; margin: 20px 0; border-left: 5px solid #3498db;
            }
            .file-input { margin: 15px 0; }
            .file-input input[type="file"] { padding: 10px; border: 1px solid #ddd; border-radius: 5px; width: 100%; }
        </style>
    </head>
    <body>
        <nav class="navbar">
            <div class="logo">GRUPO 3</div>
            <ul>
                <li><a href="/">INICIO</a></li>
                <li><a href="/evaluar">EVALUACIÓN</a></li>
                <li><a href="/dataset-info">DATASET</a></li>
                <li><a href="/subir-csv">EVALUAR CSV</a></li>
            </ul>
        </nav>
        
        <div class="container">
            <a href="/" class="btn-back">← Volver al Inicio</a>
            <h1 class="main-title">Evaluación de Archivos CSV</h1>
            
            <div class="warning-box">
                <p><strong>⚠️ Importante:</strong> Esta herramienta es solo informativa y no sustituye un diagnóstico médico profesional.</p>
            </div>
            
            <div class="info-box">
                <h3>📋 Formato del archivo CSV requerido:</h3>
                <p>El archivo debe contener las siguientes columnas:</p>
                <p><strong>age, sg, al, su, sc, bu, bgr, hemo, pcv, rc, wc, dm, htn, ane, appet, rbc, pc</strong></p>
                <br>
                <p>Los valores categóricos deben estar codificados como:</p>
                <ul style="margin-left: 20px;">
                    <li><strong>dm, htn, ane:</strong> 0 (No) o 1 (Sí)</li>
                    <li><strong>appet:</strong> 0 (bueno) o 1 (pobre)</li>
                    <li><strong>rbc, pc:</strong> 0 (normal) o 1 (anormal)</li>
                </ul>
            </div>
            
            <form action="/procesar-csv" method="post" enctype="multipart/form-data">
                <div class="upload-area">
                    <h3>📁 Seleccionar archivo CSV</h3>
                    <p>Arrastra tu archivo aquí o haz clic para seleccionar</p>
                    <div class="file-input">
                        <input type="file" name="file" accept=".csv" required>
                    </div>
                    <button type="submit" class="btn-upload">EVALUAR ARCHIVO CSV</button>
                </div>
            </form>
            
            <div class="info-box">
                <h3>ℹ️ ¿Qué hace esta herramienta?</h3>
                <p>Esta funcionalidad te permite evaluar múltiples pacientes a la vez subiendo un archivo CSV. 
                El sistema procesará cada fila del archivo y generará predicciones para todos los casos, 
                mostrando un resumen de los resultados.</p>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(upload_template)

@app.route('/procesar-csv', methods=['POST'])
def procesar_csv():
    """Procesar archivo CSV subido"""
    if 'file' not in request.files:
        flash('No se seleccionó ningún archivo')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No se seleccionó ningún archivo')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Leer el archivo CSV
            df = pd.read_csv(file)
            
            # Validar columnas requeridas
            columnas_requeridas = ['age', 'sg', 'al', 'su', 'sc', 'bu', 'bgr', 'hemo', 'pcv', 'rc', 'wc', 'dm', 'htn', 'ane', 'appet', 'rbc', 'pc']
            columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
            
            if columnas_faltantes:
                error_msg = f"Columnas faltantes: {', '.join(columnas_faltantes)}"
                return render_template_string(resultado_csv_template, 
                                            error=error_msg, 
                                            total_filas=0, 
                                            resultados=[])
            
            # Realizar predicciones
            if modelo is None:
                return render_template_string(resultado_csv_template, 
                                            error="Modelo no disponible", 
                                            total_filas=0, 
                                            resultados=[])
            
            predictions = modelo.predict(df[columnas_requeridas])
            probabilities = modelo.predict_proba(df[columnas_requeridas])
            
            # Crear resultados
            resultados = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                resultado = {
                    'fila': i + 1,
                    'prediccion': 'Alto riesgo de ERC' if pred == 1 else 'Sin indicios de ERC',
                    'probabilidad': int(prob[pred] * 100),
                    'clase': 'danger' if pred == 1 else 'success'
                }
                resultados.append(resultado)
            
            # Estadísticas generales
            total_alto_riesgo = sum(1 for r in resultados if r['clase'] == 'danger')
            total_sin_riesgo = len(resultados) - total_alto_riesgo
            
            return render_template_string(resultado_csv_template, 
                                        resultados=resultados,
                                        total_filas=len(df),
                                        total_alto_riesgo=total_alto_riesgo,
                                        total_sin_riesgo=total_sin_riesgo,
                                        error=None)
            
        except Exception as e:
            return render_template_string(resultado_csv_template, 
                                        error=f"Error al procesar el archivo: {str(e)}", 
                                        total_filas=0, 
                                        resultados=[])
    
    return redirect(url_for('subir_csv'))

# Template para mostrar resultados del CSV
resultado_csv_template = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resultados CSV - ERC</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background-color: #f5f5f5; color: #333; }
        .navbar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem 2rem; display: flex; justify-content: space-between;
            align-items: center; position: fixed; top: 0; width: 100%; z-index: 1000;
        }
        .logo { font-size: 1.5rem; font-weight: bold; color: white; }
        .navbar ul { display: flex; list-style: none; gap: 1.5rem; }
        .navbar ul li a { color: white; text-decoration: none; font-weight: 500; }
        .container {
            max-width: 1200px; margin: 100px auto 20px; padding: 20px;
            background: white; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .main-title { text-align: center; color: #333; margin-bottom: 30px; font-size: 2rem; }
        .btn-back {
            display: inline-block; background: #6c757d; color: white;
            padding: 10px 20px; text-decoration: none; border-radius: 5px; margin-bottom: 20px;
        }
        .summary-card {
            background: #f8f9fa; padding: 20px; margin: 20px 0;
            border-radius: 8px; border-left: 4px solid #667eea;
        }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-top: 15px; }
        .summary-item { background: white; padding: 15px; border-radius: 5px; text-align: center; }
        .summary-item.success { border-left: 4px solid #28a745; }
        .summary-item.danger { border-left: 4px solid #dc3545; }
        .summary-item.info { border-left: 4px solid #17a2b8; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: bold; }
        .row-success { background-color: #d4edda; }
        .row-danger { background-color: #f8d7da; }
        .error-box {
            background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;
            padding: 15px; margin: 20px 0; border-left: 5px solid #dc3545;
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="logo">GRUPO 3</div>
        <ul>
            <li><a href="/">INICIO</a></li>
            <li><a href="/evaluar">EVALUACIÓN</a></li>
            <li><a href="/dataset-info">DATASET</a></li>
            <li><a href="/subir-csv">EVALUAR CSV</a></li>
        </ul>
    </nav>
    
    <div class="container">
        <a href="/subir-csv" class="btn-back">← Volver a Subir CSV</a>
        <h1 class="main-title">Resultados de Evaluación CSV</h1>
        
        {% if error %}
        <div class="error-box">
            <h3>❌ Error</h3>
            <p>{{ error }}</p>
        </div>
        {% else %}
        <div class="summary-card">
            <h3>📊 Resumen de Resultados</h3>
            <div class="summary-grid">
                <div class="summary-item info">
                    <h4>{{ total_filas }}</h4>
                    <p>Total de Pacientes</p>
                </div>
                <div class="summary-item success">
                    <h4>{{ total_sin_riesgo }}</h4>
                    <p>Sin Indicios de ERC</p>
                </div>
                <div class="summary-item danger">
                    <h4>{{ total_alto_riesgo }}</h4>
                    <p>Alto Riesgo de ERC</p>
                </div>
            </div>
        </div>
        
        <div class="summary-card">
            <h3>📋 Resultados Detallados</h3>
            <table>
                <thead>
                    <tr>
                        <th>Fila</th>
                        <th>Predicción</th>
                        <th>Probabilidad</th>
                        <th>Estado</th>
                    </tr>
                </thead>
                <tbody>
                    {% for resultado in resultados %}
                    <tr class="row-{{ resultado.clase }}">
                        <td>{{ resultado.fila }}</td>
                        <td>{{ resultado.prediccion }}</td>
                        <td>{{ resultado.probabilidad }}%</td>
                        <td>
                            {% if resultado.clase == 'success' %}
                                ✅ Normal
                            {% else %}
                                ⚠️ Requiere Atención
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <div class="summary-card">
            <h3>ℹ️ Nota Importante</h3>
            <p><strong>Advertencia:</strong> Estos resultados son generados por un modelo de Machine Learning y tienen fines informativos únicamente. 
            No constituyen un diagnóstico médico oficial. Se recomienda encarecidamente consultar con un profesional de la salud 
            para cualquier paciente que muestre alto riesgo de enfermedad renal crónica.</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
  