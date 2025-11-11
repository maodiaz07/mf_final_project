🧠 ML Final Project — Clasificación Binaria
Este proyecto implementa un flujo completo de Machine Learning para resolver un problema de clasificación binaria, integrando herramientas modernas de control, versionamiento y despliegue:

Python 3.13

Scikit-learn para modelado

DVC (Data Version Control) para versionar datasets

MLflow para seguimiento de experimentos y métricas

Gradio para construir una interfaz interactiva

📂 Estructura del Proyecto
bash
Copiar código
ML_FINAL_PROJECT/
├── data/
│   ├── raw/                 # Datos originales sin procesar
│   │   ├── dataset.csv
│   │   └── dataset.csv.dvc  # Archivo de control de DVC
│   └── processed/           # Conjuntos divididos (train, val, test)
│
├── models/                  # Modelos entrenados y guardados (.joblib)
│   ├── RandomForestClassifier.joblib
│   ├── LogisticRegression.joblib
│   └── SupportVectorMachine.joblib
│
├── src/                     # Código fuente principal
│   ├── add_data.py          # Agrega registros al dataset
│   ├── remove_data.py       # Elimina registros del dataset
│   ├── split_data.py        # Divide datos en train/val/test
│   ├── train.py             # Entrena y registra modelos en MLflow
│   ├── evaluate.py          # Evalúa modelos y métricas
│   ├── app_gradio.py        # Interfaz web con Gradio
│   └── data.py              # Funciones auxiliares de manejo de datos
│
├── notebooks/               # Notebooks opcionales de análisis
├── test/                    # Carpeta reservada para tests unitarios
├── mlruns/                  # Almacenamiento local de MLflow
├── mlflow.db                # Base de datos de tracking (SQLite)
├── .dvc/                    # Configuración interna de DVC
├── .gitignore
├── .dvcignore
├── README.md
└── requirements.txt
⚙️ Instalación
Clonar el repositorio:

bash
Copiar código
git clone <URL_DEL_REPO>
cd ml_final_project
Crear y activar un entorno virtual:

bash
Copiar código
python -m venv venv
venv\Scripts\activate
Instalar dependencias:

bash
Copiar código
pip install -r requirements.txt
🧩 Flujo de Trabajo
Versionamiento de datos con DVC

bash
Copiar código
dvc add data/raw/dataset.csv
dvc push
División de datos

bash
Copiar código
python src/split_data.py
Entrenamiento y seguimiento con MLflow

bash
Copiar código
python src/train.py
Inicia la interfaz de MLflow para explorar resultados:

bash
Copiar código
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1
Luego abre: http://127.0.0.1:5000

Evaluación de métricas

bash
Copiar código
python src/evaluate.py
Interfaz de usuario con Gradio

bash
Copiar código
python src/app_gradio.py
📊 Modelos Entrenados
Modelo	Accuracy	F1-score	Experimento MLflow
RandomForestClassifier	0.8940	0.8940	RandomForest_Experiment
LogisticRegression	0.8543	0.8535	LogisticRegression_Experiment
Support Vector Machine	0.9404	0.9404	SVM_Experiment

🧠 Tecnologías Utilizadas
Python 3.13

pandas, scikit-learn, joblib

DVC — versionamiento de datasets

MLflow — tracking y comparación de experimentos

Gradio — interfaz web interactiva

SQLite — backend local para MLflow

🚀 Ejecución del Proyecto
Ejecutar todos los pasos previos (split, train, evaluate).

Iniciar MLflow UI para visualizar métricas y versiones.

Iniciar Gradio para probar predicciones en tiempo real.

👨‍💻 Autor
Fabián Caicedo
Administrador de empresas y estudiante de Inteligencia Artificial.
Desarrollador de soluciones de automatización e IA aplicada a la gestión operativa y analítica de datos.

🧾 Licencia
Este proyecto es de uso académico y puede adaptarse libremente citando la fuente.