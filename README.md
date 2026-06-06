# 📉 Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-1A7F37?style=for-the-badge)

Predicción de la fuga de clientes (*churn*) en una empresa de telecomunicaciones mediante Machine Learning. Anticipar qué clientes tienen más probabilidad de abandonar el servicio permite activar estrategias de retención antes de perderlos.

---

## 📖 Descripción

El dataset (**Telco Customer Churn**) recoge información demográfica, detalles del contrato y servicios contratados de cada cliente, junto con la etiqueta de si abandonó o no. El objetivo es entrenar un clasificador que estime esa probabilidad de fuga.

El reto principal es el **desbalanceo de clases** (los clientes que se van son minoría) y la prioridad de maximizar el **recall** sobre la clase positiva: en retención, un falso negativo (no detectar a quien va a irse) cuesta más que un falso positivo.

---

## 🛠️ Tecnologías

- **Python** · Pandas · NumPy
- **scikit-learn** (modelado, `GridSearchCV`, métricas)
- **Matplotlib** · Seaborn (visualización)
- **Jupyter Notebook**

---

## 📂 Estructura del repositorio

```text
Customer-Churn-Prediction/
├── data/
│   ├── raw/                      # Telco_Customer_Churn.csv (dataset original)
│   └── processed/               # Train/test ya divididos y procesados
├── notebooks/
│   ├── 01_EDA.ipynb             # Análisis exploratorio
│   ├── 02_Preprocessing.ipynb   # Limpieza, encoding y split
│   └── 03_Model_Training.ipynb  # Entrenamiento y evaluación
├── src/
│   └── data_preprocessing.py    # Funciones de preprocesado reutilizables
├── requirements.txt
└── README.md
```

---

## 🚀 Cómo ejecutarlo

```bash
# 1. Clonar
git clone https://github.com/nicotimoneda/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Abrir los notebooks en orden (01 → 02 → 03)
jupyter notebook
```

El flujo es: **EDA** (`01`) → **preprocesado y split** (`02`) → **entrenamiento y evaluación** (`03`).

---

## 🔬 Metodología

1. **EDA** — distribución de la variable objetivo, correlaciones y patrones de fuga por tipo de contrato y servicios.
2. **Preprocesado** — codificación de variables categóricas, escalado y división train/test estratificada.
3. **Modelado** — comparación de varios clasificadores (Regresión Logística y Random Forest como líneas base, **XGBoost** como modelo principal), con `GridSearchCV` para la búsqueda de hiperparámetros.
4. **Evaluación** — foco en *recall* y *F1* sobre la clase de fuga, además de la importancia de variables para interpretar el modelo.

---

## 📬 Contacto

Nicolás Timoneda · [nicotimoneda@gmail.com](mailto:nicotimoneda@gmail.com) · [@nicotimoneda](https://github.com/nicotimoneda)

---

## 📝 Licencia

MIT — ver [`LICENSE`](LICENSE).
