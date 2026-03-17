9.10. Proyecto Integrador de Aprendizaje Automático
9.10.1. Objetivo
Construir un modelo de clasificación supervisada para predecir si un préstamo emitido por la plataforma Lending Club resultará en default (1) o será pagado completamente (0). El objetivo es comparar el desempeño de los modelos construidos con scikit-learn y PySpark, además de aplicar LIME para interpretar predicciones.

9.10.2. Dataset
Nombre: Lending Club Loan Data (2007–2020)

Fuente: Kaggle Dataset - Lending Club

Tamaño: más de 1,3 millones de registros

Características: variables socioeconómicas, financieras, de crédito, empleo y propósito del préstamo

9.10.3. Target
Variable binaria: loan_status

0 = Fully Paid

1 = Charged Off (default)

Generar la variable default basada en la columna loan_status:

df["default"] = df["loan_status"].apply(lambda x: 1 if x == "Charged Off" else 0)
9.10.4. Estructura del proyecto
9.10.4.1. Exploración de datos (EDA)
Cargar el dataset CSV

Ver distribución de la variable default

Verificar valores faltantes y tipos de datos

Visualizaciones:

Histogramas de variables numéricas

Boxplots por clase

Correlaciones

9.10.4.2. Preprocesamiento
9.10.4.2.1. scikit-learn
Seleccionar variables relevantes: loan_amnt, int_rate, fico_range_high, emp_length, annual_inc, purpose, home_ownership, dti, addr_state, etc.

Codificación de variables categóricas (OneHotEncoder o LabelEncoder)

Escalado con StandardScaler si es necesario

División train_test_split (80/20), estratificada por clase

9.10.4.2.2. PySpark
Leer CSV en SparkSession

StringIndexer y OneHotEncoder para categóricas

VectorAssembler + StandardScaler (opcional)

División randomSplit, estratificada si es posible

9.10.4.3. Modelado con scikit-learn
Usar RandomForestClassifier

Aplicar GridSearchCV sin Pipeline, sobre:

n_estimators: [10, 50, 100]

max_depth: [5, 10, 15]

Métricas:

Accuracy, Precision, Recall, F1-score, ROC AUC

Matriz de confusión

Medir tiempo de entrenamiento y predicción con time.time() o %time

9.10.4.4. Modelado con PySpark
Usar pyspark.ml.classification.RandomForestClassifier

Probar manualmente las mismas combinaciones de hiperparámetros

Evaluar con BinaryClassificationEvaluator

Calcular precisión, F1-score, matriz de confusión (manual si es necesario)

Medir tiempo total de entrenamiento y predicción

9.10.4.5. Interpretabilidad con LIME
Instalar LIME: pip install lime

Seleccionar una o dos instancias clasificadas erróneamente

Aplicar lime.lime_tabular.LimeTabularExplainer

Visualizar:

Variables más influyentes en la predicción

Gráficos locales de explicación

9.10.4.6. Comparación de Resultados
Tabla comparativa con:

Métricas para sklearn vs PySpark

Tiempo de cómputo

Gráficos:

Curva ROC

Comparación de tiempos

9.10.5. Entregable
Un Jupyter Book bien documentado con:

Análisis exploratorio

Preprocesamiento (sklearn y PySpark)

Modelado + tuning de hiperparámetros

Evaluación de métricas

Interpretabilidad con LIME

Reflexión crítica sobre:

¿Qué entorno fue más rápido?

¿Cuál fue más preciso?

¿Cuándo es útil PySpark?

¿Qué aporta LIME?

