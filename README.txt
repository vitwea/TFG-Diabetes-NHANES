Predicción de Diabetes Tipo 2 con Machine Learning
==================================================

Este repositorio contiene el Proyecto Final de Grado de Pablo Monclús Radigales (BAC, CESTE, 2025), titulado:
"Predicción de Diabetes Tipo 2 con Machine Learning: Análisis, Modelado y Aplicación Clínica".

Objetivo
--------
Desarrollar un modelo predictivo capaz de identificar el riesgo de desarrollar diabetes tipo 2 a partir de datos clínicos, demográficos y de estilo de vida, utilizando el dataset NHANES y técnicas avanzadas de machine learning.

Resumen del proyecto
--------------------
- La diabetes tipo 2 afecta a más de 422 millones de personas en el mundo.
- Se aplicaron algoritmos de ML para predecir riesgo de diabetes tipo 2.
- El modelo final fue un Random Forest optimizado con Optuna, alcanzando:
  * AUC-ROC: 0.95
  * Sensibilidad: 78.7%
  * Especificidad: 94.7%
- Factores clave: hemoglobina glicosilada, edad y circunferencia de cintura.
- Se desarrolló una aplicación web para profesionales de la salud.
- Se realizó una encuesta sobre percepción de herramientas predictivas basadas en IA.

Estructura del repositorio
--------------------------
- src/ → Código fuente
  * app_diabetes.py → Aplicación web 
  * main.py → Script de entrenamiento del modelo 
  * combinacion_ciclos_NHANES.py → Preprocesamiento de datos

- data/ → Datos procesados
  * nhanes_diabetes_merged.csv

- models/ → Modelos entrenados
  * mejor_modelo_diabetes_final.pkl.zip → Modelo entrenado comprimido (hay que descomprimirlo antes de usarlo)

- graphics/ → Visualizaciones y resultados

- docs/ → Documentación
  * PFG-BAC-MONCLUSRADIGALES-PABLO.pdf (Trabajo escrito completo)

Instalación
-----------
1. Clona el repositorio:
   git clone https://github.com/vitwea/TFG-Diabetes-NHANES.git
   cd TFG-Diabetes-NHANES

2. Crea un entorno virtual e instala dependencias (solo necesario si quieres reentrenar el modelo o trabajar con el código completo):
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Linux/Mac
   pip install -r requirements.txt

Uso
---
⚠️ Nota importante: No es necesario ejecutar `main.py`, ya que el modelo está entrenado y guardado en `models/mejor_modelo_diabetes_final.pkl`.

### Pasos para usar la aplicación web

1. **Descomprime el modelo entrenado**:
   - Ve a la carpeta `models/`
   - Descomprime el archivo `mejor_modelo_diabetes_final.pkl.zip`
   - Asegúrate de que el archivo `mejor_modelo_diabetes_final.pkl` quede disponible en la carpeta `models/`

2. Instala únicamente Streamlit (si solo quieres probar la web, no necesitas instalar todas las dependencias):
   pip install streamlit

3. Lanza la aplicación:
   streamlit run src/app_diabetes.py

4. Abre tu navegador en:
   http://localhost:8501

Licencia
--------
Este proyecto está protegido por derechos de autor de CESTE, Escuela Internacional de Negocios.
Uso académico y personal permitido, no se autoriza la reproducción comercial sin permiso expreso.

Autor
-----
Pablo Monclús Radigales
Bachelor in Applied Computing – CESTE
2025
