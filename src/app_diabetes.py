# app_diabetes.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIGURACIÓN INICIAL ===
st.set_page_config(
    page_title="Predictor de Diabetes Tipo 2",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CONTEXTO CIENTÍFICO ===
with st.expander("📚 Base Científica y Referencias", expanded=False):
    st.markdown("""
    **Fuente de Datos:**
    Modelo entrenado con datos del [NHANES](https://wwwn.cdc.gov/nchs/nhanes/) 2013-2018.

    **Variables Clave:**
    - Edad, IMC, glucosa en ayunas, HbA1c
    - Presión arterial, perfil lipídico, antecedentes familiares
    - Variables derivadas: HOMA-IR, ratio glucosa/HbA1c, ratio triglicéridos/HDL

    **Parámetros de Riesgo:**
    - Umbral optimizado mediante F2-Score: 0.4157
    - Métricas clave: AUC 0.9525, Matthews Correlation 0.6228, Balanced Accuracy 0.8671
    """)

# === CARGA DEL MODELO ===
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("models/mejor_modelo_diabetes_final.pkl")
    return modelo

try:
    modelo = cargar_modelo()
except Exception as e:
    st.error(f"Error al cargar el modelo: {str(e)}")
    st.stop()

# === PREPROCESAMIENTO ===
def crear_caracteristicas_derivadas(datos):
    """Calcula las variables derivadas necesarias para el modelo"""
    datos_nuevos = datos.copy()
    
    # Ratio Glucosa/HbA1c
    if 'glu_LBXGLU' in datos.columns and 'ghb_LBXGH' in datos.columns:
        datos_nuevos['derivada_ratio_glu_hba1c'] = datos['glu_LBXGLU'] / datos['ghb_LBXGH']
    
    # Índice HOMA-IR
    if 'glu_LBXGLU' in datos.columns and 'glu_LBXIN' in datos.columns:
        datos_nuevos['derivada_homa_ir'] = (datos['glu_LBXGLU'] * datos['glu_LBXIN']) / 405
    
    # Ratio cintura/altura
    if 'bmx_BMXWAIST' in datos.columns and 'bmx_BMXHT' in datos.columns:
        datos_nuevos['derivada_ratio_cintura_altura'] = datos['bmx_BMXWAIST'] / datos['bmx_BMXHT']
    
    # Ratio triglicéridos/LDL
    if 'trigly_LBXTR' in datos.columns and 'trigly_LBDLDL' in datos.columns:
        datos_nuevos['derivada_ratio_tg_hdl'] = datos['trigly_LBXTR'] / datos['trigly_LBDLDL']
    
    return datos_nuevos

def preprocesar_inputs(inputs):
    """Preprocesa los inputs para que sean compatibles con el modelo"""
    # Mapeo categórico
    mapeo_categorico = {
        'demo_RIDRETH1': {
            'Mexicano-americano': 1, 'Otro hispano': 2,
            'Blanco': 3, 'Negro': 4, 'Otro': 5
        },
        'demo_DMDEDUC2': {
            'Primaria incompleta': 1, 'Primaria completa': 2,
            'Secundaria incompleta': 3, 'Secundaria completa': 4,
            'Universidad incompleta': 5, 'Universidad completa': 6
        }
    }

    datos = pd.DataFrame([inputs])
    
    # Aplicar mapeo categórico
    for var, mapeo in mapeo_categorico.items():
        if var in datos.columns:
            datos[var] = datos[var].map(mapeo)
    
    # Calcular variables derivadas
    datos = crear_caracteristicas_derivadas(datos)
    
    return datos

# === INTERFAZ DE USUARIO ===
st.title('🩺 Sistema Predictivo de Diabetes Tipo 2')
st.markdown("""
**Metodología Validada:**
- Modelo Random Forest optimizado con Optuna
- Balanceo de clases con SMOTE y variables derivadas
- Umbral óptimo: 0.4157 (maximizando F2-Score)
- Matthews Correlation Coefficient: 0.6228, AUC: 0.9525
""")

with st.expander("📋 Introducir datos clínicos", expanded=True):
    # Navegación por pestañas temáticas
    tabs = st.tabs(["👤 Demográficos", "📏 Antropometría", "🩸 Metabólicos",
                   "💓 Cardiovascular", "🍽️ Nutrición", "🏃 Estilo de vida"])

    # === 1. DATOS DEMOGRÁFICOS ===
    with tabs[0]:
        st.subheader("Información Demográfica")
        col1, col2 = st.columns(2)
        with col1:
            edad = st.slider('Edad', 20, 80, 45, 
                           help="Rango validado por NHANES para población adulta")
            genero = st.radio('Género', ['Hombre', 'Mujer'], index=0)
        with col2:
            etnia = st.selectbox('Etnia', [
                'Mexicano-americano', 'Otro hispano', 'Blanco', 'Negro', 'Otro'
            ])
            educacion = st.selectbox('Nivel educativo', [
                'Primaria incompleta', 'Primaria completa',
                'Secundaria incompleta', 'Secundaria completa',
                'Universidad incompleta', 'Universidad completa'
            ])

    # === 2. ANTROPOMETRÍA ===
    with tabs[1]:
        st.subheader("Medidas Antropométricas")
        col1, col2 = st.columns(2)
        with col1:
            altura = st.number_input('Altura (cm)', 140, 210, 170)
            bmi = st.number_input('Índice de masa corporal', 15.0, 50.0, 25.0,
                                 help="Normal: 18.5-24.9, Sobrepeso: 25-29.9, Obesidad: ≥30")
        with col2:
            cintura = st.number_input('Circunferencia de cintura (cm)', 60, 150, 90,
                                     help="Riesgo elevado: >102cm (hombres), >88cm (mujeres)")
            
            # Mostrar el riesgo según cintura/altura
            ratio_cintura_altura = round(cintura / altura, 2) if altura > 0 else 0
            
            # Contenedor con color según el riesgo
            if ratio_cintura_altura > 0.5:
                st.error(f"Ratio cintura/altura: {ratio_cintura_altura} - RIESGO ELEVADO")
            elif ratio_cintura_altura > 0.4:
                st.warning(f"Ratio cintura/altura: {ratio_cintura_altura} - RIESGO MODERADO")
            else:
                st.success(f"Ratio cintura/altura: {ratio_cintura_altura} - RIESGO BAJO")

    # === 3. MARCADORES METABÓLICOS ===
    with tabs[2]:
        st.subheader("Marcadores Metabólicos")
        col1, col2 = st.columns(2)
        with col1:
            glucosa = st.number_input('Glucosa en sangre (mg/dL)', 70, 200, 100,
                                     help="Valores normales: 70-99 mg/dL (ayunas)")
            insulina = st.number_input('Insulina (μU/mL)', 2, 50, 10)
        with col2:
            hba1c = st.number_input('Hemoglobina glicosilada (%)', 4.0, 15.0, 5.5,
                                   help="Valor diagnóstico ≥6.5%")
        
        # Calculamos y mostramos HOMA-IR
        homa_ir = (glucosa * insulina) / 405
        
        # Contenedor con color según el valor
        if homa_ir > 5:
            st.error(f"HOMA-IR: {homa_ir:.2f} - RESISTENCIA SEVERA")
        elif homa_ir > 2.5:
            st.warning(f"HOMA-IR: {homa_ir:.2f} - RESISTENCIA MODERADA")
        else:
            st.success(f"HOMA-IR: {homa_ir:.2f} - SENSIBILIDAD NORMAL")

    # === 4. PERFIL CARDIOVASCULAR ===
    with tabs[3]:
        st.subheader("Perfil Cardiovascular")
        col1, col2 = st.columns(2)
        with col1:
            presion = st.number_input('Presión arterial sistólica (mmHg)', 90, 200, 120,
                                     help="Valor normal <120 mmHg")
            colesterol = st.number_input('Colesterol total (mg/dL)', 100, 400, 200,
                                        help="Valor deseable <200 mg/dL")
        with col2:
            trigliceridos = st.number_input('Triglicéridos (mg/dL)', 50, 500, 150,
                                           help="Valor normal <150 mg/dL")
            ldl = st.number_input('LDL (mg/dL)', 50, 300, 100,
                                 help="Valor óptimo <100 mg/dL")
        
        # Calculamos y mostramos ratio TG/HDL
        hdl = colesterol - ldl - (trigliceridos/5) if ldl > 0 else 0
        ratio_tg_hdl = trigliceridos / hdl if hdl > 0 else 0
        
        # Contenedor con color según el valor
        if ratio_tg_hdl > 4:
            st.error(f"Ratio TG/HDL: {ratio_tg_hdl:.2f} - RIESGO ALTO")
        elif ratio_tg_hdl > 2:
            st.warning(f"Ratio TG/HDL: {ratio_tg_hdl:.2f} - RIESGO MODERADO")
        else:
            st.success(f"Ratio TG/HDL: {ratio_tg_hdl:.2f} - RIESGO BAJO")

    # === 5. NUTRICIÓN ===
    with tabs[4]:
        st.subheader("Información Nutricional")
        calorias = st.number_input('Ingesta calórica diaria (kcal)', 800, 5000, 2000,
                                  help="Calorías totales consumidas diariamente")
        st.markdown("**Distribución de macronutrientes (%)**")
        col1, col2, col3 = st.columns(3)
        with col1:
            carbohidratos_pct = st.slider('Carbohidratos (%)', 45, 65, 55,
                                         help="AMDR recomendado: 45-65% de calorías totales")
        with col2:
            proteinas_pct = st.slider('Proteínas (%)', 10, 35, 20,
                                     help="AMDR recomendado: 10-35% de calorías totales")
        with col3:
            grasas_pct = st.slider('Grasas (%)', 20, 35, 25,
                                  help="AMDR recomendado: 20-35% de calorías totales")
        
        # Validar que los porcentajes sumen 100%
        total_pct = carbohidratos_pct + proteinas_pct + grasas_pct
        if abs(total_pct - 100) > 1:  # Permitir un margen de error de ±1%
            st.warning(f"Los porcentajes de macronutrientes suman {total_pct}%. Ajusta los valores para que la suma sea cercana a 100%.")
        
        # Cálculo de gramos basado en porcentajes
        carbohidratos_g = round((carbohidratos_pct / 100 * calorias) / 4, 1)
        proteinas_g = round((proteinas_pct / 100 * calorias) / 4, 1)
        grasas_g = round((grasas_pct / 100 * calorias) / 9, 1)
        
        # Mostrar equivalencia en gramos con gráfico
        st.markdown("**Equivalencia en gramos:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Carbohidratos", f"{carbohidratos_g}g")
        with col2:
            st.metric("Proteínas", f"{proteinas_g}g")
        with col3:
            st.metric("Grasas", f"{grasas_g}g")
        

    # === 6. ESTILO DE VIDA ===
    with tabs[5]:
        st.subheader("Factores de Estilo de Vida")
        col1, col2 = st.columns(2)
        with col1:
            actividad_fisica = st.radio('¿Realiza actividad física vigorosa?', ['Sí', 'No'], index=1,
                                       help="Al menos 20 min, 3 veces por semana")
            actividad_moderada = st.radio('¿Realiza actividad física moderada?', ['Sí', 'No'], index=0,
                                         help="Al menos 30 min, 5 veces por semana")
        with col2:
            fumador = st.radio('Fumador actual', ['Sí', 'No'], index=1)
        
        # Crear una visualización simple del nivel de actividad física
        nivel_actividad = 0
        if actividad_fisica == 'Sí':
            nivel_actividad += 2
        if actividad_moderada == 'Sí':
            nivel_actividad += 1
        
        st.markdown("**Nivel de actividad física:**")
        if nivel_actividad == 0:
            st.error("⚠️ SEDENTARIO - Aumenta tu actividad física")
        elif nivel_actividad == 1:
            st.warning("🚶 LIGERO - Considera añadir actividad vigorosa")
        elif nivel_actividad == 2:
            st.warning("🏃 MODERADO - Considera añadir actividad moderada regular")
        else:
            st.success("🏆 ACTIVO - ¡Excelente nivel de actividad!")

# === PREDICCIÓN Y RESULTADOS ===
if st.button('🔍 Calcular riesgo'):
    try:
        # Construcción de inputs
        input_dict = {
            'demo_RIDAGEYR': edad,
            'demo_RIAGENDR': 1 if genero == 'Hombre' else 2,
            'demo_RIDRETH1': etnia,
            'demo_DMDEDUC2': educacion,
            'glu_LBXGLU': glucosa,
            'glu_LBXIN': insulina,
            'ghb_LBXGH': hba1c,
            'bmx_BMXBMI': bmi,
            'bmx_BMXWAIST': cintura,
            'bmx_BMXHT': altura,
            'bpx_BPXSY1': presion,
            'tchol_LBXTC': colesterol,
            'trigly_LBXTR': trigliceridos,
            'trigly_LBDLDL': ldl,
            'dr1tot_DR1TKCAL': calorias,
            'dr1tot_DR1TPROT': proteinas_g,
            'dr1tot_DR1TCARB': carbohidratos_g,
            'dr1tot_DR1TTFAT': grasas_g,
            'paq_PAQ605': 1 if actividad_fisica == 'Sí' else 2,
            'paq_PAQ620': 1 if actividad_moderada == 'Sí' else 2,
            'smq_SMQ020': 1 if fumador == 'Sí' else 2
        }

        # Preprocesar datos
        datos_usuario = preprocesar_inputs(input_dict)
        
        # Calcular variables derivadas para mostrar
        homa_ir = (glucosa * insulina) / 405
        ratio_glu_hba1c = glucosa / hba1c
        ratio_cintura_altura = cintura / altura
        ratio_tg_ldl = trigliceridos / ldl if ldl > 0 else 0
        
        # Mostrar variables derivadas calculadas
        st.subheader("🔬 Variables derivadas calculadas")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("HOMA-IR", f"{homa_ir:.2f}",
                     help="Indicador de resistencia a la insulina. Valores >2.5 indican resistencia")
        with col2:
            st.metric("Ratio Glucosa/HbA1c", f"{ratio_glu_hba1c:.2f}",
                     help="Relación entre glucosa y hemoglobina glicosilada")
        with col3:
            st.metric("Ratio Cintura/Altura", f"{ratio_cintura_altura:.2f}",
                     help="Predictor de riesgo metabólico. >0.5 indica mayor riesgo")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Ratio Triglicéridos/LDL", f"{ratio_tg_ldl:.2f}",
                     help="Indicador de perfil lipídico aterogénico")

        # Predicción
        y_proba = modelo.predict_proba(datos_usuario)
        proba = y_proba[0][1]
        riesgo = "Alto riesgo" if proba > 0.4157 else "Bajo riesgo"
        
        # Visualización de resultados
        st.subheader("📊 Resultados Clínicos")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Probabilidad de Diabetes", f"{proba*100:.1f}%")
            st.progress(proba)
            
            if riesgo == "Alto riesgo":
                st.error("⚠️ Recomendación: Consulta urgente con endocrinólogo")
            else:
                st.success("✅ Riesgo dentro de parámetros normales")
            
            st.markdown("""
            **Interpretación del Umbral:**
            - >41.57%: Alto riesgo (Balanced Accuracy: 86.71%)
            - Validación con Matthews Correlation: 0.62
            """)

        with col2:
            # Gráfica de variables importantes con sus valores
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Variables más importantes según los resultados finales
            variables = ['ghb_LBXGH', 'demo_RIDAGEYR', 'bmx_BMXWAIST', 'x2_4.0', 'bmx_BMXBMI']
            valores = [hba1c, edad, cintura, 4.0, bmi]  # Valor temporal para x2_4.0
            nombres = ['HbA1c (%)', 'Edad (años)', 'Cintura (cm)', 'Variable x2_4.0', 'IMC (kg/m²)']
            
            # Rangos normales para evaluación de riesgo
            rangos_normales = {
                'ghb_LBXGH': (4.0, 5.7),  # Normal: <5.7, Prediabetes: 5.7-6.4, Diabetes: >6.5
                'demo_RIDAGEYR': (20, 80),  # No hay rango "normal", solo para visualización
                'bmx_BMXWAIST': (80, 102) if genero == 'Hombre' else (80, 88),
                'bmx_BMXBMI': (18.5, 24.9),
                'x2_4.0': (0, 4.0)  # Rango temporal, ajustar según definición real
            }
            
            # Determinar colores según el rango
            colores = []
            for i, var in enumerate(variables):
                valor = valores[i]
                min_val, max_val = rangos_normales[var]
                
                if var == 'demo_RIDAGEYR':
                    colores.append('blue')  # La edad siempre azul
                elif valor > max_val:
                    colores.append('red')  # Por encima del rango (riesgo)
                elif valor < min_val:
                    colores.append('yellow')  # Por debajo del rango
                else:
                    colores.append('green')  # En rango normal
            
            # Crear gráfico de barras
            bars = ax.barh(nombres, valores, color=colores)
            
            # Añadir etiquetas con valores
            for i, bar in enumerate(bars):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{valores[i]:.1f}", va='center')
            
            ax.set_title('Principales Factores de Riesgo')
            ax.set_xlabel('Valor')
            
            # Añadir líneas para límites normales
            for i, var in enumerate(variables):
                if var != 'demo_RIDAGEYR':
                    _, max_val = rangos_normales[var]
                    y_pos = i
                    ax.axvline(x=max_val, ymin=(y_pos-0.4)/len(variables),
                              ymax=(y_pos+0.4)/len(variables),
                              color='red', linestyle='--', alpha=0.7)
                    ax.text(max_val, y_pos+0.2, f"Límite: {max_val}",
                           color='red', fontsize=8, ha='center')
            
            st.pyplot(fig)
            plt.close()

            # Gráfica de escala de riesgo
            fig, ax = plt.subplots(figsize=(8, 2))
            
            # Crear una escala visual del riesgo
            ax.axhspan(0, 0.5, xmin=0, xmax=0.4157, color='green', alpha=0.3)
            ax.axhspan(0, 0.5, xmin=0.4157, xmax=1, color='red', alpha=0.3)
            
            # Marcar posición actual y umbral
            ax.axvline(x=0.4157, color='black', linestyle='--', label='Umbral: 0.4157')
            ax.plot(proba, 0.25, 'o', markersize=12, color='blue', label=f'Riesgo: {proba:.2f}')
            
            # Etiquetas
            ax.set_yticks([])
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probabilidad')
            ax.set_title('Escala de Riesgo')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
            
            st.pyplot(fig)
            plt.close()

        # Recomendaciones basadas en evidencia
        st.subheader("📌 Guía de Actuación Clínica")
        if riesgo == "Alto riesgo":
            st.markdown("""
            1. **Confirmación diagnóstica:**
               - Prueba de tolerancia oral a la glucosa (75g)
               - Hemoglobina glicosilada repetida
            
            2. **Intervenciones inmediatas:**
               - Dieta mediterránea (evidencia grado A)
               - 150 min/semana de ejercicio aeróbico
               - Control mensual de parámetros metabólicos
            """)
        else:
            st.markdown("""
            1. **Prevención primaria:**
               - Mantener IMC <25 kg/m²
               - Consumo diario de fibra ≥30g
               - Revisiones anuales
            
            2. **Monitorización:**
               - Auto-chequeo glucosa cada 6 meses
               - Perfil lipídico anual
            """)

        # Sección técnica expandible
        with st.expander("🔧 Detalles técnicos del modelo", expanded=False):
            st.markdown("""
            **Hiperparámetros optimizados:**
            - k_neighbors: 6
            - n_estimators: 164
            - max_depth: 28
            - min_samples_split: 14
            - max_features: log2
            
            **Métricas de rendimiento:**
            - F1-Score: 0.6447
            - Average Precision: 0.6864
            - ROC-AUC: 0.9525
            """)

    except Exception as e:
        st.error(f"Error en el procesamiento: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        st.markdown("""
        **Solución de Problemas:**
        1. Verifique que todos los valores ingresados estén dentro de los rangos especificados
        2. Asegúrese de completar todos los campos obligatorios
        3. Si persiste el error, contacte al soporte técnico
        """)

# === REFERENCIAS ===
st.markdown("""
---
**Referencias Científicas:**
1. American Diabetes Association (2023). Standards of Medical Care in Diabetes.
2. Estruch R, et al. (2018). Primary Prevention of Cardiovascular Disease with a Mediterranean Diet. NEJM.
3. Matthews Correlation Coefficient como medida robusta para evaluación de modelos de clasificación desbalanceados.
4. Variables derivadas como HOMA-IR han demostrado ser predictores independientes de diabetes tipo 2.
""")
