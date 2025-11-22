# === 1. IMPORTACIONES ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
f1_score, roc_auc_score, precision_recall_curve,
roc_curve, auc, confusion_matrix, classification_report,
matthews_corrcoef, balanced_accuracy_score,
average_precision_score, cohen_kappa_score)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# === 2. FUNCIONES DE VISUALIZACIÓN ===
def mostrar_distribucion_clases(y, titulo):
    plt.figure(figsize=(8, 5))
    valores, conteos = np.unique(y, return_counts=True)
    plt.bar(['No Diabetes', 'Diabetes'], conteos)
    plt.title(f"Distribución de Clases - {titulo}")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad")
    for i, v in enumerate(conteos):
        plt.text(i, v + 10, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(f"graphics/distribucion_clases_{titulo}.png")
    plt.close()

def crear_diccionario_nombres_descriptivos():
    """Diccionario completo con nombres de variables codificadas"""
    return {
        # Variables demográficas
        'demo_RIDAGEYR': 'Edad (años)',
        'demo_RIAGENDR': 'Género',
        'demo_RIDRETH1': 'Etnia/Raza',
        'demo_DMDEDUC2': 'Nivel educativo',
        # Variables clínicas
        'ghb_LBXGH': 'Hemoglobina glicosilada (%)',
        'glu_LBXGLU': 'Glucosa en ayunas (mg/dL)',
        'glu_LBXIN': 'Insulina en ayunas (uU/mL)',
        'bmx_BMXBMI': 'Índice de masa corporal',
        'bmx_BMXWAIST': 'Circunferencia de cintura (cm)',
        'bmx_BMXHT': 'Altura (cm)',
        'bpx_BPXSY1': 'Presión arterial sistólica',
        'bpx_BPXDI1': 'Presión arterial diastólica',
        'tchol_LBXTC': 'Colesterol total (mg/dL)',
        'trigly_LBXTR': 'Triglicéridos (mg/dL)',
        'trigly_LBDLDL': 'Colesterol LDL (mg/dL)',
        # Variables de estilo de vida
        'dr1tot_DR1TKCAL': 'Calorías diarias (kcal)',
        'dr1tot_DR1TPROT': 'Proteínas diarias (g)',
        'dr1tot_DR1TCARB': 'Carbohidratos diarios (g)',
        'dr1tot_DR1TTFAT': 'Grasas totales diarias (g)',
        'smq_SMQ020': 'Historial de tabaquismo',
        'paq_PAQ605': 'Actividad física intensa',
        'paq_PAQ620': 'Actividad física moderada',
        # Variables derivadas
        'derivada_ratio_glu_hba1c': 'Índice de control glucémico',
        'derivada_homa_ir': 'Índice HOMA-IR',
        'derivada_ratio_cintura_altura': 'Relación cintura-altura',
        'derivada_ratio_tg_hdl': 'Ratio Triglicéridos/HDL',
        # Variables categóricas codificadas
        'demo_RIAGENDR_1': 'Género: Masculino',
        'demo_RIAGENDR_2': 'Género: Femenino',
        'demo_RIDRETH1_1': 'Etnia: Mexicano-Americano',
        'demo_RIDRETH1_2': 'Etnia: Otra Hispana',
        'demo_RIDRETH1_3': 'Etnia: Blanca no Hispana',
        'demo_RIDRETH1_4': 'Etnia: Negra no Hispana',
        'demo_RIDRETH1_5': 'Etnia: Otra/Multiracial',
        'demo_DMDEDUC2_1.0': 'Nivel Educativo: Menos de 9º grado',
        'demo_DMDEDUC2_2.0': 'Nivel Educativo: 9º-11º grado',
        'demo_DMDEDUC2_3.0': 'Nivel Educativo: Graduado secundaria',
        'demo_DMDEDUC2_4.0': 'Nivel Educativo: Algo de universidad',
        'demo_DMDEDUC2_5.0': 'Nivel Educativo: Graduado universitario',
        'smq_SMQ020_1.0': 'Fumador: Sí',
        'smq_SMQ020_2.0': 'Fumador: No',
        'smq_SMQ020_7.0': 'Fumador: Sin respuesta',
        'smq_SMQ020_9.0': 'Fumador: No sabe',
        # Compatibilidad con nombres antiguos OneHotEncoder
        'x0_1': 'Género: Masculino',
        'x0_2': 'Género: Femenino',
        'x1_1': 'Etnia: Mexicano-Americano',
        'x1_2': 'Etnia: Otra Hispana',
        'x1_3': 'Etnia: Blanca no Hispana',
        'x1_4': 'Etnia: Negra no Hispana',
        'x1_5': 'Etnia: Otra/Multiracial',
        'x2_1.0': 'Nivel Educativo: Menos de 9º grado',
        'x2_2.0': 'Nivel Educativo: 9º-11º grado',
        'x2_3.0': 'Nivel Educativo: Graduado secundaria',
        'x2_4.0': 'Nivel Educativo: Algo de universidad',
        'x2_5.0': 'Nivel Educativo: Graduado universitario',
        'x3_1.0': 'Fumador: Sí',
        'x3_2.0': 'Fumador: No',
        'x3_7.0': 'Fumador: Sin respuesta',
        'x3_9.0': 'Fumador: No sabe'
    }

def traducir_nombres_variables(nombres_tecnicos):
    """Traduce automáticamente nombres técnicos a descriptivos"""
    dict_nombres = crear_diccionario_nombres_descriptivos()
    nombres_traducidos = []
    for nombre in nombres_tecnicos:
        # Si existe directamente en el diccionario
        if nombre in dict_nombres:
            nombres_traducidos.append(dict_nombres[nombre])
        else:
            # Para nombres codificados por OneHotEncoder
            partes = nombre.split('_')
            if len(partes) >= 2:
                # Intentar encontrar la variable base
                var_base = '_'.join(partes[:-1]) # Todo excepto el último elemento
                valor = partes[-1] # Último elemento
                # Buscar traducción para la variable base
                if var_base in dict_nombres:
                    base_traducida = dict_nombres[var_base]
                    nombres_traducidos.append(f"{base_traducida}: {valor}")
                    continue
            # Si todo lo demás falla, usar el nombre original
            nombres_traducidos.append(nombre)
    return nombres_traducidos

def graficar_importancia_gini(modelo, nombres, titulo="importancia_gini"):
    if 'classifier' in modelo.named_steps:
        clf = modelo.named_steps['classifier']
    else:
        clf = modelo.named_steps['xgb']
    
    importancias = clf.feature_importances_
    
    # Obtener nombres de características después del preprocesamiento
    cat_indices = []
    cat_names = []
    for name, transformer, cols in modelo.named_steps['preprocessor'].transformers_:
        if name == 'cat' and 'onehot' in transformer.named_steps:
            try:
                cat_names = transformer.named_steps['onehot'].get_feature_names_out()
            except Exception as e:
                print(f"Advertencia: No se pudieron obtener nombres de características: {e}")
                cat_names = [f"cat_feature_{i}" for i in range(len(cols))]
    
    # Combinar nombres de características numéricas y categóricas
    nombres_procesados = []
    for name, _, cols in modelo.named_steps['preprocessor'].transformers_:
        if name == 'num':
            nombres_procesados.extend(cols)
        elif name == 'cat':
            nombres_procesados.extend(cat_names)
    
    # Aplicar traducción de nombres técnicos a descriptivos
    nombres_descriptivos = traducir_nombres_variables(nombres_procesados)
    
    # Crear DataFrame con nombres e importancias
    importancia_df = pd.DataFrame({
        'Feature': nombres_procesados,
        'Nombre_Descriptivo': nombres_descriptivos,
        'Importancia': importancias
    })
    
    # Ordenar por importancia
    importancia_df = importancia_df.sort_values(by='Importancia', ascending=False).head(15)
    
    # Crear gráfica con nombres descriptivos
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importancia_df, x='Importancia', y='Nombre_Descriptivo', palette='viridis')
    plt.title('Top 15 Características por Importancia en Predicción de Diabetes', fontsize=14)
    plt.xlabel('Importancia Relativa', fontsize=12)
    plt.ylabel('Característica', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"graphics/{titulo}.png")
    plt.close()
    
    return importancia_df[['Feature', 'Importancia']]

def mostrar_classification_report(y_true, y_pred, titulo="reporte_clasificacion"):
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).iloc[:-1, :].T
    
    plt.figure(figsize=(8, 4))
    sns.heatmap(df_report, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Reporte de Clasificación")
    plt.tight_layout()
    plt.savefig(f"graphics/{titulo}.png")
    plt.close()
    
    return df_report

def graficar_evolucion_optuna(study):
    plt.figure(figsize=(10, 5))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("Evolución del AUC durante la Optimización con Optuna")
    plt.tight_layout()
    plt.savefig("graphics/optuna_optimization_history.png")
    plt.close()

# === 3. CARGA Y PREPROCESAMIENTO DE DATOS ===
def cargar_preprocesar_datos(ruta_archivo="data/nhanes_diabetes_merged.csv",
                            demo_weight=1.0):
    print("Cargando y preprocesando datos...")
    df = pd.read_csv(ruta_archivo)
    
    # Definir variable objetivo (diabetes)
    df['diabetes_binaria'] = df['diq_DIQ010'].apply(lambda x: 1 if x == 1.0 else 0)
    
    # Definir variables relevantes
    variables_demograficas = [
        'demo_RIDAGEYR', 'demo_RIAGENDR', 'demo_RIDRETH1', 'demo_DMDEDUC2'
    ]
    
    variables_clinicas = [
        'glu_LBXGLU', 'glu_LBXIN', 'ghb_LBXGH', 'bmx_BMXBMI', 'bmx_BMXWAIST',
        'bpx_BPXSY1', 'tchol_LBXTC', 'trigly_LBXTR', 'trigly_LBDLDL',
        'dr1tot_DR1TKCAL', 'dr1tot_DR1TPROT', 'dr1tot_DR1TCARB',
        'dr1tot_DR1TTFAT', 'smq_SMQ020', 'paq_PAQ605', 'paq_PAQ620'
    ]
    
    # Filtrar variables que existen en el dataset
    variables_demo_exist = [col for col in variables_demograficas if col in df.columns]
    variables_clinicas_exist = [col for col in variables_clinicas if col in df.columns]
    variables_todas = variables_demo_exist + variables_clinicas_exist
    
    print(f"Variables demográficas: {variables_demo_exist}")
    print(f"Variables clínicas: {variables_clinicas_exist}")
    
    # Filtrar registros con datos de la variable objetivo
    df_modelo = df[['diabetes_binaria'] + variables_todas].dropna(subset=['diabetes_binaria'])
    
    # Crear variables derivadas (características avanzadas)
    df_modelo = crear_caracteristicas_avanzadas(df_modelo)
    
    # Actualizar las listas de variables
    variables_derivadas = [col for col in df_modelo.columns 
                          if col.startswith('derivada_') and col != 'diabetes_binaria']
    print(f"Variables derivadas creadas: {variables_derivadas}")
    
    X = df_modelo[[col for col in df_modelo.columns if col != 'diabetes_binaria']]
    y = df_modelo['diabetes_binaria']
    
    # Definir tipos de variables
    cat_vars = ['demo_RIAGENDR', 'demo_RIDRETH1', 'demo_DMDEDUC2', 'smq_SMQ020']
    cat_vars = [col for col in cat_vars if col in X.columns]
    num_vars = [col for col in X.columns if col not in cat_vars]
    
    # Preprocesador
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())]), num_vars),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_vars)
    ])
    
    # División en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nDistribución de clases original: {np.bincount(y_train)}")
    mostrar_distribucion_clases(y_train, "Antes_SMOTE")
    
    return X_train, X_test, y_train, y_test, preprocessor, {
        'todas': X.columns.tolist(),
        'demograficas': variables_demo_exist,
        'clinicas': variables_clinicas_exist + variables_derivadas
    }

def crear_caracteristicas_avanzadas(df):
    """Crea características derivadas basadas en conocimiento clínico"""
    df_nuevo = df.copy()
    
    # Ratio Glucosa/HbA1c (importante marcador clínico)
    if 'glu_LBXGLU' in df.columns and 'ghb_LBXGH' in df.columns:
        df_nuevo['derivada_ratio_glu_hba1c'] = df['glu_LBXGLU'] / df['ghb_LBXGH'].replace(0, np.nan)
    
    # Índice HOMA-IR (resistencia a la insulina)
    if 'glu_LBXGLU' in df.columns and 'glu_LBXIN' in df.columns:
        # HOMA-IR = (glucosa * insulina) / 405 (glucosa en mg/dL)
        df_nuevo['derivada_homa_ir'] = (df['glu_LBXGLU'] * df['glu_LBXIN']) / 405
    
    # Ratio cintura/altura (mejor predictor que BMI)
    if 'bmx_BMXWAIST' in df.columns and 'bmx_BMXHT' in df.columns:
        df_nuevo['derivada_ratio_cintura_altura'] = df['bmx_BMXWAIST'] / df['bmx_BMXHT'].replace(0, np.nan)
    
    # Ratio triglicéridos/HDL (marcador de resistencia a la insulina)
    if 'trigly_LBXTR' in df.columns and 'trigly_LBDLDL' in df.columns:
        df_nuevo['derivada_ratio_tg_hdl'] = df['trigly_LBXTR'] / df['trigly_LBDLDL'].replace(0, np.nan)
    
    # Reemplazar infinito y NaN con medianas
    for col in df_nuevo.columns:
        if col.startswith('derivada_'):
            median_val = df_nuevo[col].median()
            df_nuevo[col] = df_nuevo[col].replace([np.inf, -np.inf], np.nan)
            df_nuevo[col] = df_nuevo[col].fillna(median_val)
    
    return df_nuevo

# === 4. FUNCIONES DE EVALUACIÓN MEJORADAS ===
def calcular_metricas_avanzadas(y_true, y_pred, y_proba=None):
    """Calcula métricas avanzadas especialmente útiles para datos desbalanceados"""
    metricas = {}
    
    # Métricas básicas
    metricas['accuracy'] = accuracy_score(y_true, y_pred)
    metricas['precision'] = precision_score(y_true, y_pred)
    metricas['recall'] = recall_score(y_true, y_pred)
    metricas['f1'] = f1_score(y_true, y_pred)
    
    # Métricas avanzadas para desbalance
    metricas['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metricas['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    metricas['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # Métricas basadas en probabilidad (si están disponibles)
    if y_proba is not None:
        metricas['roc_auc'] = roc_auc_score(y_true, y_proba)
        metricas['average_precision'] = average_precision_score(y_true, y_proba)
    
    return metricas

def visualizar_metricas(metricas, titulo="metricas_modelo"):
    """Visualiza las métricas en un gráfico de barras"""
    plt.figure(figsize=(12, 6))
    
    # Seleccionar métricas para visualización
    metricas_viz = {k: v for k, v in metricas.items() 
                   if k not in ['matriz_confusion']}
    
    # Ordenar métricas por valor
    metricas_ordenadas = sorted(metricas_viz.items(), key=lambda x: x[1])
    nombres, valores = zip(*metricas_ordenadas)
    
    # Crear gráfico
    colors = sns.color_palette("viridis", len(nombres))
    bars = plt.barh(nombres, valores, color=colors)
    
    # Añadir etiquetas
    for bar, valor in zip(bars, valores):
        plt.text(valor + 0.01, bar.get_y() + bar.get_height()/2,
                f'{valor:.3f}', va='center')
    
    plt.title('Métricas del Modelo')
    plt.xlim([0, 1.1])
    plt.tight_layout()
    plt.savefig(f"graphics/{titulo}.png")
    plt.close()

# === 5. OPTIMIZACIÓN Y SELECCIÓN DE MODELOS ===
def optimizar_modelo_seleccionado(X_train, y_train, X_test, y_test, preprocessor, tipo_modelo="randomforest"):
    """Optimiza hiperparámetros para el modelo seleccionado"""
    print(f"\nOptimizando modelo {tipo_modelo}...")
    
    def objective(trial):
        if tipo_modelo == "randomforest":
            k_neighbors = trial.suggest_int('k_neighbors', 3, 10)
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'class_weight': 'balanced',
                'random_state': 42
            }
            
            # Crear modelo
            model = ImbPipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(k_neighbors=k_neighbors, random_state=42)),
                ('classifier', RandomForestClassifier(**params))
            ])
            
            # Usar datos sin preprocesar (el pipeline lo hará)
            X_train_prep = X_train
            X_test_prep = X_test
        
        # Entrenar modelo
        model.fit(X_train_prep, y_train)
        
        # Predecir
        y_proba = model.predict_proba(X_test_prep)[:, 1]
        
        # Calcular métrica objetivo (Matthews correlation coefficient)
        threshold = 0.5
        y_pred = (y_proba >= threshold).astype(int)
        score = matthews_corrcoef(y_test, y_pred)
        
        return score
    
    # Crear y ejecutar estudio Optuna
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    print(f"\nMejor valor de Matthews correlation: {study.best_value:.4f}")
    print("Mejores hiperparámetros:", study.best_params)
    
    # Graficar evolución
    graficar_evolucion_optuna(study)
    
    return study

# === 6. EVALUACIÓN FINAL Y UMBRAL ÓPTIMO ===
def evaluar_umbral_optimo(model, X_test, y_test, metrica_objetivo='f2'):
    """Encuentra el umbral óptimo para un modelo dado"""
    print("\nOptimizando umbral de decisión...")
    
    # Obtener probabilidades
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Para modelos en pipeline
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcular curva precision-recall
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    
    # Calcular métricas para cada umbral
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    f2_scores = 5 * (precision * recall) / (4 * precision + recall + 1e-9)
    
    # Seleccionar métrica objetivo
    if metrica_objetivo == 'f1':
        scores = f1_scores
        nombre_metrica = 'F1-Score'
    else:
        scores = f2_scores
        nombre_metrica = 'F2-Score'
    
    # Encontrar umbral óptimo
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    # Predecir con umbral óptimo
    y_pred_opt = (y_proba >= optimal_threshold).astype(int)
    
    # Calcular métricas finales
    metricas_finales = calcular_metricas_avanzadas(y_test, y_pred_opt, y_proba)
    metricas_finales['umbral_optimo'] = optimal_threshold
    metricas_finales['matriz_confusion'] = confusion_matrix(y_test, y_pred_opt)
    
    # Visualizar curva precision-recall
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label='Curva PR')
    plt.scatter(recall[optimal_idx], precision[optimal_idx], c='red', 
                label=f'Umbral óptimo: {optimal_threshold:.3f}')
    plt.xlabel('Recall (Sensibilidad)')
    plt.ylabel('Precisión')
    plt.title(f'Curva Precision-Recall con Umbral Óptimo para {nombre_metrica}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('graphics/curva_precision_recall.png')
    plt.close()
    
    # Visualizar matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(metricas_finales['matriz_confusion'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title('Matriz de Confusión con Umbral Óptimo')
    plt.ylabel('Real')
    plt.xlabel('Predicción')
    plt.savefig('graphics/confusion_matrix_optimal.png')
    plt.close()
    
    print(f"\nUmbral óptimo: {optimal_threshold:.4f}")
    print(f"Balanced Accuracy: {metricas_finales['balanced_accuracy']:.4f}")
    print(f"Matthews Correlation: {metricas_finales['matthews_corrcoef']:.4f}")
    print(f"F1-Score: {metricas_finales['f1']:.4f}")
    
    return metricas_finales, y_pred_opt, y_proba, optimal_threshold

# === 7. FUNCIÓN PRINCIPAL OPTIMIZADA ===
def main():
    try:
        print("=== ANÁLISIS DE PREDICCIÓN DE DIABETES CON NHANES ===")
        
        # 1. Cargar y preprocesar datos
        X_train, X_test, y_train, y_test, preprocessor, vars_dict = cargar_preprocesar_datos()
        
        # 2. Optimizar hiperparámetros del modelo completo
        print("\nOptimizando modelo completo...")
        estudio = optimizar_modelo_seleccionado(
            X_train, y_train, X_test, y_test, preprocessor, tipo_modelo="randomforest")
        
        # 3. Crear modelo final con mejores hiperparámetros
        modelo_final = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(
                k_neighbors=estudio.best_params['k_neighbors'],
                random_state=42)),
            ('classifier', RandomForestClassifier(
                n_estimators=estudio.best_params['n_estimators'],
                max_depth=estudio.best_params['max_depth'],
                min_samples_split=estudio.best_params['min_samples_split'],
                max_features=estudio.best_params['max_features'],
                class_weight='balanced',
                random_state=42))
        ])
        
        # 4. Entrenar modelo final
        print("\nEntrenando modelo final con parámetros optimizados...")
        modelo_final.fit(X_train, y_train)
        
        # 5. Evaluar modelo final con umbral óptimo
        metricas_finales, y_pred_opt, y_proba, umbral_opt = evaluar_umbral_optimo(
            modelo_final, X_test, y_test, metrica_objetivo='f2')
        
        # 6. Graficar importancia de características del modelo final
        importancia_final = graficar_importancia_gini(
            modelo_final, vars_dict['todas'], "importancia_gini_final")
        
        # 7. Guardar modelo final
        joblib.dump(modelo_final, 'models/mejor_modelo_diabetes_final.pkl')
        print("\nModelo final guardado como 'mejor_modelo_diabetes_final.pkl'")
        
        # 8. Resumen final
        print("\n=== RESULTADOS FINALES ===")
        print(f"Modelo: Modelo completo optimizado")
        print(f"Matthews Correlation Coefficient: {metricas_finales['matthews_corrcoef']:.4f}")
        print(f"Balanced Accuracy: {metricas_finales['balanced_accuracy']:.4f}")
        print(f"ROC-AUC: {metricas_finales['roc_auc']:.4f}")
        print(f"Average Precision: {metricas_finales['average_precision']:.4f}")
        print(f"Umbral óptimo F2: {umbral_opt:.4f}")
        
        # Imprimir top 5 características más importantes
        print("\nTop 5 características más importantes:")
        print(importancia_final.head(5))
        
    except Exception as e:
        print(f"Error en la ejecución: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()