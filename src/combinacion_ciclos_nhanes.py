import zipfile
import os
zip_path = "/content/nhanes_data.zip"
extract_path = "/content"

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Carpeta descomprimida en:", extract_path)

import os
import pyreadstat

# Define tus ciclos y la ruta base
CYCLES = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
BASE_PATH = './nhanes_data' 

# Variables clave que quieres buscar
key_vars = [
    'LBXGLU', 'LBXIN', 'LBXGH', 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST',
    'BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4', 'BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4',
    'LBXTC', 'LBDHDL', 'LBXTR', 'LBDLDL',
    'DR1TKCAL', 'DR1TPROT', 'DR1TCARB', 'DR1TTFAT',
    'PAQ605', 'PAQ620', 'SMQ020', 'SMD030',
    'SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'DMDEDUC2',
    'DIQ010', 'DIQ160', 'DIQ050'
]

print("----- Exploración de archivos y variables NHANES -----\n")

for cycle in CYCLES:
    cycle_path = os.path.join(BASE_PATH, cycle)
    if not os.path.exists(cycle_path):
        print(f"Carpeta no encontrada para ciclo {cycle}: {cycle_path}")
        continue
    print(f"\nCiclo {cycle}:")
    files = [f for f in os.listdir(cycle_path) if f.lower().endswith('.xpt')]
    if not files:
        print("  No hay archivos .XPT en esta carpeta.")
        continue
    for file in files:
        file_path = os.path.join(cycle_path, file)
        try:
            df, meta = pyreadstat.read_xport(file_path, metadataonly=True)
            columns = list(meta.column_names)
            intersection = [var for var in key_vars if var in columns]
            print(f"  {file}:")
            print(f"    Variables encontradas: {intersection if intersection else 'Ninguna de interés'}")
        except Exception as e:
            print(f"  Error leyendo {file}: {e}")

print("\n----- Fin de la exploración -----")

"""COMBINAR TODOS LOS DATOS EN UN DATASET ÚNICO"""

import pyreadstat
import pandas as pd

# Configuración
CYCLES = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
BASE_PATH = '/content/nhanes_data'

# Variables clave por archivo
var_map = {
    'DEMO': ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'DMDEDUC2'],
    'DIQ': ['SEQN', 'DIQ010', 'DIQ160', 'DIQ050'],
    'GLU': ['SEQN', 'LBXGLU', 'LBXIN'],  # D-G: ambas, H-J: solo LBXGLU
    'INS': ['SEQN', 'LBXIN'],            # Solo H-J
    'GHB': ['SEQN', 'LBXGH'],
    'BMX': ['SEQN', 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST'],
    'BPX': ['SEQN', 'BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4', 'BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4'],
    'TCHOL': ['SEQN', 'LBXTC'],
    'HDL': ['SEQN', 'LBDHDL'],
    'TRIGLY': ['SEQN', 'LBXTR', 'LBDLDL'],
    'DR1TOT': ['SEQN', 'DR1TKCAL', 'DR1TPROT', 'DR1TCARB', 'DR1TTFAT'],
    'PAQ': ['SEQN', 'PAQ605', 'PAQ620'],
    'SMQ': ['SEQN', 'SMQ020', 'SMD030']
}

# Función para cargar los datos de un ciclo
def load_cycle_data(cycle):
    cycle_path = os.path.join(BASE_PATH, cycle)
    dfs = []
    for file_prefix, var_list in var_map.items():
        # Lógica especial para GLU/INS según ciclo
        if file_prefix == 'INS' and cycle in ['D', 'E', 'F', 'G']:
            continue  # No hay archivo INS en estos ciclos
        if file_prefix == 'GLU' and cycle in ['H', 'I', 'J']:
            # En H-J, solo extraer LBXGLU de GLU
            var_list = ['SEQN', 'LBXGLU']
        file_name = f"{file_prefix}_{cycle}.xpt"
        file_path = os.path.join(cycle_path, file_name)
        if os.path.exists(file_path):
            try:
                df, _ = pyreadstat.read_xport(file_path)
                # Solo conservar columnas que existen en el archivo
                cols = [col for col in var_list if col in df.columns]
                if cols:
                    # Evita duplicar SEQN al unir más tarde
                    prefix = file_prefix.lower()
                    # No añadir prefijo a SEQN
                    renamer = {col: f"{prefix}_{col}" if col != 'SEQN' else col for col in cols}
                    df = df[cols].rename(columns=renamer)
                    dfs.append(df)
            except Exception as e:
                print(f"Error leyendo {file_path}: {e}")
    if not dfs:
        return None
    # Unir todos los DataFrames del ciclo por SEQN
    from functools import reduce
    df_merged = reduce(lambda left, right: pd.merge(left, right, on='SEQN', how='outer'), dfs)
    df_merged['CYCLE'] = cycle
    return df_merged

# Cargar y combinar todos los ciclos
all_cycles = []
for cycle in CYCLES:
    print(f"Cargando ciclo {cycle}...")
    df_cycle = load_cycle_data(cycle)
    if df_cycle is not None:
        all_cycles.append(df_cycle)
    else:
        print(f"  ⚠️ No se encontraron datos para el ciclo {cycle}")

# Concatenar todo en un único DataFrame
final_df = pd.concat(all_cycles, ignore_index=True)
print(f"\n¡Datos combinados! Total de registros: {final_df.shape[0]} y variables: {final_df.shape[1]}")

# Mostrar las primeras filas
final_df.head()

"""CONVERTIMOS EN ARCHIVO CSV PARA SU FUTURO USO"""

final_df.to_csv('/data/nhanes_diabetes_merged.csv', index=False)