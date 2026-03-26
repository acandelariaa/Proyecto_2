# Exploración de los datos

Habiendo hablado sobre la motivación del estudio, pasemos a explorar y contextualizar los datos.

Con el fin de entender la naturaleza del dataset, vamos a cargar los datos que tengamos y entender las variables que lo conforman.

>Nota: el dataset viene compuesto de 3 archivos csv, los cuales son:
- Patient_info (informacion del paciente)
- Tratment_outcome (Resultado del tratamiento)
- Tumor_clinical_data (Datos clinicos)

Sabiendo esto, para manejar los datos de forma mas sencilla, tendremos que hacer un merge, donde juntemos estos 3 archivos haciendolos coincidir por medio de una variable de ID.


>Python Code



```python
from google.colab import drive
drive.mount('/content/drive')

# cargar datos y juntar datasets en un solo dataframe

import pandas as pd

# ── Carga de archivos ──────────────────────────────────────────────
patient_info     = pd.read_csv("/content/drive/MyDrive/Inteligencia_Artificial_1/Proyecto 2/Patient_Info.csv")
clinical_data    = pd.read_csv("/content/drive/MyDrive/Inteligencia_Artificial_1/Proyecto 2/Tumor_Clinical_Data.csv")
treatment_outcome = pd.read_csv("/content/drive/MyDrive/Inteligencia_Artificial_1/Proyecto 2/Treatment_Outcome.csv")

# ── Merge de los tres archivos usando Patient_ID como llave ────────
df = patient_info \
    .merge(clinical_data,      on="Patient_ID", how="inner") \
    .merge(treatment_outcome,  on="Patient_ID", how="inner")

# ── Verificación ───────────────────────────────────────────────────
print(f"✅ Filas:   {df.shape[0]}")
print(f"✅ Columnas: {df.shape[1]}")
print(f"\n📋 Columnas del dataset unificado:\n{list(df.columns)}")
df.head()

```



>Output

| Patient_ID | Age | Gender | Tumor_Type | Tumor_Size_mm | Tumor_Location | Tumor_Grade | Headache | Seizures | Vision_Problems | Treatment_Type | Surgery_Performed | Chemotherapy | Radiation_Therapy | Survival_Months | Tumor_Progression | Recurrence | Outcome_Status | Early_Detection_Label | Treatment_Response |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 61 | Male | Meningioma | 8.07 | Temporal | I | 1 | 0 | 0 | Radiation | 0 | 0 | 1 | 64 | 0 | 1 | Recovered | 1 | Poor |
| 1 | 24 | Female | Meningioma | 45.54 | Occipital | IV | 1 | 1 | 1 | Chemotherapy | 0 | 1 | 1 | 95 | 0 | 0 | Critical | 0 | Good |
| 2 | 37 | Male | Meningioma | 35.03 | Temporal | III | 0 | 0 | 1 | Radiation | 1 | 0 | 1 | 67 | 1 | 0 | Recovered | 0 | Poor |
| 3 | 43 | Female | Meningioma | 43.82 | Frontal | I | 1 | 1 | 0 | Radiation | 1 | 1 | 1 | 81 | 1 | 0 | Critical | 0 | Moderate |
| 4 | 53 | Female | Glioma | 58.28 | Temporal | I | 0 | 1 | 0 | Surgery | 0 | 1 | 0 | 16 | 0 | 1 | Stable | 1 | Good |



Listo!, ya tenemos nuestro dataset unificado en el dataframe `df`, ahora si vamos a comprender un poco del contexto de los datos.

# Glosario de variables

---

## 1. Información del Paciente

| Variable | Tipo | Valores Posibles | Descripción |
|---|---|---|---|
| `Patient_ID` | Entero | ID único (ej. 1, 2, 3...) | Identificador único de cada paciente. Sirve como llave primaria para unir los tres archivos. |
| `Age` | Entero | Rango numérico (ej. 12 – 70+) | Edad del paciente en años al momento del diagnóstico. |
| `Gender` | Categórico | `Male`, `Female` | Género biológico del paciente. |

---

## 2. Datos Clínicos del Tumor

### Características del Tumor

| Variable | Tipo | Valores Posibles | Descripción |
|---|---|---|---|
| `Tumor_Type` | Categórico | `Glioma`, `Meningioma`, `Pituitary` | Tipo histológico del tumor cerebral diagnosticado. |
| `Tumor_Size_mm` | Decimal | Rango continuo (ej. 8.07 – 58.28) | Tamaño del tumor medido en milímetros. |
| `Tumor_Location` | Categórico | `Frontal`, `Temporal`, `Parietal`, `Occipital` | Región del cerebro donde se localiza el tumor. |
| `Tumor_Grade` | Ordinal | `I`, `II`, `III`, `IV` | Grado de malignidad del tumor según clasificación de la OMS. El grado IV es el más agresivo. |

### Síntomas Clínicos
> Las siguientes variables son **binarias**: `1 = Presente`, `0 = Ausente`

| Variable | Tipo | Valores Posibles | Descripción |
|---|---|---|---|
| `Headache` | Binario | `0`, `1` | Indica si el paciente presenta dolores de cabeza frecuentes o severos. |
| `Seizures` | Binario | `0`, `1` | Indica si el paciente ha presentado convulsiones. |
| `Vision_Problems` | Binario | `0`, `1` | Indica si el paciente reporta problemas de visión relacionados al tumor. |
| `Nausea` | Binario | `0`, `1` | Indica la presencia de náuseas como síntoma asociado. |
| `Cognitive_Changes` | Binario | `0`, `1` | Indica cambios cognitivos como pérdida de memoria, confusión o dificultad para concentrarse. |

### Características de Imagen (MRI)

| Variable | Tipo | Valores Posibles | Descripción |
|---|---|---|---|
| `MRI_Intensity_Mean` | Decimal | 0.0 – 1.0 | Intensidad promedio de la señal en la imagen de resonancia magnética. Valores más altos pueden indicar mayor actividad tumoral. |
| `MRI_Texture_Feature` | Decimal | 0.0 – 1.0 | Característica de textura extraída del MRI. Representa la heterogeneidad del tejido tumoral. |
| `Edema_Presence` | Binario | `0`, `1` | Indica si se observa edema (inflamación) alrededor del tumor en la imagen. |
| `Contrast_Enhancement` | Binario | `0`, `1` | Indica si el tumor muestra realce con contraste en el MRI, lo cual sugiere mayor agresividad. |

### Factores de Riesgo

| Variable | Tipo | Valores Posibles | Descripción |
|---|---|---|---|
| `Family_History` | Binario | `0`, `1` | Indica si existe historial familiar de tumores cerebrales. |
| `Smoking_Status` | Categórico | `Yes`, `No` | Indica si el paciente es o fue fumador. |
| `Radiation_Exposure` | Binario | `0`, `1` | Indica si el paciente tuvo exposición previa a radiación ionizante. |
| `Genetic_Risk` | Ordinal | `Low`, `Medium`, `High` | Nivel de riesgo genético estimado para desarrollar tumor cerebral. |

---

## 3. Resultado del Tratamiento

### Tipo de Tratamiento Aplicado

| Variable | Tipo | Valores Posibles | Descripción |
|---|---|---|---|
| `Treatment_Type` | Categórico | `Surgery`, `Chemotherapy`, `Radiation` | Tipo principal de tratamiento que recibió el paciente. |
| `Surgery_Performed` | Binario | `0`, `1` | Indica si se realizó una intervención quirúrgica. |
| `Chemotherapy` | Binario | `0`, `1` | Indica si el paciente recibió quimioterapia. |
| `Radiation_Therapy` | Binario | `0`, `1` | Indica si el paciente recibió radioterapia. |

### Evolución y Pronóstico

| Variable | Tipo | Valores Posibles | Descripción |
|---|---|---|---|
| `Survival_Months` | Entero | Rango numérico (ej. 16 – 106) | Número de meses de supervivencia del paciente desde el diagnóstico. |
| `Tumor_Progression` | Binario | `0`, `1` | Indica si el tumor mostró progresión (crecimiento o avance) durante o después del tratamiento. |
| `Recurrence` | Binario | `0`, `1` | Indica si el tumor reapareció después de haber sido tratado. |
| `Early_Detection_Label` | Binario | `0`, `1` | Indica si el tumor fue detectado en una etapa temprana (`1`) o tardía (`0`). |

### Variable Objetivo (Target)

> Esta es la variable más relevante para construir el modelo de clasificación.

| Variable | Tipo | Valores Posibles | Descripción |
|---|---|---|---|
| `Treatment_Response` | Categórico | `Good`, `Moderate`, `Poor` | Nivel de respuesta del paciente al tratamiento recibido. Puede usarse como variable objetivo alternativa. |

---


Bien, habiendo adentrado en el contexto de los datos, pasemos a el tratamiento de estos mismos.

-----

[Siguiente pagina][]
