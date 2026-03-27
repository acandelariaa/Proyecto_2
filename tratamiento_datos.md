# Tratamiento de los datos

Con el fin de poder manejar el conjunto de datos de forma mas sencilla, transformafermos ciertas variables
a conveniencia y buscaremos valores faltantes y valorea atipicos en caso de que existan.

## Transformación de variables

Recordemos que algunas variables son cualitativas (categoricas), las cuales no podemos trabajar, asi que 
para eso las convertiremos a binarias.

### Genero

Convertiremos la variable de genero donde 1= Mujer y 0=Hombre, esto para poder tener un dato numerico
con el que podamos trabajar en vez de tener un dato categorico.


>Python Code




```python

# Convertir Gender a variable binaria (1 = Female, 0 = Male)
df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})

# Verificación
print(df['Gender'].value_counts())
print(f"\nTipo de dato: {df['Gender'].dtype}")
```


>Output


```text
Gender
1    107
0     93
Name: count, dtype: int64

Tipo de dato: int64

```



Vemos que tenemos 107 mujeres y 93 hombres en el dataset ya nuevo con la variable binaria.
Ahora, del mismo modo utilizemos one-hot-encoding para convertir a binario la variable de `Tumor_type`, donde tendremos 3 columnas derivadas
de esa donde tendra la siguient logica 1= presente, 0= ausente.


>Python code



```python
# One-Hot Encoding con valores 0 y 1 (enteros)
df = pd.get_dummies(df, columns=['Tumor_Type'], prefix='Tumor').astype({
    'Tumor_Glioma': int,
    'Tumor_Meningioma': int,
    'Tumor_Pituitary': int
})

# Verificación
print(df[['Tumor_Glioma', 'Tumor_Meningioma', 'Tumor_Pituitary']].head(10))
print(f"\nTipo de datos:\n{df[['Tumor_Glioma', 'Tumor_Meningioma', 'Tumor_Pituitary']].dtypes}")

```


>Output

```text
   Tumor_Glioma  Tumor_Meningioma  Tumor_Pituitary
0             0                 1                0
1             0                 1                0
2             0                 1                0
3             0                 1                0
4             1                 0                0
5             0                 1                0
6             0                 1                0
7             0                 0                1
8             0                 1                0
9             0                 0                1

Tipo de datos:
Tumor_Glioma        int64
Tumor_Meningioma    int64
Tumor_Pituitary     int64
dtype: object

```

Listo!, ahora vemos que la variable `Tumor_type` ya no esta, en su lugar, hay otras 3 variables
donde se describe si ese tipo de tumor esta presente, 
es cierto que ahora generamos mas variables 
con las cuales trabajar, pero, ahora podemos tratarlas de forma mas sencilla


### Localización del Tumor

Tenemos una variable llamada `Tumor_location`, la cual nos describe la ubicacion del tumor en el cerebro, 
a esta tambien le aplicaremos el metodo de One-hot-encoding, para generar distintas variables a partir de sus clases.


>Python Code



```python
# One-Hot Encoding para Tumor_Location con valores 0 y 1
df = pd.get_dummies(df, columns=['Tumor_Location'], prefix='Location')
df = df.apply(lambda col: col.astype(int) if col.dtype == bool else col)

# Verificación
location_cols = ['Location_Frontal', 'Location_Occipital', 'Location_Parietal', 'Location_Temporal']
print(df[location_cols].head(10))
print(f"\nTipo de datos:\n{df[location_cols].dtypes}")

```


>Output


```text
   Location_Frontal  Location_Occipital  Location_Parietal  Location_Temporal
0                 0                   0                  0                  1
1                 0                   1                  0                  0
2                 0                   0                  0                  1
3                 1                   0                  0                  0
4                 0                   0                  0                  1
5                 0                   0                  0                  1
6                 0                   0                  1                  0
7                 0                   1                  0                  0
8                 0                   0                  1                  0
9                 0                   0                  1                  0

Tipo de datos:
Location_Frontal      int64
Location_Occipital    int64
Location_Parietal     int64
Location_Temporal     int64
dtype: object

```

Perfecto!, ahora tenemos 4 variables que nos describen la ubicación del tumor, asi mismo,
la varible `Tumor_location`, ha desaparecido.

## Otras variables

Siguiendo esta misma lógica, haremos el mismo proceso a las variables de:
- `Tumor_Grade`
- `Genetic_Risk`
- `Treatment_Type`
- `Smoking_Status`

Con la consideración de que excluiremos ciertas variables como:
- `Patient_ID`
- `Outcome_Status`
- `Treatment_Response`

Esto ya que el ID no nos sirve para analizar, pues solo es un identificador.
En cuanto a `Treatment_Response`, es nuestra **variable objetivo**, por lo que
la guardaremos por separado antes de cualquier transformación.

Finalmente, `Outcome_Status` también se excluye ya que es otra variable de salida
y podría estar correlacionada con `Treatment_Response`, lo que causaría
**fuga de datos** si se incluyera como feature.

Por lo tanto, guardaremos `Treatment_Response` en la variable `y`, y el resto
de las features en un nuevo dataframe llamado `df_features`.


>Python Code




```python
# PASO 1 — Guardar y quitar objetivos
y = df['Treatment_Response']
df_features = df.drop(columns=['Patient_ID', 'Outcome_Status', 'Treatment_Response'])

# PASO 2 — Convertir Smoking_Status a binaria
df_features['Smoking_Status'] = df_features['Smoking_Status'].map({'Yes': 1, 'No': 0})

# PASO 3 — One-Hot Encoding solo a features
variables_categoricas = ['Tumor_Grade', 'Genetic_Risk', 'Treatment_Type']
prefijos              = ['Grade',        'Risk',          'Treatment']

df_features = pd.get_dummies(df_features, columns=variables_categoricas, prefix=prefijos)
df_features = df_features.apply(lambda col: col.astype(int) if col.dtype == bool else col)

# ── Verificación — mostrar columnas generadas por el One-Hot Encoding ──
print("✅ Columnas generadas por One-Hot Encoding:\n")
for prefijo in prefijos:
    cols = [col for col in df_features.columns if col.startswith(prefijo)]
    print(f"  {variables_categoricas[prefijos.index(prefijo)]}:")
    for col in cols:
        print(f"    → {col}")
```


>Output


```text
✅ Columnas generadas por One-Hot Encoding:

  Tumor_Grade:
    → Grade_I
    → Grade_II
    → Grade_III
    → Grade_IV
  Genetic_Risk:
    → Risk_High
    → Risk_Low
    → Risk_Medium
  Treatment_Type:
    → Treatment_Chemotherapy
    → Treatment_Radiation
    → Treatment_Surgery
```



Bien, ya tenemos nuestras variables convertidas a binarias, el siguiente paso 
es buscar valores atipicos y valores faltantes, en caso de que existan, los trataremos de forma correspondiente.


### Datos Faltantes

Busquemos si es que hay huecos en el dataset, verifiquemos si faltan datos por columna.


>Python Code


```python
# ── Verificación de valores faltantes ─────────────────────────────
missing = df_features.isnull().sum()
missing_pct = (missing / len(df_features)) * 100

# Crear tabla resumen
missing_df = pd.DataFrame({
    'Valores Faltantes': missing,
    'Porcentaje (%)':    missing_pct.round(2)
}).sort_values('Porcentaje (%)', ascending=False)

# Mostrar solo columnas con al menos 1 faltante
faltantes = missing_df[missing_df['Valores Faltantes'] > 0]

if faltantes.empty:
    print(" No hay valores faltantes en el dataset.")
else:
    print(f"⚠️ Se encontraron valores faltantes en {len(faltantes)} columnas:\n")
    print(faltantes.to_string())
```


Excelente, no hay valores faltantes, esto puede deberse a que el dataset se rescato de `kaggle`, donde 
seguramente la persona que publico el conjunto de datos ya habia echo ciertos cambios al dataset, posiblemente
hasta quito datos atipicos o relleno con algun metodo de imputación, de cualquier forma, es bueno verificar. 
Ahora pasemos a ver si hay datos atipicos.


### Outliers

Ahora, verifiquemos si hay datos atipicos, esto lo haremos con el metodo de tukey, donde definiremos un rango
intercuartil de datos Q1-Q3, donde los datos que caigan fuera de este rango, seran considerados atipicos.


>Python Code



```python
# ── Detección de Outliers con método de Tukey (IQR) ───────────────

# Solo aplicar a columnas numéricas continuas
cols_numericas = df_features.select_dtypes(include=['float64', 'int64']).columns

# Excluir columnas binarias (solo tienen 0 y 1, no tiene sentido buscar outliers)
cols_binarias  = [col for col in cols_numericas if df_features[col].nunique() == 2]
cols_continuas = [col for col in cols_numericas if col not in cols_binarias]

print(f"📋 Columnas analizadas ({len(cols_continuas)}): {cols_continuas}\n")

# ── Calcular outliers por columna ──────────────────────────────────
resultados = []
outlier_mask = pd.Series(False, index=df_features.index)  # Para % del dataset

for col in cols_continuas:
    Q1  = df_features[col].quantile(0.25)
    Q3  = df_features[col].quantile(0.75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    es_outlier = (df_features[col] < limite_inferior) | (df_features[col] > limite_superior)
    outlier_mask = outlier_mask | es_outlier  # Acumular filas con al menos un outlier

    n_outliers = es_outlier.sum()
    pct_col    = (n_outliers / len(df_features)) * 100

    resultados.append({
        'Columna':            col,
        'Q1':                 round(Q1, 3),
        'Q3':                 round(Q3, 3),
        'IQR':                round(IQR, 3),
        'Límite Inferior':    round(limite_inferior, 3),
        'Límite Superior':    round(limite_superior, 3),
        'Outliers':           n_outliers,
        '% por Columna':      round(pct_col, 2)
    })

# ── Resultados por columna ─────────────────────────────────────────
resultados_df = pd.DataFrame(resultados).sort_values('% por Columna', ascending=False)
print("📊 Outliers por columna:")
print(resultados_df.to_string(index=False))

# ── Porcentaje del dataset ─────────────────────────────────────────
filas_con_outlier = outlier_mask.sum()
pct_dataset       = (filas_con_outlier / len(df_features)) * 100

print(f"\n📐 Filas con al menos un outlier: {filas_con_outlier} de {len(df_features)}")
print(f"📐 Porcentaje del dataset afectado: {pct_dataset:.2f}%")

```


```text
📋 Columnas analizadas (5): ['Age', 'Tumor_Size_mm', 'MRI_Intensity_Mean', 'MRI_Texture_Feature', 'Survival_Months']

📊 Outliers por columna:
            Columna     Q1     Q3    IQR  Límite Inferior  Límite Superior  Outliers  % por Columna
                Age 24.000 63.000 39.000          -34.500          121.500         0            0.0
      Tumor_Size_mm 19.742 45.795 26.053          -19.336           84.874         0            0.0
 MRI_Intensity_Mean  0.394  0.681  0.287           -0.036            1.110         0            0.0
MRI_Texture_Feature  0.318  0.728  0.410           -0.297            1.343         0            0.0
    Survival_Months 33.750 96.250 62.500          -60.000          190.000         0            0.0

📐 Filas con al menos un outlier: 0 de 200
📐 Porcentaje del dataset afectado: 0.00%

```

Buena noticia, parece que no hay outliers, como lo sospechabamos, el autor del dataset se tomo la molestia 
de limpiar y preparar de forma parcial los datos, de cualquier modo, es bueno verificar.


### Split Train y Test

Ahora, partiremos los datos de trai y test en una proporcion de 80/20 respectivamente, usando stratify para 
asegurar el balance de las clases.

>Pyhon Code


```python
from sklearn.model_selection import train_test_split

# ── Partición de datos 50/50 con stratify ─────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df_features, y,
    test_size    = 0.2,   # 50% train, 50% test
    random_state = 42,
    stratify     = y      # Asegura proporciones iguales de cada clase
)

# ── Verificación ───────────────────────────────────────────────────
print(f"✅ X_train: {X_train.shape}")
print(f"✅ X_test:  {X_test.shape}")
print(f"✅ y_train: {y_train.shape}")
print(f"✅ y_test:  {y_test.shape}")

# ── Distribución de clases por partición ──────────────────────────
print(f"\n📊 Distribución en y_train:")
print(y_train.value_counts(normalize=True).mul(100).round(2).astype(str) + ' %')

print(f"\n📊 Distribución en y_test:")
print(y_test.value_counts(normalize=True).mul(100).round(2).astype(str) + ' %')

```

>Output


```text
✅ X_train: (160, 39)
✅ X_test:  (40, 39)
✅ y_train: (160,)
✅ y_test:  (40,)

📊 Distribución en y_train:
Treatment_Response
Poor        36.88 %
Moderate     32.5 %
Good        30.63 %
Name: proportion, dtype: object

📊 Distribución en y_test:
Treatment_Response
Poor        37.5 %
Moderate    32.5 %
Good        30.0 %
Name: proportion, dtype: object
```

Bien, ahora que tenemos listos nuestros datos, vayamos a la construcción de modelos de clasificación.

-----

[Siguiente pagina](regresion_logistica.md)
