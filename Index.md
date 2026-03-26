# Clasificación y predicción de respuesta a tratamiento de tumores cerebrales
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)


----
**Dataset**: Brain Tumor Risk & Treatment Dataset
**Objetivo**: Identificar los determinantes físicos de la temperatura planetaria y priorizar candidatos para búsqueda de vida extraterrestre.

| Dataset |[NASA_exoplanets](PSCompPars_2026.02.16_14.28.02.csv) |
|---|---|
| **Notebook** | [.ipynb](PP1_IA.ipynb) |
----

Este proyecto tiene como finalidad, clasificar y predecir la respuesta a un tratamiento de una persona con algun tipo de tumor cerebral a partir de datos clinicos, información del paciente y resultados del tratamiento.


La razon de este proyecto es porque a lo largo de los años, el porcentaje de personas con tumores cerebrales ha aumentado significativamente. Segun articulos cientificos, aproximadamente `1.3 millones` de personas en los estados unidos viven con algun tipo de tumor cerebra, de los cuales solo `108,810`, es decir casi solo el `12%` de ellos, recibe algun diagnostico al respecto y solo el `34.8 %` de esas personas dianosticadas sobreviven.

> Brain Tumor Facts - National Brain Tumor Society. (2026, 6 marzo). National Brain Tumor Society. https://braintumor.org/brain-tumors/about-brain-tumors/brain-tumor-facts/



De modo que el entender el comportamiento  y el como los pacientes reaccionan al tratamiento de estos tumores, es una gran ayuda a las organizaciones que se dedican a estudiar el tema, combinado con la inteligencia artificial y el machine learning, nos da un alcance bastante amplio para poder abordar este tema.

Dataset obtenido de kaggke *"Brain Tumor Risk & Treatment Dataset
A Comprehensive Dataset for Tumor Classification and Treatment Study"*

## Metodología

1. **Limpieza de datos**: Eliminación de columnas con >50% nulos, imputación KNN
2. **Ingeniería de características**: Transformación logarítmica, eliminación de outliers
3. **Selección de variables**: Análisis de correlación + búsqueda exhaustiva con CV
4. **Modelado**:
   - Regresión lineal múltiple
   - Lasso con características polinomiales (grado 2)
   - Backward elimination para inferencia
5. **Evaluación**: R², MAE, RMSE + validación cruzada

---

## Procedimiento


[Contexto de los datos](contexto.md)

[Exploracion de datos](explorar_datos.md)

[Modelos/conclusion](modelos.md)

[What if??](whatif.md)

## Referencias

- **NASA Exoplanet Archive**: https://exoplanetarchive.ipac.caltech.edu/
- Kopparapu et al. (2013). *Habitable Zones Around Main-Sequence Stars*
- Kasting et al. (1993). *Habitable Zones around Main Sequence Stars*

---



**By** 

*Juan Angel Candelaria Rodriguez*

Universidad de Monterrey | Inteligencia Artificial  

---
