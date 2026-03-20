# ViolenciaTechReddit
**Detección Multinivel de Violencia Simbólica contra Mujeres en Comunidades Tecnológicas de Reddit mediante BERT**

Proyecto de tesis de licenciatura en Ciencias Computacionales — Universidad Autónoma del Estado de Hidalgo (UAEH).

**Autora:** Maricarmen Camacho Pérez

---

## Descripción

Sistema automático de cuatro niveles para detectar y analizar la violencia simbólica dirigida contra mujeres en comunidades tecnológicas de Reddit:

1. **Nivel 1 — Clasificador de reglas léxicas:** 75 patrones regex como línea base
2. **Nivel 2 — Modelo BERT:** Fine-tuning de `bert-base-uncased` (84.90% accuracy, F1-macro: 0.85)
3. **Nivel 3 — Post-procesamiento:** 24 patrones contextuales para corregir falsos positivos
4. **Nivel 4 — Clasificador de contexto:** Distingue violencia conversacional de testimonial

## Resultados principales

| Dato | Valor |
|---|---|
| Textos analizados | 184,572 |
| Período | 2016–2026 |
| Subreddits | 17 |
| Violencia detectada (BERT) | 23.3% |
| Violencia tras post-procesamiento | 22.3% |
| Violencia testimonial | 99.9% |
| Tendencia temporal | ↓ significativa (r²=0.58, p=0.0067) |

## Estructura del proyecto

```
VioLearn-Tech/
├── notebooks/                          # Pipeline completo (7 notebooks)
│   ├── 01_Extraccion_Reddit.ipynb      # Extracción con PRAW API
│   ├── 02_Analisis_Exploratorio.ipynb  # EDA del corpus
│   ├── 03_Limpieza_Preprocesamiento.ipynb  # Limpieza y filtros
│   ├── 04_Clasificacion_Reglas.ipynb   # 75 patrones regex
│   ├── 05_BERT_Colab.ipynb             # Fine-tuning (Google Colab + GPU)
│   ├── 06_Analisis_Temporal.ipynb      # Regresión lineal 2016-2026
│   └── Final_Analisis_Completo.ipynb   # Post-proc + contexto + dashboard
├── data/
│   ├── raw/                            # CSVs crudos + IDs públicos
│   ├── processed/                      # CSVs limpios y clasificados
│   └── models/                         # Modelo BERT entrenado
├── images/
│   ├── exploratorio/                   # Figuras del EDA
│   ├── resultados/                     # Figuras del artículo
│   └── diagramas/                      # Diagramas de arquitectura
├── config/
│   └── credentials_template.py         # Plantilla de credenciales Reddit
├── docs/                               # Artículo y reporte HTML
├── .gitignore
└── README.md
```

## Requisitos

### Para notebooks 01-04 y 06-Final (local)
- Python 3.10+
- pandas, numpy, matplotlib, seaborn, scipy
- PRAW 7.7.1 (solo para extracción)

### Para notebook 05 (Google Colab)
- Cuenta de Google con acceso a Colab
- GPU T4 (gratuita en Colab)
- transformers 4.40.0, torch, scikit-learn

### Instalación
```bash
pip install pandas numpy matplotlib seaborn scipy praw
```

## Cómo ejecutar el pipeline

1. **Configurar credenciales de Reddit:** Copiar `config/credentials_template.py` a `config/credentials.py` y completar con tus credenciales de la API de Reddit.

2. **Ejecutar notebooks en orden:**
   ```
   01 → 02 → 03 → 04 → 05 (en Colab) → 06 → Final
   ```

3. **Para el notebook 05 (BERT):**
   - Subir `data/processed/reddit_data_balanceado_bert.csv` a Google Drive
   - Subir `data/processed/reddit_data_clasificado_reglas.csv` a Google Drive
   - Abrir `05_BERT_Colab.ipynb` en Google Colab
   - Seleccionar GPU T4 en Entorno de ejecución → Cambiar tipo
   - Descargar `reddit_data_clasificado_BERT.csv` y colocarlo en `data/processed/`

## Datos

Por razones de privacidad y los términos de servicio de Reddit, no se incluyen los textos completos del corpus. En su lugar, se proporciona el archivo `data/raw/reddit_ids_publicos.csv` con los identificadores de cada publicación y comentario, que permite reconstruir el corpus usando la API de Reddit.

## Comunidades analizadas (17 subreddits)

**Apoyo para mujeres (5):** r/TwoXChromosomes, r/WomenInTech, r/girlsgonewired, r/LadiesofScience, r/WomenEngineers

**Profesional mixta (9):** r/cscareerquestions, r/ITCareerQuestions, r/experienceddevs, r/AskComputerScience, r/programming, r/learnprogramming, r/webdev, r/datascience, r/MachineLearning

**Laboral/general (3):** r/technology, r/antiwork, r/AskWomen

## Cita

Si usas este trabajo, por favor cita:

```
Camacho Pérez, M. (2026). Detección Multinivel de Violencia Simbólica contra
Mujeres en Comunidades Tecnológicas de Reddit mediante BERT.
Tesis de licenciatura, Universidad Autónoma del Estado de Hidalgo.
```

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Ver `LICENSE` para más detalles.
