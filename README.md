# Housing Price Prediction

Este proyecto predice los precios de viviendas utilizando **Machine Learning** sobre el dataset **Ames Housing** (Kaggle).  
El objetivo es comparar diferentes algoritmos de regresión y encontrar el que mejor se ajuste para la predicción de precios.

---

## Estructura del Proyecto

- "data/" :  datasets utilizados.  
- "notebooks/" : notebooks y reportes en HTML del análisis.  
- "src/" : scripts de Python con el flujo principal.  
- "images/" : gráficos de exploración, resultados de modelos y análisis.  
- "docs/" : documentación adicional del proyecto.  
- "requirements.txt" : librerías necesarias.  

---

## Requisitos

Instalar las dependencias desde el archivo `requirements.txt`:

```bash
pip install -r requirements.txt


## Uso

1. Clonar el repositorio:
```bash
git clone https://github.com/AVALLEJOTORRES/housing-price-prediction.git

2. Acceder a la carpeta del proyecto:
cd housing-price-prediction

3. Ejecutar los scripts principales:
python src/proyecto_uno.py

## Modelos probados
- RandomForestRegressor
- CatBoostRegressor
- XGBRegressor
- LGBMRegressor
- MLPRegressor
- ElasticNet
- HuberRegressor

## Resultados principales
- CatBoostRegressor y LGBMRegressor fueron los mejores modelos (R² ≈ 0.80).
- El MLPRegressor tuvo bajo desempeño comparado con los demás.
- Se analizaron métricas como MSE, RMSE, MAE y R².

---

**Andrés Vallejo**  
Año: 2025



