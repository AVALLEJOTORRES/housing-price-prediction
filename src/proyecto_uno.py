# %% [markdown]
# ## Importar librerias necesarias
# 
# #### En este proyecto se utilizan las siguientes librerías:
# 
# * pandas: para manipulación y análisis de datos tabulares.
# * numpy: para operaciones numéricas y manejo de arrays.
# * seaborn y matplotlib: para la visualización de datos.

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# ## Cargar el dataset trainsi
# 
# #### Cargar el dataset
# 
# El dataset corresponde al Ames Housing Dataset descargado desde Kaggle.  
# Se utiliza la función "pd.read_csv" para cargar los datos y se hace una copia (`data`) para trabajar sobre ella sin modificar el original.
# 

# %%
data_c = pd.read_csv("trainsi.csv")
data = data_c.copy()

# %% [markdown]
# ## Inspección inicial del dataset
# 
# #### Usamos los codigos:
# * data  Visualiza brevemente el contenido del dataset para tener una idea de su estructura.
# * data.shape :Devuelve las dimensiones del dataset en formato (filas, columnas).
# * data.head() :Muestra las primeras 5 filas para revisar el inicio de los datos.
# * data.tail() :Muestra las últimas 5 filas para revisar el final de los datos.
# * data.nunique() :Muestra el número de valores únicos por columna, útil para variables categóricas.
# * data.isnull().sum() :Indica cuántos valores nulos hay por columna.
# * data.info() :Muestra tipos de datos, conteo de valores no nulos y memoria usada.
# * data.describe() :Resume estadísticas descriptivas de variables numéricas (media, mediana, cuartiles, min, max, desviación estándar).

# %%
data

# %%
data.shape

# %%
data.head(10) 

# %%
data.tail(10)

# %%
data.nunique()

# %%
data.isnull().any()

# %%
data.info() 

# %%
data.describe()

# %% [markdown]
# ## Limpieza y preprocesamiento de datos

# %% [markdown]
# #### Procedemos a realizar un analisis exploratorio de los datos Dejar el dataset listo para análisis y modelado eliminando, imputando o transformando datos

# %% [markdown]
# ##### Detección y tratamiento de valores nulos

# %%
col_con_nulos = data.columns[data.isnull().any()]
col_con_nulos
total_columnas_con_nulos = len(col_con_nulos)
print(f"Las columnas que tienen valores nulos son: {total_columnas_con_nulos}, y sus nombres son:\n\n {col_con_nulos}")

# %%
total_nulos_por_columna = data[col_con_nulos].isnull().sum()
total_nulos = data[col_con_nulos].isnull().sum().sum()

print(f"El total de NULOS es de : {total_nulos}, y el total de NULOS en cada COLUMNA es de:\n{total_nulos_por_columna}")

# %% [markdown]
# ##### Porcentaje de valores nulos, para observar si se eliminan las columnas o se tratan, de ser importantes para el modelo

# %%
porcentaje_nulos = data.isnull().mean() * 100
porcentaje_nulos.head(60)

# %% [markdown]
# #### Rellenamos los valores nulos con la mediana y moda, dependiendo el caso
# 
# ##### Para garantizar la calidad de los datos antes del modelado, se creó una función rellenar_nulos() que recorre todas las columnas del DataFrame y aplica un método de imputación según el tipo de dato:
# 
# * Numéricos (float64 e int64) Se rellenan con la mediana, ya que es robusta frente a valores atípicos.
# * Booleanos (bool) Se rellenan con la moda, dado que no se puede calcular media ni mediana.
# * Categóricos (objetos o texto) Se rellenan con la moda, pues representa el valor más frecuente.
# * Este proceso asegura que no queden valores faltantes y evita la eliminación de registros innecesaria.

# %%
def rellenar_nulos(df):
    data=df.copy()
    
    for col in data.columns:
        if data[col].isnull().any():
            if data[col].dtype in ["float64","int64"]:
                data[col] = data[col].fillna(data[col].median())
            elif data[col].dtype == bool:
                data[col] = data[col].fillna(data[col].mode()[0])
            else:
                data[col] = data[col].fillna(data[col].mode()[0])
    return data 

# %% [markdown]
# ##### Rellenamos los nulos

# %%
data_sin_nulos = rellenar_nulos(data)
# Observamos el data
data_sin_nulos.info()

# %% [markdown]
# ## Visualizaciones Exploratorias EDA

# %%
# Grafico Distribucion SalePrice
moda = data_sin_nulos["SalePrice"].mode()[0]
media = data_sin_nulos["SalePrice"].mean()

# grafica 1. Vamos analizar SalePrice
sns.displot(data_sin_nulos["SalePrice"], kde=True, bins=30, height=5, aspect=1.5)
plt.title("Distribucion del precio de venta Sale Price")
plt.ylabel("Frecuencia")
plt.axvline(x=moda, color="Red", linestyle="--", label=f"Moda: {moda}")
plt.axvline(x=media, color="Orange", linestyle="--", label=f"Media: {media:.2f}")
plt.legend()
plt.grid()
plt.show()

## =================================
#              Análisis
## =================================
# Observamos que la moda (valor más frecuente) se encuentra alrededor de los 140.000
# La mayoría de los precios se concentran entre 100.000 y 240.000 dólares
# Se identifican posibles valores atípicos a partir de aproximadamente 390.000 dólares
# Se observan valores extremos que alcanzan entre 500.000 y hasta 720.000 dólares
# Aproximadamente 100 casas se vendieron en el rango de los 100.000 dólares
# Alrededor de 10 viviendas se vendieron por menos de 90.000 dólares
# Cerca de 250 viviendas se vendieron alrededor de los 145.000 dólares
# La distribución presenta una asimetría positiva (sesgo a la derecha), lo que sugiere una concentración de precios bajos y algunos valores altos aislados

# Vamos a comprobar el sesgo de la grafica
media = data_sin_nulos["SalePrice"].mean()
mediana = data_sin_nulos["SalePrice"].median()
print(f"La media es igual a: {media:.3f}\nLa mediana es igual a: {mediana:.3f}")

print("Sesgo: ")
if media > mediana:
    print("Sesgo a la derecha")
elif media < mediana:
    print("Sesgo a la izquierdad")
else:
    print("Simetrica")

kurtosis = data_sin_nulos["SalePrice"].kurt()
print(f"La Curtosis es igual a: {kurtosis:.3f}")

# si la curtosis es mayor a 1, es leptocurtica y dependiendo del valor a partir de uno, ya es moderada o fuerte
# si la curtosis es menor a 1, es platicurtica y dependiendo del valor a partir de uno, ya es moderada o fuerte
# si la curtosis esta entre -0.5 y 0.5 se considera mesocurtica, campna de gauss
# Para que nos sirve el sesgo y la curtosis:
# El skew, nos ayuda a identificar hacia donde se inclinan mas los datos
# mientras la curtosis, nos ayuda a identificar posibles outliers, gracias al peso de sus colas

# %%
# Grafico Distribucion GrLivArea
# Grafica GrLivArea
moda_1 = data_sin_nulos["GrLivArea"].mode()[0]
media_1 = data_sin_nulos["GrLivArea"].mean()

sns.displot(data_sin_nulos["GrLivArea"], kde=True,bins=30,height=5, aspect=1.5)
plt.title("Distribucion del area vivienda GrLivArea")
plt.ylabel("Frecuencia")
plt.axvline(x=moda_1, color="Red", linestyle="--", label=f"Moda: {moda_1:.2f}")
plt.axvline(x=media_1, color="Orange", linestyle="--", label=f"Media: {media_1:.2f}")
plt.legend()
plt.grid()
plt.show()


## =================================
#              Análisis
## =================================
# La moda es aproximadamente de 864 pies cuadrados area de vivienda
# 25 casas estan por debajo de 2600 pies cuadrados area de vivienda
# la mayoria de datos del area de vivienda se centran en los valores de 1020 y 1090 pies cuadrados aproximadamente
# 5 casas estan por debajo de los 3000 pies cuadrados del area de vivienda
# hay algunas casas con valores extremos de 4000 y 5000 pies cuadrados pero son pocas
# Podemos decir que entre mas aumentan la construccion de casas mas se reduce el area  de vivienda

# Vamos a comprobar el sesgo de la grafica
media = data_sin_nulos["GrLivArea"].mean()
mediana = data_sin_nulos["GrLivArea"].median()
print(f"La media es igual a: {media:.3f}\nLa mediana es igual a: {mediana:.3f}")

print("Sesgo: ")
if media > mediana:
    print("Sesgo a la derecha")
elif media < mediana:
    print("Sesgo a la izquierdad")
else:
    print("Simetrica")

kurtosis = data_sin_nulos["GrLivArea"].kurt()
print(f"La Curtosis es igual a: {kurtosis:.3f}")

#- La moda se encuentra aproximadamente en 864 pies cuadrados.
#- Unas 25 viviendas tienen menos de 2600 pies cuadrados de área construida.
#- La mayoría de las viviendas tienen entre 1020 y 1090 pies cuadrados, concentrando la densidad en ese rango.
#- Solo 5 viviendas tienen áreas inferiores a 3000 pies cuadrados
#- Existen valores atípicos entre 4000 y 5000 pies cuadrados, aunque son muy pocos.
#- Esto podría indicar que, a medida que aumenta la densidad de construcción, el área individual de las casas tiende a reducirse.

# %%
# Grafico Evolucion p1
# Agrupar por año de venta y calcular precio promedio
ventas_por_año = data_sin_nulos.groupby("YrSold")["SalePrice"].mean()

# Graficar evolución del precio promedio por año
plt.subplots(figsize=(10, 6))
plt.plot(ventas_por_año.index, ventas_por_año.values, marker='*')
plt.title("Evolución del Precio Promedio de Venta por Año")
plt.xlabel("Año de Venta")
plt.ylabel("Precio Promedio")
plt.grid()
plt.show()


# %%
# Grafico violin plot p1
plt.subplots(figsize=(14,9))
sns.violinplot(x="OverallQual", y="SalePrice", data=data_sin_nulos)
plt.title("Distribucion del Precio de Venta segun La Calidad General")
plt.xlabel("Calidad General (1 = Muy Baja, 10 = Excelente)")
plt.ylabel("Precio Venta USD")
plt.show()

# %%
# Grafico Scatter plot p1
plt.subplots(figsize=(14,8))
sns.scatterplot(x="GrLivArea", y="SalePrice", data=data_sin_nulos)
plt.title("Grafico de dispersion GrLivArea VS SalePrice")
plt.grid()
plt.show()

# %%
# Grafico boxplot p1
plt.subplots(figsize=(14,8))
sns.boxplot(x="OverallQual", y="SalePrice", data=data_sin_nulos)
plt.title("Precio Venta VS Calidad De La Casa")
plt.xlabel("Calidad Casa (1=Baja 10=Muy Alta)")
plt.ylabel("Precio Venta")
plt.grid()
plt.show()

## =================================
#              Análisis
## =================================

# El gráfico muestra la distribución del precio de venta (SalePrice) según la calidad general de la vivienda (OverallQual).
# Cada caja representa el rango intercuartílico (IQR), es decir, desde el primer cuartil (Q1, 25%) hasta el tercer cuartil (Q3, 75%).
# La línea dentro de la caja indica la mediana (Q2, 50%), mostrando el valor central de los datos para cada categoría de calidad.

# A mayor ancho de la caja, mayor dispersión de precios en esa categoría.
# Las líneas verticales (bigotes) representan la variabilidad dentro de 1.5 * IQR; los puntos fuera de estos son outliers.

# Si la mediana está más cerca de Q1 o Q3, puede indicar asimetría:
# - Si está cerca de Q1 → asimetría a la derecha (positiva)
# - Si está cerca de Q3 → asimetría a la izquierda (negativa)

# Observamos que a medida que aumenta la calidad de la casa (de 1 a 10), el precio de venta también tiende a aumentar.
# Este comportamiento indica una relación positiva entre calidad de vivienda y precio.

# Conclusión: 
# Las viviendas de mayor calidad presentan precios más altos en promedio, pero también mayor dispersión y más outliers en los rangos superiores.

# %% [markdown]
# ## Análisis gráfico de umbrales superiores e inferiores para detectar outliers
# ##### Objetivo
# 
# Identificar valores atípicos (outliers) en variables numéricas utilizando el método del rango intercuartílico (IQR), con el fin de definir límites superiores e inferiores y así determinar qué observaciones están fuera del rango esperado.

# %%
## =====================
# Definición de la función limites_sup_inf
## =====================

def limites_sup_inf(df, col):
    data=df.copy()

    if data[col].dtype in ["int32", "int64", "float64"]:
        
        # Quarttiles
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        
        # IQR: Rango intercuartílico
        IQR = Q3 - Q1
        
        # Límites inferiores y superiores según la regla del 1.5 * IQR
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR
        
    return lim_inf, lim_sup

# %%
## =====================
# Obtención y visualización de los límites
## =====================

limite_inferior, limite_superior = limites_sup_inf(data_sin_nulos, "GrLivArea")
print(f"Limites: Inferior: {limite_inferior}, Superior: {limite_superior}")

# %%
## =====================
# Análisis de valores atípicos en la variable GrLivArea
## =====================

plt.figure(figsize=(14,8))
plt.subplot(1,2,1)
sns.histplot(data_sin_nulos["GrLivArea"], kde=True, alpha=0.8, bins=30)
plt.axvline(x=limite_superior, color="Red", linestyle="--", label=f"Limite Superior: {limite_superior}")
plt.axvline(x=limite_inferior, color="Red", linestyle="--", label=f"Limite Inferior: {limite_inferior}")
plt.title("Distribucion GrLivArea")
plt.legend()
plt.grid()

plt.subplot(1,2,2)
sns.boxplot(x=data_sin_nulos["GrLivArea"], color="Red")
plt.title("Boxplot, para detectar Outlier")
plt.axvline(x=limite_superior, color="Orange", linestyle="--", label=f"Limite Superior: {limite_superior}")
plt.axvline(x=limite_inferior, color="Orange", linestyle="--", label=f"Limite Inferior: {limite_inferior}")
plt.grid(axis="x")

plt.tight_layout()

## =====================
# Análisis
## =====================
#Se realizó un análisis visual de valores atípicos para la variable `GrLivArea` utilizando histogramas y boxplots. 
#Los límites superior e inferior se calcularon mediante el rango intercuartílico (IQR), que nos permite detectar valores atípicos extremos sin hacer suposiciones de normalidad.

#- El histograma muestra una distribución asimétrica a la derecha
#- Los valores por encima de aproximadamente 2747 y por debajo de 158 son considerados outliers.
#- En el boxplot se visualizan los puntos fuera del rango intercuartílico, confirmando la presencia de outliers en el extremo superior.

#Este análisis es clave para decidir los outliers mediante eliminación, imputación o transformación, en función del impacto que puedan tener sobre los modelos predictivos.

# %%
## =====================
# Análisis de valores atípicos en la variable GarageCars
## =====================

lim_inf_gc, lim_sup_gc = limites_sup_inf(data_sin_nulos, "GarageCars")


plt.figure(figsize=(14,8))
plt.subplot(1,2,1)
sns.histplot(data["GarageCars"], kde=True, bins=30)
plt.axvline(x=lim_inf_gc, color="Red", linestyle="--", label=f"Limite Inferior: {lim_inf_gc}")
plt.axvline(x=lim_sup_gc, color="Orange", linestyle="--", label=f"Limite Superior: {lim_sup_gc}")
plt.title("Distribucion Garage de Carros")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid()


plt.subplot(1,2,2)
sns.boxplot(x=data["GarageCars"])
plt.title("Boxplot GarageCars (Detectar Outliers)")
plt.grid()
plt.tight_layout()

## =====================
# Análisis
## =====================

## =====================
# Histograma con límites marcados
## =====================

# Se representa la distribución de GarageCars con un histograma y curva de densidad (kde).
# Se añaden líneas verticales para los límites inferior (rojo) y superior (naranja), que indican el rango esperado de valores sin considerar outliers.
# Esto permite observar la concentración de datos y detectar valores extremos.

## =====================
# Boxplot
## =====================

# El diagrama de caja muestra los valores dentro del rango intercuartílico
# En este caso, se observa un valor en 4, que queda fuera del límite superior de 3.5, lo que lo clasifica como un valor atípico.

# El análisis evidencia que la mayoría de los registros se encuentran entre 0 y 3 vehículos en el garaje, 
# mientras que valores como 4 podrían ser considerados atípicos.


# %%
## =====================
# Análisis de valores atípicos en la variable TotalBsmtSF
## =====================
# TotalBsmtSF representa la superficie total del sótano en pies cuadrados

lim_inf_tb, lim_sup_tb = limites_sup_inf(data_sin_nulos, "TotalBsmtSF")

plt.figure(figsize=(14,8))
plt.subplot(1,2,1)
sns.histplot(data["TotalBsmtSF"], kde=True, bins=30)
plt.axvline(x=lim_inf_tb, color="Red", linestyle="--", label=f"Limite Inferior: {lim_inf_tb}")
plt.axvline(x=lim_sup_tb, color="Orange", linestyle="--", label=f"Limite Superior: {lim_sup_tb}")
plt.title("Distribucion de la variable Superficie total del sótano")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid()


plt.subplot(1,2,2)
sns.boxplot(x=data["TotalBsmtSF"])
plt.title("Boxplot de la variable Superficie total del sótano para (Detectar Outliers)")
plt.grid()
plt.tight_layout()

## =====================
# Análisis
## =====================

# La mayoría de los valores se concentran entre 500 y 2000 pies².
# La distribución es asimétrica a la derecha (sesgo positivo), lo que indica que hay algunas propiedades con sótanos excepcionalmente grandes que arrastran la cola hacia valores altos.
# Límite inferior: 42.0 pies² → Hay muy pocos casos cercanos a este valor, lo que indica sótanos extremadamente pequeños o prácticamente inexistentes.
# Límite superior: 2052.0 pies² → Hay varias observaciones que lo superan, llegando hasta más de 6000 pies².

# %%
## Se identificaron estas variables como candidatas a tratamiento por su distribución y presencia de valores extremos:
variables_outliers = [
    "SalePrice", "GrLivArea", "GarageArea", "TotalBsmtSF", "1stFlrSF",
    "MasVnrArea", "BsmtFinSF1", "LotFrontage", "WoodDeckSF",
    "2ndFlrSF", "OpenPorchSF", "LotArea"]

# %%
## Funcion

# Reduce la influencia de valores extremos sin necesidad de eliminar registros completos.
# Mantiene la integridad del dataset para imputaciones posteriores.

def trata_atipicos(df, columnas):
    data=df.copy()
    
    for col in columnas:
        if data[col].dtype in ["int32", "int64", "float64"]:
            
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            
            IQR = Q3 - Q1

            lim_inf = Q1 - 1.5 * IQR
            lim_sup = Q3 + 1.5 * IQR

            data[col] = np.where ((data[col] < lim_inf) | (data[col] > lim_sup), np.nan, data[col])
            
    return data, lim_inf, lim_sup

# %%
# Aplicamos la funcion:
data_sin_atipicos , lim_inf, lim_sup = trata_atipicos(data_sin_nulos, variables_outliers)

# Visualizamos el data_sin_atipicos
data_sin_atipicos.info()

# %%
## Creamos una función: Rellenar los valores faltantes generados tras el tratamiento de outliers (o ya presentes en el dataset original) 
# usando la mediana, ya que es un estadístico robusto frente a valores extremos y distribuciones sesgadas.

def imputar_nulos(df):
    data=df.copy()
    
    for col in data.select_dtypes(include=np.number):
        if data[col].isnull().any():
           if data[col].dtype in ["int64","int32","float64"]:
             data[col] = data[col].fillna(data[col].median())
    
    return data

# %%
# Aplicamos la funcion imputar nulos y desempaquetamos el data ya tratado
data_tratado = imputar_nulos(data_sin_atipicos)
data_tratado.info()

# %% [markdown]
# ## Exportación del dataset limpio para análisis en Power BI
# ##### Guardar el dataset final ya preprocesado (sin nulos, con outliers tratados e imputados) en formato .csv para su posterior uso en herramientas de visualización como Power BI.

# %%
# Guardamos este data en un excel para visualizarlo en Power Bi:
# Data :
# Sin nulos
# Codificado en numerico
# Atipicos tratados y rrellenados con NaN y luego tratados con la mediana
data_tratado.to_csv("data_para_BI.csv", index=False)

# %%
data_t = data_tratado.copy()
data_t.info()

# %% [markdown]
# ##### Codificación de variables categóricas
# 
# En este paso convertimos todas las variables categóricas a numéricas usando One-Hot Encoding con **pandas.get_dummies()**
# Es necesario porque:
# * Los algoritmos de Machine Learning no pueden trabajar directamente con texto.
# * Permite analizar correlaciones y generar mapas de calor.
# * Facilita la detección de outliers y el cálculo de métricas.
# 
# El parámetro **drop_first=True** se utiliza para evitar la multicolinealidad eliminando una de las columnas categorías como referencia

# %%
## =========================================
## CODIFICACIÓN DE VARIABLES CATEGÓRICAS
## =========================================
# Objetivo:
# Convertir todas las variables categóricas a formato numérico para:
# Facilitar el entrenamiento de modelos de Machine Learning
# Permitir la detección de outliers y correlaciones (mapas de calor)
# Evitar errores al aplicar algoritmos que solo aceptan datos numéricos

# 1. Seleccionar solo las columnas categoricas
columnas_categoricas = data_t.select_dtypes(include=["object"]).columns
columnas_categoricas

# 2. Convertir a lista de columnas python
lista_categoricos = list(columnas_categoricas)
lista_categoricos

# 3.  Codificar las variables con One-Hot Encoding

data_sin_categoricos = pd.get_dummies(data_t, columns=lista_categoricos, drop_first=True)
data_sin_categoricos

# 4.  Verificar el nuevo formato de los datos
data_sin_categoricos.info()

# Algunas columnas booleanas aparecen como True/False, hay que convertirlas a 0/1.

# %% [markdown]
# ##### CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# 
# En este paso creamos una función para transformar todas las variables de tipo booleano (True / False) a valores numéricos (1 / 0).  
# Esto es importante porque:
# * Algunos modelos de Machine Learning no aceptan valores booleanos.
# * Facilita el cálculo de métricas y la creación de visualizaciones estadísticas.
# * Unifica el formato de datos, dejando **todo el dataset en formato numérico**.
# 

# %%
## ============
# CONVERSIÓN DE VARIABLES BOOLEANAS A 0 Y 1
## ============

def codificar_booleanos(df):
    data=df.copy()
    
    for col in data.columns:
        if data[col].dtype == "bool":
            data[col] = data[col].astype(int)
    return data

# %%
# Aplicamos la función al dataset sin variables categóricas

data_sin_bool = codificar_booleanos(data_sin_categoricos)
data_sin_bool

# Visualizar el resultado
data_sin_bool.info()

# %% [markdown]
# ## Análisis de correlación y selección de variables
# ##### Objetivo:
# * En este paso se busca identificar las variables que tienen mayor relación con la variable objetivo SalePrice, con el fin de priorizar aquellas que potencialmente aportarán más información al modelo.
# * Evitar redundancia y multicolinealidad alta.

# %%
## ================================
#  Creación de la matriz de correlación
## ===============================
# Se calculó la correlación de Pearson entre todas las variables numéricas del conjunto de datos.
# Eto permite medir la relación lineal entre cada variable independiente y la variable objetivo.
# El rango de valores posibles es de -1 a 1, donde:
# 1 = correlación positiva perfecta
# -1 = correlación negativa perfecta
# 0 = sin correlación lineal

matriz_correlacion = data_sin_bool.corr()
matriz_correlacion

# %%
## ================================
#  Selección de las variables más importantes
## ===============================

# Seleccionamos las 50 variables con mayor valor absoluto de correlación con SalePrice usando .nlargest().
# Esto nos ayuda a quedarnos con aquellas características que tienen una relación más fuerte (positiva o negativa) con el precio de venta.

las_50_mas_importantes = matriz_correlacion["SalePrice"].abs().nlargest(50).index
las_50_mas_importantes

# %%
## ===============================
#  Resultado
## ===============================
# Se obtuvo un conjunto reducido de variables candidatas a ser usadas en el modelado, lo que puede mejorar el rendimiento y reducir el riesgo de sobreajuste.
data_50 = data_sin_bool[las_50_mas_importantes]
data_50

# %%
## ================================
#  ReCreación de matriz reducida (mx_50)
## ===============================
# Se construyó una nueva matriz de correlación únicamente con estas 50 variables para un análisis más enfocado.
mx_50 = data_50.corr()
mx_50

# %%
## ================================
#  Seaborn (sns.heatmap) para graficar la correlación entre las variables seleccionadas.
## ===============================
# Se estableció un rango de colores y formato para facilitar la interpretación

plt.subplots(figsize=(30,25))
sns.heatmap(mx_50, annot=True, linewidths=0.2, vmax=0.8, fmt=".1f")
plt.title("Mapa de Calor 50 Variables mas importantes de Nlargest")
plt.show()

# En este mapa de calor podemos visualizar lo siguiente:
# Las variables mas correlacionadas respecto a la target SalePrice, en el cual a partir dde 0.5 se considera una correlacion moderada y donde se entra a visualizar
# si cada una de estas variables son importantes o no para el modelo, en el cual se elimina la columna si no se ve un motivo significante para el modelo
# Las variables que tengan mas de 0.7 ya son fuertemente correlacionadas y por lo general estas no se quitan o se eliminan ya que aportan gran valor al modelo
# Tambien podemos encontrar variables que no tienen correlacion con SalePrice, pero si fuerte correlacion entre ellas mismas, por lo cual aca se decide cual de las dos se deja y cual se elimina
# se eliminan con el fin de evitar colinealidad, ya que hay modelos sensibles a estos y son los de regression lineal, ya modelos un poco mas robustos, se pueden dejar

# En el mapa de calor se evidencia que variables como OverallQual, GrLivArea, GarageCars, GarageArea y TotalBsmtSF presentan una correlación superior a 0.6 con SalePrice, 
# lo cual las posiciona como fuertes candidatas para los modelos predictivos. También se identifican grupos de variables con alta colinealidad entre sí, 
# como 1stFlrSF y TotalBsmtSF, lo que requiere decisiones informadas sobre cuáles conservar según la lógica del negocio y la naturaleza del modelo.

# %%
## ================================
#  SSelección final de variables para modelado
## ===============================
# Determinar las variables con mayor relación estadística con la variable objetivo SalePrice y conservarlas para el entrenamiento de los modelos.

relacion = mx_50["SalePrice"]
relacion

## ================================
#  Decisión de conservación
## ===============================
# Se optó por mantener las 50 variables más correlacionadas, sin eliminar ninguna en esta etapa, 
# con el objetivo de que los modelos de Machine Learning determinen de forma automática la relevancia de cada una.

# %% [markdown]
# ## División de los datos **(Train-test split)**
# ##### Objetivo: separar los datos en dos subconjuntos principales:
# 
# * Entrenamiento (train): usado para que el modelo aprenda patrones de los datos.
# * Prueba (test): reservado para evaluar el rendimiento del modelo en datos que nunca ha visto, garantizando una estimación más realista de su desempeño.

# %%
## Realizamos la respectiva división de los datos
# data_50 será nuestro DataFrame ya tratado (sin nulos y sin atípicos)

## ====================
# Cargamos librería
## ====================
from sklearn.model_selection import train_test_split

# Creamos subconjuntos de entrenamiento y prueba
train_val_df, test_df = train_test_split(data_50, test_size=0.2, random_state=42)

# Separamos las variables predictoras (X) de la variable objetivo (y)
x_train, y_train = train_val_df.drop(columns=["SalePrice"], axis=1), train_val_df["SalePrice"]
x_test, y_test = test_df.drop(columns=["SalePrice"], axis=1), test_df["SalePrice"]

# %% [markdown]
# ## Se crean y entrenan diferentes modelos de regresión no lineales, los cuales son especialmente útiles porque:
# ##### Capturan relaciones complejas entre las variables predictoras y la variable objetivo.
# * No requieren escalado de datos (a diferencia de los modelos lineales).
# * Son robustos frente a outliers y variables de distinta escala.

# %%
## ===============================
#Entrenamiento de modelos (No lineales): 
## ===============================
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor

# %%
## ====================
# Random Forest Regressor
## ====================
# Random Forest es un modelo de ensamble basado en múltiples árboles de decisión. 
# Cada árbol se entrena con una muestra distinta de los datos (bagging), y las predicciones finales resultan de promediar todos los árboles.

rfr_model = RandomForestRegressor(random_state=42)
rfr_model.fit(x_train, y_train)

# Generar predicciones
y_train_pred_rfr = rfr_model.predict(x_train)
y_test_pred_rfr = rfr_model.predict(x_test)

# %%
## ====================
# CatBoost Regressor
## ====================
# El CatBoost es un modelo de boosting basado en árboles de decisión. Se caracteriza por:
# Manejar automáticamente variables categóricas (aunque en este caso ya codificamos previamente).
# Generalizar bien sin un tuning muy complejo.
# Entrenar de forma rápida y estable frente a datos heterogéneos.

cat_model = CatBoostRegressor(verbose=0)
cat_model.fit(x_train, y_train)

# Generar predicciones
y_train_pred_cat = cat_model.predict(x_train)
y_test_pred_cat = cat_model.predict(x_test)

# %%
## ====================
#  XGBoost Regressor
## ====================
# El XGBoost es un algoritmo de boosting basado en árboles de decisión. Se diferencia de otros métodos en su:
# Alta eficiencia computacional.
# Regularización integrada (para evitar sobreajuste).
# Capacidad de manejar datos con outliers y multicolinealidad.

xgb_model = XGBRegressor()
xgb_model.fit(x_train, y_train)

# Generar predicciones
y_train_pred_xgb = xgb_model.predict(x_train)
y_test_pred_xgb = xgb_model.predict(x_test)

# %%
## ====================
#  LightGBM Regressor
## ====================
# Es un modelo de boosting con árboles de decisión, optimizado para:
# Manejar grandes volúmenes de datos.
# Ser más rápido que XGBoost en entrenamiento.
# Soportar categorías de forma nativa.

lgbm_model = LGBMRegressor()
lgbm_model.fit(x_train, y_train)

# Generar predicciones
y_train_pred_lgbm = lgbm_model.predict(x_train)
y_test_pred_lgbm = lgbm_model.predict(x_test)

# %%
## ===============================
# Entrenamiento de modelos (SI lineales): 
## ===============================
# LinearRegression: modelo de regresión lineal clásico.
# ElasticNet: combina penalización Lasso (L1) y Ridge (L2).
# HuberRegressor: robusto a outliers.
from sklearn.linear_model import HuberRegressor, ElasticNet, LinearRegression


# %% [markdown]
# ## Preparación de datos para modelos lineales
# ##### Los modelos lineales son sensibles a la escala de los datos, por lo que se debe aplicar un escalado a las variables numéricas. Para eso se crea una copia del dataset de entrenamiento:

# %%
## ====================
# Se crea copia
## ====================
## Ahora vamos a usar los modelos lineales
# 1. Hay que escalar los datos, usamos el data_e
data_e = data_50.copy()

# %%
## ====================
# Escalado de los datos
## ====================
# Se aplicará un escalador robusto (RobustScaler) ya que es menos sensible a outliers que otros métodos como MinMaxScaler o StandardScaler.

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Procedemos a escalar los respectivos datos
# Creamos el modelo de escalado
data_scaler = RobustScaler()

# Ahora entrenamos y transformamos solo el x_train y el x_test solo se transforma
x_train_scaler = data_scaler.fit_transform(x_train)
x_test_scaler = data_scaler.transform(x_test)

# %%
## ====================
#  MLPRegressor (Red Neuronal Artificial)
## ====================
# El MLPRegressor (Multi-Layer Perceptron) es una red neuronal de tipo perceptrón multicapa, usada para problemas de regresión.
# Utiliza capas ocultas de neuronas y funciones de activación no lineales
# Aprende representaciones complejas de los datos.
# Necesita escalado de variables, porque es sensible a las magnitudes de los datos (a diferencia de RandomForest o XGBoost).

# Crear el modelo con hiperparámetros básicos 
mlp_model = MLPRegressor(hidden_layer_sizes=(100,),  # 1 capa oculta con 100 neuronas
                         activation='relu',          # función de activación
                         solver='adam',              # optimizador
                         max_iter=500,               # número máximo de iteraciones
                         random_state=42)

mlp_model.fit(x_train, y_train)
y_train_pred_mlp = mlp_model.predict(x_train_scaler)
y_test_pred_mlp = mlp_model.predict(x_test_scaler)

# %%
## ====================
# ElasticNet
## ====================
# ElasticNet combina Lasso (L1) y Ridge (L2) en una sola penalización.
# alpha: controla la fuerza de la regularización (0.1 en este caso).
# l1_ratio: controla el balance entre L1 y L2 (0.5 significa mitad Lasso, mitad Ridge).

elas_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elas_model.fit(x_train_scaler, y_train)

# Generamos predicciones
y_train_pred_elas = elas_model.predict(x_train_scaler)
y_test_pred_elas = elas_model.predict(x_test_scaler)

# %%
## ====================
# HuberRegressor
## ====================
# Es una regresión robusta a outliers, combinando lo mejor de la regresión lineal y una función de pérdida menos sensible a valores extremos.
# epsilon: determina qué tan sensible es a los outliers (1.35 es un valor común).
# alpha: regularización (penalización para evitar sobreajuste).

huber_model = HuberRegressor(epsilon=1.35, alpha=0.0001)
huber_model.fit(x_train_scaler, y_train)

# Generar predicciones
y_train_pred_huber = huber_model.predict(x_train_scaler)
y_test_pred_huber = huber_model.predict(x_test_scaler)


# %% [markdown]
# ## Evaluar modelos
# ##### Importar métricas de evaluación
# * mean_squared_error : calcula el MSE (error cuadrático medio).
# * mean_absolute_error : alcula el MAE (error absoluto medio).
# * r2_score : calcula el R² (qué tan bien el modelo explica la varianza).

# %%
## =======================
# librería de métricas
## ======================
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %%
## =======================
# Se crea una función de evaluación
## ======================
# Explicación
# 1. real = np.mean(y_real) → calcula el valor medio del target (se usa como referencia para calcular los errores relativos).
# 2. Errores absolutos y relativos:
# mse : penaliza más los errores grandes.
# rmse : raíz del MSE, más interpretable porque está en la misma escala que la variable objetivo.
# error_relativo_rmse : qué porcentaje representa el RMSE respecto al promedio del target.
# 3. Errores absolutos:
# mae → error promedio sin importar si es positivo o negativo.
# error_relativo_mae → porcentaje relativo al promedio del target.
# 4. Bondad del ajuste:
# r2 : mide qué proporción de la variabilidad de la variable dependiente explica el modelo (1 es perfecto, 0 significa que no explica nada).

def evaluar_modelo(y_real, y_predicho, model=""):
    real = np.mean(y_real)
    
    mse = mean_squared_error(y_real, y_predicho)
    rmse = np.sqrt(mean_squared_error(y_real, y_predicho))
    error_relativo_rmse = (rmse/real) * 100
    
    mae = mean_absolute_error(y_real, y_predicho)
    error_relativo_mae = (mae/real)*100
    
    r2= r2_score(y_real, y_predicho)
    
    print(f"---- {model.upper()} ----")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"Error Relativo RMSE: {error_relativo_rmse:.2f} %")
    print(f"MAE: {mae}")
    print(f"Error Relativo MAE: {error_relativo_mae:.2f} %")
    print(f"R2: {r2:.2f}%")


# %%
## ===================
# Evaluación de modelos
## ===================
# Para cada modelo, se imprimirán en consola:
# MSE : error cuadrático medio
# RMSE : raíz del error cuadrático medio
# Error Relativo RMSE (%)
# MAE : error absoluto medio
# Error Relativo MAE (%)
# R² : bondad del ajuste

evaluar_modelo(y_test, y_test_pred_rfr, "test randomforestregressor")
evaluar_modelo(y_test, y_test_pred_cat, "test catboostregressor")
evaluar_modelo(y_test, y_test_pred_xgb, "test xgbregressor")
evaluar_modelo(y_test, y_test_pred_lgbm, "test lgbregressor")
evaluar_modelo(y_test, y_test_pred_mlp, "test mlpregressor")
evaluar_modelo(y_test, y_test_pred_elas, "test elasticnet")
evaluar_modelo(y_test, y_test_pred_huber, "test huberregressor")

# %% [markdown]
# ## Mejores modelos
# ##### CatBoost y LightGBM son los ganadores:
# 
# * R²: 0.80 --- Explican el 80% de la variabilidad.
# * Errores relativos bajos (16% RMSE y 11% MAE).
# * Son modelos robustos y consistentes.
# * Random Forest también es bastante competitivo (R² = 0.76), pero un poco menos preciso.

# %% [markdown]
# ## Visualización de Predicciones vs Valores Reales
# 
# * Para evaluar visualmente el desempeño de los modelos, se generaron gráficos de dispersión que muestran la relación entre los valores reales observados (en el eje X) y los valores predichos por cada modelo (en el eje Y). La línea diagonal representa la línea ideal, es decir, el caso en que las predicciones coinciden exactamente con los valores reales.

# %%
## =================================
## Graficos de dispersión de los datos : lightGBMRegressor y CatboostRegressor
## ==================================
plt.figure(figsize=(17,7))
plt.subplot(1,2,1)
sns.scatterplot(x= y_test, y= y_test_pred_lgbm, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="Red", linestyle="--")
plt.title("Dispersion de los datos del modelo LightGBMRegressor")
plt.xlabel("Precio Real Observado")
plt.ylabel("Precio Predicho por el Modelo")
plt.grid()

plt.subplot(1,2,2)
sns.scatterplot(x=y_test, y=y_test_pred_cat, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="Orange", linestyle="--")
plt.title("Distribuccion de los datos del modelo CatboostRegressor")
plt.xlabel("Precio Real Observado")
plt.ylabel("Precio Predicho por el Modelo")
plt.grid()
plt.tight_layout()

## =====================
# Análisis por modelo
## ======================
# Ambos modelos ofrecen un rendimiento sólido y mantienen un buen nivel de ajuste respecto a los datos reales.
# LightGBMRegressor muestra un comportamiento más estable y menor dispersión, lo que lo convierte en el modelo más preciso y confiable dentro de esta comparación.
# CatBoostRegressor, si bien competitivo, introduce más variabilidad en las predicciones, especialmente en los extremos de la distribución de precios.

# %%
## Ahora procedemos a elegir un modelo cualquiera para obtener la importancia de variables del modelo cat_model

importancias = cat_model.feature_importances_
importancia = pd.DataFrame({
    "Variable": x_train.columns,
    "Importancia": importancias
}).sort_values(by="Importancia", ascending=False)

importancia

# %%
plt.subplots(figsize=(20,12))
plt.barh(importancia["Variable"], importancia["Importancia"])
plt.gca().invert_yaxis()
plt.title("Variables mas importantes segun el modelo CatBoostRegressor")
plt.xlabel("Importancia de la Variable")
plt.grid(axis="x")

# El objetivo de entrenar catboost es con el fin de elegir las variables mas importantes por decir las 20 y con estas variables entrenar cada modelo para ver su desempeño al final
# y adicional se entrenaran los modelos tambien con las 50 variables y todo el data junto, para ver cual se desempeña mejor

# %%
## Procedemos a realizar una grafica de los residuos para cada uno de los modelos LightGMBRegressor y CatboostRegressor

plt.figure(figsize=(17,6))
plt.subplot(1,2,1)

# Grafico residuos lightGBMRegressor
residuos_lgbm = y_test - y_test_pred_lgbm
sns.histplot(residuos_lgbm, kde=True, color="Orange")
plt.title("Grafico de Residuos del Modelo LGBM")
plt.xlabel("Valor del Residuo")
plt.axvline(x=0, color="Red", linestyle="--")
plt.grid()

plt.subplot(1,2,2)
# Grafico residuos CatboostRegressor
residuos_cat = y_test - y_test_pred_cat
sns.histplot(residuos_cat, kde=True)
plt.title("Grafico Residuos del Modelo Catboots")
plt.xlabel("Valor del Residuo")
plt.axvline(x=0, color="Red", linestyle="--")
plt.grid()
plt.tight_layout()

## ===========================
# Analisis
## ===========================
#Gráfico de residuos del modelo LightGBM (izquierda):
#Distribución centrada en 0: El modelo predice bien en promedio.
#Simetría aceptable: Indica que el modelo no tiende a sobreestimar ni subestimar de forma sesgada.
#Presencia de colas: Hay algunos errores grandes (outliers), pero no son excesivos.
#Gráfico de residuos del modelo CatBoost (derecha):
#Muy centrado en 0: Excelente. La media del error es casi nula.
#Distribución con forma de campana (leve asimetría): Bastante aceptable.
#Menos residuos extremos que LightGBM: Indica que CatBoost podría estar generalizando mejor o es más robusto ante outliers.

# %%
## Grafico shap

import shap

explainer = shap.Explainer(lgbm_model)
shap_values = explainer(x_test)

# 1 Grafico shap  (beeswarm)
shap.plots.beeswarm(shap_values)

# %%
# 2 Grafico shap (waterfall), filtramos la casa 1 del data

shap.plots.waterfall(shap_values[1])

# %%
# 3 grafico (summary plot)

shap.summary_plot(shap_values, x_test)


