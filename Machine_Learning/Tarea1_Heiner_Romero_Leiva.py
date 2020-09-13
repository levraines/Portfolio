#!/usr/bin/env python
# coding: utf-8

# # Modelos Supervisados

# # Estudiante: Heiner Romero Leiva

# # Tarea I

# ## Ejercicio #1:

# ### Parte A)
# Programe en lenguaje Python una funcion que reciba como entrada la matriz de confusion (para el caso 2 × 2) que calcule y retorne en una lista: la Precision Global, el Error Global, la Precision Positiva (PP), la Precision Negativa (PN), los Falsos Positivos (FP), los Falsos Negativos (FN), la Asertividad Positiva (AP) y la Asertividad Negativa (NP).

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import math
import scipy.stats
import pandas.util.testing as tm


# In[2]:


def indices_general(MC):
    print(MC)
    precision_global = (MC.iloc[0,0] + MC.iloc[1,1]) / (MC.iloc[0,0] 
                        + MC.iloc[0,1] + MC.iloc[1,0] + MC.iloc[1,1]) 
    error_global  = 1 - precision_global
    precision_positiva = (MC.iloc[1,1]) / (MC.iloc[1,0] + MC.iloc[1,1])
    precision_negativa = (MC.iloc[0,0]) / (MC.iloc[0,0] + MC.iloc[0,1]) 
    falsos_positivos = (MC.iloc[0,1]) / (MC.iloc[0,0] + MC.iloc[0,1]) 
    falsos_negativos = (MC.iloc[1,0]) / (MC.iloc[1,0] + MC.iloc[1,1])
    asertividad_positiva = (MC.iloc[1,1]) / (MC.iloc[0,1] + MC.iloc[1,1])
    asertividad_negativa = (MC.iloc[0,0]) / (MC.iloc[0,0] + MC.iloc[1,0])
    return {"Precisión Global":precision_global, 
            "Error Global":error_global, 
            "Precision Positiva (PP)":precision_positiva,
            "Precision Negativa (PN)":precision_negativa,
            "Falsos Positivos (PFP)":falsos_positivos,
            "Falsos Negativos (PFN)":falsos_negativos,
            "Asertividad Positiva (AP)":asertividad_positiva,
            "Asertividad Negativa (AN)":asertividad_negativa} 


# # 
# Supongamos que tenemos un modelo predictivo para detectar Fraude en Tarjetas de Credito, la variable a predecir es Fraude con dos posibles valores Sı (para el caso en que sı fue fraude) y No (para el caso en que no fue fraude). Supongamos la matriz de confusion es:

# In[7]:


#Configuraciones de Python
import pandas as pd
pd.options.display.max_rows = 5


# In[8]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/matriz.png")


# ### Parte B)
# Calcule la Precision Global, el Error Global, la Precision Positiva (PP), la Precision Negativa (PN), los Falsos Positivos (FP), los Falsos Negativos (FN), la Asertividad Positiva (AP) y la Asertividad Negativa (NP).
# 

# In[9]:


datos = (([892254, 212], #y_actual
    [8993, 300])) #y_predicted 
df = pd.DataFrame(datos, columns = ["No", "Si"]) #No = 0, Si = 1
print (df)


# In[10]:


# Reasignando variable 
MC = df 


# In[11]:


# Retornando funcion
indices_general(MC)


# ### Parte C) 
# ¿Es bueno o malo el modelo predictivo? Justifique su respuesta

# #### Analisis:
# En este caso este modelo es bastante bueno, porque de la totalidad de predicciones que son 901759, 892554 fueron acertadas lo que da un porcentage de precisión global de un 98.97%. Por otro lado, se tiene un error muy pequeño, que es de un 1.02%. Sin embargo, al hilar más a fondo, se puede observar como la precisión positiva apenas tuve un 32% de calidad, es decir, casi un 70% de las predicciones fueron erroneas, no obstante esto no es así para detectar cuando efectivamente no es fraude, ya que la precisión alcanzó un 99%. Además, la proporción de falsos positivos es muy baja, ni siquiera un 1%, mientras que para elos falsos negativos si fue bastante alta, proque se obtuvo más de un 96%. 
# Sin embargo la asertividad negativa fue bastante buena, porque es de un 99% y la asertividad positiva es de casi un 60%. Nuevamente, en los casos en que sí es fraude, el modelo no supo identificar cuando se estaba ante fraude. 
# * Este modelo parece ajustarse mejor a cuando se quiere saber si las transacciones no corresponden a fraude, ya que identifica mejor en estos casos, no así para cuando efectivamente lo es, se recomienda utilizar en el primer caso, pero hay que ser precavido con la segunda dimensión. 

# ## Ejercicio #2:

# # 
# En este ejercicio usaremos los datos (voces.csv). Se trata de un problema de reconocimiento de genero mediante el analisis de la voz y el habla. Esta base de datos fue creada para identificar una voz como masculina o femenina, basandose en las propiedades acusticas de la voz y el habla. El conjunto de datos consta de 3.168 muestras de voz grabadas, recogidas de hablantes masculinos y femeninos.

# ### 1) Cargue la tabla de datos voces.csv en Python.

# In[3]:


voces = pd.read_csv("voces.csv", delimiter = ",", decimal = ".")
print(voces)


# In[7]:


# Extrayendo variables solamente numericas para normalizar y centrar
voice = pd.DataFrame(data = voces, columns = (['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 
'skew', 'kurt', 'sp.ent', 'sfm', 'centroid', 'meanfun', 'minfun', 'maxfun', 'meandom', 'mindom', 'maxdom', 
                                               'dfrange', 'modindx']))
voice.head()


# In[8]:


# Normalizando y centrando la tabla ya que hay valores en diferentes escalas
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values = scaler.fit_transform(voice) 
voice.loc[:,:] = scaled_values
print(voice)


# ### 2) Realice un analisis exploratorio (estadısticas basicas) que incluya: el resumen numerico (media, desviacion estandar, etc.), los valores atıpicos, la correlacion entre las variables, el poder predictivo de las variables predictoras. Interprete los resultados.
# 

# In[4]:


class exploratorio:
    def __init__(self, datos = pd.DataFrame()):
        self.__datos = pd.DataFrame(datos)
    @property
    def datos(self):
        return self.__datos
    # Definiendo variables para analisis exploratorio de datos
    def head(self):
        return self.__datos.head()
    def dimension(self):
        return self.__datos.shape
    def estadisticas(self):
        return self.__datos.describe()
    def valores_atipicos(self):
        boxplots = self.__datos.boxplot(return_type='axes')
        return boxplots
    def correlaciones(self):
        corr = self.__datos.corr()
        return corr
    def histograma(self):
        plt.style.use('seaborn-white')
        return plt.hist(self.__datos)
    def grafico_densidad(self):
        grafo = self.__datos.plot(kind='density')
        return grafo
    def test_normalidad(self):
        X = self.__datos['kurt'] 
        print(X)
        shapiro_resultados = scipy.stats.shapiro(X)
        print(shapiro_resultados)
        p_value = shapiro_resultados[1]
        print(p_value)
        # interpretación
        alpha = 0.05
        if p_value > alpha:
            print('Sí sigue la curva Normal (No se rechaza H0)')
        else:
            print('No sigue la curva Normal (Se rechaza H0)')


# In[9]:


datos = exploratorio(voice)


# In[18]:


datos.head()


# In[19]:


datos.dimension()


# In[20]:


datos.estadisticas()


# In[21]:


datos.valores_atipicos()


# In[22]:


datos.correlaciones()
f, ax = plt.subplots(figsize=(10, 8)) 
sns.heatmap(datos.correlaciones(), mask=np.zeros_like(datos.correlaciones(), dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)


# In[23]:


datos.histograma()


# In[24]:


datos.grafico_densidad()


# datos.test_normalidad()

# ### Función para graficar la distribución de la variable a predecir

# In[9]:


def distribucion_variable_predecir(data:DataFrame,variable_predict:str):
    colors = list(dict(**mcolors.CSS4_COLORS))
    df = pd.crosstab(index=data[variable_predict],columns="valor") / data[variable_predict].count()
    fig = plt.figure(figsize=(10,9))
    g = fig.add_subplot(111)
    countv = 0
    titulo = "Distribución de la variable %s" % variable_predict
    for i in range(df.shape[0]):
        g.barh(1,df.iloc[i],left = countv, align='center',color=colors[11+i],label= df.iloc[i].name)
        countv = countv + df.iloc[i]
    vals = g.get_xticks()
    g.set_xlim(0,1)
    g.set_yticklabels("")
    g.set_title(titulo)
    g.set_ylabel(variable_predict)
    g.set_xticklabels(['{:.0%}'.format(x) for x in vals])
    countv = 0 
    for v in df.iloc[:,0]:
        g.text(np.mean([countv,countv+v]) - 0.03, 1 , '{:.1%}'.format(v), color='black', fontweight='bold')
        countv = countv + v
    g.legend(loc='upper center', bbox_to_anchor=(1.08, 1), shadow=True, ncol=1)


# ### Funciones para ver la distribución de una variable respecto a la predecir (poder predictivo)

# #### Función para ver la distribución de una variable categórica respecto a la predecir
# 

# In[10]:


def poder_predictivo_categorica(data:DataFrame, var:str, variable_predict:str):
    df = pd.crosstab(index= data[var],columns=data[variable_predict])
    df = df.div(df.sum(axis=1),axis=0)
    titulo = "Distribución de la variable %s según la variable %s" % (var,variable_predict)
    g = df.plot(kind='barh',stacked=True,legend = True, figsize = (10,9),                 xlim = (0,1),title = titulo, width = 0.8)
    vals = g.get_xticks()
    g.set_xticklabels(['{:.0%}'.format(x) for x in vals])
    g.legend(loc='upper center', bbox_to_anchor=(1.08, 1), shadow=True, ncol=1)
    for bars in g.containers:
        plt.setp(bars, width=.9)
    for i in range(df.shape[0]):
        countv = 0 
        for v in df.iloc[i]:
            g.text(np.mean([countv,countv+v]) - 0.03, i , '{:.1%}'.format(v), color='black', fontweight='bold')
            countv = countv + v


# #### Función para ver la distribución de una variable numérica respecto a la predecir

# In[4]:


def poder_predictivo_numerica(data:DataFrame, var:str, variable_predict:str):
    sns.FacetGrid(data, hue=variable_predict, height=6).map(sns.kdeplot, var, shade=True).add_legend()


# In[7]:


print(voces.info())


# ### Selección de Variables - Análisis del poder predictivo de cada una de las variables predictoras

# In[8]:


poder_predictivo_numerica(voces,"meanfreq","genero")


# In[11]:


poder_predictivo_numerica(voces,"sd","genero")


# In[37]:


poder_predictivo_numerica(voces,"median","genero")


# In[38]:


poder_predictivo_numerica(voces,"Q25","genero")


# In[39]:


poder_predictivo_numerica(voces,"Q75","genero")


# In[40]:


poder_predictivo_numerica(voces,"IQR","genero")


# In[41]:


poder_predictivo_numerica(voces,"skew","genero")


# In[42]:


poder_predictivo_numerica(voces,"kurt","genero")


# In[43]:


poder_predictivo_numerica(voces,"sp.ent","genero")


# In[44]:


poder_predictivo_numerica(voces,"sfm","genero")


# In[45]:


poder_predictivo_numerica(voces,"mode","genero")


# In[46]:


poder_predictivo_numerica(voces,"centroid","genero")


# In[47]:


poder_predictivo_numerica(voces,"meanfun","genero")


# In[48]:


poder_predictivo_numerica(voces,"minfun","genero")


# In[49]:


poder_predictivo_numerica(voces,"maxfun","genero")


# In[50]:


poder_predictivo_numerica(voces,"meandom","genero")


# In[51]:


poder_predictivo_numerica(voces,"mindom","genero")


# In[52]:


poder_predictivo_numerica(voces,"maxdom","genero")


# In[53]:


poder_predictivo_numerica(voces,"dfrange","genero")


# In[54]:


poder_predictivo_numerica(voces,"modindx","genero")


# ### Interpretacion de resultados
# * El problema cuenta con 20 variables, de las que 19 son numericas continuas y la ultima variable es la predictora que es una variable categorica.
# * En el boxplot se ve que la mayoria de variables contienen valores atipicos (valores que se alejan de su desviacion estandar) y que se pueden ver como "outliers", solo dos variables de las 19 no contienen valores atipicos.
# * Con respecto a las correlaciones, se presentan fuertes correlaciones tanto positivas como negativas, con lo que se puede ver que hay muchas variables dependientes entre si, se encontro por ejemplo, que las variables meandom y sd tienen una correlacion altamente negativa, es decir cuando meandom (promedio de la frecuencia dominante medida a traves de la senal acustica) aumenta, sd (entropıa espectral) va a disminuir, y si sd aumemta, meandom va a disminuir, lo mismo pasa con: centroid y sd, Q25 y IQR, entre otras. Mientras que otras como es el caso de: skew y kurt, presentan correlaciones altamente positivas, por ejemplo cuando skew (sesgo) aumenta, tambien lo hara kurt (curtosis); lo mismo sucede con: centroid y meanfreq o maxdom y meando, entre otras mas. Pero hay unas variables que no presentan correlaciones como las antes descritas, por ejemplo: skew y sfm, es decir el sesgo y la plenitud espectral no tienen ningun tipo de correlacion y lo que le suceda a una no afectara a la otra. Esto se repite para otras como: mindom y Q75, minfum y mindom, IQR y Q75, entonces si se quisiera hacer una reduccion de la dimensionalidad en definitiva estas variables se podrian tomar para evitar sesgos en las predicciones. 
# * El histograma muestra como las variables se tienden a aglomerar en tres picos muy cerca del centro, haciendo una distribucion relativamente normal, lo que indica que la mayoria de los datos estan por debajo de la curva de normalidad, excepto por algunos datos que se salen de la desviacion (las variables numericas fueron estandarizadas y normalizadas para poder apreciar este grafico).
# * El grafico de densidad muestra que los datos tienden a tener una distribucion normal. 
# * Con lo referente al analisis de poder predictivo, se observa que muchos datos estan traspuestos entre los generos (estan encima del otro) lo que quiere decir, que no seran buenos predictores, ya que tenderan a generalizar de mas y creara una mala prediccion entre la variable meta que es genero. Sin embargo, se observa como las variables que tienen mayor poder predictor, ya que sus datos se encuentran alejados de sus distribuciones son: IQR, Q25, meanfun y sd. Sin embargo, con IQR y Q25 aunque sus datos tienen buen poder predicitivo, si se combinan estas variables cuentan con correlaciones altamente negativas, entonces se incurrira en un sesgo, se puede usar una de las dos pero mezclada con otra que tenga muy poca correlacion o correlacion nula. 

# ### 3) ¿Es este problema equilibrado o desequilibrado? Justifique su respuesta.

# In[9]:


distribucion_variable_predecir(voces,"genero")


# ### Interpretacion
# En este caso estamos ante la presencia de un problema equilibrado, ya que existe en cada distribucion de las variables genero un 50% en cada caso (hombre y mujer), es decir, ambos generos cuentan con la misma cantidad de datos, lo que permite crear una prediccion mas certera y generalizable para el modelo de test, ya que se evitara parcializar los resultados (overfitting o underfitting) que vayan a quitarle fidelidad al modelo.

# ### 4) Use el metodo de K vecinos mas cercanos en Python (con los parametros por defecto) para generar un modelo predictivo para la tabla voces.csv usando el 80 % de los datos para la tabla aprendizaje y un 20 % para la tabla testing, luego calcule para los datos de testing la matriz de confusion, la precision global y la precision para cada una de las dos categorıas. ¿Son buenos los resultados? Explique.

# In[10]:


# Prueba para saber si estandarizar y centrar tabla 
voces.iloc[:,7:10]
voces.iloc[:,7:10].describe() # Se ve que es necesario estandarizar y centrar la tabla ya que hay escalas diferentes


# ### Normalizando y centrando la tabla

# In[11]:


# Extrayendo variables solamente numericas para normalizar y centrar
voice = pd.DataFrame(data = voces, columns = (['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 
'skew', 'kurt', 'sp.ent', 'sfm', 'centroid', 'meanfun', 'minfun', 'maxfun', 'meandom', 'mindom', 'maxdom', 
                                               'dfrange', 'modindx']))

# Normalizando y centrando la tabla ya que hay valores en diferentes escalas
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values = scaler.fit_transform(voice) 
voice.loc[:,:] = scaled_values
print(voice)


# In[12]:


X = voice.iloc[:,0:19]
print(X.head())


# ### Deja la variable a predecir en y

# In[13]:


y = voces.iloc[:,20:21] 
print(y.head())


# ### Se separan los datos con el 80% de los datos para entrenamiento y el 20% para testing
# 

# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


# ### Mediante el constructor inicializa el atributo n_neighbors= 50 (3168*80%= √2534.4 = 50)

# In[15]:


instancia_knn = KNeighborsClassifier(n_neighbors=50)


# ### Entrenando el modelo llamando al método fit

# In[16]:


instancia_knn.fit(X_train,y_train.iloc[:,0].values)


# ### Se imprimen las predicciones en testing

# In[17]:


print("Las predicciones en Testing son: {}".format(instancia_knn.predict(X_test)))


# In[5]:


# Llamando funcion predefinida

def indices_generales (MC, nombres = None):
    precision_global = np.sum(MC.diagonal()) / np.sum(MC)
    error_global = 1 - precision_global
    precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
    if nombres!=None:
        precision_categoria.columns = nombres
    return {"Matriz de Confusión":MC, 
            "Precisión Global":precision_global, 
            "Error Global":error_global, 
            "Precisión por categoría":precision_categoria}


# ### Índices de Calidad del Modelo

# In[19]:


prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_generales(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ### Analisis
# Si se dejan todos los parametros por defecto, es decir, no se elimina ninguna variable, se ve que la precision global del modelo es de un 94.79% lo que indica que el modelo es bastante bueno, pero se hace la salvedad de que hay muchas variables que contienen altas correlaciones, ademas que no se toman en cuenta las variables predictoras mas fuertes. El error global es de un 5.20%, sin embargo si se analiza la precision por categoria se ve que estan balanceadas, no obstante se obtiene una mejor clasificacion del genero masculino que del femenino y al ser la variable "genero" balanceada deberian de dar muy parecido. 

# ### 5) Repita el item d), pero esta vez, seleccione las 6 variables que, segun su criterio, tienen mejor poder predictivo. ¿Mejoran los resultados?

# In[20]:


print(voice.head())


# #### Elimina la variable categórica, deja como variables predictoras en X "sd", "Q25", "IQR", "sp.ent", "sfm", "meanfun" y deja la variable a predecir en y.

# In[21]:


X = voice.iloc[:,[1,3,5,8,9,12]] 
print(X.head())
y = voces.iloc[:,20:21] 
print(y.head())


# ### Genera las tablas de training y testing con el 80% de los datos para entrenamiento y el 20% para test

# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)


# ### Generacion de modelos, predicciones y medicion de las calidades de las variables escogidas

# In[23]:


instancia_knn = KNeighborsClassifier(n_neighbors=50) #se usa 50 por el calculo indicado arriba
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_generales(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ### Analisis
# Con las variables seleccionadas que fueron: sd, Q25, IQR, sp.ent, sfm y meanfun, se puede observar como las predicciones son relativamente iguales, ya que se tiene una precision global de casi un 93%, mientras que el error es de un 7%, por otro lado, la precision por categoria tambien es bastante parecido con el modelo por defecto, y aqui tambien se puede observar como el modelo clasifica mejor a los hombres que a las mujeres, ya que casi en un 96% de los casos siempre atina, mientras que las mujeres es casi en un 90% de los casos. Tambien hay que considerar algo, estos datos estan balanceados, por eso tambien es que ambos modelos dan muy parecido. 
# * Importante: los datos fueron centrados y reducidos. 

# ### 6) Usando la funcion programada en el ejercicio 1, los datos voces.csv y los modelos generados arriba construya un DataFrame de manera que en cada una de las filas aparezca un modelo predictivo y en las columnas aparezcan los indices Precision Global, Error Global Precision Positiva (PP), Precision Negativa (PN), Falsos Positivos (FP), los Falsos Negativos (FN), la Asertividad Positiva (AP) y la Asertividad Negativa (AN). ¿Cual de los modelos es mejor para estos datos?

# In[27]:


# Desplegando funcion programada y modifcada 
def indices_general_KNN_parametros_defecto(estandar): # ---------- KNN Parametros por defecto ----------
    print(estandar)
    precision_global = (estandar.iloc[0,0] + estandar.iloc[1,1]) / (estandar.iloc[0,0] 
                        + estandar.iloc[0,1] + estandar.iloc[1,0] + estandar.iloc[1,1]) 
    error_global  = 1 - precision_global
    precision_positiva = (estandar.iloc[1,1]) / (estandar.iloc[1,0] + estandar.iloc[1,1])
    precision_negativa = (estandar.iloc[0,0]) / (estandar.iloc[0,0] + estandar.iloc[0,1]) 
    falsos_positivos = (estandar.iloc[0,1]) / (estandar.iloc[0,0] + estandar.iloc[0,1]) 
    falsos_negativos = (estandar.iloc[1,0]) / (estandar.iloc[1,0] + estandar.iloc[1,1])
    asertividad_positiva = (estandar.iloc[1,1]) / (estandar.iloc[0,1] + estandar.iloc[1,1])
    asertividad_negativa = (estandar.iloc[0,0]) / (estandar.iloc[0,0] + estandar.iloc[1,0])
    return {"Precisión Global":precision_global, 
            "Error Global":error_global, 
            "Precision Positiva (PP)":precision_positiva,
            "Precision Negativa (PN)":precision_negativa,
            "Falsos Positivos (PFP)":falsos_positivos,
            "Falsos Negativos (PFN)":falsos_negativos,
            "Asertividad Positiva (AP)":asertividad_positiva,
            "Asertividad Negativa (AN)":asertividad_negativa} 
def indices_general_KNN_parametros_definidos(modificado): # -------- KNN Parametros definidos por usuario -------
    print(modificado)
    precision_global = (modificado.iloc[0,0] + modificado.iloc[1,1]) / (modificado.iloc[0,0] 
                        + modificado.iloc[0,1] + modificado.iloc[1,0] + modificado.iloc[1,1]) 
    error_global  = 1 - precision_global
    precision_positiva = (modificado.iloc[1,1]) / (modificado.iloc[1,0] + modificado.iloc[1,1])
    precision_negativa = (modificado.iloc[0,0]) / (modificado.iloc[0,0] + modificado.iloc[0,1]) 
    falsos_positivos = (modificado.iloc[0,1]) / (modificado.iloc[0,0] + modificado.iloc[0,1]) 
    falsos_negativos = (modificado.iloc[1,0]) / (modificado.iloc[1,0] + modificado.iloc[1,1])
    asertividad_positiva = (modificado.iloc[1,1]) / (modificado.iloc[0,1] + modificado.iloc[1,1])
    asertividad_negativa = (modificado.iloc[0,0]) / (modificado.iloc[0,0] + modificado.iloc[1,0])
    return {"Precisión Global":precision_global, 
            "Error Global":error_global, 
            "Precision Positiva (PP)":precision_positiva,
            "Precision Negativa (PN)":precision_negativa,
            "Falsos Positivos (PFP)":falsos_positivos,
            "Falsos Negativos (PFN)":falsos_negativos,
            "Asertividad Positiva (AP)":asertividad_positiva,
            "Asertividad Negativa (AN)":asertividad_negativa} 


# In[24]:


# Asignando MC
datos_1 = (([290, 26],[7, 311])) 
df_1 = pd.DataFrame(datos_1, columns = ["Masculino", "Femenino"])

datos_2 = (([270,31],[14,319]))
df_2 = pd.DataFrame(datos_2, columns = ["Masculino", "Femenino"])


# In[25]:


# Cambiando parametros de funcion
estandar = df_1
modificado  = df_2 


# ### Modelo KNN Estandar

# In[28]:


indices_general_KNN_parametros_defecto(estandar)


# ### Modelo KNN con variables definidas por usuario

# In[29]:


indices_general_KNN_parametros_definidos(modificado)


# ### Analisis
# En este caso se puede observar como de los dos modelos realizados, el KNN por defecto, asi como en el que se escogieron las variables arrojan resultados muy parecidos, aunque el modelo con todas las variables sin escoger sobre el poder predictor, dio levemente mejor con casi un 95% de precision y con mejores precisiones positivas, negativas, aunque se hace la salvedad de que los falsos positivos que es la cantidad de casos negativos que fueron clasificados incorrectamente como negativos esta alta en ambos, ya que en el KNN estandar de un 82% mientras en el que se le definieron las variables es de un 102%, mientras que los falsos negativos en ambos esta relativamente baja, es decir hay pocos casos positivos que se clasifican como negativos. A nivel global se preferiria el KNN estandar, ya que es el que tiene levemente mejor los indices de precision y la asertividad. 
# * Para este caso, el mejor modelo fue el que se trabajo de forma estandar, es decir con todas las variables. 
# 
# ** Todos los datos fueron centrados y reducidos para ambos casos. 

# ### 7) Repita el ejercicio 4, pero esta vez use en el metodo KNeighborsClassifier utilice los 4 diferentes algoritmos auto, ball tree, kd tree y brute. ¿Cual da mejores resultados?

# ### Ejecución con otros parámetros

# ### Algoritmo Brute

# In[30]:


instancia_knn = KNeighborsClassifier(n_neighbors=3,algorithm='brute',p=3)
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_generales(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ### Algoritmo Auto 

# In[31]:


instancia_knn = KNeighborsClassifier(n_neighbors=3,algorithm='auto',p=2)
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_generales(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ### Algoritmo Ball Tree

# In[32]:


instancia_knn = KNeighborsClassifier(n_neighbors=3,algorithm='ball_tree',p=3)
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_generales(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ### Algoritmo Kd_tree

# In[34]:


instancia_knn = KNeighborsClassifier(n_neighbors=3,algorithm='kd_tree',p=3)
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_generales(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ### Analisis:
# En realidad todos estos algoritmos dan resultados satisfactorios ya que dan completamente igual al KNN estandar con todas las variables contempladas, aunque con respecto al algoritmo estandar de KNN estos cuatro algoritmos con otros parametros cuentan con una precision por categoria mejor con respecto al KNN estandar, ya que en los mismos se puede ver como la precision para femenino es de un 93% mientras que para el estandar es de un 91%, pero en el caso de de la precision para genero masculino, estos alcanzan una precision de un 95% mientras que el KNN estandar alcanza una precision de un 97%, es decir estos clasifican mejor mujeres que hombres y el KNN estandar el contrario, aunque en realidad estan bastante bien y se pueden usar a discresion y dependiendo que se quiera encontrar en cada caso. 
# * Importante: En KNN es frecuente que los resultados cambiando el algoritmos den igual.

# ## Pregunta #3: 

# ### Esta pregunta utiliza los datos (tumores.csv). Se trata de un conjunto de datos de caracteristicas del tumor cerebral que incluye cinco variables de primer orden y ocho de textura y cuatro parametros de evaluacion de la calidad con el nivel objetivo. La variables son: Media, Varianza, Desviacion estandar, Asimetrıa, Kurtosis, Contraste, Energıa, ASM (segundo momento angular), Entropıa, Homogeneidad, Disimilitud, Correlacion, Grosor, PSNR (Pico de la relacion senal-ruido), SSIM (´Indice de Similitud Estructurada), MSE (Mean Square Error), DC (Coeficiente de Dados) y la variable a predecir tipo (1 = Tumor, 0 = No-Tumor).

# ### Parte 1) 
# Use el metodo de K vecinos mas cercanos en Python para generar un modelo predictivo para la tabla tumores.csv usando el 70 % de los datos para la tabla aprendizaje y un
# 30 % para la tabla testing.

# In[52]:


# Importando dataset
tumores = pd.read_csv("tumores.csv", delimiter = ",", decimal = ".")
print(tumores.head())
tumores.info()


# In[40]:


# Extrayendo datos numericos y evaluando calidad. 
data = tumores.iloc[:,1:17]
print(data)


# ### Evaluando normalidad de las variables 

# In[43]:


data.iloc[:,5:8].describe() # como se puede ver los datos no estan normalizados, y tienen escalas diferentes


# In[44]:


# Normalizando y centrando la tabla ya que hay valores en diferentes escalas
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values = scaler.fit_transform(data) 
data.loc[:,:] = scaled_values
print(data)


# In[49]:


distribucion_variable_predecir(tumores,"tipo")


# ### Analisis:
# Importante: la variable a predicr no esta balanceada, ya que hay mas casos donde hay tumores (92.4%), lo que puede provocar un desajuste en el modelo.  

# In[46]:


X = data.iloc[:,0:17]
print(X.head())


# ### Deja la variable a predecir en Y

# In[48]:


y = tumores.iloc[:,17:18] 
print(y.head())


# ### Se separan los datos en un 70% para training y un 30% para testing

# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)


# ### Se escoge la cantidad de Ns del modelo, donde N = 30 (1275*70%= √893 = 30 (redondeado))

# In[53]:


instancia_knn = KNeighborsClassifier(n_neighbors=30)


# ### Entrenando el modelo llamando al método fit

# In[54]:


instancia_knn.fit(X_train,y_train.iloc[:,0].values)


# ### Se imprimen las predicciones en testing

# In[55]:


print("Las predicciones en Testing son: {}".format(instancia_knn.predict(X_test)))


# ### Indices de Calidad del Modelo

# In[56]:


prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_generales(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ### 2) Genere un Modelo Predictivo usando K vecinos mas cercanos para cada uno de los siguientes nucleos auto, ball tree, kd tree y brute ¿Cual produce los mejores resultados en el sentido de que predice mejor los tumores, es decir, Tumor = 1.

# ### Ejecucion con otros parametros

# ### Algoritmo Brute

# In[61]:


instancia_knn = KNeighborsClassifier(n_neighbors=30,algorithm='brute',p=3)
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_generales(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ### Algoritmo Auto

# In[62]:


instancia_knn = KNeighborsClassifier(n_neighbors=30,algorithm='auto',p=3)
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_generales(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ### Algoritmo ball_tree

# In[63]:


instancia_knn = KNeighborsClassifier(n_neighbors=30,algorithm='ball_tree',p=3)
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_generales(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ### Algoritmo KD_tree

# In[64]:


instancia_knn = KNeighborsClassifier(n_neighbors=30,algorithm='kd_tree',p=3)
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_generales(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ### Analisis
# En realidad al aplicar el modelo de KNN con diferntes algoritmos, todos dan el mismo resultado de prediccion para 1 (tumor) debido a que todos los modelos de KNN usados con los diferentes algoritmos: auto, ball_tree, kd_tree y brute, logran tener una precision de un 99.71% en cuanto a la deteccion de un tumor, la cual es casi perfecta, ya que el modelo tiene overfitting para el caso de cuando se tenga un tumor. (Recorda que tumor = 1). 
# * Por otro lado en KNN es frecuente que los resultados cambiando el algoritmos den igual.

# ## Pregunta #4: 

# # 
# En este ejercicio vamos a predecir numeros escritos a mano (Hand Written Digit Recognition), la tabla de aprendizaje esta en el archivo ZipDataTrainCod.csv y la tabla de testing esta en el archivo ZipDataTestCod.csv. En la figura siguiente se ilustran los datos:

# In[65]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/Numeros.png")


# #
# Los datos de este ejemplo vienen de los codigos postales escritos a mano en sobres del correo postal de EE.UU. Las im agenes son de 16 × 16 en escala de grises, cada pixel va de intensidad de −1 a 1 (de blanco a negro). Las imagenes se han normalizado para tener aproximadamente el mismo tamano y orientacion. La tarea consiste en predecir, a partir de la matriz de 16 × 16 de intensidades de cada pixel, la identidad de cada imagen (0, 1, . . . , 9) de forma rapida y precisa. Si es lo suficientemente precisa, el algoritmo resultante se utiliza como parte de un procedimiento de seleccion automatica para sobres. 
# Este es un problema de clasificacion para el cual la tasa de error debe mantenerse muy baja para evitar la mala direccion de correo. La columna 1 tiene la variable a predecir Numero codificada como sigue: 0=‘cero’; 1=‘uno’; 2=‘dos’; 3=‘tres’; 4=‘cuatro’; 5=‘cinco’;6=‘seis’; 7=‘siete’; 8=‘ocho’ y 9=‘nueve’, las demas columnas son las variables predictivas, ademas cada fila de la tabla representa un bloque 16 × 16 por lo que la matriz tiene 256 variables predictoras.

# ### 1) Ejercicio:
# Usando K vecinos mas cercanos un modelo predictivo para la tabla de aprendizaje, con los parametros que usted estime mas convenientes.

# In[68]:


# Importando datos de train 
ZipDataTrainCod = pd.read_csv("ZipDataTrainCod.csv", delimiter= ';', decimal = '.')
print(ZipDataTrainCod)


# In[69]:


# Importando datos de test
ZipDataTestCod = pd.read_csv("ZipDataTestCod.csv", delimiter = ';', decimal = '.')
print(ZipDataTestCod)


# In[70]:


distribucion_variable_predecir(ZipDataTrainCod,"Numero")


# #### La distribucion muestra que la matriz no esta del todo balancedada, porque, el 0 y el uno tienen muchos datos, mientras que otros como el 5 y el 8 tienen menos. 

# ### Eliminando la variable predictora en el conjunto de entrenamiento 

# In[73]:


X_train = ZipDataTrainCod.iloc[:,1:257] 
print(X_train.head())


# ### Definiendo la variable predictora en el conjunto de entrenamiento 

# In[74]:


y_train = ZipDataTrainCod.iloc[:,0:1] 
print(y_train.head())


# ### Eliminando la variable predictora en el conjunto de test 

# In[76]:


X_test = ZipDataTestCod.iloc[:,1:257] 
print(X_test.head())


# ### Definiendo la variable predictora en el conjunto de test

# In[77]:


y_test = ZipDataTestCod.iloc[:,0:1] 
print(y_test.head())


# ### Se escoge la cantidad de Ns del modelo, donde N = 85 (√7291  = 85 (redondeado))

# In[78]:


instancia_knn = KNeighborsClassifier(n_neighbors=85)


# ### Entrenando modelo con datos de training

# In[79]:


instancia_knn.fit(X_train,y_train.iloc[:,0].values)


# ### Imprimiendo las predicciones en testing

# In[80]:


print("Las predicciones en Testing son: {}".format(instancia_knn.predict(X_test)))


# ### 2) Ejercicio: 
# Con la tabla de testing calcule la matriz de confusion, la precision global, el error global y la precision en cada unos de los dıgitos. ¿Son buenos los resultados?

# ### Indices de Calidad del Modelo

# In[81]:


prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_generales(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ### Analisis:
# Los resultados obtenidos dan una precision global de un 87% y un error global de un 12.75%, lo cual es bueno, sin embargo analizando con detalle la precision por categoria, es decir, por numeros se ve que en los numeros que tiene mayor precision es en el 0, 1, 9 y 7, pero en los que tiene el peor desempeno es en el 4, 8, 2 y 5.

# ### 3)  Repita los ejercicios 1, 2 y 3 pero usando solamente los 3s, 5s y los 8s. ¿Mejora la prediccion?

# ### Importando datos que solo contengan 3, 5 y 8 del dataset de entrenamiento. 

# In[6]:


# Importando datos de train 
ZipDataTrainCod = pd.read_csv("ZipDataTrainCod.csv", delimiter= ';', decimal = '.')

nuevo = ZipDataTrainCod[(ZipDataTrainCod.Numero == "tres") | 
                        (ZipDataTrainCod.Numero == "cinco") | (ZipDataTrainCod.Numero == "ocho")]
print(nuevo)


# In[7]:


# Importando datos de test
ZipDataTestCod = pd.read_csv("ZipDataTestCod.csv", delimiter = ';', decimal = '.')
nuevo_2 = ZipDataTestCod[(ZipDataTestCod.Numero == "tres") | (ZipDataTestCod.Numero == "cinco")
                        | (ZipDataTestCod.Numero == "ocho")]
print(nuevo_2)


# In[11]:


distribucion_variable_predecir(nuevo,"Numero")


# #### Los numeros 3, 5 y 8 estan bastante balanceados entre si, salvo el tres que tiene un poco mas de datos

# ### Eliminando la variable predictora en el conjunto de entrenamiento

# In[12]:


X_train = nuevo.iloc[:,1:257] 
print(X_train.head())


# ### Definiendo la variable predictora en el conjunto de entrenamiento

# In[13]:


y_train = nuevo.iloc[:,0:1] 
print(y_train.head())


# ### Eliminando la variable predictora en el conjunto de test

# In[14]:


X_test = nuevo_2.iloc[:,1:257] 
print(X_test.head())


# ### Definiendo la variable predictora en el conjunto de test

# In[15]:


y_test = nuevo_2.iloc[:,0:1] 
print(y_test.head())


# ### Se escoge la cantidad de Ns del modelo, donde N = 42 (√1756 = 42 (redondeado)) 

# In[16]:


instancia_knn = KNeighborsClassifier(n_neighbors=42)


# ### Entrenando modelo con datos de training

# In[17]:


instancia_knn.fit(X_train,y_train.iloc[:,0].values)


# ### Imprimiendo las predicciones en testing

# In[18]:


print("Las predicciones en Testing son: {}".format(instancia_knn.predict(X_test)))


# ### Indices de Calidad del Modelo

# In[20]:


prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_generales(MC,list(np.unique(y_train)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ### Analisis
# Utilizando solo las columnas que contengan los numeros 5, 8 y 3, se obtiene ahora una precision global mejor de un 91% y un error global de un 8.5%, no obstante, la precision en la categorias aumentaron de estar en promedio en 70 - 80 a pasar a estar en mas de 89+, con lo que se ve, que haciendo reduccion de la dimensionalidad se pueden obtener mejores resultados, y ahora el 5, tiene una precision de un 91.25%, el 8 una de 89.75% y finalmente, el 3 una de 93.37%. 
# * En este caso, el modelo si mejora con respecto al anterior para la prediccion de estas variables. 

# ### 4) Repita los ejercicios 1, 2 y 3 pero reemplazando cada bloque 4 × 4 de pıxeles por su promedio. ¿Mejora la prediccion? Recuerde que cada bloque 16×16 esta representado por una fila en las matrices de aprendizaje y testing. Despliegue la matriz de confusion resultante

# In[23]:


# Importando datos de train 
datos = pd.read_csv("ZipDataTrainCod.csv", delimiter= ';', decimal = '.')
print(datos)


# ### Definiendo funcion que redimensione en matriz 4X4 Train set

# In[24]:


def redimension (datos, pixeles = 4):
    lista_df = []
    def reducir(fila, bloque, N=16):
        matriz = np.zeros((N,N))
        inicio = 1
        final = N + 1
        resultado = [fila.iloc[0]]
        for n in range(0,N):
            matriz[n] = fila.iloc[inicio:final]
            inicio = final 
            final += N
        for f in range(int(matriz.shape[0] / bloque)):
            for c in range(int(matriz.shape[1] / bloque)):
                suma = 0
                for f_2 in range(f * bloque, (f + 1) * bloque):
                    for c_2 in range(c * bloque, (c + 1) * bloque):
                        suma += matriz[f_2, c_2]
                resultado.append(suma / N)
        return resultado
    for x in range(datos.shape[0]):
        lista_df.append(reducir(datos.iloc[x, :], pixeles)) 
    return pd.DataFrame(lista_df)


# In[30]:


# Llamando parametros de funcion
redimension(datos)
datos_reducidos = redimension(datos)


# In[27]:


print(datos_reducidos)


# In[31]:


# Importando datos de test
datos_2 = pd.read_csv("ZipDataTestCod.csv", delimiter = ';', decimal = '.')
print(datos_2)


# ### Definiendo funcion que redimensione en matriz 4X4 Test set

# In[32]:


def redimension (datos_2, pixeles = 4):
    lista_df = []
    def reducir(fila, bloque, N=16):
        matriz = np.zeros((N,N))
        inicio = 1
        final = N + 1
        resultado = [fila.iloc[0]]
        for n in range(0,N):
            matriz[n] = fila.iloc[inicio:final]
            inicio = final 
            final += N
        for f in range(int(matriz.shape[0] / bloque)):
            for c in range(int(matriz.shape[1] / bloque)):
                suma = 0
                for f_2 in range(f * bloque, (f + 1) * bloque):
                    for c_2 in range(c * bloque, (c + 1) * bloque):
                        suma += matriz[f_2, c_2]
                resultado.append(suma / N)
        return resultado
    for x in range(datos.shape[0]):
        lista_df.append(reducir(datos.iloc[x, :], pixeles)) 
    return pd.DataFrame(lista_df)


# In[48]:


# Llamando funcion
redimension(datos_2)
datos_reducidos_2 = redimension(datos_2)


# In[35]:


print(datos_reducidos_2)


# ### Distrubucion de la variable a predecer

# In[36]:


distribucion_variable_predecir(datos_reducidos,0)


# ### Eliminando la variable predictora en el conjunto de entrenamiento

# In[37]:


X_train = datos_reducidos.iloc[:,1:17] 
print(X_train.head())


# ### Definiendo la variable predictora en el conjunto de entrenamiento

# In[38]:


y_train = datos_reducidos.iloc[:,0:1] 
print(y_train.head())


# ### Eliminando la variable predictora en el conjunto de test

# In[39]:


X_test  = datos_reducidos_2.iloc[:,1:17] 
print(X_test.head())


# ### Definiendo la variable predictora en el conjunto de test

# In[40]:


y_test = datos_reducidos_2.iloc[:,0:1] 
print(datos_reducidos_2.head())


# ### Se escoge la cantidad de Ns del modelo, donde N = 85 (√7291 = 85 (redondeado))

# In[41]:


instancia_knn = KNeighborsClassifier(n_neighbors=85)


# ### Entrenando modelo con datos training

# In[42]:


instancia_knn.fit(X_train,y_train.iloc[:,0].values)


# ### Imprimiendo las predicciones en testing 

# In[43]:


print("Las predicciones en Testing son: {}".format(instancia_knn.predict(X_test)))


# ### Indices de Calidad del Modelo

# In[47]:


prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_generales(MC,list(np.unique(y_train)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ### Analisis
# * En realidad la prediccion continua dando muy parecida a la original, ya que la precision global es de un 86% y la del original es de un 87%, los errores globales son muy parecidos (por lo que ya se menciono). 
# 
# * Pero por otro lado, con esta matriz redimensionada en 4X4 el 0, 5, 6, 8, 3 dan levemente menor, es decir, la precision para estos numeros es menor que la original, no asi para 1, 2, 3, 4, 7, 9 ya que mas bien en este caso estos aumentaron considerablemente su precision; es decir, con esta redimension, se pudo obtener mayor precision en numeros determinados. 

# ### 5):  Repita los ejercicios 1, 2 y 3 pero reemplazando cada bloque p × p de pıxeles por su promedio. ¿Mejora la prediccion? (pruebe con algunos valores de p). Despliegue las matrices de confusion resultantes.

# In[49]:


# Importando datos de train 
datos = pd.read_csv("ZipDataTrainCod.csv", delimiter= ';', decimal = '.')
print(datos)


# ### Definiendo funcion que reemplaza pixeles por promedio Train matriz 8x8

# In[59]:


def redimension (datos, pixeles = 2):
    lista_df = []
    def reducir(fila, bloque, N=16):
        matriz = np.zeros((N,N))
        inicio = 1
        final = N + 1
        resultado = [fila.iloc[0]]
        for n in range(0,N):
            matriz[n] = fila.iloc[inicio:final]
            inicio = final 
            final += N
        for f in range(int(matriz.shape[0] / bloque)):
            for c in range(int(matriz.shape[1] / bloque)):
                suma = 0
                for f_2 in range(f * bloque, (f + 1) * bloque):
                    for c_2 in range(c * bloque, (c + 1) * bloque):
                        suma += matriz[f_2, c_2]
                resultado.append(suma / N)
        return resultado
    for x in range(datos.shape[0]):
        lista_df.append(reducir(datos.iloc[x, :], pixeles)) 
    return pd.DataFrame(lista_df)


# In[60]:


# Llamando parametros de funcion
redimension(datos)
datos_reducidos = redimension(datos)


# In[61]:


print(datos_reducidos)


# In[62]:


# Importando datos de test
datos_2 = pd.read_csv("ZipDataTestCod.csv", delimiter = ';', decimal = '.')
print(datos_2)


# ### Definiendo funcion que reemplaza pixeles por promedio test matriz 8x8

# In[79]:


def redimension (datos_2, pixeles = 2):
    lista_df = []
    def reducir(fila, bloque, N=16):
        matriz = np.zeros((N,N))
        inicio = 1
        final = N + 1
        resultado = [fila.iloc[0]]
        for n in range(0,N):
            matriz[n] = fila.iloc[inicio:final]
            inicio = final 
            final += N
        for f in range(int(matriz.shape[0] / bloque)):
            for c in range(int(matriz.shape[1] / bloque)):
                suma = 0
                for f_2 in range(f * bloque, (f + 1) * bloque):
                    for c_2 in range(c * bloque, (c + 1) * bloque):
                        suma += matriz[f_2, c_2]
                resultado.append(suma / N)
        return resultado
    for x in range(datos.shape[0]):
        lista_df.append(reducir(datos.iloc[x, :], pixeles)) 
    return pd.DataFrame(lista_df)


# In[80]:


# Llamando funcion
redimension(datos_2)
datos_reducidos_2 = redimension(datos_2)


# In[81]:


print(datos_reducidos_2)


# ### Distribucion de variables a predecir

# In[66]:


distribucion_variable_predecir(datos_reducidos,0)


# ### Eliminando la variable predictora en el conjunto de entrenamiento¶

# In[68]:


X_train = datos_reducidos.iloc[:,1:65] 
print(X_train.head())


# ### Definiendo la variable predictora en el conjunto de entrenamiento¶

# In[69]:


y_train = datos_reducidos.iloc[:,0:1] 
print(y_train.head())


# ### Eliminando la variable predictora en el conjunto de test

# In[70]:


X_test  = datos_reducidos_2.iloc[:,1:65] 
print(X_test.head())


# ### Definiendo la variable predictora en el conjunto de test

# In[71]:


y_test = datos_reducidos_2.iloc[:,0:1] 
print(datos_reducidos_2.head())


# ### Se escoge la cantidad de Ns del modelo, donde N = 85 (√7291 = 85 (redondeado))

# In[72]:


instancia_knn = KNeighborsClassifier(n_neighbors=85)


# ### Entrenando modelo con datos training

# In[73]:


instancia_knn.fit(X_train,y_train.iloc[:,0].values)


# ### Imprimiendo las predicciones en testing

# In[74]:


print("Las predicciones en Testing son: {}".format(instancia_knn.predict(X_test)))


# ### Indices de calidad del Modelo 

# In[75]:


prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_generales(MC,list(np.unique(y_train)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ### Analisis:
# * Sustituyendo la matriz original por una de 8X8, la precision global mejora aun mas, ya que ahora es de 91% y el error es de un 8.6%. 
# 
# * Por otro lado, se ve que con las precisiones especificas por categorias todas aumentan, entonces el ideal seria hacer uno de 8X8 porque es en el que se alcanza una mejor precision de deteccion de los numeros escritos a mano. 
# 
