#!/usr/bin/env python
# coding: utf-8

# # Tarea #2

# # Estudiante: Heiner Romero Leiva

# ### Importacion de paquetes

# In[1]:


import os
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
from matplotlib import colors as mcolors


# ### Función para calcular los índices de calidad de la predicción

# In[2]:


def indices_general(MC, nombres = None):
    precision_global = np.sum(MC.diagonal()) / np.sum(MC)
    error_global = 1 - precision_global
    precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
    if nombres!=None:
        precision_categoria.columns = nombres
    return {"Matriz de Confusión":MC, 
            "Precisión Global":precision_global, 
            "Error Global":error_global, 
            "Precisión por categoría":precision_categoria}


# ### Función para graficar la distribución de la variable a predecir

# In[3]:


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

# ### Función para ver la distribución de una variable categórica respecto a la predecir

# In[4]:


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


# ### Función para ver la distribución de una variable numérica respecto a la predecir

# In[5]:


def poder_predictivo_numerica(data:DataFrame, var:str, variable_predict:str):
    sns.FacetGrid(data, hue=variable_predict, height=6).map(sns.kdeplot, var, shade=True).add_legend()


# ### Ejercicio 1:

# ### Pregunta 1: [25 puntos] En este ejercicio usaremos los datos (voces.csv). Se trata de un problema de reconocimiento de genero mediante el analisis de la voz y el habla. Esta base de datos fue creada para identificar una voz como masculina o femenina, basandose en las propiedades acusticas de la voz y el habla. El conjunto de datos consta de 3.168 muestras de voz grabadas, recogidas de hablantes masculinos y femeninos.

# ### 1. Cargue la tabla de datos voces.csv en Python

# In[6]:


voces = pd.read_csv("voces.csv", delimiter = ',', decimal = '.')
voces


# In[7]:


voces.info()
voces.shape
voces.info


# In[8]:


voces.describe()
# Se sacan estadisticas basicas para ver distribuciones y si es necesario centrar y normalizar las tabla. 


# In[9]:


# Normalizando y centrando la tabla ya que hay valores en diferentes escalas
voices = voces.iloc[:,0:20]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values = scaler.fit_transform(voices) 
voices.loc[:,:] = scaled_values
voices.head()


# #### Distribución de la variable a predecir¶

# In[10]:


distribucion_variable_predecir(voces,"genero")
# La variable a predecir esta completamente balanceada, por lo que, en el testing las predicciones 
# deben de dar muy parecidas. 


# ### 2. Genere al azar una tabla de testing con una 20 % de los datos y con el resto de los datos genere una tabla de aprendizaje.

# #### Elimina la variable categorica, deja las variables predictoras en X

# In[11]:


X = voices.iloc[:,0:20] 
X.head()


# #### Deja la variable a predecir en y

# In[12]:


y = voces.iloc[:,20:21] 
y.head()


# #### Se separan los datos con el 80% de los datos para entrenamiento y el 20% para testing

# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


# ### 3. Usando los metodos de Bayes, Discriminante Lineal y Discriminante Cuadratico genere modelos predictivos para la tabla de aprendizaje

# In[65]:


cadena = "-- Utilizando Metodo de Bayes --"
print(cadena.center(120," "))


# In[15]:


# Se usan los parámetros por defecto
bayes = GaussianNB()
print(bayes)


# #### Entrenando Modelo

# In[16]:


bayes.fit(X_train, y_train.iloc[:,0].values)


# #### Imprimiendo prediccion

# In[17]:


print("Las predicciones en Testing son: {}".format(bayes.predict(X_test)))


# #### Indices de Calidad del Modelo

# In[18]:


prediccion = bayes.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# In[64]:


cadena = "-- Utilizando Metodo de Discriminante Lineal --"
print(cadena.center(120, " "))


# In[26]:


# Se usan los parámetros por defecto
lda = LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage = 'auto')
print(lda)


# #### Entrenando el Modelo

# In[27]:


lda.fit(X_train, y_train.iloc[:,0].values)


# #### Imprimiendo predicciones

# In[28]:


print("Las predicciones en Testing son: {}".format(lda.predict(X_test)))


# #### Indices de Calidad del Modelo

# In[29]:


prediccion = lda.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# In[66]:


cadena = "-- Utilizando Metodo de Discriminante Cuadratico --"
print(cadena.center(120, " "))


# In[15]:


# Se usan los parámetros por defecto
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
print(qda)


# #### Entrenando Modelo

# In[16]:


qda.fit(X_train, y_train.iloc[:,0].values)


# #### Python genera un warning: 
# 
#    Explicacion: dado que el Discriminante cuadratico implica computar la inversion de una matriz, lo que no incorrecto si el determinante esta cerco de 0 (puede pasar que dos variables sean caso la combinacion lineal una de la otra), Lo que genera que los coeficientes sean imposibles de interpretar, ya que un incremento en X1 creara un decrecimiento en X2 y viceversa. 
#      
#    Hay que evaluar el modelo con la precision global y ver si la precision es mayor de un 85% para tratar la colinearidad, como la primera vez que se corrio el modelo, dio muy malos resultados (menos de un 70% de precision) se cambia el store_covariance de False a True para que calcule y guarde la matriz de covarianza en el atributo covariance y pueda aumentar la precision). 
# 

# #### Imprime las predicciones

# In[17]:


print("Las predicciones en Testing son: {}".format(qda.predict(X_test)))


# #### Indices de Calidad del Modelo 

# In[18]:


prediccion = qda.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ### 4. Con la tabla de testing calcule la matriz de confusion, la precision, la precision positiva, la precision negativa, los falsos positivos, los falsos negativos, la acertividad positiva y la acertividad negativa. Luego construya un cuadro comparativo.

# In[67]:


cadena = "-- Desplegando indices Globales Personalizados de Metodo de Bayes --"
print(cadena.center(120, " "))


# In[22]:


# Desplegando funcion programada
def indices_personalizados(MC):
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


# #### Desplegando Indices Personalizados

# In[23]:


datos = (([284, 42],[33, 275])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 
indices_personalizados(MC)


# In[68]:


cadena = "-- Desplegando indices Globales Personalizados de Metodo Discriminante Lineal --"
print(cadena.center(120, " "))


# #### Desplegando Indices Personalizados

# In[25]:


datos = (([315, 11],[5, 303])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 
indices_personalizados(MC)


# In[69]:


cadena = "-- Desplegando indices Globales Personalizados de Metodo Discriminante Cuadratico --"
print(cadena.center(120, " "))


# #### Desplegando Indices Personalizados

# In[28]:


datos = (([276, 39],[11, 308])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 
indices_personalizados(MC)


# ### Construya un cuadro comparativo con respecto a la tarea anterior y las tareas del curso anterior. ¿Cual metodo es mejor?

# In[31]:


cadena = "Cuadro Comparativo entre Modelos Supervisados"
print(cadena.center(35," "))
print(" ========================================")
print(" Modelo K Vecinos Mas Cercanos:\n**************************") 
print("Precisión Global: 0.9479495268138801\nError Global: 0.05205047318611988\nPrecision Positiva (PP): 0.9779874213836478\nPrecision Negativa (PN): 0.9177215189873418\nFalsos Positivos (PFP): 0.08227848101265822\nFalsos Negativos (PFN): 0.0220125786163522\nAsertividad Positiva (AP): 0.9228486646884273\nAsertividad Negativa (AN): 0.9764309764309764\n**************************")
print(" Arbol de decision:\n**************************")
print("Precisión Global: 0.9684542586750788\nError Global: 0.03154574132492116\nPrecision Positiva (PP): 0.9688473520249221\nPrecision Negativa (PN): 0.9680511182108626\nFalsos Positivos (PFP): 0.03194888178913738\nFalsos Negativos (PFN): 0.03115264797507788\nAsertividad Positiva (AP): 0.9688473520249221\nAsertividad Negativa (AN): 0.9680511182108626\n**************************")
print(" Arboles Aleatorios:\n**************************")
print("Precisión Global: 0.9889589905362776\nError Global: 0.01104100946372244\nPrecision Positiva (PP): 0.99375\nPrecision Negativa (PN): 0.9840764331210191\nFalsos Positivos (PFP): 0.01592356687898089\nFalsos Negativos (PFN): 0.00625\nAsertividad Positiva (AP): 0.9845201238390093\nAsertividad Negativa (AN): 0.9935691318327974\n**************************")
print(" Modelo ADA Boosting:\n**************************")
print("Precisión Global: 0.9810725552050473,\nError Global: 0.018927444794952675\nPrecision Positiva (PP): 0.990625\nPrecision Negativa (PN): 0.9713375796178344\nFalsos Positivos (PFP): 0.028662420382165606\nFalsos Negativos (PFN): 0.009375\nAsertividad Positiva (AP): 0.9723926380368099\nAsertividad Negativa (AN): 0.9902597402597403\n**************************")
print(" Modelo XG Boosting:\n**************************")
print("Precisión Global: 0.9889589905362776,\nError Global: 0.01104100946372244\nPrecision Positiva (PP): 0.99375\nPrecision Negativa (PN): 0.9840764331210191\nFalsos Positivos (PFP): 0.01592356687898089\nFalsos Negativos (PFN): 0.00625\nAsertividad Positiva (AP): 0.9845201238390093\nAsertividad Negativa (AN): 0.9935691318327974\n**************************")
print(" Modelo Maquinas de Soporte Vectorial:\n**************************")
print("Precisión Global: 0.9826498422712934\nError Global: 0.017350157728706628\nPrecision Positiva (PP): 0.9821958456973294\nPrecision Negativa (PN): 0.9831649831649831\nFalsos Positivos (PFP): 0.016835016835016835\nFalsos Negativos (PFN): 0.017804154302670624\nAsertividad Positiva (AP): 0.9851190476190477\nAsertividad Negativa (AN): 0.9798657718120806\n**************************")
print(" Modelo Redes Neuronales - MLPClassifier\n**************************")
print("Precisión Global: 0.9842271293375394\nError Global: 0.01577287066246058\nPrecision Positiva (PP): 0.9797101449275363\nPrecision Negativa (PN): 0.9896193771626297\nFalsos Positivos (PFP): 0.010380622837370242\nFalsos Negativos (PFN): 0.020289855072463767\nAsertividad Positiva (AP): 0.9912023460410557\nAsertividad Negativa (AN): 0.9761092150170648\n**************************")
print(" Modelo Redes Neuronales - Keras - TensorFlow\n**************************")
print("Precisión Global: 0.9794952681388013\nError Global: 0.02050473186119872\nPrecision Positiva (PP): 0.975975975975976\nPrecision Negativa (PN): 0.9833887043189369\nFalsos Positivos (PFP): 0.016611295681063124\nFalsos Negativos (PFN): 0.024024024024024024\nAsertividad Positiva (AP): 0.9848484848484849\nAsertividad Negativa (AN): 0.9736842105263158\n**************************")
print(" Modelo Metodo de Bayes\n**************************")
print("Precisión Global: 0.8817034700315457\nError Global: 0.1182965299684543\nPrecision Positiva (PP): 0.8928571428571429\nPrecision Negativa (PN): 0.8711656441717791\nFalsos Positivos (PFP): 0.12883435582822086\nFalsos Negativos (PFN): 0.10714285714285714\nAsertividad Positiva (AP): 0.8675078864353313\nAsertividad Negativa (AN): 0.8958990536277602\n**************************")
print(" Modelo Metodo de Discriminante Lineal\n**************************")
print("Precisión Global: 0.9747634069400631\nError Global: 0.025236593059936863\nPrecision Positiva (PP): 0.9837662337662337\nPrecision Negativa (PN): 0.9662576687116564\nFalsos Positivos (PFP): 0.03374233128834356\nFalsos Negativos (PFN): 0.016233766233766232\nAsertividad Positiva (AP): 0.964968152866242\n Asertividad Negativa (AN): 0.984375\n**************************")
print(" Modelo Metodo de Discriminante Cuadratico\n**************************")
print("Precisión Global: 0.9211356466876972\nError Global: 0.07886435331230279\nPrecision Positiva (PP): 0.9655172413793104\nPrecision Negativa (PN): 0.8761904761904762\nFalsos Positivos (PFP): 0.12380952380952381\nFalsos Negativos (PFN): 0.034482758620689655\nAsertividad Positiva (AP): 0.8876080691642652\nAsertividad Negativa (AN): 0.9616724738675958\n**************************")
print(" ========================================")


# #### Analisis
# 
# * Haciendo la comparacion con todos los modelos que se han visto hasta el momento y con respecto al cuadro comparativo se puede ver que el Modelo que da los mejores resultados es el de Arboles Aleatorios junto con el XG Boosting, ya que ambos tienen la precision global mas alta de casi un 99%, ademas que la Asertividad Positiva es de mas de un 98% mientras que la negativa es de mas de un 99% lo que los hace modelos bastante confiables. Sin embargo, para este caso el Modelo de Discriminante Lineal da muy buenos resultados, junto con el Discriminante cuadratico, no asi el de Bayes. 

# ### Ejercicio 2: 

# ### Esta pregunta utiliza los datos (tumores.csv). Se trata de un conjunto de datos de caracteristicas del tumor cerebral que incluye cinco variables de primer orden y ocho de textura y cuatro parametros de evaluacion de la calidad con el nivel objetivo. La variables son: Media, Varianza, Desviacion estandar, Asimetria, Kurtosis, Contraste, Energia, ASM (segundo momento angular), Entropıa, Homogeneidad, Disimilitud, Correlacion, Grosor, PSNR (Pico de la relacion senal-ruido), SSIM (Indice de Similitud Estructurada), MSE (Mean Square Error), DC (Coeficiente de Dados) y la variable a predecir tipo (1 = Tumor, 0 = No-Tumor).

# ### 1. Usando Bayes, Discriminante Lineal y Discriminante Cuadratico genere modelospredictivos para la tabla tumores.csv usando 70 % de los datos para tabla aprendizaje y un 30 % para la tabla testing.

# In[32]:


tumores = pd.read_csv("tumores.csv", delimiter = ',', decimal = '.')
tumores.head()


# In[33]:


tumores.info()


# In[34]:


# Convierte las variables de object a categórica
tumores['imagen'] = tumores['imagen'].astype('category')
print(tumores.info())
print(tumores.head())
# Recodifica las categorías usando números
tumores["imagen"] = tumores["imagen"].cat.codes
print(tumores.info())
print(tumores.head())
# Convierte las variables de entero a categórica
tumores['imagen'] = tumores['imagen'].astype('category')
print(tumores.info())
print(tumores.head())


# In[35]:


tumores.tail() #variable categorica ha sido convertida a numero 


# #### Distribucion de la variable a Predecir

# In[37]:


distribucion_variable_predecir(tumores,"tipo") #Problema altamente desequilibrado. 


# In[38]:


# Normalizando y centrando la tabla ya que hay valores en diferentes escalas
tumores_1 = tumores.iloc[:,0:17]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values = scaler.fit_transform(tumores_1) 
tumores_1.loc[:,:] = scaled_values
tumores_1.head()

# Variables con escalas diferentes han sido reescaladas.


# #### Elimina la variable catégorica, deja las variables predictoras en X

# In[39]:


X = tumores_1.iloc[:,0:17] 
X.head()


# #### Deja la variable a predecir en y

# In[40]:


y = tumores.iloc[:,17:18] 
y.head()


# #### Se separan los datos con el 70% de los datos para entrenamiento y el 30% para testing

# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)


# In[70]:


cadena = "-- Utilizando Metodo de Bayes --"
print(cadena.center(120," "))


# In[43]:


# Se usan los parámetros por defecto
bayes = GaussianNB()
print(bayes)


# #### Entrenamiento del Modelo

# In[44]:


bayes.fit(X_train, y_train.iloc[:,0].values)


# #### Imprimiendo predicciones del testing

# In[45]:


print("Las predicciones en Testing son: {}".format(bayes.predict(X_test)))


# #### Indices de Calidad del Modelo

# In[46]:


prediccion = bayes.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# In[71]:


cadena = "-- Utilizando Metodo de Discriminante Lineal --"
print(cadena.center(120, " "))


# In[48]:


# Se usan los parámetros por defecto
lda = LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage = 'auto')
print(lda)


# #### Entrenando Modelo

# In[50]:


lda.fit(X_train, y_train.iloc[:,0].values)


# #### Imprimiendo las predicciones

# In[52]:


print("Las predicciones en Testing son: {}".format(lda.predict(X_test)))


# #### Indices de Calidad del Modelo

# In[53]:


prediccion = lda.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# In[72]:


cadena = "-- Utilizando Metodo de Discriminante Cuadratico --"
print(cadena.center(120, " "))


# In[55]:


# Se usan los parámetros por defecto
qda = QuadraticDiscriminantAnalysis()
print(qda)


# #### Entrenando el Modelo

# In[56]:


qda.fit(X_train, y_train.iloc[:,0].values)


# #### Imprimiendo predicciones

# In[57]:


print("Las predicciones en Testing son: {}".format(qda.predict(X_test)))


# #### Indices de Calidad del Modelo

# In[58]:


prediccion = qda.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ### 2. Calcule para los datos de testing la precision global y la matriz de confusion. Interprete la calidad de los resultados. Ademas compare respecto a los resultados obtenidos en la tarea anterior y los resultados del curso anterior.

# In[60]:


cadena = "Cuadro Comparativo entre Calidades de los Modelos Supervisados"
print(cadena.center(100," "))
print(" ========================================")
print(" Modelo K Vecinos Mas Cercanos:\n**************************") 
print("Precisión Global: 0.9479495268138801\nError Global: 0.05205047318611988\n**************************")
print(" Arbol de decision:\n**************************")
print("Precisión Global: 0.9684542586750788\nError Global: 0.03154574132492116\n**************************")
print(" Arboles Aleatorios:\n**************************")
print("Precisión Global: 0.9889589905362776\nError Global: 0.01104100946372244\n**************************")
print(" Modelo ADA Boosting:\n**************************")
print("Precisión Global: 0.9810725552050473,\nError Global: 0.018927444794952675\n**************************")
print(" Modelo XG Boosting:\n**************************")
print("Precisión Global: 0.9889589905362776,\nError Global: 0.01104100946372244\n**************************")
print(" Modelo Maquinas de Soporte Vectorial:\n**************************")
print("Precisión Global: 0.9826498422712934\nError Global: 0.017350157728706628\n**************************")
print(" Modelo utilizando paquete MLPClassifier\n**************************")
print("Precisión Global: 0.9686684073107049\nError Global: 0.031331592689295085\n**************************")
print(" Modelo Redes Neuronales - TensorFlow y Keras\n**************************")
print("Precisión Global: 0.9712793733681462\nError Global: 0.02872062663185382\n**************************")
print(" Modelo Metodo de Bayes\n**************************")
print("Precisión Global: 0.9817232375979112\nError Global:0.018276762402088753\n**************************")
print(" Modelo Metodo de Discriminante Lineal\n**************************")
print("Precisión Global: 0.9765013054830287\nError Global: 0.023498694516971286\n**************************")
print(" Modelo Metodo de Discriminante Cuadratico\n**************************")
print("Precisión Global: 0.9869451697127938\nError Global: 0.01305483028720622\nn**************************")
print(" ========================================")


# #### Analisis
# 
# * De acuerdo a las predicciones obtenidas con el Metodo de Bayes, Discriminante Linel y Cuadratico, se puede ver como las tres dan bastante bien, no obstante, la que da mejor es la del Discriminante Cuadratico, ya que la Precision Global es de un 98.69%. La precision por categoria del "no posee cancer" es de un 100%, mientars que la del "posee cancer" es de un 98.62%. Seguida del Metodo de Bayes con valores muy similares, la del Discriminante Lineal pese a que la prediccion no es mala, la prediccion del no es de un 75% mucho menor a las de los dos otros metodos. 
# 
# * Comparando los resultados obtenidos con las redes neuronales con los de las tareas anteriores se puede ver como se mantienen los mejores resultados usando los Arboles Aleatorios y el XG Boosting a nivel de Precision Global (es casi de un 99%) mientras que el error es poco mas de un 1%.

# ### Ejercicio 3: 

# ### [Filmina 19 de Bayes] Supongamos que se tiene una nueva fila o registro de la base de datos t = (Pedro, M, 4, ?), prediga (a mano) si Pedro corresponde a la clase pequeno, mediano o alto.

# In[61]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/probabilidad.png")


# ### Ejercicio 4: 

# #### [Filmina 24 de Bayes] Realice la predicciones (a mano) para el registro numero 101.

# In[62]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/evento.png")


# In[73]:


cadena = "=============== FIN =============== "
print(cadena.center(120, " "))


# In[ ]:




