#!/usr/bin/env python
# coding: utf-8

# # Tarea #1

# ## Estudiante: Heiner Romero Leiva

# In[2]:


import os
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
from matplotlib import colors as mcolors
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler


# #### Función para calcular los índices de calidad de la predicción

# In[3]:


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


# #### Función para graficar la distribución de la variable a predecir

# In[4]:


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


# #### Función para ver la distribución de una variable categórica respecto a la predecir

# In[5]:


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

# In[6]:


def poder_predictivo_numerica(data:DataFrame, var:str, variable_predict:str):
    sns.FacetGrid(data, hue=variable_predict, height=6).map(sns.kdeplot, var, shade=True).add_legend()


# ### Ejercicio 1: 

# #### En este ejercicio usaremos los datos (voces.csv). Se trata de un problema de reconocimiento de genero mediante el analisis de la voz y el habla. Esta base de datos fue creada para identificar una voz como masculina o femenina, basandose en las propiedades acusticas de la voz y el habla. El conjunto de datos consta de 3.168 muestras de voz grabadas, recogidas de hablantes masculinos y femeninos

# #### 1. Cargue la tabla de datos voces.csv en Python.

# In[6]:


voces = pd.read_csv("voces.csv", delimiter = ',', decimal = '.')


# In[7]:


voces.info()
voces.shape
voces.info


# In[8]:


voces.describe()
# Se sacan estadisticas basicas para ver distribuciones y si es necesario centrar y normalizar las tabla, ya que Redes 
# Neuronales es un metodo basado en distancias.


# In[9]:


# Normalizando y centrando la tabla ya que hay valores en diferentes escalas
voices = voces.iloc[:,0:20]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values = scaler.fit_transform(voices) 
voices.loc[:,:] = scaled_values
voices.head()


# #### 2. Genere al azar una tabla de testing con una 20% de los datos y con el resto de los datos genere una tabla de aprendizaje.

# #### Distribución de la variable a predecir

# In[10]:


distribucion_variable_predecir(voces,"genero")
# La variable a predecir esta completamente balanceada, por lo que, en el testing las predicciones 
# deben de dar muy parecidas. 


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


# #### 3. Usando MLPClassifier genere un modelo predictivo para la tabla de aprendizaje. Utilice una cantidad suficiente de capas ocultas y nodos para que la prediccion sea buena.

# #### Usando 10000 capas ocultas de 150 nodos cada una

# In[14]:


instancia_red = MLPClassifier(solver='lbfgs', random_state=0,hidden_layer_sizes=[10000, 150])
instancia_red.fit(X_train,y_train.iloc[:,0].values)


# #### 4. Con la tabla de testing calcule la matriz de confusion, la precision, la precision positiva, la precision negativa, los falsos positivos, los falsos negativos, la acertividad positiva y la acertividad negativa. Luego construya un cuadro comparativo.

# #### Indices de Calidad del Modelo

# In[15]:


prediccion = instancia_red.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# In[16]:


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


# #### Desplegando indices personalizados

# In[19]:


datos = (([286, 3],[7, 338])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 
indices_personalizados(MC)


# #### 5. Construya un cuadro comparativo con respecto a las tareas del curso anterior. ¿Cual metodo es mejor?

# In[74]:


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
print(" ========================================")


# ##### Analisis
# * Con respecto al cuadro comparativo se puede ver que el Modelo que da los mejores resultados es el de Arboles Aleatorios junto con el XG Boosting, ya que ambos tienen la precision global mas alta de casi un 99%, ademas que la Asertividad Positiva es de mas de un 98% mientras que la negativa es de mas de un 99% lo que los hace modelos bastante confiables. 

# #### Repita los ejercicios anteriores, pero esta vez utilice el paquete Keras, utilice la misma cantidad de capas ocultas y nodos que la usada arriba. ¿Mejora la prediccion?

# #### Reescalando las variables predictoras y haciendo el split del 80% para el training y el 20% para el testing

# In[89]:


dummy_y = pd.get_dummies(y)
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_X  = pd.DataFrame(scaler.fit_transform(X), columns = list(X))
X_train, X_test, y_train, y_test = train_test_split(scaled_X, dummy_y, train_size = 0.8, random_state = 0)
print(X_train.head())
print(dummy_y)


# #### Creando el Modelo en keras junto con Capas

# In[98]:


model = Sequential()
model.add(Dense(20, input_dim = 20, activation = 'relu'))  # primera capa oculta con 10 neuronas, 20 features
model.add(Dense(100, activation = 'sigmoid'))  # segunda capa oculta con 10 neuronas
model.add(Dense(30, activation = 'relu'))  # tercera capa oculta con 10 neuronas
model.add(Dense(2, activation = 'softmax')) # capa output con 10 neuronas, el 2 son la cantidad de categorias a predecir 


# #### Compilacion del Modelo

# In[99]:


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# #### Resumen del Modelo

# In[100]:


print(model.summary())


# #### Se usan 10000 etapas de entrenamiento (Epochs) y se actualizan los pesos cada 150 observaciones procesadas

# In[101]:


model.fit(X_train, y_train, epochs = 10000, batch_size = 150, verbose = 0)
y_pred = model.predict(X_test)
y_test_class = np.argmax(np.asanyarray(y_test), axis = 1) 
y_pred_class = np.argmax(y_pred, axis = 1)


# #### Predicciones y Calidad del Modelo

# In[102]:


scores = model.evaluate(X_test, y_test)
MC = confusion_matrix(y_test_class, y_pred_class)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# In[103]:


datos = (([296, 5],[8, 325])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 
indices_personalizados(MC)


# #### 7. Compare los resultados con los obtenidos en las tareas del curso anterior.

# In[104]:


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
print(" Modelo Redes Neuronales - TensorFlow y Keras\n**************************")
print("Precisión Global: 0.9794952681388013\nError Global: 0.02050473186119872\nPrecision Positiva (PP): 0.975975975975976\nPrecision Negativa (PN): 0.9833887043189369\nFalsos Positivos (PFP): 0.016611295681063124\nFalsos Negativos (PFN): 0.024024024024024024\nAsertividad Positiva (AP): 0.9848484848484849\nAsertividad Negativa (AN): 0.9736842105263158\n**************************")
print(" ========================================")


# ##### Analisis
# * Comparando los resultados, se puede ver que la Red Neuronal usando TensorFlow da muy buenos resultados, sin embargo, el que sigue dando los mejores resultados es el Arbol de Decision junto con el Modelo XG Boosting (igual que el ejercicio previo) ya que se alcanza casi un 99% de precision global, y mas de un 98% de Asertividad Positiva y mas de un 99% de Asertividad Negativa. 

# #### Ejercicio 2:

# #### Esta pregunta utiliza los datos (tumores.csv). Se trata de un conjunto de datos de caracterısticas del tumor cerebral que incluye cinco variables de primer orden y ocho de textura y cuatro parametros de evaluacion de la calidad con el nivel objetivo. La variables son: Media, Varianza, Desviacion estandar, Asimetrıa, Kurtosis, Contraste, Energıa, ASM (segundo momento angular), Entropıa, Homogeneidad, Disimilitud, Correlacion, Grosor, PSNR (Pico de la relacion senal-ruido), SSIM (Indice de Similitud Estructurada), MSE (Mean Square Error), DC (Coeficiente de Dados) y la variable a predecir tipo (1 = Tumor, 0 = No-Tumor).

# #### 1. Usando el paquete MLPClassifier y el paquete Keras en Python genere modelos predictivos para la tabla Tumores.csv usando 70 % de los datos para tabla aprendizaje y un 30 % para la tabla testing. Utilice una cantidad suficiente de capas ocultas y nodos para que la prediccion sea buena.

# #### Utilizando paquete MLPClassifier

# In[7]:


tumores = pd.read_csv("tumores.csv", delimiter = ',', decimal = '.')
tumores.head()


# In[8]:


tumores.info()


# In[9]:


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


# In[10]:


tumores.tail() #variable categorica ha sido convertida a numero 


# #### Distribución de la variable a predecir

# In[11]:


distribucion_variable_predecir(tumores,"tipo") #Problema altamente desequilibrado. 


# In[12]:


# Normalizando y centrando la tabla ya que hay valores en diferentes escalas
tumores_1 = tumores.iloc[:,0:17]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values = scaler.fit_transform(tumores_1) 
tumores_1.loc[:,:] = scaled_values
tumores_1.head()

# Variables con escalas diferentes han sido reescaladas.


# #### Elimina la variable catégorica, deja las variables predictoras en X

# In[13]:


X = tumores_1.iloc[:,0:17] 
X.head()


# #### Deja la variable a predecir en y

# In[14]:


y = tumores.iloc[:,17:18] 
y.head()


# #### Se separan los datos con el 70% de los datos para entrenamiento y el 30% para testing

# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)


# #### Usando 10000 capas ocultas de 150 nodos cada una

# In[17]:


instancia_red = MLPClassifier(solver='lbfgs', random_state=0,hidden_layer_sizes=[10000, 150])
print(instancia_red)


# #### Entrenando al modelo mediante el metodo Fit

# In[18]:


instancia_red.fit(X_train,y_train.iloc[:,0].values)
print("Las predicciones en Testing son: {}".format(instancia_red.predict(X_test)))


# #### Índices de Calidad del Modelo

# In[19]:


prediccion = instancia_red.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### Usando Paquete Keras

# In[20]:


distribucion_variable_predecir(tumores,"tipo") # Es un problema desbalanceado.


# In[21]:


tumores.info() # Hay una categoria dentro de los datos, sin embargo esa variable ya que se habia convertido. 


# tumores_1.info()

# #### Elimina la variable catégorica, deja las variables predictoras en X

# In[22]:


X = tumores_1.iloc[:,0:17] 
X.head()


# #### Deja la variable a predecir en y 

# #### Como la variable a predecir la dan en terminos de 0 y 1, es necesario con vertirla a Si y No. 

# In[24]:


import pandas as pd
d = tumores
df = pd.DataFrame(data=d)
df


# In[40]:


df.replace({0: "No", 1: "Si"}, inplace = True)
print(df.iloc[:,17:18]) #Resultado fue reemplazado con exito. 


# In[26]:


y = df.iloc[:,17:18] 
y.head()


# #### Se separan los datos con el 70% de los datos para entrenamiento y el 30% para testing

# #### Como la variable a predecir ya viene dada por "0" y "1" no es necesario utilizar codigo disyuntivo ni reescalar

# In[27]:


dummy_y = pd.get_dummies(y)
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_X  = pd.DataFrame(scaler.fit_transform(X), columns = list(X))
X_train, X_test, y_train, y_test = train_test_split(scaled_X, dummy_y, train_size = 0.7, random_state = 0)
print(dummy_y)


# #### Creando Modelo en Keras 

# In[33]:


model = Sequential()
model.add(Dense(1000, input_dim = 17, activation = 'relu'))  # primera capa oculta con 5000 neuronas
model.add(Dense(500, activation = 'sigmoid'))  # segunda capa oculta con 10000 neuronas
model.add(Dense(300, activation = 'sigmoid'))  # tercera capa oculta con 5000 neuronas
model.add(Dense(50, activation = 'relu'))  # Agregamos tercera capa oculta con 4998 neuronas
model.add(Dense(2, activation = 'softmax')) # Agregamos capa output con 2 neuronas


# #### Complilando el Modelo

# In[34]:


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# #### Resumen del Modelo

# In[35]:


print(model.summary())


# #### Ajustes del Modelo

# #### Usamos 10000 etapas de entrenamiento (epochs) y actualizando los pesos de la red cada 150 observaciones procesadas (batch_size).

# In[36]:


model.fit(X_train, y_train, epochs = 10000, batch_size = 150, verbose = 0)
# La predicción es una matriz con 3 columnas
y_pred = model.predict(X_test)
# Convertimos a columna
y_test_class = np.argmax(np.asanyarray(y_test), axis = 1)  # Convertimos a array
y_pred_class = np.argmax(y_pred, axis = 1)


# #### Predicciones y Calidad del Modelo

# In[37]:


scores = model.evaluate(X_test, y_test)
MC = confusion_matrix(y_test_class, y_pred_class)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### 2. Calcule para los datos de testing la precision global y la matriz de confusion. Interprete la calidad de los resultados. Ademas compare respecto a los resultados obtenidos en las tareas del curso anterior.

# In[43]:


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
print(" ========================================")


# #### 3. Compare los resultados con los obtenidos en las tareas del curso anterior.

# #### Analisis
# * Comparando los resultados obtenidos con las redes neuronales con los de las tareas anteriores se puede ver como se mantienen los mejores resultados usando los Arboles Aleatorios y el XG Boosting a nivel de Precision Global (es casi de un 99%) mientras que el error es poco mas de un 1%. 
# * Se rescata que utilizando Keras y usando el paquete de TensoFlow de Google, los resultados por la categoria especifica de "no" cuenta con un tumor son considerablemente mejores que en casi todos los modelos, ya que al ser un problema desequilibrado, tiende a solo dar buenos resultados en el "Si" ya quie es donde hay mas datos para hacer la prediccion. 

# ### Ejercicio 3: 

# #### [no usar MLPClassifier ni Keras] Disene una Red Neuronal de una capa (Perceptron) para la tabla de verdad del nand:

# In[45]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/cadena.png")


# #### Es decir, encuentre los pesos w1, w2 y el umbral θ para la Red Neuronal que se muestra en el siguiente grafico, usando una funcion de activacion tipo Sigmoidea:

# In[46]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/screen.png")


# In[2]:


import numpy as np
import pandas as pd
import math

def sigmoidea(pesos, predictoras):
  x = 0
  for i in range(len(predictoras)):
    x += pesos[i] * predictoras[i]
  
  x -= pesos[len(pesos) - 1]
  return (1 / (1 + math.exp(-(x))))

def NAND(datos, resultado, pesos, ACTIVACION, corte = 0.5):
  calc = []
  for i in range(datos.shape[0]):
    calc.append(ACTIVACION(pesos, datos.iloc[i, :].values))
  
  return(pd.DataFrame({
    'res': calc, 
    'pred': [1 if a >= corte else 0 for a in calc], 
    'real': resultado
  }))

x = pd.DataFrame({'X1': [0, 1, 0, 1], 'X2': [0, 0, 1, 1]})
z = [1, 1, 1, 0]
NAND(x, z, [-0.5, -0.5, -0.5], sigmoidea) # El utlimo valor corresponde al umbral


# ### Ejercicio 4:

# [no usar MLPClassifier ni Keras] Para la Tabla de Datos que se muestra seguidamente donde x j para j = 1, 2, 3 son las variables predictoras y la variable a predecir es z disene y programe a pie una Red Neuronal de una capa (Perceptron):

# In[47]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/mini.png")


# #### Es decir, encuentre todos los posibles pesos w1, w2, w3 y umbrales θ para la Red Neuronal que se muestra en el siguiente grafico:

# In[48]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/perce.png")


# #### Use una funcion de activacion tipo Tangente hiperbolica, para esto escriba una Clase en Python que incluya los metodos necesarios pra implementar esta Red Neuronal.Se deben hacer variar los pesos wj con j = 1, 2, 3 en los siguientes valores v=(-1,-0.9,-0.8,...,0,...,0.8,0.9,1) y haga variar θ en u=(0,0.1,...,0.8,0.9,1). Escoja los pesos wj con j = 1, 2, 3 y el umbral θ de manera que se minimiza el error cuadratico medio. 

# In[3]:


def tan_hiperbolica(pesos, predictoras):
  x = 0
  for i in range(len(predictoras)):
    x += pesos[i] * predictoras[i]
  
  x -= pesos[len(pesos) - 1]
  return ((2 / (1 + math.exp(-2 * x))) - 1)

def ECM(pred, real):
  ecm = [(pred[i] - real[i])**2 for i in range(len(pred))]
  return(sum(ecm) / len(pred))

x = pd.DataFrame({'X1': [1, 1, 1, 1], 'X2': [0, 0, 1, 1], 'X3': [0, 1, 0, 1]})
z = [1, 1, 1, 0]

resultados = pd.DataFrame()
for w1 in np.arange(-10, 11)/10:
  for w2 in np.arange(-10, 11)/10:
    for w3 in np.arange(-10, 11)/10:
      for umbral in np.arange(0, 11)/10:
        aux = NAND(x, z, [w1, w2, w3, umbral], tan_hiperbolica, corte = 0)
        e = ECM(aux["pred"].values, aux["real"].values)
        nuevo = pd.DataFrame({"w1": [w1], "w2": [w2], "w3": [w3], "umbral": [umbral], "ECM": [e]})
        resultados = resultados.append(nuevo)
        
resultados.loc[resultados["ECM"] == 0, :]


# #### Como se puede ver, la funcion programada puede predecir los valores dados en las Z. 

# In[149]:


############################################################# FIN ###################################################################

