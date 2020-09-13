#!/usr/bin/env python
# coding: utf-8

# ## Modelos No Supervisados

# ## Estudiante: Heiner Romero Leiva

# ## Tarea **#4:**

# In[2]:


# Importacion de paquetes
import os
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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

# #### Función para ver la distribución de una variable categórica respecto a la predecir

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


# #### Función para ver la distribución de una variable numérica respecto a la predecir

# In[5]:


def poder_predictivo_numerica(data:DataFrame, var:str, variable_predict:str):
    sns.FacetGrid(data, hue=variable_predict, height=6).map(sns.kdeplot, var, shade=True).add_legend()


# ### **Pregunta 1)**

# #### En este ejercicio usaremos los datos (voces.csv). Se trata de unproblema de reconocimiento de genero mediante el analisis de la voz y el habla. Esta base de datos fue creada para identificar una voz como masculina o femenina, basandose en las propiedades acusticas de la voz y el habla. El conjunto de datos consta de 3.168 muestras de voz grabadas, recogidas de hablantes masculinos y femeninos.

# #### 1) Cargue la tabla de datos voces.csv en Python.

# In[23]:


data = pd.read_csv("voces.csv", delimiter = ',', decimal = '.')
data.head()


# In[24]:


print(data.info()) # Se ve que todas las variables son de tipo float 64 y solo la variable a predecir es de tipo objeto. 


# #### Distribucion de la variable a predecir

# In[40]:


distribucion_variable_predecir(voces,"genero")


# ##### Se puede observar como el problema esta completamente balanceado, por lo que, deberian los predictores globales tanto para masculino como femenino dar muy parecido. 

# In[25]:


data.describe() #Se ve que el dataframe presenta varias variables con diferentes escalas, como es el caso de 
# skew, kurt, etc., por lo que es necesario centrar y reducir la tabla, ya que el SVM es un metodo basado en distancias.


# #### 2) Use Maquinas de Soporte Vectorial en Python (con los parametros por defecto) para generar un modelo predictivo para la tabla voces.csv usando el 80 % de los datos para la tabla aprendizaje y un 20 % para la tabla testing, luego calcule para los datos de testing la matriz de confusion, la precision global y la precision para cada una de las dos categorias. ¿Son buenos los resultados? Explique.

# #### Elimina la variable catégorica, deja las variables predictoras en X - Separacion y seleccion de variables 

# In[37]:


X = data.iloc[:, :20]
print(X)


# #### Dejando la variable a predecir en y

# In[38]:


y = data.iloc[:,20:21]
print(y)


# #### Centrando y normalizando la tabla 

# In[41]:


# Normalizando y centrando la tabla 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values = scaler.fit_transform(X) 
X.loc[:,:] = scaled_values
print(X)


# #### Se separan los datos con el 80% de los datos para entrenamiento y el 20% para testing

# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


# #### Se usan los parámetros por defecto de SVM

# In[43]:


instancia_svm = SVC()
print(instancia_svm)


# #### Entrenando el modelo llamando al método fit

# In[44]:


instancia_svm.fit(X_train,y_train.iloc[:,0].values)


# #### Imprimiendo las predicciones en testing

# In[46]:


print("Las predicciones en Testing son: {}".format(instancia_svm.predict(X_test)))


# #### Índices de Calidad del Modelo

# In[49]:


prediccion = instancia_svm.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### Analisis
# * Para este modelo, los resultados son bastante buenos, ya que se logra llegar a una precision global de un 98.26% y el error es de poco mas de un 1.5%. 
# * Por otro lado, se obtienen precisiones bastante buenas en cuanto a las precisiones especificas por categoria, tanto masculino como femenino, ya que en ambos casos da poco mas de un 98%. Como se vio desde el inicio el problema es completamente equilibrado, y se tienen precisiones muy similares, dando como resultado una buena generalizacion del modelo. 

# #### 3) Usando la funcion programada en el ejercicio 1 de la tarea anterior, los datos voces.csv y los modelos generados arriba construya un DataFrame de manera que en cada una de las filas aparezca un modelo predictivo y en las columnas aparezcan los indices Precision Global, Error Global Precision Positiva (PP), Precision Negativa (PN), Falsos Positivos (FP), los Falsos Negativos (FN), la Asertividad Positiva (AP) y la Asertividad Negativa (AN). Compare con todos los medelos generados en las tareas anteriores ¿Cual de los modelos es mejor para estos datos?

# In[50]:


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


# In[51]:


# Asignando MC
datos = (([292, 5],[6, 331])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 


# #### Modelo con funcion personalizada

# In[52]:


indices_personalizados(MC)


# #### Analisis 
# 
# * Comparando con todos los modelos anteriores, en cuanto a los arboles de decision se ve que la precision global es de un 96%, al igual que la asertividad positiva y negativa. 
# * Con respecto al KNN, este indice de precision global era de un 94%, por otro lado las predicciones de los generos rondaban el 90 a 91% para ambos casos. 
# * Con arboles aleatorios, la precision era de un 97% a nivel global, pero predecia mejor a las observaciones masculinas que a las femeninas, es decir, no habia un balance. Lo mismo pasaba con ADA boosting y XG Boosting, las predicciones eran muy similares, pero se penalizan las predicciones especificas por categoria. 
# * En cuanto al SVM se ve que la prediccion es bastante alta, y ademas por categorias especificas por predecir son bastante parecidas, por lo que, el mejor modelo para estos casos es el de Support Vector Machine. 
# 

# #### 4) Repita los ejercicios 1-3, pero esta vez use otro nucleo (Kernel). ¿Mejora la prediccion?.

# #### La importacion y separacion de los datos continua en memoria, entonces solo se importara el metodo.

# #### Usando otro nucleo Kernel - "poly"

# In[61]:


instancia_svm = SVC(kernel='poly')
print(instancia_svm)
instancia_svm.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_svm.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# In[57]:


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


# In[62]:


# Asignando MC
datos = (([279, 18],[6, 331])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 


# #### Indices Generales con funcion personalizada

# In[63]:


indices_personalizados(MC)


# #### Analisis
# 
# * Cambiando los parametros por defecto del SVM, se como las precisiones generales desmejoran y se ve como las categorias de prediccion se desbalancean, dando mejores predicciones a favor del genero masculino. 
# * En cuanto a los otros metodos, el KNN con cambio de parametros tenia una precision global de un 94% y tenia el mismo problema, generalizaba mejor a las voces masculinas que femeninas. 
# * Los arboles de decision con cambio de parametros, tenia una precision global de poco mas de un 96% y para las voces masculinas tenia una prediccion global de un 98% y femeninas de mas de un 95%. 
# * Con lo referente al ADA Boosting cambiando parametros por defecto, se obtuvo una precision bastante mala de un 46%, pero con Random Forest y XG Boosting, las predicciones fueron de mas de un 97%. Y para ambos modelos, las categorias de prediccion masculina fueron de 97% y 96% respectivamemte y las de femenino de 98% y 99% respectivamente, con esto se puede ver como los mejores resultados con cambios en los metodos han sido modelados con los potenciadores, ya sea Random Forest o XG Boosting.  

# #### 5) Repita los ejercicios 1-4, pero esta vez use 2 combinaciones diferentes de seleccion de 6 variables predictoras. ¿Mejora la prediccion?.

# #### Importando las mejores 6 variables

# In[71]:


# Importando las 6 mejores variables predictoras
X = data.iloc[:,[1,3,5,8,9,12]] 
print(X.head())
y = data.iloc[:,20:21] 
print(y.head())


# #### Las variables elegidas ya se encuentran centradas y normalizadas. 

# #### Usando la primera combinacion de seleccion

# In[73]:


# Definiendo parametros personalizados de modelo y creando predicciones

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)

# Usando la primera combinacion de seleccion
instancia_svm = SVC(kernel='rbf')

print(instancia_svm)
instancia_svm.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_svm.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### Indices Generales con funcion personalizada - Primera combinacion con 6 variables

# In[75]:


# Asignando MC
datos = (([303, 16],[10, 305])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 


# In[76]:


indices_personalizados(MC)


# #### Usando la segunda combinacion de seleccion

# In[74]:


# Definiendo parametros personalizados de modelo y creando predicciones

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)

# Usando la primera combinacion de seleccion
instancia_svm = SVC(kernel='linear')

print(instancia_svm)
instancia_svm.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_svm.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### Indices Generales con funcion personalizada - Segunda combinacion con 6 variables

# In[77]:


# Asignando MC
datos = (([288, 36],[8, 302])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 


# In[78]:


indices_personalizados(MC)


# #### Analisis General 
# * Si se cambian los parametros por defecto y se usan solo 6 variables se obtienen buenois resultados en ambos escenarios, no obstante la que genera mejores resultados es la que tiene el parametro: "rbf" ya que, tiene una precision global de casi un 96% y en las predicciones por categoria, estan muy de la mano, ya que las predicciones masculinas ascienden a casi 97% y las femeninas a casi 95%. 
# * Con los arboles de decision estas predicciones rondaban un 88% y las categorias a nivel especifico (masculino y femenino) rondaban el 88% y 89% respectivamente. 
# * Con KNN y con los diferentes algoritmos, se alcanzo una precision maxima de un 94%, en cuanto a las predicciones masculinas y femeninas rondaban el 93 y el 95% respectivamente. 
# * Usando Random Forest, ADA boosting y XG Boosting, las predicciones con parametros personalizados y con 6 variables rondaban el 97% al 98% y las predicciones por categoria eran de un 98 y 97% (Femenino y Masculino respectivamente). 
# 
# * Se puede observar como los metodos que generan las mejores precisiones seleccionando diferentes parametros por defecto y solo 6 variables es el del Random Forest y XG Boosting. 

# ## **Pregunta 2:** 

# #### Esta pregunta utiliza los datos (tumores.csv). Se trata de un conjunto de datos de caracteristicas del tumor cerebral que incluye cinco variables de primer orden y ocho de textura y cuatro parametros de evaluacion de la calidad con el nivel objetivo. La variables son: Media, Varianza, Desviacion estandar, Asimetria, Kurtosis, Contraste, Energia, ASM (segundo momento angular), Entropia, Homogeneidad, Disimilitud, Correlacion, Grosor, PSNR (Pico de la relacion senal-ruido), SSIM (Indice de Similitud Estructurada), MSE (Mean Square Error), DC (Coeficiente de Dados) y la variable a predecir tipo (1 = Tumor, 0 = No-Tumor).

# #### 1) Use Maquinas de Soporte Vectorial en Python para generar un modelo predictivo para la tabla tumores.csv usando el 70 % de los datos para la tabla aprendizaje y un 30 % para la tabla testing.

# In[79]:


# Importando datos 
tumores = pd.read_csv("tumores.csv", delimiter = ',', decimal = '.')
print(tumores)


# In[80]:


tumores.info()


# In[81]:


# Hay un objeto en la primera columna que es un objeto y que es necesario recategorizar.   
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


# In[83]:


tumores.describe() # Se puede observar como las variables estan en diferentes escalas, por lo que se deben normalizar


# #### Equilibrio de la Variable a Predecir

# In[85]:


distribucion_variable_predecir(tumores,"tipo") 
# Se puede observar como estamos ante de la presencia de una variable desbalanceada a predecir, 
# entonces las predicciones puede que no sean tan altas


# #### Elimina la variable catégorica, deja las variables predictoras en X

# In[115]:


X = tumores.iloc[:,0:17] 
print(X.head())


# #### Normalizando y centrando la tabla 

# In[116]:


# Normalizando y centrando la tabla 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values = scaler.fit_transform(X) 
X.loc[:,:] = scaled_values
print(X)


# #### Deja la variable a predecir en y

# In[117]:


y = tumores.iloc[:,17:18] 
print(y.head())


# In[118]:


#Se separan los datos con el 70% de los datos para entrenamiento y el 30% para testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70)


# #### Se usan los parametros por defecto del SVM

# In[119]:


instancia_svm = SVC()
print(instancia_svm)


# In[120]:


# Entrenando el modelo
instancia_svm.fit(X_train,y_train.iloc[:,0].values)


# In[125]:


# Imprimiendo las predicciones en el testing
print("Las predicciones en Testing son: {}".format(instancia_svm.predict(X_test)))


# #### Indices de Calidad del Modelo

# In[126]:


prediccion = instancia_svm.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### 2. Usando la funcion programada en el ejercicio 1 de la tarea anterior, los datos tumores.csv y los modelos generados arriba construya un DataFrame de manera que en cada una de las filas aparezca un modelo predictivo y en las columnas aparezcan los indices Precision Global, Error Global Precision Positiva (PP), Precision Negativa (PN), Falsos Positivos (FP), los Falsos Negativos (FN), la Asertividad Positiva (AP) y la Asertividad Negativa (AN). Compare los resultados con todos los modelos generados en las tareas antariores ¿Cual de los modelos es mejor para estos datos?

# #### Indices con funcion personalizada

# In[2]:


# Asignando MC
datos = (([6, 31],[2, 344])) 
df = pd.DataFrame(datos, columns = ["No tumor", "Tumor"])
MC = df 


# In[128]:


indices_personalizados(MC)


# #### Analisis
# * Con SVM, se observa como la precision global es de un 91.38% y de se tiene que el modelo es malo prediciendo cuando una persona no tiene un tumor, ya que la mayoria siempre se predicen como con que cuenta con un tumor. 
# * Con KNN se tenia el mismo indice de precision global, pero la precision cuando no tenia tumores era mejor, casi un 32%. 
# * Con los arboles de decision, la precision global que se alcanzo fue de un 98% y para ambas categorias tanto cuando existe tumor de cuanto no existe, las predicciones son bastante buenas, 96 y 98% respectivamente. 
# * En cuanto al Random Forest, El XG boosting y el ADA boosting, las precisiones rondan entre el 97 y 98% y las predicciones por categoroa entre el 94 y el 96%. 
# * Con lo referente a este tipo de datos, el que hace un mejor analisis sin duda fue el de Arboles de Decision. 

# #### 3) Repita los ejercicios 1-2, vez use otro nucleo (Kernel). ¿Mejora la prediccion?

# #### La importacion y separacion de los datos continua en memoria, entonces solo se importara el metodo.

# #### Usando otro nucleo Kernel - "Linear"¶

# In[132]:


instancia_svm = SVC(kernel='linear')
print(instancia_svm)
instancia_svm.fit(X_train,y_train.iloc[:,0].values)
prediccion = instancia_svm.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### Indices con funcion personalizada

# In[134]:


# Asignando MC
datos = (([27, 10],[9, 337])) 
df = pd.DataFrame(datos, columns = ["No tumor", "Tumor"])
MC = df 


# In[135]:


indices_personalizados(MC)


# #### Analisis 
# * Cambiando los parametros por defecto, y usando "linear" los resultados mejoran, ya que se pasa de 91% a un 95% y tambien la prediccion por categoria tambien aumentan [0] de un 16% a un 73% mientras que la de [1] mas bien disminuye de un 99% a un 97%, pero da mejores resultados. 

# ### **Pregunta #3:**

# ### Suponga que se tiene la siguiente tabla de datos:

# In[136]:


#Configuraciones de Python
import pandas as pd
pd.options.display.max_rows = 5

from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/graph.png")


# ### 1) Investigue sobre paquete en Python que permiten realizar graficacion en 3D. Escoja en que mejor considere para resolver los siguientes ejercicios.

# ### 2) Dibuje con colores los puntos de ambas clases en R3. 

# In[34]:


tabla = {'Observaciones': np.array([[1,0,1], [1,0,2], [1,1,2],[3,1,4], [1,1,3],[3,2,3],[1,2,1],[3,2,1],[1,1,0]]), 
         'Clase': np.array([0,0,0,0,0,1,1,1,1]), 'Clase_nombre': np.array(['Rojo', 'Azul'])}

print(tabla)


# In[18]:


import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC 


# In[19]:


# Se importa paquete con Java para graficas en 3D. 
get_ipython().run_line_magic('matplotlib', 'notebook')
plt.rcParams['figure.figsize'] = (6,4)
plt.rcParams['figure.dpi'] = 150


# In[35]:


from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
X = tabla['Observaciones'] # Se toma la columna X, Y y Z. 
Y = tabla['Clase'] # Se toma toda la clase. 
print(X)
print(Y)


# In[15]:


# Creando una variable binaria 
X = X[np.logical_or(Y==0,Y==1)]
Y = Y[np.logical_or(Y==0,Y==1)]
model = svm.SVC(kernel='linear') # Asignando parametro "lineal"
clf = model.fit(X, Y)
tmp = np.linspace(-5,5,30)
x,y = np.meshgrid(tmp,tmp)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'sr') #Ploteando ejes con colores
ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'ob') # Ploteando ejes con colores
plt.show()


# ### Segunda vista, ya que no se aprecia bien

# In[16]:


# Creando una variable binaria 
X = X[np.logical_or(Y==0,Y==1)]
Y = Y[np.logical_or(Y==0,Y==1)]
model = svm.SVC(kernel='linear') # Asignando parametro "lineal"
clf = model.fit(X, Y)
tmp = np.linspace(-5,5,30)
x,y = np.meshgrid(tmp,tmp)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'sr') #Ploteando ejes con colores
ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'ob') # Ploteando ejes con colores
plt.show()


# ### 3. Dibuje el hiperplano optimo de separacion e indique la ecuacion de dicho hiperplano de la forma ax+by +cz +d = 01 Nota: Se debe observar con detenimiento los puntos de ambas clases para encontrar los vectores de soporte de cada margen y trazar con estos puntos los hiperplanos de los margenes luego trazar el hiperplano de soporte justo en el centro.

# #### Encontrando plano de separacion

# In[4]:


# como el problema que se pide representar esta en 3D, se pasa a 2D para poder visualizar mejor los puntos 

# Se sustituye el Rojo por el 0 y el Azul por el 1 para hacer una variable binaria 

tabla = pd.DataFrame({'X': [1, 1, 1, 3, 1, 3, 1, 3, 1], 'Y': [0,0,1,1,1,2,2,2,1], 'Z': [1,2,2,4,3,3,1,1,0],
                     'Clase': ['0', '0','0','0','0','1','1','1','1']}, 
                  columns=['X','Y','Z','Clase'])

X=tabla.iloc[:,0:3] # Ploteando solo en dos dimensiones y en el eje x y y 
y=tabla['Clase']
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50, cmap='autumn')


# #### Encontrando vectores propios 

# In[5]:


from sklearn.svm import SVC 
model = SVC(kernel='linear', C=1E10)
model.fit(X, y)
 
model.support_vectors_
# Los vectores de soporte son (2D): 


# #### Ploteando los Vectores

# In[8]:


plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50, cmap='autumn')
plt.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1])


# #### Analisis
# * Los vectores de soporte encontrados son 4. 
# * Respectivamente se puede ver como el primer punto esta entre los puntos 3, 2 y , el segundo entre 1, 1 y 2, 3, 1 y 4, y 1,1 y 0 respectivamente. 
# * Como se sabe que, los vectores de soporte representan los puntos especificos en el eje "x" y "y", entonces, se procede a utilizar el punto medio entre los puntos para encontrar la ecuaciacion de dicho hiperplano (se toma solamente los vectores en 2D para hacer mas facil la interpretacion como en el grafico), entonces se tiene lo siguiente: 
#     * Se toma como referencia el punto 1 y 1, y el punto medio de este es 0.5, y para el segundo es 3 y 2, punto    medio, seria 1.5, entonces se puede construir la ecuacion de la siguiente forma: 
#         * f(x): Puntos P1=(0.5,0.5) y P2=(1.5,1.5), lo que es igual a = Pendiente m= (Y2-Y1)/(X2-X1) =(1.5-0.5)/(1.5-0.5)=1, Usando P1=(1.5,1.5) b=y-mx=1.5-(1*1.5)=0, entonces la recta de separación es f(x)=x. 

# #### Ploteando el Hiper Plano

# In[18]:


# Creando una variable binaria 
X = X[np.logical_or(Y==0,Y==1)]
Y = Y[np.logical_or(Y==0,Y==1)]
model = svm.SVC(kernel='linear')
clf = model.fit(X, Y)

# La ecuacion de la separacion del plano esta dada por = np.dot(svc.coef_[0], x) + b = 0 o 
# b=y-mx=1.5-(1*1.5)=0, entonces la recta de separación es f(x)=x.

# Solucion para w = (z) - Regla de clasificacion (explicada en el punto 4)
z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x -clf.coef_[0][1]*y) / clf.coef_[0][2]

tmp = np.linspace(-5,5,30)
x,y = np.meshgrid(tmp,tmp)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'sr')
ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'ob')
ax.plot_surface(x, y, z(x,y))
ax.view_init(30, 60)
plt.show()


# ### 4. Escriba la regla de clasificacion para el clasificador con margen maximo. Debe ser algo como lo siguiente: w = (w1, w2, w3) se clasifica como Rojo si ax + by + cz + d > 0 y otro caso se clasifica como Azul.

# #### Analisis:
# * En el punto 3 se encontraron los vectores de soporte que son: 3 y 2, 1 y 1, 3 y 1, 1 y 1, que corresponden a los puntos representados por el Scatter plot en la grafica que tiene el titulo: "Ploteando los Vectores". 
#    * Dado lo anterior, el procedimiento para encontrar h(x): Puntos es igual a Punto=(3,2) y segundo punto =(1,1).
#    * Entonces: la Pendiente m= (Y2-Y1)/(X2-X1) =(2-1)/(3-1)= 0.5 b=y-mx =2-1*3=-1 y g(x) se calcula análogamente. 
#        * Se calcula como rojo si ax + by + cz + d > -1 y se clasifica como azul si ax + by + cz + d < -1, entonces se grafica de la siguiente forma: 

# In[133]:


#Creando una variable binaria
X = X[np.logical_or(Y==0,Y==1)]
Y = Y[np.logical_or(Y==0,Y==1)]
model = svm.SVC(kernel='linear')
clf = model.fit(X, Y)
# La ecuacion de la separacion del plano esta dada por = np.dot(svc.coef_[0], x) + b = 0.

# Solucion para w = (z) - Regla de clasificacion 
# m= (Y2-Y1)/(X2-X1) =(2-1)/(3-1)= 0.5 b=y-mx =2-1*3=-1 y g(x)
# ax + by + cz + d > -1 y se clasifica como azul si ax + by + cz + d < -1
z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x -clf.coef_[0][1]*y) / clf.coef_[0][2]

tmp = np.linspace(-5,5,30)
x,y = np.meshgrid(tmp,tmp)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'sr')
ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'ob')
ax.plot_surface(x, y, z(x,y)) # Se utiliza "z" con la regla de clasificacion
ax.view_init(30, 60)
plt.show()


# ### 5) Indique el margen para el hiperplano optimo y los vectores de soporte.

# #### Analisis:
# * En el punto 3 se encontraron los vectores de soporte que son: 3 y 2, 1 y 1, 3 y 1, 1 y 1, que corresponden a los puntos representados por el Scatter plot en la grafica que tiene el titulo: "Ploteando los Vectores". 
#    * Dado lo anterior, el procedimiento para encontrar h(x): Puntos es igual a Punto=(3,2) y segundo punto =(1,1).
#    * Entonces: la Pendiente m= (Y2-Y1)/(X2-X1) =(2-1)/(3-1)= 0.5 b=y-mx =2-1*3=-1 y g(x) se calcula análogamente. 
#        * Ahora bien, tenemos la "y", la "m" y la "b", entonces solo hay que despejar la ecuacion  (y tomando el segundo punto (1,1)) y = mx + b, donde y = 1, m = 0.5, y b = -1, entonces, (y - b) / m = (1-0.5)/-1 = -0.5 y las rectas que delimitan el margen vienen dadas por h(x)=x+2 y g(x)=x-2, entonces -0.5 + 2 y -0.5 - 2 = 1.5 y - 2.5, por lo tanto el margen para el hiperplano optimo es el rango [1.5, -2.5]. 

# #### Vectores de Soporte:

# In[11]:


from sklearn.svm import SVC 
model = SVC(kernel='linear', C=1E10)
model.fit(X, y) 

model.support_vectors_


# ### 6) Explique por que un ligero movimiento de la octava observacion no afectaria el hiperplano de margen maximo.

# #### La octava observacion es [X = 3, Y = 2, Z =1] y es azul, un ligero movimiento no afectaria el hiperplano, porque al ser azul quiere decir que:  ax + by + cz + d < -1, y ademas, el margen esta modelado por -2.5 (porque es negativo) lo que hace que este por encima de el (sigue siendo separable) entonces no habria ninguna afectacion con el margen maximo. 

# ### 7) Dibuje un hiperplano que no es el hiperplano optimo de separacion y proporcione la ecuacion para este hiperplano.

# #### Simplemente en lugar de tener la ecuacion por defecto (y = mx + b), se cambia a y = m+x+b donde y = 1, m = 0.5, y b = -1, entonces, y - b - m = x , entonces = 1--1-0.5  = 1.5  y las rectas que delimitan el margen vienen dadas por h(x)=x+2 y g(x)=x-2, entonces -1.5 + 2 y -1.5 - 2 = 0.5 y - 3.5, por lo tanto el margen para el hiperplano quedaria entre [0.5 y -3.5] con lo que ya no separaria todos los puntos. 

# In[39]:


#Creando una variable binaria
X = X[np.logical_or(Y==0,Y==1)]
Y = Y[np.logical_or(Y==0,Y==1)]
model = svm.SVC(kernel='linear')
clf = model.fit(X, Y)
# La ecuacion de la separacion del plano esta dada por = np.dot(svc.coef_[0], x) + b = 0.

# Solucion para w = (z) - Regla de clasificacion 
# m= (Y2-Y1)/(X2-X1) =(2-1)/(3-1)= 0.5 b=y-mx =2-1*3=-1 y g(x)
# ax + by + cz + d > -1 y se clasifica como azul si ax + by + cz + d < -1
z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]**x -clf.coef_[0][1]*y) / clf.coef_[0][2] # Se hace la modificacion 
 
tmp = np.linspace(-5,5,30)
x,y = np.meshgrid(tmp,tmp)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'sr')
ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'ob')
ax.plot_surface(x, y, z(x,y)) # Se utiliza "z" con la regla de clasificacion
ax.view_init(30, 60)
plt.show()


# ### 8) Dibuje un hiperplano de separacion pero que no es el hiperplano optimo de separacion, y escriba la ecuacion correspondiente.

# #### Se vuelve a hacer otro ejemplo: en lugar de tener la ecuacion por defecto (y = mx + b), se cambia a y = mxb donde y = 1, m = 0.5, y b = -1, entonces, y / mb = x , entonces = 1/0.5*-1 = -2 y las rectas que delimitan el margen vienen dadas por h(x)=x+2 y g(x)=x-2, entonces -2 + 2 y -2 - 2 = 0 y - 4, por lo tanto el margen para el hiperplano quedaria entre [0 y -4] con lo que ya no separaria todos los puntos.

# In[40]:


#Creando una variable binaria
X = X[np.logical_or(Y==0,Y==1)]
Y = Y[np.logical_or(Y==0,Y==1)]
model = svm.SVC(kernel='linear')
clf = model.fit(X, Y)
# La ecuacion de la separacion del plano esta dada por = np.dot(svc.coef_[0], x) + b = 0.

# Solucion para w = (z) - Regla de clasificacion 
# m= (Y2-Y1)/(X2-X1) =(2-1)/(3-1)= 0.5 b=y-mx =2-1*3=-1 y g(x)
# ax + by + cz + d > -1 y se clasifica como azul si ax + by + cz + d < -1
z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]**x - clf.coef_[0][1]-y) / clf.coef_[0][2] # Se hace la modificacion 
 
tmp = np.linspace(-5,5,30)
x,y = np.meshgrid(tmp,tmp)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'sr')
ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'ob')
ax.plot_surface(x, y, z(x,y)) # Se utiliza "z" con la regla de clasificacion
ax.view_init(30, 60)
plt.show()


# #### 9) Dibuje una observacion adicional de manera que las dos clases ya no sean separables por un hiperplano.

# In[30]:


tabla = {'Observaciones': np.array([[1,0,1], [1,0,2], [1,1,2],[3,1,4], [1,1,3],[3,2,3],[1,2,1],[3,2,1],[1,1,0], [4,3,4]]), 
         'Clase': np.array([0,0,0,0,0,0,1,1,1,1]), 'Clase_nombre': np.array(['Rojo', 'Azul'])}

print(tabla)


# In[31]:


from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
X = tabla['Observaciones'] # Se toma la columna X, Y y Z. 
Y = tabla['Clase'] # Se toma toda la clase. 
print(X)
print(Y)


# In[32]:


# Creando una variable binaria 
X = X[np.logical_or(Y==0,Y==1)]
Y = Y[np.logical_or(Y==0,Y==1)]
model = svm.SVC(kernel='linear')
clf = model.fit(X, Y)

# La ecuacion de la separacion del plano esta dada por = np.dot(svc.coef_[0], x) + b = 0 o 
# b=y-mx=1.5-(1*1.5)=0, entonces la recta de separación es f(x)=x.

# Solucion para w = (z) - Regla de clasificacion (explicada en el punto 4)
z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x -clf.coef_[0][1]*y) / clf.coef_[0][2]

tmp = np.linspace(-5,5,30)
x,y = np.meshgrid(tmp,tmp)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'sr')
ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'ob')
ax.plot_surface(x, y, z(x,y))
ax.view_init(30, 60)
plt.show()


# #### Analisis:
# * Ya no es separable, porque se agrego otra observacion tomando en cuenta los rangos minimos y maximos, y por ende, la observacion de color rojo quedo al lado de las azules, por lo que, ya no se puede separar.

# In[33]:


################################################## FIN ########################################################

