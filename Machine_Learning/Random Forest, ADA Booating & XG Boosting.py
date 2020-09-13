#!/usr/bin/env python
# coding: utf-8

# ## Modulo Modelos Supervisados

# ## Tarea #3

# ## Estudiante: Heiner Romero Leiva

# In[1]:


# Importacion de paquetes
import os
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import colors as mcolors
import random


# ### Pregunta 1: 
# En este ejercicio usaremos los datos (voces.csv). Se trata de un problema de reconocimiento de genero mediante el analisis de la voz y el habla. Esta base de datos fue creada para identificar una voz como masculina o femenina, basandose en las propiedades acusticas de la voz y el habla. El conjunto de datos consta de 3.168 muestras de voz grabadas, recogidas de hablantes masculinos y femeninos.

# #### 1) Cargue la tabla de datos voces.csv en Python.

# In[4]:


voces = pd.read_csv("voces.csv", delimiter = ",", decimal = ".")
print(voces.head())


# In[5]:


# Inspeccionando dataset
voces.info()


# In[6]:


voces.describe()


# ##### El dataset contiene varias variables con escalas diferentes, sin embargo, como lo que se va a realizar es Random Forest, ADA Boosting y XG Boosting, no es necesario normalizar los datos. Se observa que todas las variables son continuas, excepto la ultila columna que es una variable categorica binaria. 

# #### 2) Use Bosques Aleatorios, ADABoosting y XGBoosting en Python (con los parametros pordefecto) para generar un modelo predictivo para la tabla voces.csv usando el 80 % de los datos para la tabla aprendizaje y un 20 % para la tabla testing, luego calcule para los datos de testing la matriz de confusion, la precision global y la precision para cada una de las dos categorıas. ¿Son buenos los resultados? Explique

# #### Función para calcular los índices de calidad de la predicción

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


# #### Función para graficar la distribución de la variable a predecir

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


# ##### Función para ver la distribución de una variable categórica respecto a la predecir

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


# ##### Función para ver la distribución de una variable numérica respecto a la predecir

# In[5]:


def poder_predictivo_numerica(data:DataFrame, var:str, variable_predict:str):
    sns.FacetGrid(data, hue=variable_predict, height=6).map(sns.kdeplot, var, shade=True).add_legend()


# #### Distribución de la variable a predecir

# In[11]:


distribucion_variable_predecir(voces,"genero")


# ##### Variable completamente balanceada, entonces las predicciones para que sean buenas deben dar muy parecidas en cuanto a su precision

# ##### Elimina la variable catégorica, deja las variables predictoras en X

# In[12]:


X = voces.iloc[:,:20] 
print(X.head())


# ##### Deja la variable a predecir en y

# In[13]:


y = voces.iloc[:,20:21] 
print(y.head())


# ##### Se separan los datos con el 80% de los datos para entrenamiento y el 20% para testing

# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


# #### **Analisis con Bosques Aleatorios**

# ##### Mediante el constructor inicializa el atributo n_estimators= 120

# ##### En este caso con los n_estimators, se hicieron varias pruebas con n_estimators de 10, 50, 100 y 120, dando como resultado el 120, el mejor de los estimadores de N, ya que fue en el que se "normaliza el error" (se mantiene estable) y con el que, se alcanza una precision global mas alta, al igual que con las variables a predecer, por eso se toma el 120. Este numero tambien se puede estimar con ayuda del Codo de Jambu. 

# In[15]:


instancia_bosque = RandomForestClassifier(n_estimators=120)


# ##### Entrena el modelo llamando al método fit

# In[16]:


instancia_bosque.fit(X_train,y_train.iloc[:,0].values)


# ##### Imprime las predicciones en testing

# In[17]:


print("Las predicciones en Testing son: {}".format(instancia_bosque.predict(X_test)))


# ##### Índices de Calidad del Modelo

# In[18]:


prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ##### **Analisis**
# En cuanto a los resultados obtenidos usando el metodo de arboles aleatorios con 120 estimadores, se puede ver que la precision global es de casi un 99% dejando un error muy pequeno de solo 1%. Por otro lado, la precision en cuanto a femenino es de poco mas de un 98% y la de masculino es mas de un 99%, dando sin lugar a dudas unas de las mejores predicciones de los metodos hasta ahora estudiados (siempre y cuando se haga el estudio para determinar la cantidad de estimadores que dan los mejores resultados). 

# ##### **Graficando Importancia de las variables**

# In[19]:


importancia = instancia_bosque.feature_importances_
print(importancia)
etiquetas = X_train.columns.values
y_pos = np.arange(len(etiquetas))
plt.barh(y_pos, importancia, align='center', alpha=0.5)
plt.yticks(y_pos, etiquetas)


# ##### **Analisis**
# En cuanto a las mejores variables a mencionar estar meanfun, IQR, Q25, y sd. 
# 

# #### **Analisis con Potenciación - ADA Boosting**

# ##### Mediante el constructor inicializa el atributo n_estimators=120

# In[21]:


instancia_potenciacion = AdaBoostClassifier(n_estimators=120)


# ##### Entrena el modelo llamando al método fit

# In[22]:


instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)


# ##### Imprime las predicciones en testing

# In[23]:


print("Las predicciones en Testing son: {}".format(instancia_potenciacion.predict(X_test)))


# ##### Índices de Calidad del Modelo

# In[56]:


prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ##### **Analisis**
# Con respecto al ADA boosting, se puede observar como la precision global tambien es bastante buena dando poco mas de un 98% y con un error de un casi 2% (siendo el metodo de random forest mejor por unos poco decimales de precision) pero se obtienen muy buenas predicciones. Por otro lado, se tienen las predicciones del genero dando muy buenos resultados de igual forma, debido a que la precision para femenino es de poco mas de un 97% y la de masculino de mas de un 99%. 

# ##### **Importancia de las Variables**

# In[25]:


importancia = instancia_potenciacion.feature_importances_
print(importancia)
etiquetas = X_train.columns.values
y_pos = np.arange(len(etiquetas))
plt.barh(y_pos, importancia, align='center', alpha=0.5)
plt.yticks(y_pos, etiquetas)


# ##### **Analisis**
# En cuanto a la importancia de las variables estan: minfun, sfm, IQR y sd. 

# #### **Analisis con Potenciación - XG Boosting**

# ##### Mediante el constructor inicializa el atributo n_estimators= 120

# In[27]:


instancia_potenciacion = GradientBoostingClassifier(n_estimators=120)


# ##### Entrena el modelo llamando al método fit

# In[28]:


instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)


# ##### Imprime las predicciones en testing

# In[29]:


print("Las predicciones en Testing son: {}".format(instancia_potenciacion.predict(X_test)))


# ##### Índices de Calidad del Modelo

# In[30]:


prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ##### **Analisis**
# Con respecto al XG Boosting, con este metodo podemos ver como la precision glopbal es de casi un 99% y el error es poco mas de un 1%. Se tienen muy buenas predicciones para la el genero femenino (mas de un 98%) y para el masculino se tiene poco mas de un 99% y solo no pudo clasificar 2 observaciones de las 309 que se tenian, lo que lo hace un modelo muy fiable para predicciones. 

# ##### **Importancia de las variables**

# In[32]:


importancia = instancia_potenciacion.feature_importances_
print(importancia)
etiquetas = X_train.columns.values
y_pos = np.arange(len(etiquetas))
plt.barh(y_pos, importancia, align='center', alpha=0.5)
plt.yticks(y_pos, etiquetas)


# ##### **Analisis**
# Se puede observar como en este caso la mayoria de variables decaen y solo se tiene como la variable mas importante meanfun e IQR. 

# #### 3) Usando la funcion programada en el ejercicio 1 de la tarea anterior, los datos voces.csv y los modelos generados arriba construya un DataFrame de manera que en cada una de las filas aparezca un modelo predictivo y en las columnas aparezcan los indices Precision Global, Error Global Precision Positiva (PP), Precision Negativa (PN), Falsos Positivos (FP), los Falsos Negativos (FN), la Asertividad Positiva (AP) y la Asertividad Negativa (AN). ¿Cual de los modelos es mejor para estos datos?

# In[28]:


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


# In[34]:


# Asignando MC ---- Bosques aleatorios ---- 
datos = (([309, 5],[2, 318])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 


# ##### **Modelo Arboles Aleatorios con funcion personalizada de indices**

# In[35]:


indices_personalizados(MC)


# ##### **Modelo ADA Boosting con funcion personalizada de indices**

# In[36]:


# Asignando MC ---- ADA Boosting ---- 
datos = (([305, 9],[3, 317])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 


# In[37]:


indices_personalizados(MC)


# ##### **Modelo XG Boosting con funcion personalizada de indices**

# In[38]:


# Asignando MC ---- XG Boosting ---- 
datos = (([309, 5],[2, 318])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 


# In[39]:


indices_personalizados(MC)


# ##### **Analisis**
# 
# En cuanto a los metodos utilizados, tanto el de "Arboles aleatorios" como el de XG Boosting, dan exactamemte los mismos resultados, los cuales son bastante buenos entre si y logran una casi perfecta prediccion y su error global es minimo. Por otra aprte, la PP, PN, PFP, PFN la AP y AN, son bastante buenas y se puede ver como estos dos metodos pueden detectar los positivos muy facilmente al igual que los negativos, y como el porcentaje de positivos clasificados como negativos es muy bajo y con los casos de los negativos clasificados como positivos es apenas significativo. Con lo referente al ADA boosting tambien arroja resultados muy buenos y cercanos a los mencionados anteriormente, sin embargo los mejores con el Random Forest y el XG boosting. 
# 

# #### 4) Repita los ejercicios 1-3, pero esta vez use una combinacion diferente de los parametros de los metodos. ¿Mejora la prediccion?

# In[41]:


# Llamando variables x y y, y haciendo el split. 

X = voces.iloc[:,:20] 
y = voces.iloc[:,20:21] 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


# ##### **Analisis con Bosques Aleatorios con diferentes parametros de los metodos** 

# In[42]:


instancia_bosque = RandomForestClassifier(n_estimators=120, max_depth =6, min_samples_split = 6)
instancia_bosque.fit(X_train,y_train.iloc[:,0].values)


# ##### **Indices de Calidad del Modelo** 

# In[43]:


prediccion = instancia_bosque.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ##### **ADA boosting con diferentes parametros de los metodos**

# In[44]:


instancia_potenciacion = AdaBoostClassifier(n_estimators=120, algorithm='SAMME', learning_rate=2.0)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)


# ##### **Indices de Calidad del Modelo**

# In[45]:


prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ##### **XG Boosting con diferentes parametros de los metodos**
# 

# In[46]:


instancia_potenciacion = GradientBoostingClassifier(n_estimators=120, max_depth =6, min_samples_split = 6)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)


# ##### **Indices de Calidad del Modelo**

# In[47]:


prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# In[48]:


# Asignando MC ---- Bosques aleatorios ---- 
datos = (([325, 5],[9, 295])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 


# In[49]:


# -----------------Modelo Arboles Aleatorios con funcion personalizada de indices¶-----------------------
indices_personalizados(MC)


# In[50]:


# Asignando MC ---- ADA Boosting ---- 
datos = (([43, 287],[50, 254])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 


# In[51]:


# ------------------ Modelo ADA Boosting con funcion personalizada de indices ---------------------------
indices_personalizados(MC)


# In[54]:


# Asignando MC ---- XG Boosting ---- 
datos = (([327, 3],[12, 292])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 


# In[55]:


# --------------------- Modelo XG Boosting con funcion personalizada de indices ------------------------
indices_personalizados(MC)


# ##### Analisis
# A nivel global se tiene el siguiente analisis: 
# 
#   * En cuanto al arbol aleatorio, con parametros por defecto y cambiando los n en el original, y en el personalizado cambiando los parametros de los metodos, se obtiene una precision global bastante similar, es decir, 97.94% versus 97.79% y el error, 2.05% versus 2.20% es decir en este caso es muy parecido y solo desmejora un poco. 
#   * ADA Boosting en el metodo original se obtiene un 97.47% versus los parametros personalizados en el metodo se obtienen peores resultados, 46.86% y en el error global se tiene un 2.5% versus el error obtenido son el los metodos personalizados que se obtienen 53.15%, es decir, el error es bastante grande, los FN son bastantes grandes asi como los FP, en este caso el metodo mas bien desmejoro. 
#   * Por ultimo en el XG Boosting en el primer metodo por defecto y solo cambiando la cantidad de estimadores, se obtienen 97.63% versus 97.63%, y en cuanto al error global es el mismo. 
#     
#   * Resumen: A nivel general cuando se cambian los parametros por defecto las predicciones cambian levemente en cuanto a los arboles aleatorios, con el ADA Boosting las predicciones si son bastante diferentes y son peores con los parametros modificados. Por ultimo en el XG Boosting, las predicciones con las por defecto, versus las modificadas en los parametros dan completamente similar. 

# #### 5) Repita los ejercicios 1-4, pero esta vez use 2 combinaciones diferentes de seleccion de 6 variables predictoras. ¿Mejora la prediccion?

# ##### Paso 1, importando dataset con las principales 6 variables predictoras para cada metodo, segun importancia de las variables vistas en los graficos del punto 2. 

# #### **Arboles Aleatorios**

# In[62]:


# Importando las 6 mejores variables predictoras
X = voces.iloc[:,[1,3,5,8,9,12]] 
print(X.head())
y = voces.iloc[:,20:21] 


# In[63]:


# Definiendo parametros de modelo y creando predicciones

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

instancia_bosque = RandomForestClassifier(n_estimators=120)

instancia_bosque.fit(X_train,y_train.iloc[:,0].values)

print("Las predicciones en Testing son: {}".format(instancia_bosque.predict(X_test)))


# In[64]:


# ------------ Indices de Calidad del Modelo -----------------

prediccion = instancia_bosque.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### **ADA Boosting**

# In[67]:


# Definiendo parametros de variables elegidas
X = voces.iloc[:,[1,2,5,9,12,19]] 
print(X.head())
y = voces.iloc[:,20:21] 


# In[75]:


# Definiendo parametros de modelo y creando predicciones

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

instancia_potenciacion = AdaBoostClassifier(n_estimators=120)

instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)

print("Las predicciones en Testing son: {}".format(instancia_bosque.predict(X_test)))


# In[76]:


# -------------- Indices de Calidad del modelo -------------- 

prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### **Potenciación - XG Boosting**

# In[78]:


# Definiendo parametros de variables elegidas
X = voces.iloc[:,[1,3,5,8,9,12]] 
print(X.head())
y = voces.iloc[:,20:21] 


# In[79]:


# Definiendo parametros del Modelo y creando predicciones

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

instancia_potenciacion = GradientBoostingClassifier(n_estimators=120)

instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)

print("Las predicciones en Testing son: {}".format(instancia_potenciacion.predict(X_test)))


# In[80]:


# ----------------- Indices de Calidad del Modelo ---------------------

prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### **Analisis General**
# 
# En cuanto a los resultados trabajando solo con seis variables y dejando los parametros por defecto, estos dan resultados muy positivos, ya que en cuanto a los arboles aleatorios estos dan mas de un 98% y menos de 2% de error (muy similar a los resultados obtenidos con todas las variables, aunque seleccionando las variables dan mejor). Para el caso del ADA Boosting y del XG Boosting, los resultados tambien son buenos, se tiene mas de un 97% de precision global y poco mas de un 2% de error. Estos resultados para el ADA Boosting y el XG Boosting con variables elegidas, dan mejores resultados que con todas. las variables y la precision por categoria es bastante buena y el modelo puede generalizar bastante bien. 

# #### **Usando funcion programada para desplegar indices globales y especificos**

# In[82]:


# Asignando MC ---- Bosques aleatorios ---- 
datos = (([308, 5],[7, 314])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 

indices_personalizados(MC)


# In[83]:


# Asignando MC ---- ADA Boosting ---- 
datos = (([322, 5],[10, 297])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 

indices_personalizados(MC)


# In[84]:


# Asignando MC ---- XG Boosting ---- 
datos = (([311, 7],[7, 309])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 

indices_personalizados(MC)


# #### **Analisis General con 6 variables**

# #### **Trabajando metodos con 6 variables predictoras y cambiando parametros de los metodos**

# #### **Analisis con Bosques Aleatorios con diferentes parametros de los metodos**

# In[86]:


# Llamando variables x y y, y haciendo el split. 
# Importando las 6 mejores variables predictoras
X = voces.iloc[:,[1,3,5,8,9,12]] 
print(X.head())
y = voces.iloc[:,20:21] 


# In[87]:


# Definiendo parametros de modelo y creando predicciones

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

instancia_bosque = RandomForestClassifier(n_estimators=120, max_depth =6, min_samples_split = 6)

instancia_bosque.fit(X_train,y_train.iloc[:,0].values)


# In[88]:


# --------------Indices de Calidad del Modelo------------------

prediccion = instancia_bosque.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### **ADA boosting con diferentes parametros de los metodos**

# In[89]:


# Definiendo parametros de variables elegidas
X = voces.iloc[:,[1,2,5,9,12,19]] 
print(X.head())
y = voces.iloc[:,20:21] 


# In[90]:


# Definiendo parametros de modelo y creando predicciones

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

instancia_potenciacion = AdaBoostClassifier(n_estimators=120, algorithm='SAMME', learning_rate=2.0)

instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)


# In[91]:


# -------------- Indices de Calidad del Modelo ------------------

prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### **XG Boosting con diferentes parametros de los metodos**

# In[92]:


# Definiendo parametros de variables elegidas
X = voces.iloc[:,[1,3,5,8,9,12]] 
print(X.head())
y = voces.iloc[:,20:21] 


# In[93]:


# Definiendo parametros del Modelo y creando predicciones

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

instancia_potenciacion = GradientBoostingClassifier(n_estimators=120, max_depth =6, min_samples_split = 6)

instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)


# In[94]:


# --------------- Indices de Calidad del Modelo ------------------

prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### **Analisis General con 6 variables y cambio de parametros en los metodos**
# 
# A nivel general, cambiando los parametros por defecto y trabajando solo con seis variables, los resultados en los Bosques Aleatorios son bastante buenos al igual que en el XG Boosting (y dan muy similares a los resultados obtenidos cuando se usaron todas las variables y se cambiaron los parametros de los metodos), sin embargo el modelo que esta dando peores resultados es el ADA Boosting, ya que la precision global es de poco mas de 5% y el error es de poco mas de 94% haciendolo un modelo muy poco fiable si se cambian los parametros aunque se seleccionen buenas variables predictoras. 
# Por lo demas, los modelos (XG Boosting y Random Forest) tienen bastante buena prediccion. 

# In[95]:


# Asignando MC ---- Bosques aleatorios ---- 
datos = (([291, 3],[8, 332])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 

# -----------------Modelo Arboles Aleatorios con funcion personalizada de indices y 6 variables ¶-----------------------
indices_personalizados(MC)


# In[96]:


# Asignando MC ---- ADA Boosting ---- 
datos = (([17, 298],[299, 20])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 

# ------------------ Modelo ADA Boosting con funcion personalizada de indices ---------------------------
indices_personalizados(MC)


# In[97]:


# Asignando MC ---- XG Boosting ---- 
datos = (([326, 8],[6, 294])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 

# --------------------- Modelo XG Boosting con funcion personalizada de indices ------------------------
indices_personalizados(MC)


# #### **Analisis con 6 variables, cambios en los parametos de los metodos y metodo programado de indices**
# 
# A nivel general, cambiando los parametros por defecto y trabajando solo con seis variables, los resultados en los Bosques Aleatorios son bastante buenos al igual que en el XG Boosting (y dan muy similares a los resultados obtenidos cuando se usaron todas las variables y se cambiaron los parametros de los metodos), sin embargo el modelo que esta dando peores resultados es el ADA Boosting, ya que la precision global es de poco mas de 5% y el error es de poco mas de 94% haciendolo un modelo muy poco fiable si se cambian los parametros aunque se seleccionen buenas variables predictoras. 
# Por lo demas, los modelos (XG Boosting y Random Forest) tienen bastante buena prediccion. 
# Por otro lad la AP, AN en estos ultimos modelos mencionados es bastante buena y los PFP y PFN son muy bajos, dando a entender que son modelos bastante fiables cuando se cambian los metodos por defecto del algoritmo. 

# ### Pregunta 2: 
# Esta pregunta utiliza los datos (tumores.csv). Se trata de un conjunto de datos de caracterısticas del tumor cerebral que incluye cinco variables de primer orden y ocho de textura y cuatro parametros de evaluacion de la calidad con el nivel objetivo. La variables son: Media, Varianza, Desviacion estandar, Asimetria, Kurtosis, Contraste, Energıa, ASM (segundo momento angular), Entropıa, Homogeneidad, Disimilitud, Correlacion, Grosor, PSNR (Pico de la relacion senal-ruido), SSIM (Indice de Similitud Estructurada), MSE (Mean Square Error), DC (Coeficiente de Dados) y la variable a predecir tipo (1 = Tumor, 0 = No-Tumor).

# #### 1) Use Bosques Aleatorios, ADABoosting y XGBoosting en Python para generar un modelo predictivo para la tabla tumores.csv usando el 70 % de los datos para la tabla aprendizaje y un 30 % para la tabla testing.

# In[29]:


# Importando datos 
tumores = pd.read_csv("tumores.csv", delimiter = ',', decimal = '.')
print(tumores)


# In[7]:


tumores.info()


# In[31]:


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


# ##### **Equilibrio de la Variable a predecir**

# In[32]:


distribucion_variable_predecir(tumores,"tipo") 
# Se puede observar como estamos ante de la presencia de una variable desbalanceada a predecir entonces las predicciones
# puede que no sean tan altas


# ##### **Genera el modelo, las predicciones y mide la calidad**

# In[33]:


# Elimina la variable catégorica, deja las variables predictoras en X
X = tumores.iloc[:,0:17] 
print(X.head())


# In[34]:


#Deja la variable a predecir en y¶
y = tumores.iloc[:,17:18] 
print(y.head())


# In[56]:


#Se separan los datos con el 70% de los datos para entrenamiento y el 30% para testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70)


# #### **Analisis con Bosques Aleatorios**

# In[57]:


instancia_bosque = RandomForestClassifier(n_estimators=120)

instancia_bosque.fit(X_train,y_train.iloc[:,0].values)


# #### **Indices de Calidad del Modelo**

# In[58]:


prediccion = instancia_bosque.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### **Importancia de las variables**

# In[59]:


importancia = instancia_bosque.feature_importances_
print(importancia)
etiquetas = X_train.columns.values
y_pos = np.arange(len(etiquetas))
plt.barh(y_pos, importancia, align='center', alpha=0.5)
plt.yticks(y_pos, etiquetas)


# #### **Analisis con ADA boosting**

# In[60]:


instancia_potenciacion = AdaBoostClassifier(n_estimators=120)

instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)


# #### **Indices de Calidad del Modelo**

# In[61]:


prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### **Graficando la Importancia de las Variables**

# In[62]:


importancia = instancia_potenciacion.feature_importances_
print(importancia)
etiquetas = X_train.columns.values
y_pos = np.arange(len(etiquetas))
plt.barh(y_pos, importancia, align='center', alpha=0.5)
plt.yticks(y_pos, etiquetas)


# #### **Analisis con XG Boosting**

# In[63]:


instancia_potenciacion = GradientBoostingClassifier(n_estimators=120)

instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)


# #### **Indices de Calidad del Modelo**

# In[64]:


prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### **Graficando la Importancia de las Variables**

# In[65]:


importancia = instancia_potenciacion.feature_importances_
print(importancia)
etiquetas = X_train.columns.values
y_pos = np.arange(len(etiquetas))
plt.barh(y_pos, importancia, align='center', alpha=0.5)
plt.yticks(y_pos, etiquetas)


# #### **Analisis General**
# 
# A nivel general se ve una precision entre 97 y 98% entre los modelos, lo cual es una precison bastante buena. 
# En cuanto a los arboles aleatorios se tiene una precision global de un 97.42% y un error global de un 2.5%, en cuanto a las predicciones se una un 79.71% de prediccion para "No hay tumor" y un 98.90% para "Si hay tumor" lo cual se explica dicha diferencia ya que el problema esta desequilbrado. 
# En cuanto al ADA Boosting, se tiene una precision global de un 97% un error de poco mas de un 2%, en cuanto a sus precisiones, esta para el "no hay tumor" se posiciona en un 81% y para el "si hay tumor" en un 98.78%. 
# Finalmente para el XG Boosting la precision global es de un 98.32% y un error de poco mas de un 1.5%, sin embargo con este metodo, la variable a predecir "no hay tumor" aumenta a un 92.75% dando un resultado bastante bueno, pese al desequilibro que hay de la variable en el problema, por otro lado, la variable "si tiene un tumor" tambien genera muy buenos resultados, ya que un 98.78% es su porcentaje de prediccion, dando como mejor modelo el XG boosting para este caso. 

#  #### 2. Usando la funcion programada en el ejercicio 1 de la tarea anterior, los datos tumores.csv y los modelos generados arriba construya un DataFrame de manera que en cada una de las filas aparezca un modelo predictivo y en las columnas aparezcan los ındices Precision Global, Error Global Precision Positiva (PP), Precision Negativa (PN), Falsos Positivos (FP), los Falsos Negativos (FN), la Asertividad Positiva (AP) y la Asertividad Negativa (AN). ¿Cual de los modelos es mejor para estos datos?

# In[66]:


# Asignando MC ---- Bosques aleatorios ---- 
datos = (([55, 9],[9, 815])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 

# -----------------Modelo Arboles Aleatorios con funcion personalizada de indices¶-----------------------
indices_personalizados(MC)


# In[67]:


# Asignando MC ---- ADA Boosting ---- 
datos = (([56, 13],[10, 814])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 

# ------------------ Modelo ADA Boosting con funcion personalizada de indices ---------------------------
indices_personalizados(MC)


# In[68]:


# Asignando MC ---- XG Boosting ---- 
datos = (([64, 5],[10, 814])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 

# --------------------- Modelo XG Boosting con funcion personalizada de indices ------------------------
indices_personalizados(MC)


# #### **Analisis** 
# El metodo que da el mejor resultado en definitiva es el XG Boosting, ya que logra tener una precision global de un 98.32% y la precision positiva es de un 98.78% mientras que a negativa es de un 92.75% y para ser un problema con una variable desbalanceada esta dando valores bastante buenos, por otro lado, los falsos positivos son de un 7.24% y los falsos negativos de un 1.21%. Ademas posee una Asertividad Positiva de un 99.38% (casi perfecta) y la asertividad negativa es de un 86.48%, ya que hay my poco casos de "no tiene tumor" en el dataset de test, sin embargo predice bastante bien la variable. 

# #### Repita los ejercicios 1-2, pero esta vez use una combinacion de los parametros del metodo de cada uno de los metodos citados arriba. ¿Mejora la prediccion?

# In[69]:


# Elimina la variable catégorica, deja las variables predictoras en X
X = tumores.iloc[:,0:17] 

#Deja la variable a predecir en y
y = tumores.iloc[:,17:18] 


# In[70]:


#Se separan los datos con el 70% de los datos para entrenamiento y el 30% para testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=70)


# #### **Analisis con Bosques Aleatorios y combinacion de los parametros**

# In[71]:


instancia_bosque = RandomForestClassifier(n_estimators=120, max_depth =6, min_samples_split = 6)

instancia_bosque.fit(X_train,y_train.iloc[:,0].values)

# -------------- Indices de Calidad del Modelo ---------------

prediccion = instancia_bosque.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### **Analisis con ADA Boosting y combinacion de los parametros**

# In[72]:


instancia_potenciacion = AdaBoostClassifier(n_estimators=120, algorithm='SAMME', learning_rate=2.0)

instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)


# ----------------- Indices de Calidad del Modelo -------------------------

prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### **Analisis con XG Boosting y combinacion de los parametros**

# In[73]:


instancia_potenciacion = GradientBoostingClassifier(n_estimators=120, max_depth =6, min_samples_split = 6)

instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)

prediccion = instancia_potenciacion.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### **Usando parametros de la funcion programada**

# In[74]:


# Asignando MC ---- Bosques aleatorios ---- 
datos = (([45, 12],[7, 829])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 

# -----------------Modelo Arboles Aleatorios con funcion personalizada de indices¶-----------------------
indices_personalizados(MC)


# In[75]:


# Asignando MC ---- ADA Boosting ---- 
datos = (([56, 1],[19, 817])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 

# ------------------ Modelo ADA Boosting con funcion personalizada de indices ---------------------------
indices_personalizados(MC)


# In[76]:


# Asignando MC ---- XG Boosting ---- 
datos = (([49, 8],[7, 829])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
MC = df 

# --------------------- Modelo XG Boosting con funcion personalizada de indices ------------------------
indices_personalizados(MC)


# #### **Analisis General**
# 
# A grandes rasgos las precciones si mejoran, debido a que en los arboles aleatorios sin cambiar parametros se tiene una precision de un 97% y la prediccion para el "No tumor" es de un 79.71% y la del "si tumor" es de un 98.90% mientras que con los parametros por defecto modificados, se tiene un 78% y un 99% respectivamente. 
# En el ADA Boosting con los parametros por defectos se tiene un 97.42% mientras que con el cambio de los parametos se tiene un 97.76% y la prediccion del "no tumor" original fue de 81% pero ahora la nueva prediccion es de 98%. 
# Por ultimo, con el XG Boosting, la precision global con los parametros por defecto fue de 98.32% mientras que con los parametros modificados se tiene ahora una precision de un 98.32% (la misma) sin embargo en cuanto a la produccion de "no tumor" se tiene un 92.75% con el parametro original una vez modificado se tiene un 85% y la prediccion del "si hay tumor" solo aumemta un 1% con respecto a la original. 
# 
# En cuanto a las nuevas predicciones cambiando los metodos por defecto y sustiyendolos por nuevos, se tiene una mejoria en el modelo de ADA Boosting, en los demas se mantienen muy parecidas las predicciones o bajan sensiblemente. 

# ### Pregunta 3:
# 
# La idea de este ejercicio es programar una Clase en Python para un nuevo metodo de Consenso Propio, esto basado en los metodos K-vecinos mas cercanos, Arboles de Decision, Metodo de Potenciacion (XGBoosting) y Metodo de Potenciacion (ADABoosting), para esto realice los siguiente:

# #### 1) **Programe una Clase en Python denominada ConsensoPropio que tiene, ademas del constructor, al menos los siguientes metodos fit(X train, y train, ...) que recibe la tabla de entrenamiento y genera 4 muestras aleatorias con reemplazo (Boostraps) de los datos de aprendizaje y luego aplica en cada una de estas muestras uno de los metodos predictivos mencionados arriba. Este metodo debe generar un nuevo modelo predictivo que es un atributo de clase, tipo diccionario, que incluya los 4 modelos generados (todos los metodos usaran todas las variables) y las 4 de precisiones globales, respectivamente de cada modelo1 , que denotamos por (P G1, P G2, . . . , P G4), donde 0 ≤ P Gj ≤ 1 para j = 1, 2, . . . , 4.**
# 
# #### **2) Programe una funcion predict(X test) que recibe la tabla de testing. Luego, para predecir aplica en cada una de las filas de la tabla de testing los 4 modelos predictivos que estan almacenados dentro de la Clase en el atributo incluido para este efecto; y se establece un consenso de todos los resultados. Se debe programar una formula en Python que le de mayor importancia a los metodos con mejor precision global.**
# 
# 

# In[16]:


# Programando respuesta 1 y 2 de las preguntas


# In[20]:


class ConsensoPropio:
    def __init__(self, train = 0.8, neighbors = 3, estimators=50, criterion_gradient_boosting='friedman_mse',
                criterion_decision_tree='gini'):
        self.__train = train # Se trabaja con un 80% de los datos totales. 
        self.__neighbors = neighbors
        self.__estimators = estimators
        self.__mp = {'Models': [], 'Precisions' : []}
        self.__unique_values = 0
        self.__criterion_gradient_boosting = criterion_gradient_boosting
        self.__criterion_decision_tree = criterion_decision_tree
    @property
    def train(self):
        return self.__train
    @property
    def neighbors(self):
        return self.__neighbors
    @property
    def estimators(self):
        return self.__estimators
    @property
    def mp(self):
        return self.__mp
    @property
    def unique_values(self):
        return self.__unique_values
    @unique_values.setter
    def unique_values(self, nv):
        self.__unique_values = nv
    @property
    def criterion_gradient_boosting(self):
        return self.__criterion_gradient_boosting
    @property
    def criterion_decision_tree(self):
        return self.__criterion_decision_tree
    # ---------------------------------- Metodo para Split --------------------------------------.
    def __split(self, data):
        x = data.iloc[:,:data.shape[1] - 1]
        y = data.iloc[:,data.shape[1] - 1:]
        return train_test_split(x, y, train_size=self.train, random_state=0)
    # --------------------------------- Precisiones totales -------------------------------------.
    def __models_precision(self, precision, model):
        self.mp['Precisions'].append(precision)
        self.mp['Models'].append(model)
    # ---------------------------------------- Modelos -----------------------------------------.
    def __KNN(self, data):
        x_train, x_test, y_train, y_test = self.__split(data)
        knn = KNeighborsClassifier(self.neighbors)
        knn.fit(x_train,y_train.iloc[:,0].values)
        predic = knn.predict(x_test)
        mc = confusion_matrix(y_test, predic)
        self.__models_precision(np.sum(mc.diagonal()) / np.sum(mc), knn)
    def __DecisionTree(self, data):
        x_train, x_test, y_train, y_test = self.__split(data)
        dt = DecisionTreeClassifier(criterion=self.criterion_decision_tree)
        dt.fit(x_train,y_train.iloc[:,0].values)
        predic = dt.predict(x_test)
        mc = confusion_matrix(y_test, predic)
        self.__models_precision(np.sum(mc.diagonal()) / np.sum(mc), dt)
    def __ADABoosting(self, data):
        x_train, x_test, y_train, y_test = self.__split(data)
        ada = AdaBoostClassifier(n_estimators=self.estimators)
        ada.fit(x_train,y_train.iloc[:,0].values)
        predic = ada.predict(x_test)
        mc = confusion_matrix(y_test, predic)
        self.__models_precision(np.sum(mc.diagonal()) / np.sum(mc), ada)
    def __XGBoosting(self, data):
        x_train, x_test, y_train, y_test = self.__split(data)
        xgb = GradientBoostingClassifier(n_estimators=self.estimators,criterion=self.criterion_gradient_boosting)
        xgb.fit(x_train,y_train.iloc[:,0].values)
        predic = xgb.predict(x_test)
        mc = confusion_matrix(y_test, predic)
        self.__models_precision(np.sum(mc.diagonal()) / np.sum(mc), xgb)
    # ------------------------------- Metodo Fit ----------------------------------------------.
    def fit(self,x_train, y_train):
        df = pd.concat([x_train, y_train], axis=1)
        self.unique_values = np.unique(y_train)
        self.__KNN(df.sample(frac=1))
        self.__ADABoosting(df.sample(frac=1))
        self.__XGBoosting(df.sample(frac=1))
        self.__DecisionTree(df.sample(frac=1))
    # --------------------------------- Predictor Global -------------------------------------.
    def predict(self, data):
        precisions = []
        sumatory = np.zeros([data.shape[0], len(self.unique_values)])
        final = []
        for precision in self.mp['Precisions']:
            precisions.append(precision / sum(self.mp['Precisions']))
        for i in range(len(self.mp['Models'])):
            model_tmp = self.mp['Models'][i].predict_proba(data) * precisions[i]
            sumatory = np.add(sumatory, model_tmp)      
        for element in sumatory:
            for index in range(len(self.unique_values)):
                if element[index] == max(element):
                    final.append(self.unique_values[index])          
        return np.array(final)


# #### **Importacion de datos**

# In[18]:


voces = pd.read_csv("voces.csv", delimiter = ',', decimal = '.')
voces.describe()


# #### **Haciendo pruebas con la clase programada**

# #### **3) Usando la tabla de datos voces.csv genere al azar una tabla de testing con un 20 % de los datos y con el resto de los datos construya una tabla de aprendizaje.**

# In[21]:


X = voces.iloc[:,:20]
y = voces.iloc[:,20:21]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)


# In[37]:


# Prueba de Ejemplo con XGBoosting
cp = ConsensoPropio()
cp.fit(X_train, y_train)
predict = cp.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y_test)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### 4) Genere modelos predictivos usando la Clase ConsensoPropio y el metodo fit de la clase RandomForestClassifier (con solamente 4 arboles, es decir, 4 boostraps), luego para la tabla de testing calcule, para ambos metodos, calcule la precison global, el error global y la precision por clases. ¿Cual metodo es mejor?

# In[28]:


# Importando datos para trabajar con 4 boostraps

voces = pd.read_csv("voces.csv", delimiter = ',', decimal = '.')
voces.describe()
X = voces.iloc[:,:20]
y = voces.iloc[:,20:21]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)


# #### Definiendo instancia con Random Forest

# In[39]:


instancia_bosque = RandomForestClassifier(n_estimators = 4) # aqui se define el 4 como la cantidad de arboles 
instancia_bosque.fit(X_train, y_train.iloc[:,0].values)
prediccion = instancia_bosque.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### Usando Clase ConsensoPropio para testing

# In[40]:


cp = ConsensoPropio()
cp.fit(X_train, y_train)
predict = cp.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# ##### **Analisis**
# En realidad ambos metodos dan similar, la diferencia son unos cuantos decimales, por ende se puede utilizar cualquiera de los dos para un analisis. 

# In[41]:


################################################# FIN ######################################################

