#!/usr/bin/env python
# coding: utf-8

# # Modelos Supervisados 

# ## Estudiante: Heiner Romero Leiva

# ## Tarea II 

# In[1]:


# Importando paquetes


# In[276]:


import os
import graphviz 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from pandas import DataFrame
import seaborn as sns


# ## Pregunta 1: 
# [no usar Python] Considere los datos de entrenamiento que se muestran en la siguiente Tabla para un problema de clasificacion binaria.

# In[22]:


#Configuraciones de Python
import pandas as pd
pd.options.display.max_rows = 5


# In[23]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/tabla.png")


# ### 1) Calcule el indice de Gini para la tabla completa, observe que el 50 % de las filas son de la clase C0 y el 50 % son de la clase C1.

# In[24]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/arbol.png") #las cajas rojas senalan la seleccion que se hizo entre las combinaciones


# ### 2) Calcule el indice de Gini Split para la variable Genero.

# In[25]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/genero.png")


# ### 3) Calcule el indice de Gini Split para la variable Tipo-Carro.

# In[26]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/tipo.png")


# ### 4) Calcule el indice de Gini Split para la variable Talla.

# In[27]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/talla.png")


# ### 5) ¿Cual variable es mejor Genero, Tipo-Carro o Talla?
# 
# De acuerdo a los calculos en los Gini splits hechos, se puede apreciar como la variable tipo-carro es la que tiene un Gini mas cercano a "0" por ende, es la mejor de las tres, ya que tiene la mayor cantidad de informacion ganada. 

# ## Pregunta 2:

# ##### En este ejercicio usaremos los datos (voces.csv). Se trata de un problema de reconocimiento de genero mediante el analisis de la voz y el habla. Esta base de datos fue creada para identificar una voz como masculina o femenina, basandose en las propiedades acusticas de la voz y el habla. El conjunto de datos consta de 3.168 muestras de voz grabadas, recogidas de hablantes masculinos y femeninos.

# #### 1) Cargue la tabla de datos voces.csv en Python.

# In[28]:


voces = pd.read_csv("voces.csv", delimiter = ",", decimal = ".")
print(voces)


# In[29]:


# Inspeccionando datos
voces.tail()
# Como se puede observar todas las columnas corresponden a variables numericas y continuas, solo se cuenta con una variable categorica 
# en la columna 21, que es la de genero, y es la que hay que predecir. 


# In[31]:


# Analisis de variables
voces.describe()
# Se ve que las variables estan en diferentes escalas, sin embargo, como el metodo que se va a aplicar es un arbol de decision no se 
# deben normalizar. 


# #### 2) Use Arboles de Decision en Python (con los parametros por defecto) para generar un modelo predictivo para la tabla voces.csv usando el 80 % de los datos para la tabla aprendizaje y un 20 % para la tabla testing, luego calcule para los datos de testing la matriz de confusion, la precision global y la precision para cada una de las dos categorıas. ¿Son buenos los resultados? Explique.

# #### Importando funciones

# #### Función para calcular los índices de calidad de la predicción
# 

# In[32]:


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

# In[33]:


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


# #### Funciones para ver la distribución de una variable respecto a la predecir (poder predictivo)

# #### Función para ver la distribución de una variable categórica respecto a la predecir

# In[34]:


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

# In[35]:


def poder_predictivo_numerica(data:DataFrame, var:str, variable_predict:str):
    sns.FacetGrid(data, hue=variable_predict, height=6).map(sns.kdeplot, var, shade=True).add_legend()


# In[37]:


print(voces.info())


# #### Distribucion de la variable a predecer 

# In[39]:


distribucion_variable_predecir(voces,"genero") # Se observa que la variable esta completamente balanceada. 


# #### Elimina la variable catégorica, deja las variables predictoras en X

# In[41]:


X = voces.iloc[:,0:19]
print(X.head())


# #### Deja la variable a predecir en y

# In[44]:


y = voces.iloc[:,20:21] 
print(y.head())


# #### Se separan los datos con el 80% de los datos para entrenamiento y el 20% para testing

# In[45]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


# #### Mediante el constructor inicializa el atributo DecisionTreeClassifier

# In[46]:


instancia_arbol = DecisionTreeClassifier(random_state=0)


# #### Entrena el modelo llamando al método fit

# In[47]:


instancia_arbol.fit(X_train,y_train)


# #### Imprime las predicciones en testing

# In[48]:


print("Las predicciones en Testing son: {}".format(instancia_arbol.predict(X_test)))


# #### Índices de Calidad del Modelo

# In[49]:


prediccion = instancia_arbol.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### Analisis
# * Los resultados son bastante buenos con arboles de decision, ya que se puede observar como la precision global asciende a casi un 97% y el error es de solo un 3%. Por otro lado, la precision para ambas categorias de la variable genero son bastante buenas, porque son de casi un 97% para ambos casos, lo que indica que el modelo puede generalizar bien y puede hacer las predicciones que se requieren. 

# #### 3) Usando la funcion programada en el ejercicio 1 de la tarea anterior, los datos voces.csv y los modelos generados arriba construya un DataFrame de manera que en cada una de las filas aparezca un modelo predictivo y en las columnas aparezcan los indices Precision Global, Error Global Precision Positiva (PP), Precision Negativa (PN), Falsos Positivos (FP), los Falsos Negativos (FN), la Asertividad Positiva (AP) y la Asertividad Negativa (AN). ¿Cual de los modelos es mejor para estos datos?

# In[57]:


# Desplegando funcion programada
def indices_arbol_decision(MC):
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


# In[52]:


# Asignando MC
datos = (([303, 10],[10, 311])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])


# In[54]:


# Cambiando parametros de funcion
MC = df


# #### Modelo Arbol de decision con funcion personalizada de indices

# In[58]:


indices_arbol_decision(MC)


# #### Analisis
# * Con respecto a la precision global se puede ver que es de casi un 97% y su error global es de un 3%, lo que indica que es un modelo bastante fiable desde el punto que generaliza y clasifica la variable que se esta buscando predecir. 
# * Por otro lado, si se ven aspectos como precision positiva (que es la cantidad de aciertos positivos correctos) es de casi un 97% y la precision negativa que al contrario, son la cantidad de aciertos negativos correctos es de igual forma un 97%. 
# * Los Falsos positivos es de solo un 3%, esto quiere decir que del test set, un 3% de los datos fueron clasificados como positivos, pero en realidad eran casos negativos clasificados incorrectamente. Tambien se tiene el mismo porcentaje para la cantidad de falsos negativos (3%) este al contrario se interpreta como la cantidad de casos que siendo positivos, se clasificaron erroneamente como negativos. 
# * Finalmente La Asertividad Positiva es de casi un 97%, esto indica la proporción de buenas predicciones para los positivos y por ultimo, la Asertividad Negativa, que es la proporcion de buenas predicciones para los negativos. 
# * Nota adicional, si se compara con el KNN, el Arbol de Decision tiene mejor poder predictivo. 
# 

# #### 4) Grafique el arbol generado e interprete al menos dos reglas que se puedan extraer del mismo. Si es necesario pode el arbol para que las reglas sean legibles.

# #### Función para mostrar graficar el árbol

# In[59]:


def graficar_arbol(grafico = None):
    grafico.format = "png"
    archivo  = grafico.render()
    img = mpimg.imread(archivo)
    imgplot = plt.imshow(img)
    plt.axis('off')


# #### Graficando el arbol

# In[60]:


datos_plotear = export_graphviz(instancia_arbol, out_file=None,class_names=["Masculino", "Femenino"],
                feature_names=list(X.columns.values), filled=True)
grafico = graphviz.Source(datos_plotear) 
plt.rcParams['figure.figsize'] = [15, 15] # Tamaño del gráfico
graficar_arbol(grafico)


# #### Podando el árbol

# ##### Como no se aprecian bien las "hojas" del arbol se procede a cortarlo para poder dar una interpretacion.

# In[65]:


instancia_arbol2 = DecisionTreeClassifier(min_samples_leaf=100)
instancia_arbol2.fit(X_train,y_train)
prediccion = instancia_arbol2.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### Graficando el arbol nuevamente con menos hojas

# In[66]:


datos_plotear = export_graphviz(instancia_arbol2, out_file=None,class_names=["Masculino", "Femenino"],
                feature_names=list(X.columns.values), filled=True)
grafico = graphviz.Source(datos_plotear) 
plt.rcParams['figure.figsize'] = [30, 30] # Tamaño del gráfico
graficar_arbol(grafico)


# #### Analisis
# * En este caso tomando en cuenta el nodo raiz para el analisis (se empieza en 0) y si el meanfun es menor o igual a 0.139, si esto es cierto, entonces el nodo de la izquierda, va a ser femenino (primer nivel), aqui se puede ver que dentro de este primer nivel, el primer nodo de izquierda a derecha, se observa que el gini es de 0.075, lo cual lo convierte en un nodo impuro, que contiene 1254 observaciones y de esas 49 van a ser Masculinos y 1205 van a ser femenino, es decir, la clase predominante es Femenino. Si continuamos con la interpretacion para el segundo nodo, se ve como tambien es impuro, ya que contiene un gini de 0.087, contiene 1280 observaciones, de las que, 1222 corresponden a Masculino y 58 es Femenino, es decir es predominantemente de genero Masculino este nodo. 
# * Con el segundo nivel, se ve que si el IQR es menor o igual 0.083 y si esto es cierto, entonces el primer nodo de izquierda a derecha del nivel 2, va a tener un gini de 0.449 lo que lo convierte en un nodo impuro, con la cantidad de 100 observaciones, de las que, 34 corresponden a Masculino y 66 a Femenino, lo que predomina es la clase Femenina, con el nodo de la par (segundo nodo) sucede lo mismo, es un nodo impuro pero un poco mas puro que el de la par (primer nodo del nivel 2), aqui se ve como hay un gini de 0.026 hay 1132 observaciones y 15 corresponden a Masculino mientras que, 1139 son Femenino y la clase dominante es Femenino. Por otro lado, si en el nivel 1, el meanfun es igual o menor a 0.148 y si esto es cierto, el tercer nodo del segundo nivel (de izquierda a derecha) sera un nodo impuro con un gini de 0.443 con una cantidad total de observaciones de 118 y de las 118, 79 seran masculino y 39 femenino, dando esto a una clase dominante de masculino, pero si el meanfun no es menor a 0.148, dara lugar a un nodo un poco mas puro (ultimo nodo del nivel 2, de izquierda a derecha) que estara compuesto por un Gini de 0.032, con 1162 observaciones de las que, 1143 son Masculino y 19 Femenino, lo que lo hace un nodo con lcase Masculino. 

# #### 5) Repita los ejercicios 1-4, pero esta vez use 2 combinaciones diferentes de los parametros del metodo DecisionTreeClassifier. ¿Mejora la prediccion?.

# In[67]:


# Cargando datos
voces = pd.read_csv("voces.csv", delimiter = ",", decimal = ".")
print(voces)


# In[68]:


# Eliminando la variable categorica 
X = voces.iloc[:,0:19]
print(X.head())


# In[69]:


# Dejando la variable a predecir en y 
y = voces.iloc[:,20:21] 
print(y.head())


# In[72]:


# Se separan los datos con el 80% de los datos para entrenamiento y el 20% para testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


# #### Ejecucion con otros parametros 2 parametros

# In[74]:


# Para ver los parámetros del modelo
DecisionTreeClassifier()


# In[75]:


# Cambiando los parametros

instancia_arbol2 = DecisionTreeClassifier(criterion='entropy',max_depth=2)
instancia_arbol2.fit(X_train,y_train)
prediccion = instancia_arbol2.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# In[76]:


# Desplegando funcion programada
def indices_arbol_decision(MC):
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


# In[77]:


# Asignando MC
datos = (([304, 4],[16, 310])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])


# In[78]:


# Cambiando parametros de funcion
MC = df


# #### Modelo Arbol de decision con indices modificados y parametros

# In[79]:


indices_arbol_decision(MC)


# #### Analisis con criterio de Entropia
# * Con respecto a la precision global se puede ver que es de casi un 97% y su error global es de un 3%, lo que indica que es un modelo bastante fiable desde el punto que generaliza y clasifica la variable que se esta buscando predecir, en este caso esto es exactamente igual a lo que se habia obtenido con Gini. 
# * Por otro lado, si se ven aspectos como precision positiva (que es la cantidad de aciertos positivos correctos) es de un 95% y la precision negativa que al contrario, son la cantidad de aciertos negativos correctos es de un 98%.
# * Los Falsos positivos es de solo un 1.29%, esto quiere decir que del test set, un 1.29% de los datos fueron clasificados como positivos, pero en realidad eran casos negativos clasificados incorrectamente. Tambien se tiene el porcentaje para la cantidad de falsos negativos de un 4.90% este al contrario se interpreta como la cantidad de casos que siendo positivos, se clasificaron erroneamente como negativos.
# * Finalmente La Asertividad Positiva es de casi un 98%, esto indica la proporción de buenas predicciones para los positivos y por ultimo, la Asertividad Negativa es de un 95% respectivamente; que es la proporcion de buenas predicciones para los negativos.
# * Se puede ver como al hacerlo con el metodo de entropia se obtienen mejores parametros a nivel de AP, PN Y PFP, pero otros como el PFN y el AN mas bien dan indices menores, sin embargo los resultados si mejoran relativamente usando el criterio de Entropia. 

# #### Graficando el Arbol

# In[80]:


datos_plotear = export_graphviz(instancia_arbol, out_file=None,class_names=["Masculino", "Femenino"],
                feature_names=list(X.columns.values), filled=True)
grafico = graphviz.Source(datos_plotear) 
plt.rcParams['figure.figsize'] = [15, 15] # Tamaño del gráfico
graficar_arbol(grafico)


# #### Podando el arbol - ya que no se aprecia bien para hacer analisis - 

# In[81]:


instancia_arbol2 = DecisionTreeClassifier(min_samples_leaf=120)
instancia_arbol2.fit(X_train,y_train)
prediccion = instancia_arbol2.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### Graficando el arbol nuevamente con menos hojas

# In[82]:


datos_plotear = export_graphviz(instancia_arbol2, out_file=None,class_names=["Masculino", "Femenino"],
                feature_names=list(X.columns.values), filled=True)
grafico = graphviz.Source(datos_plotear) 
plt.rcParams['figure.figsize'] = [30, 30] # Tamaño del gráfico
graficar_arbol(grafico)


# #### Analisis
# * En este caso tomando en cuenta el nodo raiz para el analisis (se empieza en 0) y si el meanfun es menor o igual a 0.142, si esto es cierto, entonces el nodo de la izquierda, va a ser femenino (primer nivel), aqui se puede ver que dentro de este primer nivel, el primer nodo de izquierda a derecha, se observa que el gini es de 0.112, lo cual lo convierte en un nodo impuro, que contiene 1290 observaciones y de esas 77 van a ser Masculinos y 1290 van a ser femenino, es decir, la clase predominante es Femenino. Si continuamos con la interpretacion para el segundo nodo, se ve como tambien es impuro, ya que contiene un gini de 0.07, contiene 1244 observaciones, de las que, 1244 corresponden a Masculino y 45 es Femenino, es decir es predominantemente de genero Masculino este nodo. 
# * Con el segundo nivel, se ve que si el IQR es menor o igual a 0.083 y si esto es cierto, entonces el primer nodo de izquierda a derecha del nivel 2, va a tener un gini de 0.498 lo que lo convierte en un nodo impuro, con la cantidad de 120 observaciones, de las que, 56 corresponden a Masculino y 64 a Femenino, lo que predomina es la clase Femenina, con el nodo de la par (segundo nodo) sucede lo mismo, es un nodo impuro pero un poco mas puro que el de la par (primer nodo del nivel 2), aqui se ve como hay un gini de 0.035 hay 1170 observaciones y 21 corresponden a Masculino mientras que, 1170 son Femenino y la clase dominante es Femenino. Por otro lado, si en el nivel 1, el meanfun es igual o menor a 0.149 y si esto es cierto, el tercer nodo del segundo nivel (de izquierda a derecha) sera un nodo impuro con un gini de 0.354 con una cantidad total de observaciones de 126 y de las 126, 97 seran masculino y 29 femenino, dando esto a una clase dominante de masculino, pero si el meanfun no es menor a 0.149, dara lugar a un nodo un poco mas puro (ultimo nodo del nivel 2, de izquierda a derecha) que estara compuesto por un Gini de 0.028, con 1118 observaciones de las que, 1102 son Masculino y 16 Femenino, lo que lo hace un nodo con lcase Masculino. 

# #### Resumen analisis cambiando el criterio a Entropia
# * Si se cambia el criterio a entropia, la prediccion para el arbol aumenta levemente y se obtienen mejores indices de Asertividad Positiva y Falsos Positivos. 

# #### 6) Repita los ejercicios 1-4, pero esta vez use 2 combinaciones diferentes de seleccion de 6 variables predictoras. ¿Mejora la prediccion?.

# #### Selección de Variables - Análisis del poder predictivo de cada una de las variables predictoras¶

# In[105]:


poder_predictivo_numerica(voces,"meanfreq","genero")


# In[106]:


poder_predictivo_numerica(voces,"sd","genero")


# In[107]:


poder_predictivo_numerica(voces,"median","genero")


# In[109]:


poder_predictivo_numerica(voces,"Q75","genero")


# In[110]:


poder_predictivo_numerica(voces,"IQR","genero")


# In[112]:


poder_predictivo_numerica(voces,"kurt","genero")


# In[113]:


poder_predictivo_numerica(voces,"sp.ent","genero")


# In[114]:


poder_predictivo_numerica(voces,"sfm","genero")


# In[115]:


poder_predictivo_numerica(voces,"mode","genero")


# In[116]:


poder_predictivo_numerica(voces,"centroid","genero")


# In[117]:


poder_predictivo_numerica(voces,"meanfun","genero")


# In[118]:


poder_predictivo_numerica(voces,"minfun","genero")


# In[119]:


poder_predictivo_numerica(voces,"maxfun","genero")


# In[120]:


poder_predictivo_numerica(voces,"meandom","genero")


# In[121]:


poder_predictivo_numerica(voces,"mindom","genero")


# In[122]:


poder_predictivo_numerica(voces,"maxdom","genero")


# In[123]:


poder_predictivo_numerica(voces,"dfrange","genero")


# In[124]:


poder_predictivo_numerica(voces,"modindx","genero")


# In[126]:


poder_predictivo_numerica(voces,"Q25","genero")


# #### Primera combinacion:
# * sd
# * Q25
# * IQR
# * Sp.ent
# * Sfm
# * Meanfun

# In[129]:


voces.info()


# #### Deja como variables predictoras en X únicamente a sd, Q25, IQR, Sp.ent, Sfm y Meanfun. En "y" deja la variable a predecir:

# In[131]:


X = voces.iloc[:,[1,3,5,8,9,12]] 
print(X.head())
y = voces.iloc[:,20:21] 
print(y.head())


# #### Genera el modelo predictivo, las predicciones y mide la calidad

# In[133]:


# Con el 80% de los datos para entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)

instancia_arbol2 = DecisionTreeClassifier()

instancia_arbol2.fit(X_train,y_train)

prediccion = instancia_arbol2.predict(X_test)
MC = confusion_matrix(y_test, prediccion)

indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# In[135]:


# Desplegando funcion programada
def indices_arbol_decision(MC):
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

# Asignando MC
datos = (([304, 8],[21, 301])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
# Cambiando parametros de funcion
MC = df


# Modelo Arbol de decision con indices modificados y parametros
indices_arbol_decision(MC)


# #### Analisis con primeras 6 combinaciones
# * Con respecto a la precision global se puede ver que bajo, ya que es de un 95% y su error global es de un 4.57%, lo que indica que es un modelo bastante fiable desde el punto que generaliza y clasifica la variable que se esta buscando predecir, sin embargo es menos fiable que el original con todas sus variables. 
# * Por otro lado, si se ven aspectos como precision positiva (que es la cantidad de aciertos positivos correctos) es de un 93% y la precision negativa que al contrario, son la cantidad de aciertos negativos correctos es de un 97%.
# * Los Falsos positivos es de solo un 1.29%, esto quiere decir que del test set, un 1.29% de los datos fueron clasificados como positivos, pero en realidad eran casos negativos clasificados incorrectamente. Tambien se tiene el porcentaje para la cantidad de falsos negativos de un 6.52% este al contrario se interpreta como la cantidad de casos que siendo positivos, se clasificaron erroneamente como negativos.
# * Finalmente La Asertividad Positiva es de casi un 97%, esto indica la proporción de buenas predicciones para los positivos y por ultimo, la Asertividad Negativa es de un 93% respectivamente; que es la proporcion de buenas predicciones para los negativos.
# * Se puede ver como al hacerlo seleccionando 6 variables segun las mejores distribuciones, no se obtienen mejores resultados que dejando todas las variables. 

# In[139]:


# Graficando el Arbol

datos_plotear = export_graphviz(instancia_arbol2, out_file=None,class_names=["Masculino", "Femenino"],
                feature_names=list(X.columns.values), filled=True)
grafico = graphviz.Source(datos_plotear) 
plt.rcParams['figure.figsize'] = [30, 30] # Tamaño del gráfico
graficar_arbol(grafico)


# In[140]:


# Podando el arbol - ya que no se aprecia bien para hacer analisis -

instancia_arbol2 = DecisionTreeClassifier(min_samples_leaf=100)
instancia_arbol2.fit(X_train,y_train)
prediccion = instancia_arbol2.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# In[141]:


# Graficando el arbol nuevamente con menos hojas
datos_plotear = export_graphviz(instancia_arbol2, out_file=None,class_names=["Masculino", "Femenino"],
                feature_names=list(X.columns.values), filled=True)
grafico = graphviz.Source(datos_plotear) 
plt.rcParams['figure.figsize'] = [30, 30] # Tamaño del gráfico
graficar_arbol(grafico)


# #### Analisis
# * En este caso tomando en cuenta el nodo raiz para el analisis (se empieza en 0) y si el meanfun es menor o igual a 0.142, si esto es cierto, entonces el nodo de la izquierda, va a ser femenino (primer nivel), aqui se puede ver que dentro de este primer nivel, el primer nodo de izquierda a derecha, se observa que el gini es de 0.105, lo cual lo convierte en un nodo impuro, que contiene 1296 observaciones y de esas 72 van a ser Masculinos y 1224 van a ser femenino, es decir, la clase predominante es Femenino. Si continuamos con la interpretacion para el segundo nodo, se ve como tambien es impuro, ya que contiene un gini de 0.06, contiene 1238 observaciones, de las que, 1200 corresponden a Masculino y 38 es Femenino, es decir es predominantemente de genero Masculino este nodo. 
# * Con el segundo nivel, se ve que si el IQR es menor o igual a 0.08 y si esto es cierto, entonces el primer nodo de izquierda a derecha del nivel 2, va a tener un gini de 0.5 lo que lo convierte en un nodo impuro, con la cantidad de 102 observaciones, de las que, 51 corresponden a Masculino y 51 a Femenino, lo que predomina es la clase Masculina (se escoge el primero), con el nodo de la par (segundo nodo) sucede lo mismo, es un nodo impuro pero un poco mas puro que el de la par (primer nodo del nivel 2), aqui se ve como hay un gini de 0.035 hay 1194 observaciones y 21 corresponden a Masculino mientras que, 1173 son Femenino y la clase dominante es Femenino. Por otro lado, si en el nivel 1, el meanfun es igual o menor a 0.148 y si esto es cierto, el tercer nodo del segundo nivel (de izquierda a derecha) sera un nodo impuro con un gini de 0.362 con una cantidad total de observaciones de 101 y de las 101, 77 seran masculino y 24 femenino, dando esto a una clase dominante de masculino, pero si el meanfun no es menor o igual a 0.148, dara lugar a un nodo un poco mas puro (ultimo nodo del nivel 2, de izquierda a derecha) que estara compuesto por un Gini de 0.024, con 1137 observaciones de las que, 1123 son Masculino y 14 Femenino, lo que lo hace un nodo con la case Masculino. 

# #### Segunda Combinacion
# * Meanfreq
# * Median
# * sfm 
# * Centroid
# * Mode
# * Maxdom

# In[142]:


voces.info()


# #### Deja como variables predictoras en X únicamente a Meanfreq, Median, SFM, Centroid, Mode y Maxdom . En "y" deja la variable a predecir

# In[145]:


X = voces.iloc[:,[0,2,9,10,11,17]] 
print(X.head())
y = voces.iloc[:,20:21] 
print(y.head())


# #### Genera el modelo predictivo, las predicciones y mide la calidad¶

# In[146]:


# Con el 80% de los datos para entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)

instancia_arbol2 = DecisionTreeClassifier()

instancia_arbol2.fit(X_train,y_train)

prediccion = instancia_arbol2.predict(X_test)
MC = confusion_matrix(y_test, prediccion)

indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# In[147]:


# Desplegando funcion programada
def indices_arbol_decision(MC):
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

# Asignando MC
datos = (([260, 35],[36, 303])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
# Cambiando parametros de funcion
MC = df


# Modelo Arbol de decision con indices modificados y parametros
indices_arbol_decision(MC)


# #### #### Analisis con segundas 6 combinaciones
# * Con respecto a la precision global se puede ver que bajo considerablemente, ya que es de un 89% y su error global es de un 8.8%, lo que indica que es un modelo fiable, pero no asi, coomo el que se construyo escogiendo las mejores variables, ya que desde el punto que generaliza y clasifica la variable que se esta buscando predecir no es tan certero. 
# * Por otro lado, si se ven aspectos como precision positiva (que es la cantidad de aciertos positivos correctos) es de un 89.38% y la precision negativa que al contrario, son la cantidad de aciertos negativos correctos es de un 88.13%.
# * Los Falsos positivos es de solo un 11.86%, esto quiere decir que del test set, un 11.86%% de los datos fueron clasificados como positivos, pero en realidad eran casos negativos clasificados incorrectamente. Tambien se tiene el porcentaje para la cantidad de falsos negativos de un 10.61% este al contrario se interpreta como la cantidad de casos que siendo positivos, se clasificaron erroneamente como negativos.
# * Finalmente La Asertividad Positiva es de casi un 90%, esto indica la proporción de buenas predicciones para los positivos y por ultimo, la Asertividad Negativa es de un 87% respectivamente; que es la proporcion de buenas predicciones para los negativos.
# * Se puede ver como al hacerlo seleccionando 6 variables al azar (no las mejores) no se obtienen mejores resultados que dejando todas las variables ni escogiendo aquellas variables que cuentan con la mejor distribucion.  

# In[148]:


# Graficando el Arbol

datos_plotear = export_graphviz(instancia_arbol2, out_file=None,class_names=["Masculino", "Femenino"],
                feature_names=list(X.columns.values), filled=True)
grafico = graphviz.Source(datos_plotear) 
plt.rcParams['figure.figsize'] = [30, 30] # Tamaño del gráfico
graficar_arbol(grafico)


# In[149]:


# Podando el arbol - ya que no se aprecia bien para hacer analisis -

instancia_arbol2 = DecisionTreeClassifier(min_samples_leaf=100)
instancia_arbol2.fit(X_train,y_train)
prediccion = instancia_arbol2.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# In[150]:


# Graficando el arbol nuevamente con menos hojas
datos_plotear = export_graphviz(instancia_arbol2, out_file=None,class_names=["Masculino", "Femenino"],
                feature_names=list(X.columns.values), filled=True)
grafico = graphviz.Source(datos_plotear) 
plt.rcParams['figure.figsize'] = [30, 30] # Tamaño del gráfico
graficar_arbol(grafico)


# #### Analisis
# * En este caso tomando en cuenta el nodo raiz para el analisis (se empieza en 0) y si el sfm es menor o igual a 0.297, si esto es cierto, entonces el nodo de la izquierda, va a ser femenino (primer nivel), aqui se puede ver que dentro de este primer nivel, el primer nodo de izquierda a derecha, se observa que el gini es de 0.275, lo cual lo convierte en un nodo impuro, que contiene 856 observaciones y de esas 715 van a ser Masculinos y 141 van a ser femeninas, es decir, la clase predominante es la Masculina. Si continuamos con la interpretacion para el segundo nodo, se ve como tambien es impuro, ya que contiene un gini de 0.45, contiene 1678 observaciones, de las que, 574 corresponden a Masculino y 1104 es Femenino, es decir es predominantemente de genero Femenino este nodo. 
# * Con el segundo nivel, se ve que si el Mode es menor o igual a 0.238 y si esto es cierto, entonces el primer nodo de izquierda a derecha del nivel 2, va a tener un gini de 0.173 lo que lo convierte en un nodo impuro, con la cantidad de 678 observaciones, de las que, 613 corresponden a Masculino y 65 a Femenino, lo que predomina es la clase Masculina, con el nodo de la par (segundo nodo) sucede lo mismo, es un nodo impuro  (primer nodo del nivel 2), aqui se ve como hay un gini de 0.489 hay 178 observaciones y 102 corresponden a Masculino mientras que, 76 son Femenino y la clase dominante es Masculino. Por otro lado, si en el nivel 1, el Mode es igual o menor a 0.145 y si esto es cierto, el tercer nodo del segundo nivel (de izquierda a derecha) sera un nodo impuro con un gini de 0.351 con una cantidad total de observaciones de 745 y de las 745, 169 seran masculino y 576 femenino, dando esto a una clase dominante de femenino, pero si el Mode no es menor o igual a 0.145, dara lugar a un nodo (ultimo nodo del nivel 2, de izquierda a derecha) que estara compuesto por un Gini de 0.491, con 933 observaciones de las que, 405 son Masculino y 528 Femenino, lo que lo hace un nodo con la clase Femenino. 

# #### Resumen analisis apartado 6
# * Se puede observar que con la escogencia de variables no se logro tener una mejoria en las predicciones y que mas bien estas bajaron, no obstante la peor prediccion de todas fue en las que se escogieron variables que no tenian sus datos completamente divididos del genero (entre si) lo que hizo fue que bajara la precision global. 
# * Las mejores predicciones correspondieron al modelo junto con todas las variables que tiene el dataset. 

# ### Pregunta 3:

# #### Esta pregunta utiliza los datos (tumores.csv). Se trata de un conjunto de datos de caracteristicas del tumor cerebral que incluye cinco variables de primer orden y ocho de textura y cuatro parametros de evaluacion de la calidad con el nivel objetivo. La variables son: Media, Varianza, Desviacion estandar, Asimetrıa, Kurtosis, Contraste, Energia, ASM (segundo momento angular), Entropia, Homogeneidad, Disimilitud, Correlacion, Grosor, PSNR (Pico de la relacion senal-ruido), SSIM (Indice de Similitud Estructurada), MSE (Mean Square Error), DC (Coeficiente de Dados) y la variable a predecir tipo (1 = Tumor, 0 = No-Tumor). Realice lo siguiente:

# #### 1) Use el metodo de Arboles de Decision en Python para generar un modelo predictivo para la tabla tumores.csv usando el 70 % de los datos para la tabla aprendizaje y un 20% para la tabla testing.

# In[153]:


tumores = pd.read_csv("tumores.csv", delimiter =",", decimal = ".")
print(tumores)


# In[152]:


tumores.info()


# In[161]:


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


# #### Equilibrio de la Variable a predecir

# In[163]:


distribucion_variable_predecir(tumores,"tipo") 
#Se puede observar como estamos ante de la presencia de una variable desbalanceada a predecir


# #### Genera el modelo, las predicciones y mide la calidad

# In[168]:


# Elimina la variable catégorica, deja las variables predictoras en X
X = tumores.iloc[:,0:17] 
print(X.head())


# In[169]:


#Deja la variable a predecir en y¶
y = tumores.iloc[:,17:18] 
print(y.head())


# In[170]:


#Se separan los datos con el 70% de los datos para entrenamiento y el 20% para testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70)


# In[173]:


# Mediante el constructor inicializa el atributo
instancia_arbol = DecisionTreeClassifier()

#Entrena el modelo llamando al método fit
instancia_arbol.fit(X_train,y_train)

#Imprime las predicciones en testing
print("Las predicciones en Testing son: {}".format(instancia_arbol.predict(X_test)))


# #### Índices de Calidad del Modelo

# In[172]:


prediccion = instancia_arbol.predict(X_test)
MC = confusion_matrix(y_test, prediccion)

indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### 2) Usando la funcion programada en el ejercicio 1 de la tarea anterior, los datos voces.csv y los modelos generados arriba construya un DataFrame de manera que en cada una de las filas aparezca un modelo predictivo y en las columnas aparezcan los indices Precision Global, Error Global Precision Positiva (PP), Precision Negativa (PN), Falsos Positivos (FP), los Falsos Negativos (FN), la Asertividad Positiva (AP) y la Asertividad Negativa (AN). ¿Cual de los modelos es mejor para estos datos?

# In[175]:


# Desplegando funcion programada
def indices_arbol_decision(MC):
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

# Asignando MC
datos = (([27, 5],[3, 350])) 
df = pd.DataFrame(datos, columns = ["Masculino", "Femenino"])
# Cambiando parametros de funcion
MC = df


# Modelo Arbol de decision con indices modificados y parametros
indices_arbol_decision(MC)


# #### Analisis
# * Con respecto a la precision global se puede ver que es de casi un 98% y su error global es de apenas un 2%, lo que indica que es un modelo bastante fiable desde el punto que generaliza y clasifica la variable que se esta buscando predecir. 
# * Por otro lado, si se ven aspectos como precision positiva (que es la cantidad de aciertos positivos correctos) es mayor a un 99%% y la precision negativa que al contrario, son la cantidad de aciertos negativos correctos es de igual forma un 84%, esto quiere decir que este modelo tiene menos precision detectando aciertos nagativos (puede ser tambien porque la mayoria de datos a predecir son de 1 [tiene tumor] lo que crea un overfitting en cuanto a la variable a predecer 1, dejando que el modelo no generalice bien los "0" [no tiene tumor]). 
# * Los Falsos positivos son de un 15.62%, esto quiere decir que del test set, un 15.62% de los datos fueron clasificados como positivos, pero en realidad eran casos negativos clasificados incorrectamente (lo cual es bastante alto, e indica que el modelo no es muy fiable detectando casos donde no existen tumores). Ademas, el porcentaje para la cantidad de falsos negativos es de apenas un 0.8% este al contrario se interpreta como la cantidad de casos que siendo positivos, se clasificaron erroneamente como negativos (el cual es muy bajo). 
# * Finalmente La Asertividad Positiva es de casi un 99%, esto indica la proporción de buenas predicciones para los positivos y por ultimo, la Asertividad Negativa, que es la proporcion de buenas predicciones para los negativos es de un 90% (se confirma lo mencioonado arriba, el modelo no es tan fiable haciendo predicciones para los casos negativos, es decir para cuando no se presenta tumores). 
# * Nota adicional, si se compara con el KNN, el Arbol de Decision tiene mejor poder predictivo a nivel general. 

# #### 3) Grafique el arbol generado e interprete al menos dos reglas que se puedan extraer del mismo. Si es necesario pode el arbol para que las reglas sean legibles. 

# In[181]:


datos_plotear = export_graphviz(instancia_arbol, out_file=None,class_names=["No Tumor", "Tumor"],
                feature_names=list(X.columns.values), filled=True)
grafico = graphviz.Source(datos_plotear) 
plt.rcParams['figure.figsize'] = [30, 30] # Tamaño del gráfico
graficar_arbol(grafico)

# Se ve que no es necesario podar el arbol porque las reglas se aprecian bien con un figure.size de 30X30.


# #### Analisis
# * En este caso tomando en cuenta el nodo raiz para el analisis, se puede ver que si el dc es igual o menor que 0, y si esto es verdadero, entonces en el nivel 1, se tendra un primer nodo con un gini de 0.39, lo que da a entender que es un nodo impuro, con 94 observaciones, de las que 69 corresponden a tumores y 25 corresponden a no tumores. En cambio, si en el nodo raiz el dc es mayor que 0, entonces estaremos ante la presencia de un nodo completamente puro, debido a que tiene un gino de 0, tiene 0 observaciones de "no tumor" y 798 de tumores, es decir, la clase de este nodo es Tumores (hay tumores).
# * Siguiendo con el analisis en el nivel 2, si la desviacion estandar del nivel 1, era menor o igual a 0.002, y si esto es verdadero, entonces en el nivel 2 existe un primer nodo de izquierda a derecha que sera impuro, tendra un gini de 0.451 y estara conformado por 35 observaciones, de las que, 12 corresponden a "No tumor" y 23 corresponden a "tumor" generando una clase predominante de "Tumor" y el segundo nodo, de izquierda a derecha, se formara porque la desviacion estandar en el nivel 1, es mayor a 5.81 lo que originara un nodo impuro, con un gini de 0.065 que tendra una cantidad total de 59 observaciones de las que 57 seran de "No tumor" y 2, de "Tumor", generando una clase de "No tumor" en realidad este ultimo nodo es casi puro, debido a que solo tiene dos observaciones que no corresponden a la clase. 

# ### Pregunta 4: 

# #### En este ejercicio vamos a predecir numeros escritos a mano (Hand Written Digit Recognition), la tabla de aprendizaje esta en el archivo ZipDataTrainCod.csv y la tabla de testing esta en el archivo ZipDataTestCod.csv. En la figura siguiente se ilustran los datos:

# In[182]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/numeros.png")


# #### Los datos de este ejemplo vienen de los codigos postales escritos a mano en sobres del correo postal de EE.UU. Las imagenes son de 16 × 16 en escala de grises, cada pixel va de intensidad de −1 a 1 (de blanco a negro). Las imagenes se han normalizado para tener aproximadamente el mismo tamano y orientacion. La tarea consiste en predecir, a partir de la matriz de 16 × 16 de intensidades de cada pixel, la identidad de cada imagen (0, 1, . . . , 9) de forma rapida y precisa. Si es lo suficientemente precisa, el algoritmo resultante se utiliza como parte de un procedimiento de selecci´on automatica para sobres.

# #### 1) Usando Arboles de Decision mas cercanos un modelo predictivo para la tabla de aprendizaje.

# In[183]:


# Importando datos de train 
ZipDataTrainCod = pd.read_csv("ZipDataTrainCod.csv", delimiter= ';', decimal = '.')
print(ZipDataTrainCod)


# In[184]:


# Importando datos de test
ZipDataTestCod = pd.read_csv("ZipDataTestCod.csv", delimiter = ';', decimal = '.')
print(ZipDataTestCod)


# In[185]:


distribucion_variable_predecir(ZipDataTrainCod,"Numero")


# ##### La distribucion muestra que la matriz no esta del todo balancedada, porque, el 0 y el uno tienen muchos datos, mientras que otros como el 5 y el 8 tienen menos

# #### Eliminando la variable predictora en el conjunto de entrenamiento

# In[186]:


X_train = ZipDataTrainCod.iloc[:,1:257] 
print(X_train.head())


# #### Definiendo la variable predictora en el conjunto de entrenamiento

# In[187]:


y_train = ZipDataTrainCod.iloc[:,0:1] 
print(y_train.head())


# #### Eliminando la variable predictora en el conjunto de test

# In[189]:


X_test = ZipDataTestCod.iloc[:,1:257] 
print(X_test.head())


# #### Definiendo la variable predictora en el conjunto de test

# In[190]:


y_test = ZipDataTestCod.iloc[:,0:1] 
print(y_test.head())


# #### Mediante el constructor inicializa el atributo

# In[191]:


instancia_arbol = DecisionTreeClassifier(random_state=0)


# #### Entrena el modelo llamando al método fit

# In[192]:


instancia_arbol.fit(X_train,y_train)


# #### Imprime las predicciones en testing

# In[193]:


print("Las predicciones en Testing son: {}".format(instancia_arbol.predict(X_test)))


# #### 2. Con la tabla de testing calcule la matriz de confusion, la precision global, el error global y la precision en cada unos de los dıgitos. ¿Son buenos los resultados? Ademas compare respecto a los resultados obtenidos en la tarea anterior.

# #### Índices de Calidad del Modelo con Arbol de Decision

# In[197]:


prediccion = instancia_arbol.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y_train)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### Índices de Calidad del Modelo con KNN

# #### Se escoge la cantidad de Ns del modelo, donde N = 85 (√7291 = 85 (redondeado))

# In[199]:


from sklearn.neighbors import KNeighborsClassifier
instancia_knn = KNeighborsClassifier(n_neighbors=85)


# #### Entrenando modelo con datos de training¶

# In[200]:


instancia_knn.fit(X_train,y_train.iloc[:,0].values)


# #### Imprimiendo las predicciones en testing

# In[201]:


print("Las predicciones en Testing son: {}".format(instancia_knn.predict(X_test)))


# #### Indices de Calidad del Modelo 

# In[204]:


prediccion = instancia_knn.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y_train)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### Resumen del Analisis
#    * Arbol de decision: 
# * Con respecto a la precision global se tiene un 82.56%, y un error global de 17.43%, y la precision por categoria solo alcanza mas de un 90% en los numeros a predecir que son cero y uno (y es porque hay muchas en el test). Por otro lado, se tiene entre 80 y 90% de precision en los numeros 6 y 7, los demas numeros a predecir tienen precisiones menores a 80%; siendo el numero 8 el que recibe la peor precision globa. 
#    * KNN: 
# * La precision global es de un 87%, mientras que el error global es de um 12.75%, por otro lado, la precision por categorias, los numeros que obtienen las mejores precisiones (mayores a 90%) son: 0, 1, 7, 9, mientras que los que tienen precisiones entre 80 y 90% son: 6, 3 y los que presentan las precisiones mas bajas (entre 70 y 80%) son: 2, 4, 5, 8, siendo el 2 el numero que recibe la peor precision global. 
# 
# Resumen: en cuanto a este dataset especifico de escritura a mano, el modelo de Arbol de Decision no tiene una tan buena precision y tiende a perder fidelidad cuando los numeros a predecir no son ni 0 ni 1, mientras que el KNN tiene una mejor precision a nivel general y las precisiones por numeros son mejores. 

# #### 3) Repita los ejercicios 1, 2 y 3 pero usando solamente los 3s, 5s y los 8s. ¿Mejora la prediccion?

# #### Importando datos que solo contengan 3, 5 y 8 del dataset de entrenamiento.

# In[205]:


# Importando datos de train 
ZipDataTrainCod = pd.read_csv("ZipDataTrainCod.csv", delimiter= ';', decimal = '.')

nuevo = ZipDataTrainCod[(ZipDataTrainCod.Numero == "tres") | 
                        (ZipDataTrainCod.Numero == "cinco") | (ZipDataTrainCod.Numero == "ocho")]
print(nuevo)


# In[206]:


# Importando datos de test
ZipDataTestCod = pd.read_csv("ZipDataTestCod.csv", delimiter = ';', decimal = '.')
nuevo_2 = ZipDataTestCod[(ZipDataTestCod.Numero == "tres") | (ZipDataTestCod.Numero == "cinco")
                        | (ZipDataTestCod.Numero == "ocho")]
print(nuevo_2)


# In[207]:


distribucion_variable_predecir(nuevo,"Numero")


# ##### Los numeros 3, 5 y 8 estan bastante balanceados entre si, salvo el tres que tiene un poco mas de datos

# #### Eliminando la variable predictora en el conjunto de entrenamiento

# In[208]:


X_train = nuevo.iloc[:,1:257] 
print(X_train.head())


# #### Definiendo la variable predictora en el conjunto de entrenamiento

# In[209]:


y_train = nuevo.iloc[:,0:1] 
print(y_train.head())


# #### Eliminando la variable predictora en el conjunto de test

# In[211]:


X_test = nuevo_2.iloc[:,1:257] 
print(X_test.head())


# #### Definiendo la variable predictora en el conjunto de test

# In[212]:


y_test = nuevo_2.iloc[:,0:1] 
print(y_test.head())


# #### Mediante el constructor inicializa el atributo

# In[213]:


instancia_arbol = DecisionTreeClassifier(random_state=0)


# #### Entrena el modelo llamando al método fit¶

# In[214]:


instancia_arbol.fit(X_train,y_train)


# #### Imprime las predicciones en testing

# In[215]:


print("Las predicciones en Testing son: {}".format(instancia_arbol.predict(X_test)))


# #### Índices de Calidad del Modelo

# In[217]:


prediccion = instancia_arbol.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y_train)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### Analisis
# * Utilizando solo las filas que contengan los numeros 5, 8 y 3, se obtiene ahora una precision global muy parecida a la que se obtiene sin hacer el filtro de los numeros mencionados, ya que la precision global es de un 82.31%, mientras que la tenia la matriz de confusion original (sin hacer el filtro de los numeros predictores) es de un 82.56%. 
# * El error global de un 17.68% con este filtrado, mientras que el error sin filtrar corresponde a un 17.43% (es menor el original).
# * Por otro lado, si se analizan especificamente la precision por categoria de los numeros 3, 5 y 8, se puede ver mas bien estos aumentaron, ya que pasaron de un 75.30% a un 78.31% (numero 3), de un 73.12% a un 84.37% (numero 5) y finalmente de un 68.67% a un 84.33% (para el numero 8). Siendo a nivel global el numero 8 el que recibio la mejor precision versus los valores originales sin filtrar estos numeros. 
# * Se observa que cuando se hace el filtro con los numeros indicados, se obtiene una mayor precision por categoria especifica y la diferencia en la precision global y el error global es apenas notable. 

# #### 4. Optativo: [10 puntos] Repita los ejercicios 1, 2 y 3 pero reemplazando cada bloque 4 × 4 de pixeles por su promedio. ¿Mejora la prediccion? Recuerde que cada bloque 16×16 esta representado por una fila en las matrices de aprendizaje y testing. Despliegue la matriz de confusion resultante.

# In[218]:


# Importando datos de train 
datos = pd.read_csv("ZipDataTrainCod.csv", delimiter= ';', decimal = '.')
print(datos)


# #### Definiendo funcion que redimensione en matriz 4X4 Train set

# In[219]:


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


# In[220]:


# Llamando parametros de funcion
redimension(datos)
datos_reducidos = redimension(datos)


# In[221]:


print(datos_reducidos)


# In[222]:


# Importando datos de test
datos_2 = pd.read_csv("ZipDataTestCod.csv", delimiter = ';', decimal = '.')
print(datos_2)


# #### Definiendo funcion que redimensione en matriz 4X4 Test set: 

# In[223]:


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


# In[224]:


# Llamando funcion
redimension(datos_2)
datos_reducidos_2 = redimension(datos_2)


# In[225]:


print(datos_reducidos_2)


# #### Distrubucion de la variable a predecer¶

# In[227]:


distribucion_variable_predecir(datos_reducidos,0)


# #### Eliminando la variable predictora en el conjunto de entrenamiento

# In[228]:


X_train = datos_reducidos.iloc[:,1:17] 
print(X_train.head())


# #### Definiendo la variable predictora en el conjunto de entrenamiento

# In[230]:


y_train = datos_reducidos.iloc[:,0:1] 
print(y_train.head())


# #### Eliminando la variable predictora en el conjunto de test

# In[231]:


X_test  = datos_reducidos_2.iloc[:,1:17] 
print(X_test.head())


# #### Definiendo la variable predictora en el conjunto de test

# In[292]:


y_test = datos_reducidos_2.iloc[:,0:1] 
print(y_test.head())


# #### Mediante el constructor inicializa el atributo

# In[233]:


instancia_arbol = DecisionTreeClassifier(random_state=0)


# #### Entrena el modelo llamando al método fit

# In[235]:


instancia_arbol.fit(X_train,y_train)


# #### Imprime las predicciones en testing

# In[236]:


print("Las predicciones en Testing son: {}".format(instancia_arbol.predict(X_test)))


# #### Índices de Calidad del Modelo

# In[239]:


prediccion = instancia_arbol.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y_train)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### Analisis
# * Si se reduce la matriz a una de 4X4 se ve que la precision global es de un 100%, no hay un error global y la precision por categoria en todos los numeros a predecir es de un 100%. Se puede observar como cambiando la matriz se logra una optmizacion en la prediccion global. 

# #### Importando datos que solo contengan 3, 5 y 8 del dataset de entrenamiento 

# In[257]:


# Importando datos de train 
ZipDataTrainCod = pd.read_csv("ZipDataTrainCod.csv", delimiter= ';', decimal = '.')

datos = ZipDataTrainCod[(ZipDataTrainCod.Numero == "tres") | 
                        (ZipDataTrainCod.Numero == "cinco") | (ZipDataTrainCod.Numero == "ocho")]
print(datos)


# In[250]:


# Definiendo funcion que redimensiona a una matriz de 4X4
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


# In[258]:


# Llamando parametros de funcion
redimension(datos)
datos_reducidos = redimension(datos)


# In[259]:


print(datos_reducidos)


# In[260]:


# Importando datos de test
ZipDataTestCod = pd.read_csv("ZipDataTestCod.csv", delimiter = ';', decimal = '.')
datos_2 = ZipDataTestCod[(ZipDataTestCod.Numero == "tres") | (ZipDataTestCod.Numero == "cinco")
                        | (ZipDataTestCod.Numero == "ocho")]
print(datos_2)


# In[261]:


# Definiendo funcion que redimensiona a matriz de 4X4
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


# In[262]:


# Llamando funcion
redimension(datos_2)
datos_reducidos_2 = redimension(datos_2)


# In[263]:


print(datos_reducidos_2)


# In[264]:


distribucion_variable_predecir(datos,"Numero")


# #### Los numeros 3, 5 y 8 estan bastante balanceados entre si, salvo el tres que tiene un poco mas de datos

# #### Eliminando la variable predictora en el conjunto de entrenamiento

# In[267]:


X_train = datos_reducidos.iloc[:,1:16] 
print(X_train.head())


# #### Definiendo la variable predictora en el conjunto de entrenamiento

# In[268]:


y_train = datos_reducidos.iloc[:,0:1] 
print(y_train.head())


# #### Eliminando la variable predictora en el conjunto de test

# In[269]:


X_test = datos_reducidos_2.iloc[:,1:16] 
print(X_test.head())


# #### Definiendo la variable predictora en el conjunto de test

# In[270]:


y_test = datos_reducidos_2.iloc[:,0:1] 
print(y_test.head())


# #### Mediante el constructor inicializa el atributo

# In[271]:


instancia_arbol = DecisionTreeClassifier(random_state=0)


# #### Entrena el modelo llamando al método fit

# In[272]:


instancia_arbol.fit(X_train,y_train)


# #### Imprime las predicciones en testing

# In[273]:


print("Las predicciones en Testing son: {}".format(instancia_arbol.predict(X_test)))


# #### Índices de Calidad del Modelo solo tomando en cuenta 3, 5 y 8

# In[275]:


prediccion = instancia_arbol.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y_train)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### Analisis
# * Si se reduce la matriz a una de 4X4 y se filtra, para que solo tome en cuenta las filas que tengan en la columna "Numero" los 3, 5 y 8, se ve que la precision global es de un 100%, no hay un error global y la precision por categoria en todos los numeros a predecir es de un 100%. Se puede observar como cambiando la matriz se logra una optmizacion en la prediccion global con el filtro. 

# #### 5) Optativo: [10 puntos] Repita los ejercicios 1, 2 y 3 pero reemplazando cada bloque p × p de pıxeles por su promedio. ¿Mejora la prediccion? (pruebe con algunos valores de p). Despliegue las matrices de confusion resultantes.

# In[277]:


# Importando datos de train 
datos = pd.read_csv("ZipDataTrainCod.csv", delimiter= ';', decimal = '.')
print(datos)


# #### Definiendo funcion que redimensione en matriz 8X8 Train set

# In[278]:


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


# In[279]:


# Llamando parametros de funcion
redimension(datos)
datos_reducidos = redimension(datos)


# In[280]:


print(datos_reducidos)


# In[281]:


# Importando datos de test
datos_2 = pd.read_csv("ZipDataTestCod.csv", delimiter = ';', decimal = '.')
print(datos_2)


# #### Definiendo funcion que redimensione en matriz 8X8 Test set:

# In[282]:


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


# In[283]:


# Llamando funcion
redimension(datos_2)
datos_reducidos_2 = redimension(datos_2)


# In[284]:


print(datos_reducidos_2)


# #### Distrubucion de la variable a predecer¶

# In[285]:


distribucion_variable_predecir(datos_reducidos,0)


# #### Eliminando la variable predictora en el conjunto de entrenamiento

# In[287]:


X_train = datos_reducidos.iloc[:,1:65] 
print(X_train.head())


# #### Definiendo la variable predictora en el conjunto de entrenamiento

# In[288]:


y_train = datos_reducidos.iloc[:,0:1] 
print(y_train.head())


# #### Eliminando la variable predictora en el conjunto de test

# In[289]:


X_test  = datos_reducidos_2.iloc[:,1:65] 
print(X_test.head())


# #### Definiendo la variable predictora en el conjunto de test

# In[291]:


y_test = datos_reducidos_2.iloc[:,0:1] 
print(y_test.head())


# #### Mediante el constructor inicializa el atributo

# In[293]:


instancia_arbol = DecisionTreeClassifier(random_state=0)


# #### Entrena el modelo llamando al método fit

# In[294]:


instancia_arbol.fit(X_train,y_train)


# #### Imprime las predicciones en testing

# In[295]:


print("Las predicciones en Testing son: {}".format(instancia_arbol.predict(X_test)))


# #### Índices de Calidad del Modelo

# In[296]:


prediccion = instancia_arbol.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y_train)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### Analisis
# Si se reduce la matriz a una de 8X8 se ve que la precision global es de un 100%, no hay un error global y la precision por categoria en todos los numeros a predecir es de un 100%. Se puede observar como cambiando la matriz se logra una optmizacion en la prediccion global. Ademas, se puede ver como al reducir la matriz se van obteniendo mejores predicciones. 

# #### Importando datos que solo contengan 3, 5 y 8 del dataset de entrenamiento

# In[297]:


# Importando datos de train 
ZipDataTrainCod = pd.read_csv("ZipDataTrainCod.csv", delimiter= ';', decimal = '.')

datos = ZipDataTrainCod[(ZipDataTrainCod.Numero == "tres") | 
                        (ZipDataTrainCod.Numero == "cinco") | (ZipDataTrainCod.Numero == "ocho")]
print(datos)


# In[298]:


# Definiendo funcion que redimensiona a una matriz de 8X8
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


# In[299]:


# Llamando parametros de funcion
redimension(datos)
datos_reducidos = redimension(datos)


# In[300]:


print(datos_reducidos)


# In[301]:


# Importando datos de test
ZipDataTestCod = pd.read_csv("ZipDataTestCod.csv", delimiter = ';', decimal = '.')
datos_2 = ZipDataTestCod[(ZipDataTestCod.Numero == "tres") | (ZipDataTestCod.Numero == "cinco")
                        | (ZipDataTestCod.Numero == "ocho")]
print(datos_2)


# In[302]:


# Definiendo funcion que redimensiona a matriz de 8X8
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


# In[303]:


# Llamando funcion
redimension(datos_2)
datos_reducidos_2 = redimension(datos_2)


# In[304]:


print(datos_reducidos_2)


# In[309]:


distribucion_variable_predecir(datos,"Numero")


# #### Los numeros 3, 5 y 8 estan bastante balanceados entre si, salvo el tres que tiene un poco mas de datos

# #### Eliminando la variable predictora en el conjunto de entrenamiento

# In[311]:


X_train = datos_reducidos.iloc[:,1:65] 
print(X_train.head())


# #### Definiendo la variable predictora en el conjunto de entrenamiento

# In[312]:


y_train = datos_reducidos.iloc[:,0:1] 
print(y_train.head())


# #### Eliminando la variable predictora en el conjunto de test

# In[313]:


X_test = datos_reducidos_2.iloc[:,1:65] 
print(X_test.head())


# #### Definiendo la variable predictora en el conjunto de test

# In[314]:


y_test = datos_reducidos_2.iloc[:,0:1] 
print(y_test.head())


# #### Mediante el constructor inicializa el atributo

# In[315]:


instancia_arbol = DecisionTreeClassifier(random_state=0)


# #### Entrena el modelo llamando al método fit

# In[316]:


instancia_arbol.fit(X_train,y_train)


# #### Imprime las predicciones en testing

# In[317]:


print("Las predicciones en Testing son: {}".format(instancia_arbol.predict(X_test)))


# #### Índices de Calidad del Modelo solo tomando en cuenta 3, 5 y 8

# In[319]:


prediccion = instancia_arbol.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y_train)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### Analisis
# * Si se reduce la matriz a una de 8X8 y se filtra, para que solo tome en cuenta las filas que tengan en la columna "Numero" los 3, 5 y 8, se ve que la precision global es de un 100%, no hay un error global y la precision por categoria en todos los numeros a predecir es de un 100%. * Se puede observar como cambiando la matriz se logra una optmizacion en la prediccion global con el filtro.
################################################################### FIN #######################################################################