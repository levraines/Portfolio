#!/usr/bin/env python
# coding: utf-8

# # Tarea #4

# # Estudiante: Heiner Romero Leiva

# ### Importando bibliotecas

# In[1]:


import numpy  as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
from matplotlib import colors as mcolors
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd
import random as rd
import matplotlib.pyplot as plt


# ### Funcion para graficar la curva ROC

# In[2]:


def plotROC(real, prediccion, color = "red", label = None):
    fp_r, tp_r, umbral = roc_curve(real, prediccion)
    plt.plot(fp_r, tp_r, lw = 1, color = color, label = label)
    plt.plot([0, 1], [0, 1], lw = 1, color = "black")
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curva ROC")


# ### Funcion para calcular los indices de la Calidad de la Prediccion

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


# ### Funcion para graficar la distribucion de la variable a predecir

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


# ## Pregunta 1: 

# #### En esta pregunta utiliza los datos (tumores.csv). Se trata de un conjunto de datos de caracteristicas del tumor cerebral que incluye cinco variables de primer orden y ocho de textura y cuatro parametros de evaluacion de la calidad con el nivel objetivo. La variables son: Media, Varianza, Desviacion estandar, Asimetria, Kurtosis, Contraste, Energia, ASM (segundo momento angular), Entropia, Homogeneidad, Disimilitud, Correlacion, Grosor, PSNR (Pico de la relacion senal-ruido), SSIM (Indice de Similitud Estructurada), MSE (Mean Square Error), DC (Coeficiente de Dados) y la variable a predecir tipo (1 = Tumor, 0 = No-Tumor).

# #### 1. Cargue la tabla de datos tumores.csv en Python.

# In[4]:


tumores = pd.read_csv("tumores.csv", delimiter = ',', decimal = '.')
tumores.head()


# In[5]:


tumores.info()


# In[6]:


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


# In[7]:


# Normalizando y centrando la tabla ya que hay valores en diferentes escalas
tumores_1 = tumores.iloc[:,0:17]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values = scaler.fit_transform(tumores_1) 
tumores_1.loc[:,:] = scaled_values
tumores_1.head()


# #### Elimina la variable catégorica, deja las variables predictoras en X

# In[8]:


X = tumores.iloc[:,0:17] 


# #### Deja la variable a predecir en y

# In[9]:


y = tumores.iloc[:,17:18] 


# #### Ajusta los datos de entrenamiento

# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75)


# #### 2. Comparare todos los metodos predictivos vistos en el curso con esta tabla de datos utilizado la curva ROC y el area bajo la curva ROC. Aqui interesa predecir en la variable tipo. Compare los metodos SVM, KNN, Arboles, Bosques, ADA Boosting, eXtreme Gradient Boosting, Bayes, LDA, QDA y Redes Neuronales del paquete Keras y del MLPClassifier. ¿Se puede determinar con claridad cual metodos es el mejor?

# #### --------> Metodo SVM 

# In[11]:


# Construímos el modelo con probability=True
instancia_svm = SVC(kernel = "rbf",gamma='scale',probability=True)
instancia_svm.fit(X_train, y_train.iloc[:,0].values)
print("Probabilidad del No y del Si:\n",instancia_svm.predict_proba(X_test))
probabilidad = instancia_svm.predict_proba(X_test)[:, 1]
print("Probabilidad de Si (o sea del 1):\n",probabilidad)
# Genera curva ROC para SVM
plt.figure(figsize=(10,10))
plotROC(y_test, instancia_svm.predict_proba(X_test)[:, 1])


# #### --------> Metodo KNN

# In[12]:


# Se construye el modelo
instancia_knn = KNeighborsClassifier(n_neighbors=30) # se utiliza 30 porque (1275*70%= √893 = 30 (redondeado))
instancia_knn.fit(X_train,y_train.iloc[:,0].values)
print("Probabilidad del No y del Si:\n",instancia_knn.predict_proba(X_test))
probabilidad = instancia_knn.predict_proba(X_test)[:, 1]
print("Probabilidad de Si (o sea del 1):\n",probabilidad)
# Genera la Curva ROC para KNN
plt.figure(figsize=(10,10))
plotROC(y_test, instancia_knn.predict_proba(X_test)[:, 1], color = "blue")


# #### --------> Metodo Arboles de Decision

# In[13]:


# Se construye el modelo
instancia_arbol = DecisionTreeClassifier()
instancia_arbol.fit(X_train,y_train)
print("Probabilidad del No y del Si:\n",instancia_arbol.predict_proba(X_test))
probabilidad = instancia_arbol.predict_proba(X_test)[:, 1]
print("Probabilidad de Si (o sea del 1):\n",probabilidad)
# Genera la Curva ROC para Decision Trees
plt.figure(figsize=(10,10))
plotROC(y_test, instancia_arbol.predict_proba(X_test)[:, 1], color = "green")


# #### --------> Metodo Random Forest

# In[14]:


instancia_bosque = RandomForestClassifier(n_estimators=120, criterion = 'gini', max_features = 3)
instancia_bosque.fit(X_train,y_train.iloc[:,0].values)
print("Probabilidad del No y del Si:\n",instancia_bosque.predict_proba(X_test))
probabilidad = instancia_bosque.predict_proba(X_test)[:, 1]
print("Probabilidad de Si (o sea del 1):\n",probabilidad)
# Genera la Curva ROC para Random Forest
plt.figure(figsize=(10,10))
plotROC(y_test, instancia_bosque.predict_proba(X_test)[:, 1], color = "purple")


# #### --------> Metodo ADA Boosting

# In[16]:


instancia_potenciacion = AdaBoostClassifier(n_estimators=120)
instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
print("Probabilidad del No y del Si:\n",instancia_potenciacion.predict_proba(X_test))
probabilidad = instancia_potenciacion.predict_proba(X_test)[:, 1]
print("Probabilidad de Si (o sea del 1):\n",probabilidad)
# Genera la Curva ROC para Ada Boosting
plt.figure(figsize=(10,10))
plotROC(y_test, instancia_potenciacion.predict_proba(X_test)[:, 1], color = "pink")


# #### --------> Metodo XG Boosting

# In[18]:


instancia_potencia = GradientBoostingClassifier(n_estimators=120)
instancia_potencia.fit(X_train,y_train.iloc[:,0].values)
print("Probabilidad del No y del Si:\n",instancia_potencia.predict_proba(X_test))
probabilidad = instancia_potencia.predict_proba(X_test)[:, 1]
print("Probabilidad de Si (o sea del 1):\n",probabilidad)
# Genera la Curva ROC para XG Boosting
plt.figure(figsize=(10,10))
plotROC(y_test, instancia_potencia.predict_proba(X_test)[:, 1], color = "darkgreen")


# #### --------> Metodo de Bayes

# In[19]:


bayes = GaussianNB()
bayes.fit(X_train, y_train.iloc[:,0].values)
print("Probabilidad del No y del Si:\n",bayes.predict_proba(X_test))
probabilidad = bayes.predict_proba(X_test)[:, 1]
print("Probabilidad de Si (o sea del 1):\n",probabilidad)
# Genera la Curva ROC para Metodo de Bayes
plt.figure(figsize=(10,10))
plotROC(y_test, bayes.predict_proba(X_test)[:, 1], color = "orange")


# #### --------> Metodo de Discriminante Lineal

# In[20]:


lda = LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage = 'auto')
lda.fit(X_train, y_train.iloc[:,0].values)
print("Probabilidad del No y del Si:\n",lda.predict_proba(X_test))
probabilidad = lda.predict_proba(X_test)[:, 1]
print("Probabilidad de Si (o sea del 1):\n",probabilidad)
# Genera la Curva ROC para Metodo de Discriminante Lineal
plt.figure(figsize=(10,10))
plotROC(y_test, lda.predict_proba(X_test)[:, 1], color = "cyan")


# #### --------> Metodo de Discriminante Cuadratico

# In[21]:


qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train.iloc[:,0].values)
print("Probabilidad del No y del Si:\n",qda.predict(X_test))
probabilidad = qda.predict(X_test)
print("Probabilidad de Si (o sea del 1):\n",probabilidad)
# Genera la Curva ROC para Metodo de Discriminante Cuadratico
plt.figure(figsize=(10,10))
plotROC(y_test, qda.predict(X_test), color = "lime")


# #### --------> Metodo Redes Neuronales MLPClassifier

# In[22]:


instancia_red = MLPClassifier(solver='lbfgs', random_state=0,hidden_layer_sizes=[10000, 350])
instancia_red.fit(X_train,y_train.iloc[:,0].values)
print("Probabilidad del No y del Si:\n",instancia_red.predict_proba(X_test))
probabilidad = instancia_red.predict_proba(X_test)[:, 1]
print("Probabilidad de Si (o sea del 1):\n",probabilidad)
# Genera la Curva ROC para MLP Classifier
plt.figure(figsize=(10,10))
plotROC(y_test, instancia_red.predict_proba(X_test)[:, 1], color = "yellow")


# #### --------> Metodo Redes Neuronales Paquete Keras

# #### Como la variable a predecir la dan en terminos de 0 y 1, es necesario convertirla a "Si" y "No".

# In[25]:


d = tumores
df = pd.DataFrame(data=d)
df


# In[26]:


df.replace({0: "No", 1: "Si"}, inplace = True)
print(df.iloc[:,17:18]) #Resultado fue reemplazado con exito. 


# In[27]:


# Verificando cambio
y = df.iloc[:,17:18] 
y.head()


# In[28]:


# Como la variable a predecir ya viene dada por "0" y "1" no es necesario utilizar codigo disyuntivo ni reescalar
# Convirtiendo en dummies para separacion
from sklearn.preprocessing import MinMaxScaler
dummy_y = pd.get_dummies(y)
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_X  = pd.DataFrame(scaler.fit_transform(X), columns = list(X))
X_train, X_test, y_train, y_test = train_test_split(scaled_X, dummy_y, train_size = 0.75, random_state = 0)
print(dummy_y)


# In[29]:


# Definiendo Modelo
model = Sequential()
model.add(Dense(100, input_dim = 17, activation = 'relu'))  # primera capa oculta con 100 neuronas
model.add(Dense(500, activation = 'sigmoid'))  # segunda capa oculta con 500 neuronas
model.add(Dense(300, activation = 'sigmoid'))  # tercera capa oculta con 300 neuronas
model.add(Dense(50, activation = 'relu'))  # Agregamos tercera capa oculta con 50 neuronas
model.add(Dense(2, activation = 'softmax')) # Agregamos capa output con 2 neuronas
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])  
model.fit(X_train, y_train, epochs = 1000, batch_size = 150, verbose = 0)
# La predicción es una matriz con 3 columnas
y_pred = model.predict(X_test)
# Convertimos a columna
y_test_class = np.argmax(np.asanyarray(y_test), axis = 1)  # Convertimos a array
y_pred_class = np.argmax(y_pred, axis = 1)
# Genera la Curva ROC para Redes Neuronales usando paquete Keras
plt.figure(figsize=(10,10))
plotROC(y_test_class, y_pred_class, color = "gold")


# #### Todos los gráficos juntos para comparar

# In[23]:


plt.figure(figsize=(10,10))
plotROC(y_test, instancia_svm.predict_proba(X_test)[:, 1], label = 'SVM')
plotROC(y_test, instancia_knn.predict_proba(X_test)[:, 1], color = "blue", label = 'KNN')
plotROC(y_test, instancia_arbol.predict_proba(X_test)[:, 1], color = "green", label = 'Arboles')
plotROC(y_test, instancia_bosque.predict_proba(X_test)[:, 1], color = "purple", label = 'RamdomForest')
plotROC(y_test, instancia_potenciacion.predict_proba(X_test)[:, 1], color = "pink", label = 'ADA')
plotROC(y_test, instancia_potencia.predict_proba(X_test)[:, 1], color = "darkgreen", label = 'XG')
plotROC(y_test, bayes.predict_proba(X_test)[:, 1], color = "orange", label = 'Bayes')
plotROC(y_test, lda.predict_proba(X_test)[:, 1], color = "cyan", label = 'LDA')
plotROC(y_test, qda.predict(X_test), color = "lime", label = 'QDA')
plotROC(y_test, instancia_red.predict_proba(X_test)[:, 1], color = "yellow", label = 'RedNeuMLP')
plt.legend(loc = "lower right")


# #### Comparando el área bajo la curva ROC

# In[24]:


SVM = roc_auc_score(y_test, instancia_svm.predict_proba(X_test)[:, 1])
KNN = roc_auc_score(y_test, instancia_knn.predict_proba(X_test)[:, 1])
Arbol = roc_auc_score(y_test, instancia_arbol.predict_proba(X_test)[:, 1])
Random = roc_auc_score(y_test, instancia_bosque.predict_proba(X_test)[:, 1])
ADA = roc_auc_score(y_test, instancia_potenciacion.predict_proba(X_test)[:, 1])
XG = roc_auc_score(y_test, instancia_potencia.predict_proba(X_test)[:, 1])
Bayes = roc_auc_score(y_test, bayes.predict_proba(X_test)[:, 1])
LDA = roc_auc_score(y_test, lda.predict_proba(X_test)[:, 1])
QDA = roc_auc_score(y_test, qda.predict(X_test))
MPL = roc_auc_score(y_test, instancia_red.predict_proba(X_test)[:, 1])


print("Área bajo la curva ROC en Maquinas de Soporte Vectorial: {:.3f}".format(SVM))
print("Área bajo la curva ROC en K Vecinos Cercanos: {:.3f}".format(KNN))
print("Área bajo la curva ROC en Arboles de Decision: {:.3f}".format(Arbol))
print("Área bajo la curva ROC en Bosques Aleatorios: {:.3f}".format(Random))
print("Área bajo la curva ROC en ADA Boosting: {:.3f}".format(ADA))
print("Área bajo la curva ROC en XG Boosting: {:.3f}".format(XG))
print("Área bajo la curva ROC en Metodo Bayes: {:.3f}".format(Bayes))
print("Área bajo la curva ROC en Metodo de Discriminante Lineal: {:.3f}".format(LDA))
print("Área bajo la curva ROC en Metodo de Discriminante Cuadratico: {:.3f}".format(QDA))
print("Área bajo la curva ROC en Redes Neuronales MPL: {:.3f}".format(MPL))


# In[30]:


Keras = roc_auc_score(y_test_class, y_pred_class)
print("Área bajo la curva ROC en Redes Neuronales Paquete Keras: {:.3f}".format(Keras))


# #### ¿Se puede determinar con claridad cual metodo es el mejor?

# #### Analisis 
# 
# * A nivel de todos los metodos utilizados, se puede ver que el metodo que da una prediccion de un 99.8% es el de Bosques Aleatorios seguido por ADA Boosting y por el XG Boosting con un acierto de un 99.7% de prediccion respectivamente. 
# 
# * Discriminante Lineal junto con Arboles de Decision y Redes Neuronales con el Paquete Keras tambien dan buenos resultados. 
# 
# * Los metodos que dan los peores resultados son el de Discriminante Cuadratico con un 50% y se debe mas que todo, a que al momento de hacer el calculo este metodo tiene que invertir una serie de matrices y por lo visto, no logro del todo invertir esas matrices de ahi que de varios warnings y una prediccion bastante deficiente y el de Redes Neuronales con MPL dando un total de 46.5% (bastante malo). 

# #### 3. ¿Que se puede concluir?

# #### Analisis:
# 
# * Se puede concluir que el metodo que da mejores resultados es el de Bosques Aleatorios con un 99.8% seguido del XG Boosting y ADA Boosting. 
# 
# * Los metodos que dan los peores resultados son el Discriminante Cuadratico con un 50% de prediccion y se debe mas que todo a que el metodo no pudo invertir las matrices asociadas a sus calculos y el de Redes Neuronales con MPL, que da un 46.5% de prediccion. 

# ## Pregunta 2: 

# #### Esta pregunta utiliza los datos sobre muerte del corazon en Sudafrica (SAheart.csv). La variable que queremos predecir es chd que es un indicador de muerte coronaria basado en algunas variables predictivas (factores de riesgo) como son el fumado, la obesidad, las bebidas alcoholicas, entre otras.

# #### 1. Usando Bosques Aleatorios para la tabla SAheart.csv con el 80 % de los datos para la tabla aprendizaje y un 20 % para la tabla testing determine la mejor Probabilidad de Corte, de forma tal que se prediga de la mejor manera posible la categoria Si de la variable chd, pero sin desmejorar de manera significativa la precision global.

# In[5]:


corazon = pd.read_csv("SAheart.csv", delimiter = ';', decimal = '.')
print(corazon)


# In[6]:


corazon.info()


# #### Convirtiendo la variable de Object a Categorica

# In[7]:


# Convierte las variables de object a categórica
corazon['famhist'] = corazon['famhist'].astype('category')

# Recodifica las categorías usando números
corazon["famhist"] = corazon["famhist"].cat.codes

# Convierte las variables de entero a categórica
corazon['famhist'] = corazon['famhist'].astype('category')


# In[8]:


corazon.head()
# Variable convertida con exito


# In[9]:


corazon.info()


# #### Distribucion de la variable a predecir

# In[10]:


distribucion_variable_predecir(corazon,"chd")
# La variable a predecir esta desbalanceada y hay mas observaciones en el "NO"


# #### Los datos se encuentran en diferentes escalas pero como los analisis que se van a hacer estan basados en arboles es mejor no normalizar ni centrar los datos.

# #### Separacion de las variables predictoras en X

# In[11]:


X = corazon.iloc[:,:9] 
X


# #### Separacion de la variable a predecir en y

# In[12]:


y = corazon.iloc[:,9:10] 
y


# #### Realizando split de los datos con un 80% de training y un 20% de testing

# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.80)


# #### Usando la Probabilidad de Corte = 0.5 (para comparar)

# In[53]:


# Construímos el modelo
instancia_bosque = RandomForestClassifier(n_estimators=120, criterion = 'gini', max_features = 3)
instancia_bosque.fit(X_train,y_train.iloc[:,0].values)
print("Probabilidad del No y del Si:\n",instancia_bosque.predict_proba(X_test))
probabilidad = instancia_bosque.predict_proba(X_test)[:, 1]
print("Probabilidad de Si (o sea del 1):\n",probabilidad)

# Se aplica una Regla de Decisión
corte = 0.5
prediccion = np.where(probabilidad > corte, "Si", "No")
print("Predicción de la Regla\n",prediccion)

# Calidad de la predicción 
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### ----> Con esto se ve que se requiere una probabilidad mas chica que 0.5, ya que una mayor que 0.5 aumentaria la incapacidad para poder predecir el "Si".

# #### Se usa un "for" para encontrar la mejor Probabilidad de Corte que mejore el "Si" y no baje tanto la precision global

# In[74]:


# Construímos el modelo
instancia_bosque = RandomForestClassifier(n_estimators=120, criterion = 'gini', max_features = 3)
instancia_bosque.fit(X_train,y_train.iloc[:,0].values)
probabilidad = instancia_bosque.predict_proba(X_test)[:, 1]

# Aplicamos una Regla de Decisión
corte = [0.401, 0.402, 0.403, 0.404, 0.405, 0.406, 0.407, 0.408, 0.409]
for c in corte:
    print("===========================")
    print("Probabilidad de Corte: ",c)
    prediccion = np.where(probabilidad > c, "Si", "No")
    # Calidad de la predicción 
    MC = confusion_matrix(y_test, prediccion)
    indices = indices_general(MC,list(np.unique(y)))
    for k in indices:
        print("\n%s:\n%s"%(k,str(indices[k])))


# #### Analisis:
# 
# Una buena Probabilidad de Corte podria ser cualquiera entre el rango de [0.401 a 0.409] ya que en estas se obtiene precision del "Si" de 0.636364 y del "No" de un 0.6 con una Precisión Global de: 61.29%. Se obtuvo mejores predicciones para el "Si" usando probabilidades de corte de 0.35, sin embargo la Precisión Global ya era de 50% con lo que ya era bastante mala y no satisfizo lo que prononia el enunciado. 

# #### 2. Repita el ejercicio anterior usando XGBoosting. ¿Cambio la probabilidad de corte? Explique.

# #### Utilizando la probabilidad de Corte de 0.5 para comparar

# In[67]:


# Construímos el modelo
instancia_potencia = GradientBoostingClassifier(n_estimators=120)
instancia_potencia.fit(X_train,y_train.iloc[:,0].values)
print("Probabilidad del No y del Si:\n",instancia_potencia.predict_proba(X_test))
probabilidad = instancia_potencia.predict_proba(X_test)[:, 1]
print("Probabilidad de Si (o sea del 1):\n",probabilidad)

# Se aplica una Regla de Decisión
corte = 0.5
prediccion = np.where(probabilidad > corte, "Si", "No")
print("Predicción de la Regla\n",prediccion)

# Calidad de la predicción 
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))


# #### ---->  Con esto se ve que se requiere una probabilidad mas chica que 0.5, ya que una mayor que 0.5 aumentaria la incapacidad para poder predecir el "Si".

# In[72]:


# Construímos el modelo
instancia_bosque = RandomForestClassifier(n_estimators=120, criterion = 'gini', max_features = 3)
instancia_bosque.fit(X_train,y_train.iloc[:,0].values)
probabilidad = instancia_bosque.predict_proba(X_test)[:, 1]

# Aplicamos una Regla de Decisión
corte = [0.351, 0.352, 0.353, 0.354, 0.355, 0.356, 0.357, 0.358, 0.359]
for c in corte:
    print("===========================")
    print("Probabilidad de Corte: ",c)
    prediccion = np.where(probabilidad > c, "Si", "No")
    # Calidad de la predicción 
    MC = confusion_matrix(y_test, prediccion)
    indices = indices_general(MC,list(np.unique(y)))
    for k in indices:
        print("\n%s:\n%s"%(k,str(indices[k])))


# #### Analisis:
# 
# Una buena Probabilidad de Corte podria ser la de 0.359, ya que en esta se obtiene una precision del "Si" de 0.727273 y del "No" de 0.583333 con una Precisión Global de: 63.44% (incluso mejor que la de Bosques Aleatorios). Para este caso la probabilidad de corte cambio al utilizar un variante, pero mas bien la variable a predecir (la categoria del "Si") dio mejores resultados y se gano un poco de precision global (cerca de 2% de precision global). 

# ### Pregunta 3: 

# #### Dada la siguiente tabla:

# In[75]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/marc.png")


# #### 1. Usando la definicion de curva ROC calcule y grafique “a mano” la curva ROC, use un umbral T = 0,1 y un paso de 0,1. Es decir, debe hacerlo variando el umbral y calculando las matrices de confusion.

# In[26]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/analisis.jpg")


# In[27]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/curva.jpg")


# #### 2. Verifique el resultado anterior usando el codigo visto en clase

# In[13]:


# Paquetes Necesarios
import numpy as np
import pandas as pd
import random as rd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


Clase = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1])
Score = np.array([0.3, 0.25, 0.8, 0.7, 0.65, 0.35, 0.6, 0.1, 0.5, 0.2])

# Se grafica curva ROC usando roc_curve de sklearn
fp_r, tp_r, umbral = roc_curve(Clase, Score)
plt.figure(figsize=(10,10))
plt.plot(fp_r, tp_r, lw = 1, color = "red")
plt.plot([0, 1], [0, 1], lw = 1, color = "black")
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC")


# Graficamos puntos con el siguiente algoritmo
i = 1  # Contador
FP_r = -1  # Para que entre al condicional en la primera iteración
TP_r = -1  # Para que entre al condicional en la primera iteración

# linspace genera una sucesión de 100 números del 1 al 0 que equivale a una sucesión del 1 al 0 con paso de 0.1
for Umbral in np.linspace(1, 0, 201):   
    Prediccion = np.where(Score >= Umbral, 1, 0)
    MC = confusion_matrix(Clase, Prediccion)   
    if (FP_r != MC[0, 1] / sum(MC[0, ])) | (TP_r != MC[1, 1] / sum(MC[1, ])):     
            FP_r = MC[0, 1] / sum(MC[0, ])  # Tasa de Falsos Positivos
            TP_r = MC[1, 1] / sum(MC[1, ])  # Tasa de Verdaderos Positivos           
            # Graficamos punto
            plt.plot(FP_r, TP_r, "o", mfc = "none", color = "blue")
            plt.annotate(round(Umbral, 3), (FP_r + 0.01, TP_r - 0.02))     
            # Imprimimos resultado
            print("=====================")
            print("Punto i = ", i, "\n")  
            print("Umbral = T = ", round(Umbral, 3), "\n")
            print("MC =")
            print(MC, "\n")
            print("Tasa FP = ", round(FP_r, 2), "\n")
            print("Tasa TP = ", round(TP_r, 2))     
            i = i + 1


# #### 3. Usando el algoritmo eficiente para la curva ROC calcule y grafique “a mano” la curva ROC, use un umbral T = 0,1 y un paso de 0,1.

# In[28]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/1.jpg")


# In[29]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/2.jpg")


# In[31]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/3.jpg")


# In[32]:


from IPython.display import Image
Image(filename="/Users/heinerleivagmail.com/4.jpg")


# #### 4. Verifique el resultado anterior usando el codigo visto en clase para el algoritmo eficiente.

# In[25]:


Clase = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 1])
Score = np.array([.8, .7, .65, .60, .50, .35, .3, .25, .2, .1])

fp_r, tp_r, umbral = roc_curve(Clase, Score)
plt.figure(figsize=(10,10))
plt.plot(fp_r, tp_r, lw = 1, color = "red")
plt.plot([0, 1], [0, 1], lw = 1, color = "black")
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC")

# Aquí se inicializan para que de igual a la corrida a pie
Umbral = 0.1
Paso = 0.1
N = 4 # ceros
P = 6 # unos
TP = 0 
FP = 0
for i in range(0, 10):    
    if Score[i] > Umbral:
        if Clase[i] == 1:
            TP = TP + 1
        else:
            FP = FP + 1
    else:
        if Clase[i] == 0:
            FP = FP + 1
        else:
            TP = TP + 1
            
    # Graficamos punto
    plt.plot(FP / N, TP / P, "o", mfc = "none", color = "blue")
    plt.annotate(round(Umbral, 2), (FP / N + 0.01, TP / P - 0.02))
        
    # Imprimimos resultado
    print("========================")
    print("Punto i = ", i + 1, "\n")  
    print("Umbral = T = ", round(Umbral, 2), "\n")
    print("Tasa FP = ", round(FP / N, 2), "\n")
    print("Tasa TP = ", round(TP / P, 2))  
    Umbral = Umbral + Paso


# In[33]:


################################################# FIN #################################################

