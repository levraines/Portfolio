#!/usr/bin/env python
# coding: utf-8

# # Tarea #3 

# # Estudiante: Heiner Romero Leiva

# ## Importando bibliotecas

# In[1]:


import os
import pandas as pd
import numpy as np 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


# ## Pregunta 1:

# #### En esta pregunta utiliza los datos (tumores.csv). Se trata de un conjunto de datos de caracteristicas del tumor cerebral que incluye cinco variables de primer orden y ocho de textura y cuatro par´ametros de evaluacion de la calidad con el nivel objetivo. La variables son: Media, Varianza, Desviacion estandar, Asimetria, Kurtosis, Contraste, Energia, ASM (segundo momento angular), Entropia, Homogeneidad, Disimilitud, Correlacion, Grosor, PSNR (Pico de la relacion senal-ruido), SSIM (Indice de Similitud Estructurada), MSE (Mean Square Error), DC (Coeficiente de Dados) y la variable a predecir tipo (1 = Tumor, 0 = No-Tumor).

# #### 1. Cargue la tabla de datos tumores.csv en Python

# In[2]:


tumores = pd.read_csv("tumores.csv", delimiter = ',', decimal = '.')
tumores.head()


# In[3]:


tumores.info()


# In[4]:


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


# In[5]:


tumores.tail() #variable categorica ha sido convertida a numero 


# In[10]:


# Normalizando y centrando la tabla ya que hay valores en diferentes escalas y al ser un metodo basado en distancias es preferible
# centrar y reducir
tumores_1 = tumores.iloc[:,0:17]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values = scaler.fit_transform(tumores_1) 
tumores_1.loc[:,:] = scaled_values
tumores_1.head()


# #### Elimina la variable catégorica, deja las variables predictoras en X

# In[11]:


X = tumores_1.iloc[:,0:17] 
X.head()


# #### Deja la variable a predecir en y

# In[12]:


y = tumores.iloc[:,17:18] 
y.head()


# #### 2. El objetivo de este ejercicio es analizar la variacion del error (usando el enfoque trainingtesting) para la prediccion de variable tipo (que indica 1 = Tumor, 0 = No-Tumor), para esto repita 5 veces el calculo de error global de prediccion usando el metodo de los k vecinos mas cercanos (use kmax=50) y con un 75 % de los datos para tabla aprendizaje y un 25 % para la tabla testing. Grafique los resultados.

# #### Enfoque Training-Testing para medir el error

# In[16]:


error_tt = []

for i in range(0, 4):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75)
    
    knn = KNeighborsClassifier(n_neighbors = 50)
    knn.fit(X_train, y_train.values.ravel())
        
    error_tt.append(1 - knn.score(X_test, y_test))
  
plt.figure(figsize=(15,10))
plt.plot(error_tt, 'o-', lw = 2)
plt.xlabel("Número de Iteración", fontsize = 15)
plt.ylabel("Error Cometido %", fontsize = 15)
plt.title("Variación del Error", fontsize = 20)
plt.grid(True)
plt.legend(['Training Testing'], loc = 'upper right', fontsize = 15)


# #### 3. El objetivo de este ejercicio es medir el error para la prediccion de variable tipo, utilizando validacion cruzada con K grupos (K−fold cross-validation). Para esto usando el metodo de los k vecinos mas cercanos (use kmax=50) realice una validacion cruzada 5 veces con 10 grupos (folds) y grafique el error obtenido en cada iteracion, agregue en este grafico los 5 errores generados en el ejercicio anterior.

# #### Enfoque: Validación Cruzada usando K grupos (K-Fold Cross-Validation, CV)

# In[17]:


error_cv = []

for i in range(0, 4):
    kfold = KFold(n_splits = 10, shuffle = True)
    error_folds = []

    for train, test in kfold.split(X, y):
        knn = KNeighborsClassifier(n_neighbors = 50)
        knn.fit(X.iloc[train], y.iloc[train].values.ravel())
        error_folds.append((1 - knn.score(X.iloc[test], y.iloc[test])))
        
    error_cv.append(np.mean(error_folds))

plt.figure(figsize=(15,10))
plt.plot(error_tt, 'o-', lw = 2)
plt.plot(error_cv, 'o-', lw = 2)
plt.xlabel("Número de Iteración", fontsize = 15)
plt.ylabel("Error Cometido", fontsize = 15)
plt.title("Variación del Error", fontsize = 20)
plt.grid(True)
plt.legend(['Training Testing',  'K-Fold CV'], loc = 'upper right', fontsize = 15)


# #### 4. ¿Que se puede concluir?

# #### Conclusion: 
# 
# * Haciendo la graficacion respectiva usando el primer enfoque de training - testing se puede ver que el error que genera brinca mucho, por lo cual no es un metodo para medir el error fiable, ya que no es constante y va en el rango de 5.65% hasta casi 9.5% con los 5 repeticiones (calculo del error) que se le hizo. Con el K-Fold Cross Validation se ve como el error es constante y esta apenas mas arriba de 7.5% y para las 5 repeticiones que se hizo en el calculo, el error se mantiene constante, creando asi confianza del error obtenido con la ultima forma de validacion. 

# ## Pregunta 2:

# #### Para esta pregunta tambien usaremos los datos tumores.csv.

# #### 1. El objetivo de este ejercicio es calibrar el metodo de RandomForestClassifier para esta Tabla de Datos. Aqui interesa predecir en la variable tipo. Para esto genere Validaciones Cruzadas con 10 grupos calibrando el modelo de acuerdo con los dos tipos de criterios que este permite para medir la calidad de cada division en los arboles, es decir, con gini y entropy. Para esto utilice KFold de sklearn y realice un grafico comparativo de barras.

# In[21]:


cadena = "=== Importando datos de nuevo ya que este metodo se basa en arboles y es mejor no centrar y reducir la tabla ==="
print(cadena.center(140, " "))


# In[28]:


tumores = pd.read_csv("tumores.csv", delimiter = ',', decimal = '.')
# Convierte las variables de object a categórica
tumores['imagen'] = tumores['imagen'].astype('category')
# Recodifica las categorías usando números
tumores["imagen"] = tumores["imagen"].cat.codes
# Convierte las variables de entero a categórica
tumores['imagen'] = tumores['imagen'].astype('category')
tumores.tail() #variable categorica ha sido convertida a numero 


# #### Elimina la variable catégorica, deja las variables predictoras en X

# In[29]:


X = tumores.iloc[:,0:17] 


# #### Dejar la variable a predecir en y

# In[30]:


y = tumores.iloc[:,17:18] 


# #### Random Forest

# #### ------> Primero con el criterio "Gini"

# In[46]:


from sklearn.ensemble import RandomForestClassifier

instancia_kfold = KFold(n_splits=10)
porcentajes = cross_val_score(RandomForestClassifier(n_estimators=10, criterion = 'gini'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_bosques_gini = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### -------> Segundo con el criterio "Entropy"

# In[47]:


from sklearn.ensemble import RandomForestClassifier

instancia_kfold = KFold(n_splits=10)
porcentajes = cross_val_score(RandomForestClassifier(n_estimators=10, criterion = 'entropy'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_bosques_entropy = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Grafico Comparativo

# In[48]:


plt.figure(figsize=(8,5))
alto = [res_bosques_gini, res_bosques_entropy]
barras = ('Gini','Entropia')
y_pos = np.arange(len(barras))
plt.bar(y_pos, alto, color=['purple', 'orange'])
plt.xticks(y_pos, barras)
plt.show()


# #### 2. ¿Cual algoritmo usaria con base en la informacion obtenida en los dos ejercicios anteriores?

# #### Analisis
# 
# * Con base en la informacion obtenida se puede usar cualesquiera de los dos algoritmos, ya que "entropy" asi como "gini", dan promedios de deteccion de 99% cada uno, lo que los hace iguales para este dataset en especifico. 

# ## Pregunta 3: 

# ####  Para esta pregunta tambien usaremos los datos tumores.csv.

# #### 1. El objetivo de este ejercicio es calibrar el metodo de KNeighborsClassifier para esta Tabla de Datos. Aqui interesa predecir en la variable tipo. Para esto genere Validaciones Cruzadas con 5 grupos calibrando el modelo de acuerdo con todos los tipos de algoritmos que permite auto, ball tree, kd tree y brute en el parametro algorithm. Realice un grafico de barras comparativo. ¿Se puede determinar con claridad cual algoritmo es el mejor? Utilice KFold de sklearn?

# In[49]:


cadena = "=== Centrando y reduciendo la tabla ya que este metodo se basa en distancias ==="
print(cadena.center(140, " "))


# In[50]:


# Normalizando y centrando la tabla ya que hay valores en diferentes escalas y al ser un metodo basado en distancias es preferible
# centrar y reducir
tumores_1 = tumores.iloc[:,0:17]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values = scaler.fit_transform(tumores_1) 
tumores_1.loc[:,:] = scaled_values
tumores_1.head()


# #### Definicion de X y y

# In[51]:


X = tumores_1.iloc[:,0:17]
y = tumores.iloc[:,17:18] 


# #### Metodo KNN

# #### -----> Primero con el algoritmo "auto"

# In[52]:


from sklearn.neighbors import KNeighborsClassifier

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(KNeighborsClassifier(n_neighbors=80, algorithm = 'auto'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_knn_auto = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### -----> Segundo con el algoritmo "ball_tree"

# In[53]:


from sklearn.neighbors import KNeighborsClassifier

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(KNeighborsClassifier(n_neighbors=80, algorithm = 'ball_tree'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_knn_ball = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### -----> Tercero con el algoritmo "kd_tree"

# In[54]:


from sklearn.neighbors import KNeighborsClassifier

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(KNeighborsClassifier(n_neighbors=80, algorithm = 'kd_tree'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_knn_kd = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### -----> Cuarto con el algoritmo "Brute"

# In[55]:


from sklearn.neighbors import KNeighborsClassifier

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(KNeighborsClassifier(n_neighbors=80, algorithm = 'brute'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_knn_brute= porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Grafico Comparativo

# In[58]:


plt.figure(figsize=(8,5))
alto = [res_knn_auto,res_knn_ball, res_knn_kd, res_knn_brute]
barras = ('Auto','Ball Tree','Kd tree','Brute')
y_pos = np.arange(len(barras))
plt.bar(y_pos, alto, color=['pink', 'yellow', 'purple', 'cyan'])
plt.xticks(y_pos, barras)
plt.show()


# #### Analisis
# 
# * No se puede determinar cual algoritmo es mejor, porque cuando se utilizan estos algortimos y se ha corrido el primero, es muy usual que los siguientes den exactamente igual (esto es muy usual en KNN), asi que no se puede tener claridad en cual escoger. 

# #### 2. ¿Cual algoritmo usaria con base en la informacion obtenida en los dos ejercicios anteriores?

# #### Analisis: 
# 
# * Como en KNN es usual que una vez habiendo corrido el modelo y despues cambiando los algoritmos estos den igual, yo usuaria cualquiera (para este caso espeficio) pero si de verdad se quiere ver cual seria mejor se tendria que trabajar cada uno por aislado para ver con cual se obtiene un mejor poder predictivo. 

# ## Pregunta 4: 

# #### Para esta pregunta tambien usaremos los datos tumores.csv

# #### 1. El objetivo de este ejercicio es comparar todos los metodos predictivos vistos en el curso con esta tabla de datos. Aqui interesa predecir en la variable tipo. Para esto genere Validaciones Cruzadas con 5 grupos para los metodos SVM, KNN, Arboles, Bosques, ADA Boosting, eXtreme Gradient Boosting, Bayes, LDA, QDA y Redes Neuronales del paquete MLPClassifier. Para KNN y Bosques use los parametros obtenidos en las calibraciones realizadas en los ejercicios anteriores (en teoria se deberian calibrar todos los metodos). Luego realice un grafico de barras para comparar los metodos. ¿Se puede determinar con claridad cual metodos es el mejor? Utilice KFold de sklearn?

# #### Metodo Arboles con criterio "Gini"

# In[63]:


from sklearn.tree import DecisionTreeClassifier

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(DecisionTreeClassifier(criterion = 'gini'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_arbol_gini = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Metodos Arbole con criterio "Entropy"

# In[64]:


from sklearn.tree import DecisionTreeClassifier

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(DecisionTreeClassifier(criterion = 'entropy'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_arbol_entropy = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Potenciación (ADA Boosting) Algoritmo "SAMME-R"

# In[65]:


from sklearn.ensemble import AdaBoostClassifier

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(AdaBoostClassifier(algorithm = "SAMME.R", n_estimators=10), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_potenciacion_sammer = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Potenciación (ADA Boosting) Algoritmo "SAMME"

# In[66]:


from sklearn.ensemble import AdaBoostClassifier

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(AdaBoostClassifier(algorithm = "SAMME", n_estimators=10), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_potenciacion_samme = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Potenciación Extrema (XGBoosting) criterio "friedman_mse"

# In[70]:


from sklearn.ensemble import GradientBoostingClassifier

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(GradientBoostingClassifier(criterion = 'friedman_mse', n_estimators=10), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_xg_potenciacion_friedman = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Potenciación Extrema (XGBoosting) criterio "mse"

# In[71]:


from sklearn.ensemble import GradientBoostingClassifier

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(GradientBoostingClassifier(criterion = 'mse', n_estimators=10), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_xg_potenciacion_mse = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Potenciación Extrema (XGBoosting) criterio "mae"

# In[73]:


from sklearn.ensemble import GradientBoostingClassifier

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(GradientBoostingClassifier(criterion = 'mae', n_estimators=10), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_xg_potenciacion_mae = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Maquinas de Soporte Vectorial, kernel "Sigmoid"

# In[74]:


from sklearn.svm import SVC

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(SVC(kernel='sigmoid', gamma = 'scale'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_svm_sigmoid= porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Maquinas de Soporte Vectorial, kernel "rbf"

# In[75]:


from sklearn.svm import SVC

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(SVC(kernel='rbf', gamma = 'scale'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_svm_rbf= porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Maquinas de Soporte Vectorial, kernel "Poly"

# In[77]:


from sklearn.svm import SVC

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(SVC(kernel='poly', gamma = 'scale'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_svm_poly = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Maquinas de Soporte Vectorial, kernel "Linear" (se usa max_iter=250000 para que no dure mucho).

# In[79]:


from sklearn.svm import SVC

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(SVC(kernel='linear', gamma = 'scale',max_iter=250000), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_svm_linear = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Redes Neuronales - MLPClassifier, Activation = "Identity"

# In[80]:


from sklearn.neural_network import MLPClassifier

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(MLPClassifier(activation = 'identity', solver='lbfgs'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_redes_MLP_iden = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Redes Neuronales - MLPClassifier, Activation = "Logistic"

# In[81]:


from sklearn.neural_network import MLPClassifier

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(MLPClassifier(activation = 'logistic', solver='lbfgs'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_redes_MLP_logis = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Redes Neuronales - MLPClassifier, Activation = "Tahn"

# In[82]:


from sklearn.neural_network import MLPClassifier

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(MLPClassifier(activation = 'tanh', solver='lbfgs'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_redes_MLP_tahn = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Redes Neuronales - MLPClassifier, Activation = "relu"

# In[83]:


from sklearn.neural_network import MLPClassifier

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(MLPClassifier(activation = 'relu', solver='lbfgs'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_redes_MLP_relu = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Método Ingenuo de Bayes

# In[84]:


from sklearn.naive_bayes import GaussianNB

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(GaussianNB(), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_bayes = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Análisis Discriminte Lineal solver = "Eigen"

# In[91]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(LinearDiscriminantAnalysis(solver = 'eigen', shrinkage = 'auto'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_dis_lineal_eigen = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Análisis Discriminte Lineal solver = "lsqr"

# In[92]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage = 'auto'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_dis_lineal_lsqr = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Análisis Discriminte Cuadrático

# In[97]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(QuadraticDiscriminantAnalysis(), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_dis_cuadratico = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


# #### Gráfico Comparativo

# In[137]:


plt.figure(figsize=(38,20))
alto = [res_bosques_gini, res_bosques_entropy , res_knn_auto, res_knn_ball , res_knn_kd, res_knn_brute, res_arbol_gini , res_arbol_entropy, res_potenciacion_sammer , res_potenciacion_samme, res_xg_potenciacion_friedman, res_xg_potenciacion_mse, res_xg_potenciacion_mae, res_svm_sigmoid, res_svm_rbf, res_svm_poly, res_svm_linear, res_redes_MLP_iden , res_redes_MLP_logis, res_redes_MLP_tahn, res_redes_MLP_relu, res_bayes, res_dis_lineal_eigen, res_dis_lineal_lsqr, res_dis_cuadratico]
barras = ('RF Gini', 'RF Entro', 'KNN auto', 'KNN ball', 'KNN kd', 'KNN brute', 'Arbol Gini', 'Arbol Entro', 'ADA Samme R', 'ADA Samme', 'XG Friedman', 'XG mse' , 'XG Mae', 'SVM Sigmo', 'SVM RBF', 'SVM Poly', 'SVM linear', 'Redes Iden','Redes Logis', 'Redes Tanh', 'Redes Relu', 'Bayes', 'Dis Lin Eigen', 'Dis Lin lsqr', 'Dis cuadra')
y_pos = np.arange(len(barras))
plt.bar(y_pos, alto,color = ["#67E568","#257F27","#08420D","#FFF000","#FFB62B","#E56124","#E53E30","#7F2353","#F911FF","#9F8CA6",'aqua', 'navy', 'plum', 'pink', 'skyblue', 'purple', 'indigo', 'blueviolet', 'crimson', 'coral', 'peru', 'cadetblue', 'gold', 'darkseagreen', 'greenyellow']
)
plt.xticks(y_pos, barras)
plt.show()


# #### Analisis
# 
# * Haciendo la calibracion con todos los metodos vistos en el curso, se puede ver que los que generan mejores resultados son: 
#     * Random Forest con criterio Gini y Entropia. 
#     * Redes Neuronales usando la activacion por identity. 
#     * Analisis Discriminante Cuadratico. 
#     * Y finalmente Bayes. 

# #### 2. ¿Se podra incluir en esta seleccion las Redes Neuronales del paquete Keras? Si la respuesta es que si entonces incluyalo.

# In[140]:


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# funcion para crear el modelo, requerido por KerasClassifier
def create_model():
  # crea el modelo
  model = Sequential()
  model.add(Dense(12, input_dim=17, activation='relu'))
  model.add(Dense(8, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  # Compila el modelo
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model
# Fija las semillas aleatorias para la reproducibilidad
seed = 7
numpy.random.seed(seed)
# carga los datos

dataset = pd.read_csv("tumores.csv", delimiter = ',', decimal = '.')
# Convierte las variables de object a categórica
dataset['imagen'] = dataset['imagen'].astype('category')
# Recodifica las categorías usando números
dataset["imagen"] = dataset["imagen"].cat.codes
# Convierte las variables de entero a categórica
dataset['imagen'] = dataset['imagen'].astype('category')

# split para la variables predictoras (X) y a predecir (y)
X = dataset.iloc[:,0:17]
Y = dataset.iloc[:,17:18]
# crea el modelo
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
# evalua usando 5 - fold validacion cruzada
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
# Asignando variable para grafico
res_keras = results.mean()


# #### Grafico Comparativo incluyendo Validacion Cruzada con Keras

# In[141]:


plt.figure(figsize=(38,20))
alto = [res_bosques_gini, res_bosques_entropy , res_knn_auto, res_knn_ball , res_knn_kd, res_knn_brute, res_arbol_gini , res_arbol_entropy, res_potenciacion_sammer , res_potenciacion_samme, res_xg_potenciacion_friedman, res_xg_potenciacion_mse, res_xg_potenciacion_mae, res_svm_sigmoid, res_svm_rbf, res_svm_poly, res_svm_linear, res_redes_MLP_iden , res_redes_MLP_logis, res_redes_MLP_tahn, res_redes_MLP_relu, res_bayes, res_dis_lineal_eigen, res_dis_lineal_lsqr, res_dis_cuadratico, res_keras]
barras = ('RF Gini', 'RF Entro', 'KNN auto', 'KNN ball', 'KNN kd', 'KNN brute', 'Arbol Gini', 'Arbol Entro', 'ADA Samme R', 'ADA Samme', 'XG Friedman', 'XG mse' , 'XG Mae', 'SVM Sigmo', 'SVM RBF', 'SVM Poly', 'SVM linear', 'Redes Iden','Redes Logis', 'Redes Tanh', 'Redes Relu', 'Bayes', 'Dis Lin Eigen', 'Dis Lin lsqr', 'Dis cuadra', 'Redes keras')
y_pos = np.arange(len(barras))
plt.bar(y_pos, alto,color = ["#67E568","#257F27","#08420D","#FFF000","#FFB62B","#E56124","#E53E30","#7F2353","#F911FF","#9F8CA6",'aqua', 'navy', 'plum', 'pink', 'skyblue', 'purple', 'indigo', 'blueviolet', 'crimson', 'coral', 'peru', 'cadetblue', 'gold', 'darkseagreen', 'greenyellow', 'teal']
)
plt.xticks(y_pos, barras)
plt.show()


# #### 3. ¿Cual metodo usaria con base en la informacion obtenida en los dos ejercicios anteriores?

# #### Analisis
# 
# Haciendo la calibracion con todos los metodos vistos en el curso, los metodos que podria utilizar serian: 
# 
# * Random Forest con criterio Gini y Entropia.
# * Redes Neuronales usando la activacion por identity.
# * Analisis Discriminante Cuadratico.
# * Y finalmente Bayes.
# 
# En todos los mencionados su promedio de deteccion es mayor a 98% haciendolos metodos bastante confiables. 

# In[142]:


############################################################## FIN ################################################################

