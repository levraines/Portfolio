#!/usr/bin/env python
# coding: utf-8

# # <<<<<<<<<<<<<<<<<<<< Tarea Número 4>>>>>>>>>>>>>>>>>>>>>>>>

# ## Estudiante: Heiner Romero Leiva 

# ## Ejercicio 1

# In[1]:


import os
import pandas as pd
import matplotlib.pyplot as plt
from   sklearn.decomposition import PCA
from   sklearn.datasets import make_blobs
from   sklearn.cluster import KMeans
import numpy as np
from   math import pi


# #### a) Cargue la tabla de datos SpotifyTop2018 40 V2

# In[2]:


# Cargando datos 

data = pd.read_csv("SpotifyTop2018_40_V2.csv", delimiter = ',', decimal = '.', index_col=0)
print(data)
print(data.head())


# In[3]:


# Normalizando y centrando la tabla 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values = scaler.fit_transform(data) 
data.loc[:,:] = scaled_values
print(data)
datos = data 


# #### b)  Ejecute el metodo k−medias para k = 3. Modificaremos los atributos de la clase KMeans(...) como sigue: max iter : int, default: 300: Numero maximo de iteraciones del algoritmo kmedias para una sola ejecucion. Para este ejercicio utilice max iter = 1000. n init : int, default: 10 (Formas Fuertes): Numero de veces que el algoritmo kmedias se ejecutara con diferentes semillas de centroides. Los resultados finales seran la mejor salida de n init ejecuciones consecutivas en terminos de inercia intra-clases. Para este ejercicio utilice n init = 100.

# In[4]:


# Función para graficar los gráficos de Barras para la interpretación de clústeres
def bar_plot(centros, labels, cluster = None, var = None):
    from math import ceil, floor
    from seaborn import color_palette
    colores = color_palette()
    minimo = floor(centros.min()) if floor(centros.min()) < 0 else 0
    def inside_plot(valores, labels, titulo):
        plt.barh(range(len(valores)), valores, 1/1.5, color = colores)
        plt.xlim(minimo, ceil(centros.max()))
        plt.title(titulo)
    if var is not None:
        centros = np.array([n[[x in var for x in labels]] for n in centros])
        colores = [colores[x % len(colores)] for x, i in enumerate(labels) if i in var]
        labels = labels[[x in var for x in labels]]
    if cluster is None:
        for i in range(centros.shape[0]):
            plt.subplot(1, centros.shape[0], i + 1)
            inside_plot(centros[i].tolist(), labels, ('Cluster ' + str(i)))
            plt.yticks(range(len(labels)), labels) if i == 0 else plt.yticks([]) 
    else:
        pos = 1
        for i in cluster:
            plt.subplot(1, len(cluster), pos)
            inside_plot(centros[i].tolist(), labels, ('Cluster ' + str(i)))
            plt.yticks(range(len(labels)), labels) if pos == 1 else plt.yticks([]) 
            pos += 1


# In[5]:


# Función para graficar los gráficos tipo Radar para la interpretación de clústeres
def radar_plot(centros, labels):
    from math import pi
    centros = np.array([((n - min(n)) / (max(n) - min(n)) * 100) if 
                        max(n) != min(n) else (n/n * 50) for n in centros.T])
    angulos = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angulos += angulos[:1]
    ax = plt.subplot(111, polar = True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angulos[:-1], labels)
    ax.set_rlabel_position(0)
    plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
           ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"], 
           color = "grey", size = 8)
    plt.ylim(-10, 100)
    for i in range(centros.shape[1]):
        valores = centros[:, i].tolist()
        valores += valores[:1]
        ax.plot(angulos, valores, linewidth = 1, linestyle = 'solid', 
                label = 'Cluster ' + str(i))
        ax.fill(angulos, valores, alpha = 0.3)
    plt.legend(loc='upper right', bbox_to_anchor = (0.1, 0.1))


# #### Formas fuertes (n_init) y Número de Iteraciones (max_iter) [Default]

# In[11]:


# Solo 3 iteraciones y una forma fuerte.
kmedias = KMeans(n_clusters=3, max_iter=300, n_init=10) # Declaración de la instancia
kmedias.fit(datos)
centros = np.array(kmedias.cluster_centers_)
print(centros)
plt.figure(1, figsize = (10, 10))
radar_plot(centros, datos.columns)


# #### Formas fuertes (n_init) y Número de Iteraciones (max_iter) [Modificado]

# In[12]:


# Volviendo a recargar datos para ver la asignacion final de los clusters 
kmedias = KMeans(n_clusters=3, max_iter=1000, n_init=100)
kmedias.fit(datos)
centros = np.array(kmedias.cluster_centers_)
print(centros)
plt.figure(1, figsize = (10, 10))
radar_plot(centros, datos.columns)


# #### c) Interprete los resultados del ejercicio anterior usando graficos de barras y graficos tipo Radar. Compare respecto a los resultados obtenidos en la tarea anterior en la que uso Clustering Jerarquico.

# In[14]:


# Ejecuta k-medias con 3 clusters
kmedias = KMeans(n_clusters=3)  
kmedias.fit(datos)
print(kmedias.predict(datos))
centros = np.array(kmedias.cluster_centers_)
print(centros) 


# In[15]:


# Ploteando grafico de barras
plt.figure(1, figsize = (12, 8))
bar_plot(centros, datos.columns)


# ### Interpretacion

# In[15]:


# En cuanto a la interpretacion se puede ver lo siguiente:

# Despues de haber corrido el K-medias con el primer set de parametros, donde init = 10 y max_iter = 300:
# se obtiene: 
# primer cluster de color azul, en el que spechiness, loudness, tempo y acoustiness son las variables mas altas, es decir
# las canciones clasificadas en un primer momento tienen registros de palabras en sus canciones, son acosticas y tienen 
# los mayores tiempos por minutos expresados en beats, ademas de time_signature y danceability, es decir, las canciones
# son bailables y hay altos volumenes de beats en cada barra, las demas variables son bajas. 
# Un segundo cluster naranja, tiene registros altos en cuanto a danceability, time signature, energy, loudness, valence
# e instrumentalness, es decir, estas canciones sosn buenas para bailar, hay altos beats por barra por minuto, tienen
# intensidades buenas, tienen alta sonoridad por pista, ademas de que son canciones bastante positivas asi como instrumen-
# tales, presentan cierto speechiness pero no mas de un 50% lo que quiere decir, es que hay moderada cantidad de palabras
# y la duracion en milisegundos de las canciones es muy baja, es decir, son canciones energeticas y buenas para bailar 
# pero duran poco. 
# En el cluster 3 (verde): se observa que son las canciones que tienen mayor duracion en milisegundos de todas, y 
# presentan cierta acustica, asi como sonoridad y cierta intensidad pero es baja, en las demas variables son bajas.


# Segunda interpretacion con init = 100 y max_iter = 1000

# En este punto, se ve como las iteraciones estabilizan los clusters y estos cambian en algunas representaciones de
# variables ya que se tiene:
#  Cluster 1 (azul):  se mantienen spechiness, time_signature, danceability, acoustiness, y se agregan liveness y valance
# lo que quiere decir que las canciones en este cluster se caracterizan por tener niveles altos de beats por cada barra 
# o medida, son canciones que registran altos registros de letras, son canciones bailables, son acusticas, y se detecta
# presencia de publica en ellas asi como alta positividad musical, es decir son canciones alegres y en la que la gente
# al escucharlas puede bailar y cantar, aunque por otro lado, son canciones cortas ya que presentan bajos registros de 
# duration_ms, es decir su duracion en milisegundo es poca, al igual que su intensidad y su deteccion de instrumentalidad.
# Cluster 2 (naranja): se caracteriza por tener las variables mas altas en time_signature, danceability, energy, loudness, 
# valence y liveness con respecto a los valores por default, no fue mucho el cambio que hubo y solo instrumentals fue el
# que se cambio, este cluster se caracteriza por tener canciones beats por barra de volumen muy altos, ser canciones
# aptas para bailar, poseen alta sonoridad en la pista medida en decibeles, son canciones que tienen alta presencia de
# positivismo en las letras y que presentan alta presencia de publico. En realidad este cluster es muy parecido al numero 1
# solo que presenta variables como energy y loudness que el cluster 1 no presenta, por otro lado en este cluster estan
# las canciones que registran baja presencia de palabras, acustica e instrumentalidad, y son canciones que tienen duraciones
# mayores en milisegundos que las del cluster 1, es decir, son aptas para bailar, son positivas pero quiza no son canciones
# aptas para cantar, porque registran indices bajos de esta variable. 
# Cluster 3 (verde): con respecto al primer cluster por default, en este nuevo cluster ahora se presenta la variable 
# de instrumentalidad, y otras como tempo y duration_ms siguen manteniendose, asi como ahora hay presencia moderada de 
# energy y loudness. En este cluster va a estar representado por todas aquellas canciones que tienen lo registros mas 
# altos de duracion por milisegundos, asi como las que poseen mayor instrumentalidad y tiempo aproximado por beats, asi
# como las que transmiten un relativo alto grado de positividad y presencia de publico pero bajos registros de intensidad
# y de sonoridad. Presenta bajos niveles de palabras en canciones y no son para nada bailables. 

# Comparacion con Clustering Jerarquico:

# Se puede ver como el cluster 1 (azul) es bastante similar, habiendo solo uno ligero cambio a nivel de duration_ms ya que
# en Clustering Jerarquico se ve como mas de un 25% de los datos presentaban algo de duration_ms (duracion en milisegundos)
# sin embargo es apenas notorio. 
# Con respecto al cluster 2 (naranja) hay muchis cambios, ya que en Jerarquico solo se tenian altas las variables de 
# duration_ms, tempo y un poco acoustiness, mientras que en k-medias estas mismas variables no se encuentra altas 
# y mas bien en k-medias estas estan bastante bajas y las que estaban bajas en Jerarquico aqui estan altas como es el caso
# de danceability, energy, etc.
# Con el cluster 3 (verde): las variables que se siguen manteniendo son intrsumentalness, tempo y un poco de loudness, aunque
# en Jerarquico instrumentalness estaba alta y en K-Medias esta en menos del 50% sin embargo este cluster sigue siendo
# caracterizado por canciones bastante instumentales y con beats por minuto bastante altos. 


# #### d) Grafique usando colores sobre las dos primeras componentes del plano principal en el Analisis en Componentes Principales los clusteres obtenidos segun k-medias (usando k =3).

# In[22]:


pca = PCA(n_components=2)
componentes = pca.fit_transform(datos)
componentes
print(datos.shape)
print(componentes.shape)
plt.scatter(componentes[:, 0], componentes[:, 1],c=kmedias.predict(datos))
plt.xlabel('componente 1')
plt.ylabel('componente 2')
plt.title('3 Cluster K-Medias')


# #### e) Usando 50 ejecuciones del metodo k−medias grafique el “Codo de Jambu” para este ejemplo. ¿Se estabiliza en algun momento la inercia inter–clases?

# In[7]:


# Solo 3 iteraciones y usando 50 ejecuciones con valores con defecto de max_iter = 300 e init = 50
kmedias = KMeans(n_clusters=3, max_iter=300, n_init=50) # Declaración de la instancia
kmedias.fit(datos)
centros = np.array(kmedias.cluster_centers_)
print(centros)


# In[10]:


Nc = range(1, 20)
kmediasList = [KMeans(n_clusters=i) for i in Nc]
varianza = [kmediasList[i].fit(datos).inertia_ for i in range(len(kmediasList))]
plt.plot(Nc,varianza,'o-')
plt.xlabel('Número de clústeres')
plt.ylabel('Varianza explicada por cada cluster (Inercia Intraclases)')
plt.title('Codo de Jambu')


# #### Interpretacion

# In[11]:


# En este caso no hay mucha claridad, ya que en realidad en ningun punto se ve que se estabilice y se forme la linea 
# recta, aunque tal vez se podria decir que en K = 5, K = 7 o K = 13 podrian ser opciones viables. 


# ## Ejercicio #2

# #### a) Repita el ejercicio 1 usando k = 3 usando esta tabla de datos, usando solo las variables numericas. Modificaremos los atributos de la clase KMeans (...) como sigue: max iter : int, default: 300: Numero maximo de iteraciones del algoritmo kmedias para una sola ejecucion. Para este ejercicio utilice max iter = 2000, n init : int, default: 10 (Formas Fuertes): Numero de veces que el algoritmo kmedias se ejecutara con diferentes semillas de centroides. Los resultados finales sera la mejor salida de n init ejecuciones consecutivas en terminos de inercia intra-clases. Para este ejercicio utilice n init = 150.

# #### Carga de la Tabla de Datos SAHeart

# In[43]:


corazon = pd.read_csv('SAheart.csv', delimiter = ';', decimal = '.')
print(corazon)


# In[44]:


# Seleccionando solo variables numericas

corazon2 = pd.DataFrame(data = corazon2, columns = (['sbp', 'tobacco', 'ldl', 'adiposity', 'typea', 'obesity', 
'alcohol', 'age']))
print(corazon2)
corazon2.head()


# In[45]:


# Normalizando y centrando la tabla 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values = scaler.fit_transform(corazon2) 
corazon2.loc[:,:] = scaled_values
print(corazon2)


# In[25]:


# Solo 3 iteraciones y valores modificables en max_iter = 2000 e init = 150
kmedias = KMeans(n_clusters=3, max_iter=2000, n_init=150) # Declaración de la instancia
kmedias.fit(datos)
centros = np.array(kmedias.cluster_centers_)
print(centros)
plt.figure(1, figsize = (10, 10))
radar_plot(centros, datos.columns)


# In[46]:


# Ejecuta k-medias con 3 clusters
kmedias = KMeans(n_clusters=3)  
kmedias.fit(corazon2)
print(kmedias.predict(corazon2))
centros = np.array(kmedias.cluster_centers_)
print(centros) 


# In[27]:


# Ploteando grafico de barras
plt.figure(1, figsize = (12, 8))
bar_plot(centros, datos.columns)


# #### Interprete los resultados del ejercicio anterior usando graficos de barras y graficos tipo Radar. Compare respecto a los resultados obtenidos en la tarea anterior en la que uso Clustering Jerarquico.

# In[41]:


# Comparando con el ejercicio pasado con Clustering Jerarquico se puede apreciar que en realidad el plot de radar con 
# K - Means es practicamente identico al plot de radar pasado, se puede observar como los clusters mantienen igual 
# casi todas sus variables, sin embargo el cambio mas grande que se tiene es en el numero de Cluster, ya que para el Jerarquico
# el Cluster 1, eran los individuos que tenian un alto typea A y las demas variables eran bastante bajas, en este caso 
# con el k-means este paso a ser el cluster 2. 
# El cluster 2 en el Jerarquico, representado por los indidvuos con un sbp alto, las edades mas altas, asi como presencia
# de alto colesterol, adiposidad y alto sobrepeso en el K - Means paso a ser el cluster 3 y ahora los individuos presentan
# mediciones mas altas de SBP y de adiposidad (llegando a lo mas alto) comparadi con el pasado.
# Finalmente el cluster 3 en el Jerarquico, ahora pasa a ser el cluster 1 en el K - medias y sigue teniendo las mismas variables
# originales, como alto colesterol, adiposidad, obesidad, relativamente alta presencia de mediciones alta de SBP y edad, 
# pero ahora el K - medias incluyo a la variable typea A alta, y no en un estado medio como el clustering Jerarquico, haciendo
# que los individuos de este cluster sean los que presentan altas edades y enfermedades como obesidad, alto colesterol y 
# adiposidad, pero ahora sumado con mayor medida un factor de tipo A asociado a personas mas competitivas y orientada a 
# resultados pero que pasan mas estresadas y ansiosas. 


# #### Grafique usando colores sobre las dos primeras componentes del plano principal en el Analisis en Componentes Principales los clusteres obtenidos segun k-medias (usando k =3).

# In[47]:


pca = PCA(n_components=2)
componentes = pca.fit_transform(corazon2)
componentes
print(corazon2.shape)
print(componentes.shape)
plt.scatter(componentes[:, 0], componentes[:, 1],c=kmedias.predict(corazon2))
plt.xlabel('componente 1')
plt.ylabel('componente 2')
plt.title('3 Cluster K-Medias')


# #### Usando 50 ejecuciones del metodo k−medias grafique el “Codo de Jambu” para este ejemplo. ¿Se estabiliza en algun momento la inercia inter–clases?

# In[48]:


# Solo 3 iteraciones y usando 50 ejecuciones con valores con defecto de max_iter = 300 e init = 50
kmedias = KMeans(n_clusters=3, max_iter=300, n_init=50) # Declaración de la instancia
kmedias.fit(corazon2)
centros = np.array(kmedias.cluster_centers_)
print(centros)


# In[49]:


Nc = range(1, 20)
kmediasList = [KMeans(n_clusters=i) for i in Nc]
varianza = [kmediasList[i].fit(corazon2).inertia_ for i in range(len(kmediasList))]
plt.plot(Nc,varianza,'o-')
plt.xlabel('Número de clústeres')
plt.ylabel('Varianza explicada por cada cluster (Inercia Intraclases)')
plt.title('Codo de Jambu')


# ### Interpretacion

# In[ ]:


# En este caso no hay mucha claridad, pero se podria decir que en K = 2 o K = 6 podrian ser opciones viables. 


# #### b)Repita los ejercicios anteriores pero esta vez incluya las variables categoricas usando codigos disyuntivos completos. ¿Son mejores los resultados?

# In[28]:


# Recodificacion
def recodificar(col, nuevo_codigo):
  col_cod = pd.Series(col, copy=True)
  for llave, valor in nuevo_codigo.items():
    col_cod.replace(llave, valor, inplace=True)
  return col_cod


# #### Cargando las variables numericas asi como categoricas y convirtiendolas a codigo disyuntivo completo

# In[54]:


# Conviertiendo la variables en Dummy
datos_dummies = pd.get_dummies(corazon)
print(datos_dummies.head())
print(datos_dummies.dtypes)


# In[57]:


# Centrando y normalizando los datos convertidos en dummies 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values = scaler.fit_transform(datos_dummies) 
datos_dummies.loc[:,:] = scaled_values
print(datos_dummies) 
dummy = datos_dummies


# In[63]:


# Solo 3 iteraciones y valores modificables en max_iter = 2000 e init = 150
kmedias = KMeans(n_clusters=3, max_iter=2000, n_init=150) # Declaración de la instancia
kmedias.fit(dummy)
centros = np.array(kmedias.cluster_centers_)
print(centros)
plt.figure(1, figsize = (10, 10))
radar_plot(centros, dummy.columns)


# In[33]:


# Ejecuta k-medias con 3 clusters
kmedias = KMeans(n_clusters=3)  
kmedias.fit(datos_dummy)
print(kmedias.predict(datos_dummy))
centros = np.array(kmedias.cluster_centers_)
print(centros) 


# In[34]:


# Ploteando grafico de barras
plt.figure(1, figsize = (12, 8))
bar_plot(centros, datos_dummy.columns)


# #### Interprete los resultados del ejercicio anterior usando graficos de barras y graficos tipo Radar. Compare respecto a los resultados obtenidos en la tarea anterior en la que uso Clustering Jerarquico.

# In[51]:


# En este caso se ve que de nuevo que los clusters junto con su asignacion de variables en cada uno comparado con el
# Jerarquico es similar, sin embargo paso el mismo problema de que se cambiaron los numeros de los cluster, por ejemplo
# el cluster 1 en el k - medias es el cluster 3 en el Jerarquico, se siguen manteniendo altas las variables de chd_no, 
# no hay historial familiar, y el comportamiento tipo A, en K - medias es alto, mientras que en el jerarquico era medio
# otra diferencia es que en el jerarquico los individuos tenian altos indices de toma de alcohol mientras que en el k 
# Medias estos sujetos presentan bajos indices de tomas de alcohol. Por lo demas estan iguales y se siguen manteniendo
# las variables no mencionadas bajas. 
# El Cluster 2 en el k - medias es el cluster 1 en el jerarquico y se siguen reportando que en este cluster estan
# los individuos que han sido diagnosticados de enfermedad del corazon, pero ahora la herencia familiar esta un poco 
# mas alta, se siguen reportando que son personas son edades altas, y ahora se suma otra variable que es una alta ingesta 
# de alcohol (con respecto al Jerarquico esta era mas baja) y se sigue manteniendo la variable de obesidad como alta, 
# pero ademas con el K-Means estos individuos ahora presentan altos indices de adiposidad, colesterol, consumen mas tabaco 
# y tienen registros de presion cardiaca mas elevados, en el Jerarquico, estas 4 ultimas variables eran exactamente iguales
# (con excepcion de la adiposidad que en el K - Medias esta un poco mas baja) con la intromision de las variables categoricas 
# varias variables tendieron a subir, y se ve una fuerte correlacion entre las variables categoricas y las numericas.
# Finalmente el Cluster 3 en el k - medias es el cluster 2 en el Jerarquico, pero muchas de las variables se mantienen
# igual como es el caso de la edad que se posiciona alta, la adiposidad, el colesterol, la ingesta de tabaco, asi como
# la medicion del ritmo cardiaco o sbp con el K - medias ahora esta mas alta que con el Jerarquico. Ambos siguen manteniendo
# que no se les ha detectado enfermedad del corazon a estos individuos y en el Jerarquico habia una alta presencia de
# historial familiar, mientras que en el K - medias este bajo levemente y la obesidad tambien bajo, pero en este nuevo
# se presenta una mayor ingesta de alcohol, misma que en el Jerarquico aparecia como baja o casi nula y ahora en el K -
# medias varia parte de los individuos presentan que no presentan historial familiar, mientras que en el Jerarquico era
# casi nulo o muy bajo. 
# En este nuevo cluster formado por K - means, se ve que estan las personas que no han sido diagnosticadas de enfermedad
# del corazon pero una fuerte parte de los datos tiene historial familiar de padecimiento sumado al hecho de que son personas
# con edades altas y otras enfermedades y que ademas, consumen altos indices de alcohol. 


# #### Grafique usando colores sobre las dos primeras componentes del plano principal en el Analisis en Componentes Principales los clusteres obtenidos segun k-medias (usando k =3).

# In[59]:


pca = PCA(n_components=2)
componentes = pca.fit_transform(dummy)
componentes
print(dummy.shape)
print(componentes.shape)
plt.scatter(componentes[:, 0], componentes[:, 1],c=kmedias.predict(dummy))
plt.xlabel('componente 1')
plt.ylabel('componente 2')
plt.title('3 Cluster K-Medias')


# #### Usando 50 ejecuciones del metodo k−medias grafique el “Codo de Jambu” para este ejemplo. ¿Se estabiliza en algun momento la inercia inter–clases?¶

# In[60]:


# Solo 3 iteraciones y usando 50 ejecuciones con valores con defecto de max_iter = 300 e init = 50
kmedias = KMeans(n_clusters=3, max_iter=300, n_init=50) # Declaración de la instancia
kmedias.fit(dummy)
centros = np.array(kmedias.cluster_centers_)
print(centros)


# In[61]:


Nc = range(1, 20)
kmediasList = [KMeans(n_clusters=i) for i in Nc]
varianza = [kmediasList[i].fit(dummy).inertia_ for i in range(len(kmediasList))]
plt.plot(Nc,varianza,'o-')
plt.xlabel('Número de clústeres')
plt.ylabel('Varianza explicada por cada cluster (Inercia Intraclases)')
plt.title('Codo de Jambu')


# ### Interpretacion

# In[62]:


# En este caso no hay mucha claridad, pero se podria decir que en K = 5 o K = 8 podrian ser opciones viables, ya que es
# donde se normaliza el codo


# ### Interpretacion Jerarquico con variables categoricas vs k - means con variables categoricas

# In[36]:


# Con la agregacion de las variables categoricas los resultados si se ven mucho mejores y se asemejan mucho a lo que 
# habia dado cuando se hizo por Jerarquico, ya que se puede ver un primer cluster (azul) que esta representado por 
# las personas "sanas" que son las que no presentan enfermedad de corazon, que no tienen historial familiar con este
# padecimiento pero que son altos en el comportamiento A, que significa que son personas mas orientadas a los resultados
# mas competitivas y que por ende pasan mas estresadas y tensas. 
# En el cluster 2 (naranja): se ve que esta representado por las personas que no han sido diagnosticas de enfermedad del
# corazon pero que son obesas, ya tienen las mayores edades, presentan adiposidad, tienen una tendencia a la alta en el
# colesterol, asi como en las mediciones de la presion cardiaca, y consumen altos indices de alcohol y tambien fuman
# y algunos presentan historial familiar de padecimientos del corazon mientras que otros no. Este es el grupo de las 
# personas que presentan cierta herencia de la enfermedad pero su condicion de salud se ve agrabada por su estilo de 
# alimentacion y de vida.
# Finalmente en el cluster 3 (verde): estas las personas que ya han sido diagnosticas de enfermedad del corazon, tambien
# son personas con las mayores edades, tienen alta su mediciones de presion cardiaca y presentan colesterol, adiposidad
# sobrepeso, son de comportamiento tipo A, consumen mucho tabaco, toman indices altos de alcohol y la enfermedad del 
# corazon a nivel hereditario esta muy altamente presente, en lo que se diferencia este cluster del 2 es que estos si han
# sido diagnosticados de enfermedad del corazon, mientras que los del cluster 2 no, pero son sujetos en riesgo. 
# Con el primer radar y graficos de las variables numericas, arrojaba informacion pero era muy escueta y no se veia 
# una alta correlacion entre las variables y como la herencia o estar diagnosticado o no jugaba un papel importante en el
# analisis.


# ## Ejercicio 3

# ### Programe la Jerarquia de clases de acuerdo al siguiente diagrama

# In[39]:


#Configuraciones para imagen
import pandas as pd
pd.options.display.max_rows = 10
from IPython.display import Image
Image(filename='/Users/heinerleivagmail.com/Jerarquia.png')


# In[1]:


import pandas as pd
import numpy as np
import scipy.linalg as la
from sklearn import preprocessing
import matplotlib.pyplot as plt
from math import sqrt
import os
import scipy.stats
import os
from math import pi
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, ward, single, complete,average,linkage, fcluster
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from math import ceil, floor
from seaborn import color_palette
from sklearn.decomposition import PCA
from   sklearn.datasets import make_blobs
from   sklearn.cluster import KMeans


# In[3]:


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
    def histograma(self):
        plt.style.use('seaborn-white')
        return plt.hist(self.__datos)
    def grafico_densidad(self):
        grafo = self.__datos.plot(kind='density')
        return grafo
    def test_normalidad(self):
        X = self.__datos['Matematicas'] 
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


# In[5]:


class PCA(exploratorio):
    def __init__(self, datos = pd.DataFrame()):
        super().__init__(datos = pd.DataFrame())
        self.__columnas = datos.columns
        self.__filas = datos.index
        self.__datos = self.__transformar(datos)
        self.__correlaciones = self.__correlaciones(self.datos)
        self.__varvec_propios = self.__varvec_propios(self.correlaciones)
        self.__componentes = self.__componentes(self.datos, self.varvec_propios)
        self.__calidades_ind = self.__calidades_ind(self.datos, self.componentes) 
        self.__coordenadas_var = self.__coordenadas_var(self.varvec_propios)
        self.__calidades_var = self.__calidades_var(self.varvec_propios)
        self.__inercias = self.__inercias(self.varvec_propios, self.datos.shape[1])
    @property
    def datos(self):
        return self.__datos
    @property
    def correlaciones(self):
        return self.__correlaciones
    @property
    def varvec_propios(self):
        return self.__varvec_propios
    @property
    def componentes(self):
        return self.__componentes
    @property
    def calidades_ind(self):
        return self.__calidades_ind
    @property
    def coordenadas_var(self):
        return self.__coordenadas_var
    @property
    def calidades_var(self):
        return self.__calidades_var
    @property
    def inercias(self):
        return self.__inercias
    def __transformar(self, datos):
        return preprocessing.StandardScaler().fit_transform(datos)
    def __correlaciones(self, datos):
        return pd.DataFrame((1/datos.shape[0])*np.mat(datos.T)*np.mat(datos), index=self.__columnas) #Se utiliza 
    # formula propuesta por el profesor, no da igual que corr y cov. 
    def __varvec_propios(self, correlaciones):
        valores_propios, vectores_propios = la.eig(correlaciones)
        valor_vector = [(np.abs(valores_propios[i]).real, vectores_propios[:,i]) for i in range(len(valores_propios))]
        valor_vector.sort(key=lambda x: x[0], reverse=True)
        return valor_vector
    def __componentes(self, datos, varvec_propios):
        df = pd.DataFrame()
        for x in range(len(varvec_propios)):
            df[x] = varvec_propios[x][1]
        return pd.DataFrame(np.dot(datos, df), index=self.__filas)
    def __calidades_ind(self, datos, componentes):
        datos_2 = datos ** 2
        componentes_2 = componentes ** 2 
        df = pd.DataFrame(datos)
        for i in range(componentes_2.shape[0]):
            for j in range(componentes_2.shape[1]):
                fila_suma = sum(datos_2[i, :])
                df.iloc[i,j] = componentes_2.iloc[i,j] / fila_suma
        return df
    def __coordenadas_var(self, varvec_propios):
        df = pd.DataFrame()
        for x in range(len(varvec_propios)):
            df[x] = (varvec_propios[x][0] ** (0.5)) * varvec_propios[x][1]
        return df
    def __calidades_var(self, varvec_propios):
        df = pd.DataFrame()
        for x in range(len(varvec_propios)):
            df[x] = varvec_propios[x][0] * (varvec_propios[x][1] ** 2)
        return df
    def __inercias(self, varvec_propios, m):
        arreglo = []
        for x in range(len(varvec_propios)):
            arreglo.append(100 * (varvec_propios[x][0] / m))
        return pd.DataFrame(np.matrix(arreglo))
    def plot_plano_principal(self, ejes = [0, 1], ind_labels = True, titulo = 'Plano Principal'):
        x = self.componentes[ejes[0]].values
        y = self.componentes[ejes[1]].values
        plt.style.use('seaborn-whitegrid')
        plt.scatter(x, y, color = 'gray')
        plt.title(titulo)
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.inercias[ejes[0]], 2)
        inercia_y = round(self.inercias[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        if ind_labels:
            for i, txt in enumerate(self.componentes.index):
                plt.annotate(txt, (x[i], y[i]))
    def plot_circulo(self, ejes = [0, 1], var_labels = True, titulo = 'Círculo de Correlación'):
        cor = self.coordenadas_var.iloc[:, ejes].values
        plt.style.use('seaborn-whitegrid')
        c = plt.Circle((0, 0), radius = 1, color = 'steelblue', fill = False)
        plt.gca().add_patch(c)
        plt.axis('scaled')
        plt.title(titulo)
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.inercias[ejes[0]], 2)
        inercia_y = round(self.inercias[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * 0.95, cor[i, 1] * 0.95, color = 'steelblue', 
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * 1.05, cor[i, 1] * 1.05, self.correlaciones.index[i], 
                         color = 'steelblue', ha = 'center', va = 'center') 
    def plot_sobreposicion(self, ejes = [0, 1], ind_labels = True, 
                      var_labels = True, titulo = 'Sobreposición Plano-Círculo'):
        x = self.componentes[ejes[0]].values
        y = self.componentes[ejes[1]].values
        cor = self.correlaciones.iloc[:, ejes]
        scale = min((max(x) - min(x)/(max(cor[ejes[0]]) - min(cor[ejes[0]]))), 
                    (max(y) - min(y)/(max(cor[ejes[1]]) - min(cor[ejes[1]])))) * 0.7
        cor = self.coordenadas_var.iloc[:, ejes].values
        plt.style.use('seaborn-whitegrid')
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.inercias[ejes[0]], 2)
        inercia_y = round(self.inercias[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        plt.scatter(x, y, color = 'gray')
        if ind_labels:
            for i, txt in enumerate(self.componentes.index):
                plt.annotate(txt, (x[i], y[i]))
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * scale, cor[i, 1] * scale, color = 'steelblue', 
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * scale * 1.15, cor[i, 1] * scale * 1.15, 
                         self.correlaciones.index[i], 
                         color = 'steelblue', ha = 'center', va = 'center')


# In[6]:


class cluster(exploratorio):
    def __init__(self, datos, num_cluster):
        super().__init__(datos)
        self.__num_cluster = num_cluster
    @property
    def num_cluster(self):
        return self.__num_cluster
    def centroide(self, num_cluster, datos, clusters):
        ind = clusters == num_cluster
        return(pd.DataFrame(datos[ind].mean()).T)
    # Función para graficar los gráficos de Barras para la interpretación de clústeres
    def bar_plot(self, centros, labels, cluster = None, var = None):
        from math import ceil, floor
        from seaborn import color_palette
        colores = color_palette()
        minimo = floor(centros.min()) if floor(centros.min()) < 0 else 0
        def inside_plot(valores, labels, titulo):
            plt.barh(range(len(valores)), valores, 1/1.5, color = colores)
            plt.xlim(minimo, ceil(centros.max()))
            plt.title(titulo)
        if var is not None:
            centros = np.array([n[[x in var for x in labels]] for n in centros])
            colores = [colores[x % len(colores)] for x, i in enumerate(labels) if i in var]
            labels = labels[[x in var for x in labels]]
        if cluster is None:
            for i in range(centros.shape[0]):
                plt.subplot(1, centros.shape[0], i + 1)
                inside_plot(centros[i].tolist(), labels, ('Cluster ' + str(i)))
                plt.yticks(range(len(labels)), labels) if i == 0 else plt.yticks([]) 
        else:
            pos = 1
            for i in cluster:
                plt.subplot(1, len(cluster), pos)
                inside_plot(centros[i].tolist(), labels, ('Cluster ' + str(i)))
                plt.yticks(range(len(labels)), labels) if pos == 1 else plt.yticks([]) 
    # Función para graficar los gráficos tipo Radar para la interpretación de clústeres
    def radar_plot(self, centros, labels):
        from math import pi
        centros = np.array([((n - min(n)) / (max(n) - min(n)) * 100) if 
                        max(n) != min(n) else (n/n * 50) for n in centros.T])
        angulos = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
        angulos += angulos[:1]
        ax = plt.subplot(111, polar = True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angulos[:-1], labels)
        ax.set_rlabel_position(0)
        plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
               ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"], 
               color = "grey", size = 8)
        plt.ylim(-10, 100)
        for i in range(centros.shape[1]):
            valores = centros[:, i].tolist()
            valores += valores[:1]
            ax.plot(angulos, valores, linewidth = 1, linestyle = 'solid', 
                    label = 'Cluster ' + str(i))
            ax.fill(angulos, valores, alpha = 0.3)
        plt.legend(loc='upper right', bbox_to_anchor = (0.1, 0.1))


# In[7]:


class Jerarquico(cluster):
    def __init__(self, datos, num_cluster):
        super().__init__(datos, num_cluster)
        self.__num_cluster = num_cluster
    def barras_jerarquico(self):
        grupos = fcluster(linkage(pdist(self.datos), method = 'ward', metric='euclidean'), 
                          self.__num_cluster, criterion = 'maxclust') 
        grupos = grupos-1 
        print(grupos)
        centros = np.array(pd.concat([cluster.centroide(self, 0, self.datos, grupos), 
                              cluster.centroide(self, 1, self.datos, grupos),
                              cluster.centroide(self, 2, self.datos, grupos)]))
        print(centros)    
        plt.figure(1, figsize = (20, 8))
        self.bar_plot(centros, self.datos.columns)
    def radar_jerarquico(self):
        grupos = fcluster(linkage(pdist(self.datos), method = 'ward', metric='euclidean'), 
                          self.__num_cluster, criterion = 'maxclust')
        grupos = grupos-1
        print(grupos)
        centros = np.array(pd.concat([cluster.centroide(self, 0, self.datos, grupos), 
                              cluster.centroide(self, 1, self.datos, grupos),
                              cluster.centroide(self, 2, self.datos, grupos)]))
        print(centros)
        plt.figure(1, figsize = (10, 10))
        self.radar_plot(centros, self.datos.columns)


# In[8]:


class K_medias(cluster):
    def __init__(self, datos, num_cluster):
        super().__init__(datos, num_cluster)
        self.__num_cluster = num_cluster
    def barras_k(self):
        kmedias = KMeans(self.__num_cluster) 
        kmedias.fit(self.datos)
        print(kmedias.predict(self.datos))
        centros = np.array(kmedias.cluster_centers_)
        print(centros) 
        plt.figure(1, figsize = (12, 8))
        self.bar_plot(centros, self.datos.columns)
    def radar_k(self):
        kmedias = KMeans(self.__num_cluster) 
        kmedias.fit(self.datos)
        print(kmedias.predict(self.datos))
        centros = np.array(kmedias.cluster_centers_)
        print(centros) 
        plt.figure(1, figsize = (10, 10))
        self.radar_plot(centros, self.datos.columns)


# In[4]:


# Importando datos 
estudiantes = pd.read_csv('EjemploEstudiantes.csv', delimiter=';', decimal=',', header=0, index_col=0)
datos = exploratorio(estudiantes)


# ## Haciendo pruebas de la clase exploratorio

# In[9]:


datos.head()


# In[15]:


datos.estadisticas()


# In[16]:


datos.dimension()


# In[10]:


datos.valores_atipicos()


# In[5]:


datos.histograma()


# In[6]:


datos.grafico_densidad()


# In[7]:


datos.test_normalidad()


# ## Haciendo pruebas con la clase PCA con herencia de la clase exploratorio

# In[25]:


nuevo = PCA(estudiantes)


# ### Prueba de analisis Exploratorio de datos con PCA

# In[34]:


nuevo.estadisticas()


# In[27]:


nuevo.histograma()


# ### Pruebas de inercia y ploteo de planos principales

# In[28]:


nuevo.inercias


# In[30]:


nuevo.plot_plano_principal()


# In[31]:


nuevo.plot_circulo()


# In[32]:


nuevo.plot_sobreposicion()


# ## Haciendo pruebas con la clase Jerarquico que hereda de la clase cluster 

# In[9]:


test = cluster(estudiantes, 3)


# ### Prueba de analisis Exploratorio de datos con Jerarquico

# In[12]:


alpha = Jerarquico(estudiantes, 3)


# In[13]:


alpha.histograma()


# In[14]:


alpha.estadisticas()


# ### Prueba de Plots de la clase Jerarquica

# In[15]:


alpha.barras_jerarquico()


# In[16]:


alpha.radar_jerarquico()


# ## Haciendo Pruebas con la clase K-Medias que hereda de Cluster

# In[10]:


beta = K_medias(estudiantes, 3)


# ### Prueba de Analisis Exploratorio con la Clase k-Medias

# In[11]:


beta.histograma()


# In[12]:


beta.estadisticas()


# ### Prueba de Plots con la clase K - Medias

# In[13]:


beta.barras_k()


# In[14]:


beta.radar_k()

