#!/usr/bin/env python
# coding: utf-8

# # >>>>>>>>>>>>>>>>>>>>Tarea número 3 <<<<<<<<<<<<<<<<<<<<<<<<

# # Estudiante: Heiner Romero Leiva

# # Ejercicio #1

# In[2]:


import os
import pandas as pd
import numpy as np
from   math import pi
from   sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, ward, single, complete,average,linkage, fcluster
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler


# In[3]:


# Función para calcular los centroides de cada cluster¶
def centroide(num_cluster, datos, clusters):
  ind = clusters == num_cluster
  return(pd.DataFrame(datos[ind].mean()).T)


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


# ### a) Cargue la tabla de datos SpotifyTop2018 40 V2.csv

# In[7]:


os.chdir("/Users/heinerleivagmail.com")
print(os.getcwd())
data = pd.read_csv('SpotifyTop2018_40_V2.csv',delimiter=',',decimal=".")
print(data)


# In[8]:


# Normalizando y centrando la tabla 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values = scaler.fit_transform(data) 
data.loc[:,:] = scaled_values
print(data)
datos = data 


# In[9]:


ward_res = ward(datos)         #Ward
single_res = single(datos)     #Salto mínimo
complete_res = complete(datos) #Salto Máxim
average_res = average(datos)   #Promedio


# ### b) Ejecute un Clustering Jerarquico con la agregacion del Salto Maximo, Salto Mınimo, Promedio y Ward. Grafique el dendograma con cortes para dos y tres clusteres.

# In[10]:


dendrogram(average_res,labels= datos.index.tolist())
plt.figure(figsize=(13,10))
dendrogram(complete_res,labels= datos.index.tolist())
plt.figure(figsize=(13,10))
dendrogram(single_res,labels= datos.index.tolist())
plt.figure(figsize=(13,10))
dendrogram(ward_res,labels= datos.index.tolist())

# Agrega cortes con 2 y 3 clústeres con agregación de Ward
ax = plt.gca()
limites = ax.get_xbound()
ax.plot(limites, [11, 11], '--', c='k')
ax.plot(limites, [9.4, 9.4], '--', c='k')
ax.text(limites[1], 11, ' dos clústeres', va='center', fontdict={'size': 15})
ax.text(limites[1], 9.4, ' tres clústeres', va='center', fontdict={'size': 15})
plt.xlabel("Orden en el eje X./nPor hacer la normalizacion de los datos el cluster 3 quedo muy cerca del 2")
plt.ylabel("Distancia o Agregación")


# ### c)  Usando tres clusteres interprete los resultados del ejercicio anterior para el caso de agregacion de Ward usando graficos de barras y graficos tipo Radar.

# In[11]:


grupos = fcluster(linkage(pdist(datos), method = 'ward', metric='euclidean'), 3, criterion = 'maxclust')
grupos = grupos-1 # Se resta 1 para que los clústeres se enumeren de 0 a (K-1), como usualmente lo hace Python
# El siguiente print es para ver en qué cluster quedó cada individuo
print(grupos)
centros = np.array(pd.concat([centroide(0, datos, grupos), 
                              centroide(1, datos, grupos),
                              centroide(2, datos, grupos)]))
print(centros)    
plt.figure(1, figsize = (20, 8))
bar_plot(centros, datos.columns)


# In[12]:


# Interpretación 3 Clústeres - Gráfico Radar plot con Ward
grupos = fcluster(linkage(pdist(datos), method = 'ward', metric='euclidean'), 3, criterion = 'maxclust')
grupos = grupos-1
print(grupos)
centros = np.array(pd.concat([centroide(0, datos, grupos), 
                              centroide(1, datos, grupos),
                              centroide(2, datos, grupos)]))
print(centros)
plt.figure(1, figsize = (10, 10))
radar_plot(centros, datos.columns)


# ### Interpretacion

# In[31]:


# Analisis: 
# Cluster 1 (azul), este cluster se caracteriza por tener los niveles mas altos (100) en accousticness, es decir, las
# canciones en este cluster son las mas acusticas, tambien, tiene el mayor speechiness, es decir, hay muchas palabras
# en las canciones que estan en este cluster, ademas cuenta  con el mayor numero en liveness (es decir hay publico en
# la cancion), tambien tiene los niveles mas altos de valence (mucha postitividad en las canciones), el time_signature 
# que representa la cantidad de beats que hay en cada barra de medida y por ultimo danceability, que son las canciones 
# que tienen mayor potencial para ser bailable, a modo general en este cluster se agrupan las canciones quee son mas 
# positivas, mas aptas para bailar, con mayor sonido, mayor presencia de publico, es decir, son las canciones mas "alegres", 
# por otro lado este cluster se caracteriza por tener canciones que tienen niveles 0 de instrumentalidad, su duracion en
# milisegundos es baja, su energy es moderada baja al igual que su loudness, es decir su sonoridad en la pista es baja. 
# Cluster 2 (naranja): este se representa por tener las canciones que tienen mayor  duracion en milisegundos, asi como
# las canciones que se encuentran en este  cluster cuentan con tempo son las que tienen mayores beats por minuto (variable
# tempo). Ademas su acousticness es moderado, es decir estas canciones presentan algo de acustica y su speechiness, que 
# es la presencia de palabras en las canciones tiende a ser bajo. En las demas variables este cluster presenta bajos niveles
# entonces se puede decir que este cluster se caracteriza por tener las canciones con mayor duracion, con mas beats por
# minuto y son canciones que combinan acustica y letras en sus estrofas. 
#Cluster 3 (verde): en este caso las canciones que pertenecen a este cluster se caracterizan por tener los mas altos
# beats por minuto, presentan mucha instrumentalidad, su time_signature es alto, lo que representa altos beats en cada
# barra o medida, su intensidad es bastante alta (energy) y su sonoridad en decibiles tambien es bastante alta. Las 
# canciones en este grupo se caracterizan por altamente instrumentales con nula cantidad de voces en sus piezas, y son 
# canciones bastante intensas y con los beats mas altos por minuto, son canciones que son relativamente bailables, y su
# positividad musical es moderada y no presenta publico en sus piezas. Son canciones por asi decirlo, meramente instrumen-
# tales con poco o nulo registro de voz por parte de tun cantante. 


# ### d) Grafique usando colores sobre las dos primeras componentes del plano principal en el Analisis en Componentes Principales los clusteres obtenidos segun la clasificacion Jerarquica (usando tres clusteres).

# In[13]:


# Importando datos 
campo = pd.read_csv('SpotifyTop2018_40_V2.csv',delimiter=',',decimal=".")

# Normalizando y centrando la tabla 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values = scaler.fit_transform(campo) 
campo.loc[:,:] = scaled_values
datosx = campo 

#Asignando variables 
cal = datosx.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]].values


# In[14]:


# Definiendo parametros de dendrograma
clustering_jerarquico = linkage(cal, 'ward')


# In[15]:


# Ploteando dendrograma 
dendrogram = sch.dendrogram(clustering_jerarquico)


# In[16]:


# Asignando cluster a cada variable  
clusters = fcluster(clustering_jerarquico, t=9.4, criterion = 'distance') #t corresponde al corte para obtener los 3 
# clusters 
clusters


# In[17]:


# Creando clusters en cada fila 
datosx['target'] = clusters


# In[18]:


# Guardando nueva variable generada
campo.to_csv("/Users/heinerleivagmail.com/SpotifyTop2018_40_V3.csv")


# In[19]:


# Llamando DF creado con la asignacion de cada cluster (tabla ya esta normalizada)
df = pd.read_csv('SpotifyTop2018_40_V3.csv',delimiter=',',decimal=".")
# Separando variables numericas
x = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]].values
# Separando los clusters obtenidos 
y = df.iloc[:,[11]].values


# In[20]:


# Definiendo parametros del nuevo PCA a partir del Dendrograma 
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(datosx)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['Componente 0', 'Componente 1'])
finalDf = pd.concat([principalDf, df.iloc[:,[12]]], axis = 1)
finalDf.head(10)


# In[21]:


# Definicion de la estructura del PCA con colores respectivos
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Componente 0', fontsize = 15)
ax.set_ylabel('Componente 1', fontsize = 15)
ax.set_title('Plano Principal', fontsize = 20)
targets = [1, 2, 3]
colors = ['g', 'r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'Componente 0']
               , finalDf.loc[indicesToKeep, 'Componente 1']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# # Ejercicio 2

# ### a) Efectue un Clustering Jerarquico usando solo las variables numericas y de una interpretacion usando 3 clusteres.

# In[6]:


os.chdir("/Users/heinerleivagmail.com")
print(os.getcwd())
corazon = pd.read_csv('SAheart.csv',delimiter=';',decimal=".")
print(corazon.head())
print(corazon.shape)


# In[7]:


corazon2 = pd.DataFrame(data=corazon, columns=['sbp', 'tobacco', 'ldl',
   'adiposity','typea','obesity','alcohol','age'])
print(corazon2)
print(corazon2.shape)
corazon2.describe()


# In[8]:


# Normalizando y centrando la tabla 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values = scaler.fit_transform(corazon2) 
corazon2.loc[:,:] = scaled_values
print(corazon2)
datos = corazon2 


# In[9]:


ward_res = ward(datos)         #Ward
single_res = single(datos)     #Salto mínimo
complete_res = complete(datos) #Salto Máximo
average_res = average(datos)   #Promedio


# In[10]:


dendrogram(average_res,labels= datos.index.tolist())
plt.figure(figsize=(13,10))
dendrogram(complete_res,labels= datos.index.tolist())
plt.figure(figsize=(13,10))
dendrogram(single_res,labels= datos.index.tolist())
plt.figure(figsize=(13,10))
dendrogram(ward_res,labels= datos.index.tolist())

# Agrega cortes solo en 3 clústeres con agregación de Ward
ax = plt.gca()
limites = ax.get_xbound()
ax.plot(limites, [20.7, 20.7], '--', c='k')
ax.text(limites[1], 20.7, ' tres clústeres', va='center', fontdict={'size': 15})
plt.xlabel("Orden en el eje X")
plt.ylabel("Distancia o Agregación")


# In[11]:


# Graficos de barras con Ward
grupos = fcluster(linkage(pdist(datos), method = 'ward', metric='euclidean'), 3, criterion = 'maxclust')
grupos = grupos-1 # Se resta 1 para que los clústeres se enumeren de 0 a (K-1), como usualmente lo hace Python
# El siguiente print es para ver en qué cluster quedó cada individuo
print(grupos)
centros = np.array(pd.concat([centroide(0, datos, grupos), 
                              centroide(1, datos, grupos),
                              centroide(2, datos, grupos)]))
print(centros)    
plt.figure(1, figsize = (30, 10))
bar_plot(centros, datos.columns)


# In[12]:


grupos = fcluster(linkage(pdist(datos), method = 'ward', metric='euclidean'), 3, criterion = 'maxclust')
grupos = grupos-1 # Se resta 1 para que los clústeres se enumeren de 0 a (K-1), como usualmente lo hace Python
# El siguiente print es para ver en qué cluster quedó cada individuo
print(grupos)
centros = np.array(pd.concat([centroide(0, datos, grupos), 
                              centroide(1, datos, grupos),
                              centroide(2, datos, grupos)]))
print(centros)
plt.figure(1, figsize = (10, 10))
radar_plot(centros, datos.columns)


# ### Interpretacion

# In[32]:


# Para este segundo caso se puede ver como el cluster 1 (azul): son los individuos que estan sanos, ya que solo presentan
# un comportamiento tipo A alto muy alto, que los hace mas competitivos, orientados al trabajo, etc., en lo demas
# no presentan ninguna otra caracteristica. 
# Cluster 2 (naranja): se caracteriza por tener a los individuos que tienen las edades mas altas, asi como la presion 
# cardiaca, adiposidad y obesidad mas altas, asi como el colesterol, mientras que en otros parametros como el comporta-
# miento del tipo A (menos de 40%) y los niveles de alcohol estan bajos, es decir, no son consumidores de alcohol.  
# En este cluster se pueden agrupar a todas aquellas personas que ya son avanzadas de edad y que presentan altos
# grados de obesidad y con ello colesterol y una presion cardiaca mas alta, y que ademas tienen una ligera tendencia 
# a ser del comportamiento tipo A. 
# En el cluster 3 (verde) se puede ver como los individuos de este grupo son los que tienen mas vicios (consumen mayores
# indices de alcohol y fuman mucho) ademas, presentan las edades altas de igual forma y su adiposidad tambien alcanza
# casi el 90%, por otro lado, presentan mas de un 60% de obesidad, y mas de un 40% de colesterol, ademas su presion 
# cardiaca tambien es muy alta, pero su comportamiento tipo A es muy bajo, al parecer en este grupo estan las personas
# que son mayores tienen vicios, y ademas cuentan con presiones sanguineas altas. 


# ### b) Efectue un Clustering Jerarquico usando las variables numericas y las variables categoricas. Luego de una interpretacion usando 3 clusteres.

# In[13]:


os.chdir("/Users/heinerleivagmail.com")
print(os.getcwd())
datos2 = pd.read_csv('SAheart.csv',delimiter=';',decimal=".")
print(datos.head())
print(datos.shape)


# In[14]:


def recodificar(col, nuevo_codigo):
  col_cod = pd.Series(col, copy=True)
  for llave, valor in nuevo_codigo.items():
    col_cod.replace(llave, valor, inplace=True)
  return col_cod


# In[15]:


# Conviertiendo la variables en Dummy
datos_dummies = pd.get_dummies(datos2)
print(datos_dummies.head())
print(datos_dummies.dtypes)


# In[16]:


# Centrando y normalizando los datos convertidos en dummies 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values = scaler.fit_transform(datos_dummies) 
datos_dummies.loc[:,:] = scaled_values
print(datos_dummies) 
datos_dummy = datos_dummies


# In[17]:


ward_res = ward(datos_dummy) # Ward
dendrogram(ward_res,labels= datos_dummy.index.tolist())
# Agrega cortes en 3 clústeres con agregación de Ward
ax = plt.gca()
limites = ax.get_xbound()
ax.plot(limites, [28, 28], '--', c='k')
ax.text(limites[1], 28, ' tres clústeres', va='center', fontdict={'size': 15})
plt.xlabel("Orden en el eje X")
plt.ylabel("Distancia o Agregación")


# In[18]:


# Grafico de Barras con Ward 
grupos = fcluster(linkage(pdist(datos_dummy), method = 'ward', metric='binary'), 3, criterion = 'maxclust')
grupos = grupos-1 # Se resta 1 para que los clústeres se enumeren de 0 a (K-1), como usualmente lo hace Python
# El siguiente print es para ver en qué cluster quedó cada individuo
print(grupos)
centros = np.array(pd.concat([centroide(0, datos_dummy, grupos), 
                              centroide(1, datos_dummy, grupos),
                              centroide(2, datos_dummy, grupos)]))
print(centros)    
plt.figure(1, figsize = (12, 8))
bar_plot(centros, datos_dummy.columns)


# In[26]:


# Graficos Radar Plot con Ward
grupos = fcluster(linkage(pdist(datos_dummy), method = 'ward', metric='binary'), 3, criterion = 'maxclust')
grupos = grupos-1 # Se resta 1 para que los clústeres se enumeren de 0 a (K-1), como usualmente lo hace Python
# El siguiente print es para ver en qué cluster quedó cada individuo
print(grupos)
centros = np.array(pd.concat([centroide(0, datos_dummy, grupos), 
                              centroide(1, datos_dummy, grupos),
                              centroide(2, datos_dummy, grupos)]))
print(centros)
plt.figure(1, figsize = (10, 10))
radar_plot(centros, datos_dummy.columns)


# ### Interpretacion

# In[91]:


# Incluyendo las variables cualitativas convertidas en Dummies y una vez que se normalizaron todas las variables se
# tiene lo siguiente:
# Cluster 1 (azul): se sigue manteniendo la variable de typea A como alta, sin embargo otras como adiposidad, edad, 
# colesterol, consumo de tabaco, presion arterial alta y diagnostico de enfermedad del corazon positivo se han ahora 
# agregado, generando que este grupo sea de los individuos que tienen mayor edad y que tienen enfermedades adicionales
# y que ademas consumen bastante tabaco y consumen alcohol y es importante mencionar que ya han sido diagnosticados
# con enfermedad cardiaca, ademas que de que tienen historial medico de que se les ha diagnosticado a familiares este 
# mismo padecimiento. Este grupo va a estar conformada por personas con diagnostico en pie de enfermedades cardiacas, 
# que ademas han tenido familiares con esta misma condicion y que ademas tienen otras enfermedades adicionales y 
# presentan algun vicio. 
# Cluster 2 (Naranja): para este cluster, los pacientes presentan los mas altos indices de obesidad, tienen adicional
# los mayores indices de herencia familiar en cuanto a enfermedades del corazon, pero no han sido diagnosticados por
# este padecimiento, pero presentan alta adiposidad, otros factores como colesterol se encuentra presenta en algunos 
# individuos pero es bajo, y algunos consumen tabaco, ne cuanto a la edad no es tan alta como en el cluster 1, pero ya
# individuos que pueden estar en una edad media. 
# Cluster 3 (verde): finalmente  se tienen a los individuos del cluster verde, que estos son los que tienen las menores 
# edades, pero presentan altos indices de toma de alcohol, cuentan con un ligero comportamiento tipo A, que los vuelve 
# individuos mas competitivos, orientados a resultados, etc., ademas no han tenido ni diagnostico de enfermedad cardiacana
# ni en su familia se ha reportado historial medico de esta enfermedad, son por decirlo, los pacientes "sanos"


# ### c)  Explique las diferencias de los dos ejercicios anteriores ¿Cual le parece mas interesante? ¿Por que?

# In[93]:


# Las diferencias radican en que sabiendo si la persona ha tenido historial familiar y si se le ha diagnosticado de 
# enfermedad coronaria se puede hacer un mejor analisis y mas completo, que solo con las variables numericas, ya que 
# ellas por si solas representan ciertos niveles, pero no se pueden sacar conclusiones ni entender excesivamente bien
# como se puede comportar esta enfermedad tomando en cuenta diferentes variables como la edad, los vicios, la obesidad
# presion alta, el typea, etc., con la intromision de las variables categoricas se ve que hay correlacion entre haber
# tenido historial familiar con esta enfermedad y sumado a un estilo de vida no saludable, se puede llegar a padecer
# esta enfermedad con el transcurso de los anos, otra como la typea A, que si una persona tienen este comportamiento
# y ademas, ingiere alcohol, aunque no haya tenido historial familiar ni han sido diagnosticados es mas probable 
# que pueda tener en algun momento alguna enfermedad cardiaca o padecerla, por otro lado se ve que las personas que tienen
# los valores mas altos (cluster azul) solo poco menos de un 40% de los casos han sido hereditarias, mientras que todos 
# en este grupo ya presentan enfermedad cardiaca y ademas, hay variables que la agraban como la obesidad, tener tipo A, 
# la edad, tener colesterol, adiposidad y fumar, se ve como esta combinacion de factores pueden agrabar la salud de las 
# personas y que no en todos los casos estas enfermedades son hereditarias, sino que tambien van relacionadas con la 
# alimentacion y estilo de vida de las personas. 

# Indudablemente me parece mas interesante el segundo analisis, ya que en realidad con este se puede ver, 
# que cualquier persona puede tener un infarto al corazon y que no van ligados necesariamente a historial familiar 
# (es algo que potencia) y que  si las personas no practican habitos saludables de alimentacion, y ademas, tienen vicios 
# y son personas que pasan muy estresadas y ansiosas (typea A) pueden tener un infarto o ser diganosticados con esta 
# enfermedad. Aqui la leccion es tener habitos alimenticios saludables, no ingerir alcohol ni tabaco en exceso y tener 
# una vida mas tranquila. 


# # Ejercicio # 3

# ### Dendrogramas construidos a ¨pie¨, en orden respectivo: Salto Mínimo, Salto Máximo y Promedio. 

# In[19]:


#Configuraciones para imagen
import pandas as pd
pd.options.display.max_rows = 10


# In[20]:


from IPython.display import Image
Image(filename='/Users/heinerleivagmail.com/Minimo.png')


# In[114]:


from IPython.display import Image
Image(filename='/Users/heinerleivagmail.com/Maximo.png')


# In[115]:


from IPython.display import Image
Image(filename='/Users/heinerleivagmail.com/Promedio.png')


# # Ejercicio #4

# ### a) Programe una clase en Python que tiene un atributo tipo DataFrame, ademas de los metodos usuales que tiene toda clase, tendra un metodo que calcula la matriz de distancias, para esto usara la distancia de Chebychev entre dos vectores que se definio arriba

# In[164]:


class Chebychev:
    def __init__(self, data):
        self.__data = data
    @property 
    def data(self):
        return self.__data    
    def __chebychev(self, x, y):
        return abs(max(x) - max(y))
    def __grupos(self):
        return fclusterdata(self.data, 3, method = 'ward', metric=self.__chebychev, criterion = 'maxclust') - 1
    def __centroide(self, num_cluster, datos, clusters):
        ind = clusters == num_cluster
        return(pd.DataFrame(datos[ind].mean()).T)
    def centros(self):
        grupos = self.__grupos()
        return np.array(pd.concat([self.__centroide(0, self.data, grupos), 
                                  self.__centroide(1, self.data, grupos),
                                  self.__centroide(2, self.data, grupos)]))
    def bar_plot(self, cluster = None, var = None):
        from math import ceil, floor
        from seaborn import color_palette
        centros = self.centros()
        labels = self.data.columns
        colores = color_palette()
        minimo = floor(centros.min()) if floor(centros.min()) < 0 else 0
        def inside_plot( valores, labels, titulo):
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
                
    def radar_plot(self):
        from math import pi
        centros = self.centros()
        labels = self.data.columns
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


# ### b) Calcule la matriz de distancias usando la distancia de Chebychev para la tabla de datos EjemploEstudiantes.csv.

# In[182]:


# Cargando dataset 
estudiantes = pd.read_csv('EjemploEstudiantes.csv', delimiter=';', decimal=',', header=0, index_col=0)
# Cargando matriz de distancias para cargar en los plots
test = Chebychev(estudiantes)


# ### c) Para la tabla de datos EjemploEstudiantes.csv ejecute un Clustering Jerarquico usando la distancia de Chebychev programada por usted y la agregacion Ward, compare el resultado respecto a usar distancia euclidiana y agregacion de Ward. (Debe investigar como usar una distancia propia en scipy.cluster.hierarchy)

# ### Clustering de Barras usando distancia de Chebyshev

# In[183]:


val = Chebychev(estudiantes)
val.bar_plot()


# ### Radar Plot usando la distancia de Chebyshev

# In[184]:


val = Chebychev(estudiantes)
val.radar_plot()


# ### Comparando resultados usando distancia euclidea y agregacion de Ward

# ### Clustering de Barras usando distancia euclidea y agregacion de Ward

# In[185]:


grupos = fcluster(linkage(pdist(estudiantes), method = 'ward', metric='euclidean'), 3, criterion = 'maxclust')
grupos = grupos-1 # Se resta 1 para que los clústeres se enumeren de 0 a (K-1), como usualmente lo hace Python
# El siguiente print es para ver en qué cluster quedó cada individuo
print(grupos)
centros = np.array(pd.concat([centroide(0, estudiantes, grupos), 
                              centroide(1, estudiantes, grupos),
                              centroide(2, estudiantes, grupos)]))
print(centros)    
plt.figure(1, figsize = (12, 8))
bar_plot(centros, estudiantes.columns)


# ### Radar Plot usando distancia Euclidea y agregacion de Ward

# In[186]:


grupos = fcluster(linkage(pdist(estudiantes), method = 'ward', metric='euclidean'), 3, criterion = 'maxclust')
grupos = grupos-1 # Se resta 1 para que los clústeres se enumeren de 0 a (K-1), como usualmente lo hace Python
# El siguiente print es para ver en qué cluster quedó cada individuo
print(grupos)
centros = np.array(pd.concat([centroide(0, estudiantes, grupos), 
                              centroide(1, estudiantes, grupos),
                              centroide(2, estudiantes, grupos)]))
print(centros)
plt.figure(1, figsize = (10, 10))
radar_plot(centros, estudiantes.columns)


# #### Resultados de la programacion propia con Chevishev y con distancia Euclidea y agregacion de Ward son diferentes. 
