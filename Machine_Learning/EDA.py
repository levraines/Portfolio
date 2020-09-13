#!/usr/bin/env python
# coding: utf-8

# In[131]:


# Importando bibliotecas
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


# In[2]:


# Importando dataset
import pandas as pd
url='https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv'
data = pd.read_csv(url,sep=",") 
data.head()


# In[3]:


# Reemplazando NaN en Min.Price por la media de la columna
# Calculando la media de la columna 
mean = data.iloc[:,3:4].mean()
mean


# In[4]:


# Reemplazando en toda la columna
data['Min.Price'] = data['Min.Price'].replace(np.nan, float(mean))
data.iloc[:,3:4]


# In[5]:


# Reemplazando NaN en Max.Price por la mediana de la columna
# Calculando la media de la columna 
median = data.iloc[:,5:6].median()
median


# In[6]:


# Reemplazando en toda la columna
data['Max.Price'] = data['Max.Price'].replace(np.nan, float(median))
data.iloc[:,5:6]


# In[7]:


# Comprobando valores ingresados
data.describe()
# Para la columna Min.Price efectivamente la media es 17.11 y para Max.Price la mediana (50% que es el segundo cuartil) es 19.15


# In[8]:


####################################### Analisis Exploratorio de Datos (EDA) ####################################


# In[9]:


data.info()
# De las 27 columnas 9 corresponden a variables categoricas


# In[10]:


# Columnas
print(data.columns)


# #### Descripcion de las columnas y especificacion de las variables: 
# * Manufacturer: empresa que fabrica el vehiculo. Variable categorica nominal. 
# * Model: modelo del vehiculo. Variable categorica nominal. 
# * Type: tipo de vehiculo, compacto, pequeno, etc. Variable categorica ordinal. 
# * Min.Price: precio minimo del vehiculo en el mercado. Variable numerica continua. * Se hace la aclaracion que es muy probable que este precio se encuentre normalizado, ya que no concuerda con un precio estandar, sin embargo como no se tiene mayor detalle se toma como precio unitario.
# * Precio: precio de salida del vehiculo. Variable numerica continua. * Se hace la aclaracion que es muy probable que este precio se encuentre normalizado, ya que no concuerda con un precio estandar, sin embargo como no se tiene mayor detalle se toma como precio unitario. 
# * Max.Precio: Precio maximo de venta del vehiculo. Variable numerica continua. * Se hace la aclaracion que es muy probable que este precio se encuentre normalizado, ya que no concuerda con un precio estandar, sin embargo como no se tiene mayor detalle se toma como precio unitario.
# * MPG.City: es el promedio de millas por galon que consume un vehiculo en la ciudad. Variable numerica discreta. 
# * MPG.highway: es le promedio de millas por galon que consume un vehiculo en carretera.  Variable numerica discreta. 
# * Airbags: numero de bolsas de aire que tiene el vehiculo.  Variable categorica nominal (en el dataset) en realidad es una variable numerica y discreta porque se pueden contar, pero fue codificada como objeto.  
# * DriveTrain: tipo de transmision. Variable categorica nominal. 
# * Cylinders: cantidad de cilindos. Mal catalogada como categorica, deberia de ser una variable numerica discreta pero se codifico como objeto. 
# * EngineSize: tamano del motor. Variable numerica continua. 
# * Horsepower: caballos de poder.  Variable numerica discreta. 
# * RPM: revoluciones por minuto. Variable numerica discreta. 
# * Rev.per.mile: Revoluciones por milla. Variable numerica discreta. 
# * Man.trans.avail: tipo de transmision. Variable categorica nominal. 
# * Fuel.tank.capacity: capacidad total del tanque. Variable categorica nominal.
# * Passengers: capacidad maxima de pasajeros por vehiculo. Variable numerica discreta. 
# * Length: longitud del vehiculo en cms. Variable numerica continua. 
# * Wheelbase: distancia entre ejes. Variable numerica continua. 
# * Width: anchura del vehiculo. Variable numerica continua. 
# * Turn.circle: diametro de giro. Variable numerica continua. 
# * Rear.seat.room: espacio entre asientos. Variable numerica continua. 
# * Luggage.room: tamano del portaequipaajes. Variable numerica continua. 
# * Weight: peso aproximado del vehiculo. Variable numerica continua. 
# * Origin: origen de la marca. Variable categorica nominal. 
# * Make: marca del vehiculo. Variable categorica nominal.

# In[11]:


# En total se tienen 9 variables categoricas de las que se van a elaborar graficos de pastel para evaluar sus diferentes dimensiones.


# In[24]:


# Ver si los datos estan completos o hay algún NaN para hacer plots
print(data.isnull().sum())
# Hay varios NaN, pero los dejare para hacer los plots y ver visualmente en que lugares no hay informacion


# In[12]:


# Pie Charts
f, ax = plt.subplots(figsize=(30, 30))
plt.subplot(2, 2, 2)
# Manufacturer
plt.title("Fabricantes de autos")
df_sums = data["Manufacturer"].value_counts() 
sums = list(df_sums.values)
labels = list(df_sums.index)
plt.pie(sums, labels=labels, autopct='%1.1f%%')
plt.show()


# In[13]:


# Pie Charts
f, ax = plt.subplots(figsize=(60, 60))
plt.subplot(2, 2, 2)
# Model
plt.title("Modelos de autos")
df_sums = data["Model"].value_counts() 
sums = list(df_sums.values)
labels = list(df_sums.index)
plt.pie(sums, labels=labels, autopct='%1.1f%%')
plt.show()


# In[14]:


# Pie Charts
f, ax = plt.subplots(figsize=(30, 30))
plt.subplot(2, 2, 2)
# Type
plt.title("Tipo de Vehiculo")
df_sums = data["Type"].value_counts() 
sums = list(df_sums.values)
labels = list(df_sums.index)
plt.pie(sums, labels=labels, autopct='%1.1f%%')
plt.show()


# In[15]:


# Pie Charts
f, ax = plt.subplots(figsize=(30, 30))
plt.subplot(2, 2, 2)
# Airbags
plt.title("Numero de Airbags")
df_sums = data["AirBags"].value_counts() 
sums = list(df_sums.values)
labels = list(df_sums.index)
plt.pie(sums, labels=labels, autopct='%1.1f%%')
plt.show()


# In[16]:


# Pie Charts
f, ax = plt.subplots(figsize=(30, 30))
plt.subplot(2, 2, 2)
# DriveTrain
plt.title("Tipos de Transmision de Conduccion")
df_sums = data["DriveTrain"].value_counts() 
sums = list(df_sums.values)
labels = list(df_sums.index)
plt.pie(sums, labels=labels, autopct='%1.1f%%')
plt.show()


# In[17]:


# Pie Charts
f, ax = plt.subplots(figsize=(30, 30))
plt.subplot(2, 2, 2)
# Cylinders
plt.title("Numero de Cilindros")
df_sums = data["Cylinders"].value_counts() 
sums = list(df_sums.values)
labels = list(df_sums.index)
plt.pie(sums, labels=labels, autopct='%1.1f%%')
plt.show()


# In[18]:


# Pie Charts
f, ax = plt.subplots(figsize=(30, 30))
plt.subplot(2, 2, 2)
# Man.trans.avail 
plt.title("Transmision Manual Si vs No")
df_sums = data["Man.trans.avail"].value_counts() 
sums = list(df_sums.values)
labels = list(df_sums.index)
plt.pie(sums, labels=labels, autopct='%1.1f%%')
plt.show()


# In[19]:


# Pie Charts
f, ax = plt.subplots(figsize=(30, 30))
plt.subplot(2, 2, 2)
# Origin 
plt.title("Procedencia del Fabricante")
df_sums = data["Origin"].value_counts() 
sums = list(df_sums.values)
labels = list(df_sums.index)
plt.pie(sums, labels=labels, autopct='%1.1f%%')
plt.show()


# In[20]:


# Pie Charts
f, ax = plt.subplots(figsize=(50, 60))
plt.subplot(2, 2, 2)
# Make 
plt.title("Marca y Modelo de Auto")
df_sums = data["Make"].value_counts() 
sums = list(df_sums.values)
labels = list(df_sums.index)
plt.pie(sums, labels=labels, autopct='%1.1f%%')
plt.show()


# #### Analisis General de los datos:
# 
# Estas primeras visualizaciones ayudan a ver ya a priori que los datos no se distibuyen de forma homogénea sino que cada grupo de datos tiene una mayor presencia frente a los otros.
# 
#    * En cuanto al fabricante de autos se ve en la primera visualizacion que hay mas autos provedientes de Chevrolet y Ford, con 9% cada uno, logrando un 18% del total, comparados con los otros que fluctuan entre 1.1% y 6.7%, lo que indica que para este dataset se tomaron mas modelos de Chevrolet y Ford para la obtencion de sus datos, mientras que de otras marcas solo se tomo un solo modelo para comparar. 
#    * Los modelos de autos solo confirman la afirmacion anterior, ya que hay modelos unicos, pero se da la relacion de que de un fabricante se puede tener mas de un modelo. 
#    * El tipo de vehiculo predominante es midSize y Small, respectivamente, juntos se llevan casi un 50% del total de las observaciones. 
#    * La mayoria de los vehiculos presentes solo cuentan con una sola bolsa de aire en el asiento del conductor (casi un 45%) mientras que casi un 37% no cuentan con bolsas de aire y solo poco mas de 18% cuenta con bolsa de aire en el asiento del conductor y del pasajero. 
#    * Casi un 71% de los modelos de autos analizados tienen un tipo de transmision delantera, un 17.4% cuenta con traccion trasera y los 4X4 solo representan un 11.6%.
#    * El numero de cilindros con que mas cuentan estos autos es con 4 (51.1%) y 34.1% tiene 6 cilindros. Un porcentaje muy bajo tiene un cilindraje rotario (1.1%) y un 2.3% cuenta con 5 cilindros. 
#    * Casi un 65% de los autos analizados cuentan con transmision manual. 
#    * Un 52.3% de los autos analizados provienen de fabricantes Estadounidenses. 
#    * En el ultimo grafico se puede observar como cada modelo es representado por la marca, que como se menciono al inicio, una marca puede tener mas de un modelo dentro del Dataset, por ejemplo Chevrolet tiene el modelo Caprice, Camaro, Lumina_APV, etc. 

# #### Analisis descriptivo especifico

# In[74]:


from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (20, 20)
data.plot(x='Make', y='Max.Price', kind='bar', color = 'grey')
plt.title("Grafico 1: Precios maximos por Modelos")
plt.ylabel("Precios Maximos")
plt.xlabel("Modelos de Vehiculo")
plt.show()


# In[75]:


from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (20, 20)
data.plot(x='Make', y='Min.Price', kind='bar', color = 'red') 
plt.title("Grafico 2: Precios minimos por Modelos")
plt.ylabel("Precios Minimos")
plt.xlabel("Modelos de Vehiculo")
plt.show()


# In[76]:


from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (20, 20)
data.plot(x='Make', y='Price', kind='bar') 
plt.title("Grafico 3: Precios estandar por Modelos")
plt.ylabel("Precios estandar")
plt.xlabel("Modelos de Vehiculo")
plt.show()


# In[77]:


from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (20, 20)
data.plot(x='Make', y='EngineSize', kind='bar', color = 'purple') 
plt.title("Grafico 4: Tamano del motor por Modelos")
plt.ylabel("tamano del motor")
plt.xlabel("Modelos de Vehiculo")
plt.show()


# In[78]:


from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (20, 20)
data.plot(x='Make', y='Length', kind='bar', color = 'navy') 
plt.title("Grafico 5: Tamano por cada modelo")
plt.ylabel("Largo")
plt.xlabel("Modelos de Vehiculo")
plt.show()


# In[79]:


from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (20, 20)
data.plot(x='Make', y='Weight', kind='bar', color = 'orange') 
plt.title("Grafico 6: Peso por cada modelo")
plt.ylabel("Peso")
plt.xlabel("Modelos de Vehiculo")
plt.show()


# In[80]:


# diagrama de dispersión
disp= data.plot(kind='scatter', x='Luggage.room', y='Length', color = 'red')
plt.title("Grafico 7: Relacion entre el tamano de los vehiculos y su espacio para equipaje")


# In[81]:


# diagrama de dispersión
disp= data.plot(kind='scatter', x='Price', y='Weight', color = 'darkgreen')
plt.title("Grafico 8: Relacion entre el precio estandar de los vehiculos y su peso total")


# In[82]:


# Estadisticas basicas
data.describe()


# #### Analisis descriptivo especifico 
# 
# * En cuanto al Grafico 1: Precios maximos por Modelos, se puede ver como el Mercedes Benz 190 E, es el que tiene el mayor precio, seguido del Audi 100, ambos no Norteamericanos, en cuanto al modelo que presenta el precio mas bajo dentro de esta categoria esta el Ford Festival, de empresa Norteamerica. 
# * Grafico 2: Precios minimos por Modelos se tiene que el auto que se puede vender con el mayor precio minimo es el Infiniti Q45, seguido del Mercedes Benz 300 E. Es curioso que el Mercedes Benz 190 E, no sea el auto que se pueda vender a mayor precio dentro de la categoria de Minimo Precio, ya que hay otros modelos incluso de marcas Japonesas que tienen precios mayores que el mencionado anteriormente, y esto nos indica, que el Mercedes Benz 190 E, es el vehiculo que se puede vender mas caro, pero en precios minimos termina siendo mas asequible que otros modelos como Audi, Lexus, Chevrolet. 
# * Grafico 3: Precios estandar por Modelos, aqui se puede observar que el auto que es mas caro, dentro de la categoria de precio estandar, es el Mercedes Benz 300 E, seguido por el Infiniti Q45, si se compara, estos dos vehiculos mantienen bastante bien su precio incluso si se llegara a vender por debajo del valor estandar, ya que ocupan el primer y segundo lugar, no asi con el Mercedes Benz 90 E, que nuevamente tiene un precio estandar bajo. En cuanto al Toyota Camry y Saturn SL no se cuenta con el precio de los vehiculos. Y el auto que se puede vender a menor precio dentro de la categoria de precio estandar es el Ford Festival que tambien va de la linea con el primer analisis del grafico 1, ya que representaba en precio mas bajo dentro de la categoria de mayores precios. 
# * Grafico 4: Tamano del motor por Modelos, aqui hay un empate entre el Chevrolet Corvette y el Buick RoadMaster, ya que son los modelos que tienen el motor mas grande, para el Audi 100 y el Pontiac Firebird no se tienen las especificaciones del tamano del motor. El auto que tiene el motor mas pequeno es el Geo metro. Si analizamos el Mercedes Benz 190 E, este tiene un motor relativamente promedio y su hermano, el Mercedes Benz 300 E, tiene un motor un poco mas grande y por ultimo el Ford Festival, tiene el penultimo motor mas pequeno, dentro de los analizados. 
# * Grafico 5: Tamano por cada modelo, el carro que es mas grande (longitud) es el Lincoln Town Car, seguido el Buick Roadmaster, es curioso que el Lincoln Town Car sea un carro tan largo, pero que su motor no sea tan grande como es el caso del Buick Roadmaster. El auto mas pequeno es el Ford Festival. No se tiene detalle de la longitud del vehiculo para los siguientes modelos: Chevrolet Lumina, Mercedes Benz 300 E, Oldsmobile Silhoutte y para el Saab 900.  
# * Grafico 6: Peso por cada modelo, se puede observar como los carros mas pesados son el Nissan Quest y el Buick Roadmaster. Con respecto al Buick Roadmaster se puede dar una ligera correlacion (vamos a verlo adelante) en si el tamano del motor influye en el peso, pero para el caso del Nissan, este no tiene uno de los motores mas grandes, de hecho es un motor de tamano promedio, pero es un carro pesado. El carro menos pesado de todos es el Geo Metro. No se tiene informacion para los modelos: Buick Century, Ford Mustang, Honda Civic, Honda RX-
# * *IMPORTANTE* La categoria NaN, en todos los graficos analizados se mantuvo dentro del promedio general del grafico, solo en el grafico 6, este no tenia observaciones asociadas. 
# * Grafico 7: Relacion entre el tamano de los vehiculos y su espacio para equipaje, con el scatter plot se puede ver como en la medida en que aumenta el tamano del vehiculo aumenta su espacio de equipaje, aunque hay unos outliers que se veran en el box plot, pero se mantiene una relacion lineal creciente, es decir, estas dos variables son dependientes una de la otra. 
# * Grafico 8: Relacion entre el precio estandar de los vehiculos y su peso total, al igual que en el grafico 7, se ve una relacion lineal dependiente, en la medida en que el precio estandar aumenta el peso del vehiculo tambien aumenta, sin embargo, esta es una relacion mas exponencial que la anterior (con algunos outliers). 
# 
# Finalmente se pueden ver las estadisticas basicas, ahi se puede ver que el total de las observaciones es de 93, sin embargo no en todas estan los datos completos, hay muchos datos que estan representados como NaN, la caracteristica que tiene mas datos "missing" es Luggage.room. Por otro lado se puede ver que la media varia dependiendo de la columna a analizar, ya que pareciera como en algunas columnas se han normalizado las cifras y en otras no, ya que hay saltos de 13 a 188, 3000 y 5000, esto se debe a que se estan combinando variables con distintas medidas, ya que no es lo mismo tener la cifra de un Price por ejemplo, que la longitud de un vehiculo, ambas tienen medidas distintas y en caso de utilizar un modelo de Machine Learning, se debe centrar y normalizar la tabla para que estos valores no afecten los analisis predictivos. *Importante*: En este apartado no se centro ni se normalizo la tabla porque de ser asi muchos variables perderian muchos de los datos iniciales y no se tendria un panorama claro de como proceder ante un eventual modelo. Con lo que respecta a las demas estadisticas como la desviacion estandar, minimos, maximos y cuartiles estan modelados por sus diferentes rangos y se podra ver en detalle con los boxplots. 

# #### Analisis mas enfocado en un proceso de Mineria de Datos

# In[92]:


# Para crear una tabla de correlaciones solo con las variable
obj_df = data.select_dtypes(include=['object']).copy() # se eligen las variables categoricas (object)
print(obj_df.columns)


# In[93]:


# Hay que transformar a numéricas las variables categóricas para poder trabajar con ellas

# Convierte las variables de object a categórica
data['Manufacturer'] = data['Manufacturer'].astype('category')
data['Model'] = data['Model'].astype('category')
data['Type'] = data['Type'].astype('category')
data['AirBags'] = data['AirBags'].astype('category')
data['DriveTrain'] = data['DriveTrain'].astype('category')
data['Cylinders'] = data['Cylinders'].astype('category')
data['Man.trans.avail'] = data['Man.trans.avail'].astype('category')
data['Origin'] = data['Origin'].astype('category')
data['Manufacturer'] = data['Manufacturer'].astype('category')
data['Make'] = data['Make'].astype('category')

# Recodifica las categorías usando números
data["Manufacturer"] = data["Manufacturer"].cat.codes
data["Model"] = data["Model"].cat.codes
data["Type"] = data["Type"].cat.codes
data["AirBags"] = data["AirBags"].cat.codes
data["DriveTrain"] = data["DriveTrain"].cat.codes
data["Cylinders"] = data["Cylinders"].cat.codes
data["Man.trans.avail"] = data["Man.trans.avail"].cat.codes
data["Origin"] = data["Origin"].cat.codes
data["Make"] = data["Make"].cat.codes


# In[94]:


# Ahora todas las variables son numericas
print(data.info())
# Ver si los datos estan completos o hay algún NaN 
print(data.isnull().sum())


# In[95]:


# Reemplazando NaN por las medias de las columnas respectivas
# Calculando la media de la columna 
uno = data.iloc[:,4:5].mean()
dos = data.iloc[:,6:7].mean()
tres = data.iloc[:,7:8].mean()
cuatro = data.iloc[:,11:12].mean()
cinco = data.iloc[:,12:13].mean()
seis = data.iloc[:,13:14].mean()
siete = data.iloc[:,14:15].mean()
ocho = data.iloc[:,16:17].mean()
nueve = data.iloc[:,17:18].mean()
diez = data.iloc[:,18:19].mean()
once = data.iloc[:,19:20].mean()
doce = data.iloc[:,20:21].mean()
trece = data.iloc[:,21:22].mean()
catorce = data.iloc[:,22:23].mean()
quince = data.iloc[:,23:24].mean()
deciseis = data.iloc[:,24:25].mean()


# In[96]:


# Reemplazando en todas las columnas con NaN
data['Price'] = data['Price'].replace(np.nan, float(uno))
data['MPG.city'] = data['MPG.city'].replace(np.nan, float(dos))
data['MPG.highway'] = data['MPG.highway'].replace(np.nan, float(tres))
data['EngineSize'] = data['EngineSize'].replace(np.nan, float(cuatro))
data['Horsepower'] = data['Horsepower'].replace(np.nan, float(cinco))
data['RPM'] = data['RPM'].replace(np.nan, float(seis))
data['Rev.per.mile'] = data['Rev.per.mile'].replace(np.nan, float(siete))
data['Fuel.tank.capacity'] = data['Fuel.tank.capacity'].replace(np.nan, float(ocho))
data['Passengers'] = data['Passengers'].replace(np.nan, float(nueve))
data['Length'] = data['Length'].replace(np.nan, float(diez))
data['Wheelbase'] = data['Wheelbase'].replace(np.nan, float(once))
data['Width'] = data['Width'].replace(np.nan, float(doce))
data['Turn.circle'] = data['Turn.circle'].replace(np.nan, float(trece))
data['Rear.seat.room'] = data['Rear.seat.room'].replace(np.nan, float(catorce))
data['Luggage.room'] = data['Luggage.room'].replace(np.nan, float(quince))
data['Weight'] = data['Weight'].replace(np.nan, float(deciseis))


# In[97]:


# Ver si los datos ahora estan completos
print(data.isnull().sum())


# In[98]:


# Descripcion estadística de los datos
data.describe()


# In[99]:


# Correlación entre variables
f,ax = plt.subplots(figsize=(20,20)) 
sns.heatmap(data.corr(method='spearman'),annot=True,fmt=".1f",linewidths=1,ax=ax)
plt.show()


# #### Analisis
# 
# El análisis de correlaciones entre variables es un punto fundamental para entender las variables presentes en el conjunto de datos.
# 
# * Hay una correlacion positiva de 0.9 entre las variables MinPrice y Price, esto indica que si una de las dos aumenta, la otra tambien aumentara postitivamente, se explica porque son variables que representan valor monetario, indudablemente si el precio de mercado de un vehiculo sube, el precio minimo tambien lo hara. 
# * Hay una correlacion positiva (0.8) de igual forma entre las variables MaxPrice y Price, por lo explicado en el punto anterior, era de esperarse que fueran homologas.
# * Hay una correlacion negativa entre MPG-City y MinPrice de -0.7, esto indica que entre mas bajo sea el precio minimo de venta de los vehiculos, el promedio de millas por galon que consume un vehiculo en la ciudad sera mayor y viceversa, esta misma correlacion negativa se da con MPG-highway pudiendose interpretar lo mismo, pero para el caso del promedio de millas por galon pero en carretera. 
# * MinPrice tambien presenta una correlacion positiva de 0.5 con RevPerMile, esto indica que si el precio minimo de venta de un vehiculo disminuye, la cantidad de revoluciones por milla tambien lo hara y viceversa. 
# * MinPrice tambien presenta correlacion positiva con Lenght de 0.5 dandose el caso de que si el precio minimo disminuye, las dimensiones del vehiculo tambien lo haran, es decir, a un veehiculo con precio menor mas pequenas sus dimensiones y esto mismo pasa con otras variables como Passengers, distancia entre ejes (wheelbase), width (anchura del vehiculo), con el diametro de giro (turn circle), espacio entre asientos (rear seat room), peso, y luggage room. Todas estas variables presentan correlaciones positivas, donde si una aumenta las demas lo haran y estas han sido modeladas por la variable MinPrice, que denota que entre mas economico un vehiculo menos dimensiones y por ende menos espacio tendran las personas. 
# * Con la variable Precio sucede exactamente lo mismo que conla variable MinPrice, por un lado entre mas bajo sea el precio de venta de un auto, el promedio de millas por galon que consume umn vehiculo en la ciudad asi como en laa carretera sera mayor, al iguaal que con el temaa de dimensiones, como se explica en el punto anterior. 
# * MaxPrice tambien contiene las mismas correlaciones antes tratadas, en este caso se hace la salvedad que entre mas costoso sea un vehiculo mayores dimensiones tendra, y por ende mas espacio. 
# * MPG-City presenta correlaciones negativas con Min.Price, Price y Max.Price, es decir, entre mayor sea el promedio de millas por galon, menor sera el precio y viceversa. 
# * Se tiene una fuerte correlacion positiva entre MPG-City y MPG-Highway de 0.9 lo que indica que entre mas promedio de millas por galon enla ciudad, tambien sera un promedio de consumo mayor en carretera y viceversa. 
# * MPG-city presenta correlaciones negativas con Cylinders, EngineSize, y Horsepower lo que indica que entre a mas promedio de millas por galon menor sera laa cantidad de cilindros, el tamano del motor sera menor y la cantidad de caballos de fuerza tambien lo seran, a su vez, la variable MPG-City presenta correlaciones negativas y fuertes con Fuel-tank-capacity, Passengers, length, wheelbase, width, turn circle, rear seat room, luggage room y weight. Si una de las anteriores mencionadas disminuye la cantidad promedio de millas aumentara y viceversa. Es importante aclarar que la variable MPG-Highway se comporta exactamente igual con respecto a las variables analizadas. 
# * Airbags presenta correlacion negativa con MinPrice, Price y MaxPrice, es decir, si alguna de las variables relacionadas a precio aumenta, la cantidad de airbags disminuye, y viceversa, sin embargo esta variable debe abordarse con cuidado, porque es una correlacion negativa que es impactada por tres dimensiones: "ningun airback", "airback solo en el asiento del conductor", "airback en el asiento del conductor y pasajero", por otro lado, esta variable presentaba varios datos NaN y del total de datos casi un 45% solo contaban con un ariback en el asiento del conductor, tambien depende mucho del tipo de auto que se este usando para la comparacion. 
# 
# *NOTA*: *Es importante mencionar que las variables presentan altos grados de relaciones lineales, es decir, son variables altamente correlatadas tanto positiva como negativamente (dependientes) y son muy pocas las variables totalmente independientes, como es el caso de Model, Type, Drive train. Importante: el coeficiente de correlación de Spearman (utilizado en esta matriz) es una medida de la correlación entre dos variables aleatorias (tanto continuas como discretas).*

# #### Dado lo anterior, es importante utilizar otro metodo para ver las correlaciones que no haga la mezcla entre variables cuantitativas y cualitativas, por ello se usara el metodo Pearson, que es una medida de dependencia lineal entre dos variables aleatorias cuantitativas, se utiliza este metodo, debido a que la correlación de Pearson es independiente de la escala de medida de las variables y puede generar datos mas fieles a la realidad del dataset sin tomar en cuenta conversion categorica. 

# In[100]:


# Una matriz de correlaciones con Pearson
# Se toman las variables numerales originales, sin las ordinales categoricas para ver si con el Criterio de Pearson se sigue dando la alta
# dependencia entre variables 
cols = [x for x in list(data.columns) if x not in list(obj_df.columns) ]
f,ax = plt.subplots(figsize=(20,20))
dataset_mat_aux = data[cols] # columnas que no esten en obj_df 
sns.heatmap(dataset_mat_aux.corr(method='pearson'),annot=True,fmt=".1f",linewidths=1,ax=ax)
plt.show()


# #### Como se sospechaba, las variables cuantitativas presentes en el dataset, son completamente dependientes y no hay ninguna que sea independiente, es decir, todo el analisis que se haga a nivel de estadistica inferencial sera modelado mediante dependencia y eventos dependientes. 

# In[101]:


plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
plt.hist(data['Min.Price'], color = 'red') 
plt.title('Histograma sobre los Precios Minimos de los Vehiculos') 
plt.xlabel('Precio')
plt.ylabel('Cantidad')
plt.show()


# In[102]:


plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
plt.hist(data['Price'], color = 'blue') 
plt.title('Histograma sobre los Precios de los Vehiculos') 
plt.xlabel('Precio')
plt.ylabel('Cantidad')
plt.show()


# In[103]:


plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
plt.hist(data['Max.Price'], color = 'cyan') 
plt.title('Histograma sobre los Precios Maximos de los Vehiculos') 
plt.xlabel('Precio')
plt.ylabel('Cantidad')
plt.show()


# #### Los histogramas relacionados a Precio se puede ver como no tienen una distribucion asimetrica, sino que tienen una tendencia a la izquierda, es decir, a precios entre 10 y 20. 

# In[104]:


plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
plt.hist(data['MPG.city'], color = 'green') 
plt.title('Histograma sobre las Millas por Galon - Ciudad') 
plt.xlabel('Millas por galon - Ciudad')
plt.ylabel('Cantidad')
plt.show()


# In[105]:


plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
plt.hist(data['MPG.highway'], color = 'pink') 
plt.title('Histograma sobre las Millas por Galon - Carretera') 
plt.xlabel('Millas por galon - Carretera')
plt.ylabel('Cantidad')
plt.show()


# #### Los histogramas relacionados a consumo de millas por Galon se puede ver como no tienen una distribucion asimetrica, sino que tienen una tendencia a la izquierda, es decir, a cantidad de bellas entre 20 y 35. 

# In[106]:


plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
plt.hist(data['EngineSize'], color = 'yellow') 
plt.title('Histograma sobre Tamano del Motor') 
plt.xlabel('Tamano del motor')
plt.ylabel('Cantidad')
plt.show()


# In[107]:


plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
plt.hist(data['Horsepower'], color = 'gray') 
plt.title('Histograma sobre Caballos de Potencia del motor') 
plt.xlabel('Caballos de Potencia del motor')
plt.ylabel('Cantidad')
plt.show()


# #### Los histogramas relacionados al tamano del motor y su potencia tampoco tienen una distribucion asimetrica, sino que tienen una tendencia a la izquierda, como se viene presentando en la demas variables. 

# In[108]:


# Hagamos un contraste de variables que no tienen tanta correlacion


# In[109]:


plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
plt.hist(data['Horsepower'], color = 'black') 
plt.title('Histograma sobre Caballos de Potencia del motor') 
plt.xlabel('Caballos de Potencia del motor')
plt.ylabel('Cantidad')
plt.show()


# In[110]:


plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
plt.hist(data['Passengers'], color = 'gold') 
plt.title('Histograma sobre la cantidad de pasajeros por vehiculo') 
plt.xlabel('Pasajeros')
plt.ylabel('Cantidad')
plt.show()


# #### Se puede ver como entre las variables Passengers y HorsePower no existe correlacion lineal, sin embargo, en ambos casos ambos histogramas tienden a ser asimetricos. Passengers por su cuenta presenta mayor cantidad de asientos endtre 5 y 6, pero otras cantidades afectan su normalidad y por ende no se puede ver como una distribucion binomial. 

# In[111]:


# Box-Plots
f, axes = plt.subplots(6, 3, figsize=(17, 25))
sns.boxplot(x= "Min.Price", data=data, orient='v' , ax=axes[0][0], palette= "Reds")
sns.boxplot(x= "Price", data=data, orient='v' , ax=axes[0][1], palette="Blues")
sns.boxplot(x= "Max.Price", data=data, orient='v' , ax=axes[0][2], palette="GnBu_d")
sns.boxplot(x= "MPG.city", data=data, orient='v' , ax=axes[1][0], palette="Purples")
sns.boxplot(x= "MPG.highway", data=data, orient='v' , ax=axes[1][1], palette="Paired")
sns.boxplot(x= "EngineSize", data=data, orient='v' , ax=axes[1][2], palette=sns.cubehelix_palette(8))
sns.boxplot(x= "Horsepower", data=data, orient='v' , ax=axes[2][0], palette="GnBu_d")
sns.boxplot(x= "RPM", data=data, orient='v' , ax=axes[2][1], palette="Blues")
sns.boxplot(x= "Rev.per.mile", data=data, orient='v' , ax=axes[2][2], palette="Reds")
sns.boxplot(x= "Fuel.tank.capacity", data=data, orient='v' , ax=axes[3][0], palette="Purples")
sns.boxplot(x= "Passengers", data=data, orient='v' , ax=axes[3][1], palette="Paired")
sns.boxplot(x= "Length", data=data, orient='v' , ax=axes[3][2], palette="Blues")
sns.boxplot(x= "Wheelbase", data=data, orient='v' , ax=axes[4][0], palette="Reds")
sns.boxplot(x= "Width", data=data, orient='v' , ax=axes[4][1], palette="Purples")
sns.boxplot(x= "Turn.circle", data=data, orient='v' , ax=axes[4][2], palette="Blues")
sns.boxplot(x= "Rear.seat.room", data=data, orient='v' , ax=axes[5][0], palette="Paired")
sns.boxplot(x= "Luggage.room", data=data, orient='v' , ax=axes[5][1], palette="Reds")
sns.boxplot(x= "Weight", data=data, orient='v' , ax=axes[5][2], palette="Blues")


# #### Analisis 
# 
# En cuanto a los box plots, se puede ver como algunas variables presentan outliers y presentan valore extremos, tal es el caso de Min.Price, Price y Max.Price, las tres tienen outliers, lo que hace que su mediana (centro de la caja) tire mas hacia abajo (valores minimos), lo mismo pasa con MPG.City y MPG.Highway. 
# 
# En otros casos se tiene una distribucion de los datos uniforme como es el caso de RPM, Passengers, Width, Rear.Seat.Room, Weight, Turn.Circle, Wheelbase. Generalmente estas se mantienen dentro de valores estandar, porque generalmente esto ya esta muy estandarizado, por ejemplo la cantidad de pasajeros en un vehiculo, el tamano, el ancho, las revoluciones por minuto, etc., entonces es raro encontrar outliers en estos casos, aunque si los hubiese, solo nos da una senal de que es muy probable que el proceso de recoleccion de los datos no se esta haciendo de la forma correcta o que hay errores de imputacion. 
# 
# Los otros box plots, presentan ligeros outliers, pero los parametros estan mejor representados y las cajas no tienen a irse hacia arriba o abajo, evidenciando que los valores extremos afectaron negativamente en el promedio. 
# 

# #### Aplicacion de un modelo no supervisado para ver como se comportan los datos a nivel general y ver si es posible hacer una separacion ya que hay muchas correlaciones

# In[148]:


# Normalizando y centrando la tabla ya que hay valores en diferentes escalas
voices = data.iloc[:,16:25]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_values = scaler.fit_transform(data) 
data.loc[:,:] = scaled_values
data.head()


# In[168]:


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
        X = self.__datos['Price'] 
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


# In[157]:


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


# In[158]:


datos = exploratorio(data)


# In[169]:


datos.test_normalidad() # se hace solo con una variable correlatada


# #### Esta prueba compara la función de distribución acumulada empírica. En este caso como la p = 0.00000053, y es menor que el nivel de significancia (α= 0.05) elegido se rechaza la hipótesis nula y se concluye que se trata de una población no normal, tal y como se viene analizando, ya que es una poblacion bastante correlatada. 

# In[161]:


nuevo = PCA(data)


# In[163]:


nuevo.plot_plano_principal()


# In[164]:


nuevo.plot_circulo()


# In[165]:


nuevo.plot_sobreposicion()


# #### Analisis: 
# 
# * Estos plots son utiles, porque es la forma grafica de ver representada la matriz de correlaciones, como se puede ver en el segundo plot, variables como AirBags, MPG.city y highway, Manufacturer, Rev per mile, Model, Type, Make, estan fuertemente correlacionadas positivamente, es decir si una aumenta las demas tambien lo haran, tambien se puede ver como por ejemplo en el caso contrario Price, Max.Price, Min.Price estan fuertemente correlacionadas pero contario a Airbags por ejemplo, esto indica que el precio de un vehiculo no indice (al menos en este dataset) en que tengan bolsas de aire, lo mismo pasa por ejemplo con Man.Trans.Avail y Luggage Room, no incide en que un carro sea manual o automatico para que tenga mas espacio de almacenamiento, o las revoluciones por minuto (RPM) con la variable Passengers, no afecta que si un carro lleva mas personas tenga menos revoluciones por minuto y asi se pueden seguir encontrando correlaciones, tanto positivas como negativas tomando las flechas que aparecen en la cuadricula y viendo la direccion de los vectores, con ello se podra determinar si son vectores y valores propios (cuando se correlacionan) de cuando no. 
# 
# * Por ultimo, se puede concluir que este dataset presenta una poblacion que no es normal, es altamente correlatada y se puede ver en los puntos en el ultimo grafico de este apartado, se puede ver como los puntos tratan de separarse pero la separacion es muy poco evidente. A simple vista se puede decir que hay 3 clusters, uno modelado por las variables de Airbags, Model Type, Manufacturer, MPG.highway y City (lado derecho superior). 
# 
# * Un segundo del lado izquierdo superior, modelado por las variables Passengers, Rear. Seat.Room, luggage Room, Wheelbase, Width, EngineSize, Fuel.Tank.Capacity, Lenght, etc.
# 
# 
# * El tercero modelado por Rev per mile, Make, Man.Trans.Aveil, RPM, Origin, Drive Train, Horsepower, Price, Min.price, Max.Price, Cilinders (lado inferior izquierdo y derecho). Sin embargo por la distribucion de los datos no se logra ver clusters bien delimitados, porque hay mucha dispersion, muchos outliers, y unos estan muy cerca del origen (punto 0.0) lo que crea que no esten bien representados. 

# #### Conclusiones: 
# 
# 1. A nivel general se logo hacer un analisis general de los datos tomando en cuenta estadisticas y graficos basicos por grandes conglomerados. 
# 
# 2. A nivel especifico se vieron incidencias mas especificas en los datos, como cuales vehiculos eran mas caros, mas baratos, cuales tenian el motor mas grande y saber si eso incidia en el tamano del vehiculo. Aqui se despejo el mito de que la correlacion no implica causalidad, ya que, no porque un auto sea el mas caro (como el ejemplo del Mercedes) y tenga las otras variables correlatadas, significa que va a ser el mas alto en todo como se vio en el ejemplo. 
# 
# 3. Por ultimo se centro en un analisis mas enfocado en mineria de datos y procesos estadisticos avanzados en el que se estudio con detalle la relacion de las variables y como estas se comportaban siguiendo el flujo de los datos, se evidencio que la poblacion en estudio no sigue un parametro de normalidad y si a futuro se quisiese hacer algun modelo supervisado tendria que tomarse otras variables diferentes para eliminar posibles sesgos por correlacion excesiva entre los datos. 

# In[ ]:




