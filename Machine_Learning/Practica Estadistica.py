#!/usr/bin/env python
# coding: utf-8

# # Tarea Número 1

# # Ejercicio 1 - A

# In[3]:


# 1
import math as mt
r=mt.pi*5**4-mt.sqrt(9)
print(r)


# In[4]:


# 2
r=12-17*(2/7)-9
mt.fabs(r)


# In[5]:


# 3
r=mt.factorial(7)
print(r)


# In[6]:


# 4
mt.log(19,5)


# In[7]:


# 5 
mt.log(5,10)


# In[8]:


# 6
mt.e**0.555457


# # Ejercicio 1 - B

# In[9]:


y=mt.pi
z=4
x=(1+y)/(1+2*z**2)
print(x)


# # Ejercico 1 - C

# In[10]:


x=-90
y=mt.pi
z=mt.sqrt(x**2+y**2)
print(z)


# # Ejercicio 2

# In[11]:


#Punto #1

x=[1,-5,31,-1,-9,-1,0,18,90,mt.pi]
y=[1,9,-3,1,-99,-10,2,-11,0,2]
print(x)
print(y)


# In[13]:


#Punto #2

import statistics as st

st.mean(x) 
st.pvariance(x) 
st.pstdev(x) 


# In[14]:


#Punto #3

st.mean(y) 
st.pvariance(y) 
st.pstdev(y) 


# In[15]:


#Punto #4

import numpy as np 
print("El coeficiente de correlación entre x y y es:",np.corrcoef(x,y)[0,1])


# In[16]:


#Punto #5

x[2:7]


# In[17]:


# Punto 6

y[2:7]


# In[18]:


# Punto 7

y[:-4:-1]


# In[19]:


# Punto 8 

print(x[:-11:-1])


# # Ejercicio 3

# In[20]:


import pandas as pd

datos = {'Genero': ["M","F","F","F","M","F"],
         'Peso': [76,67,55,57,87,48],
         'Edad': [25,23,19,18,56,13],
         'Nivel Educativo': ["Lic","Bach","Bach","Bach","Dr","MSc"]}


mi_df = pd.DataFrame(datos)
print(mi_df)


# # Ejercicio 4 

# In[21]:


variables  = {'id': range(1, 11),
                'Calificación': ["A","A","A","A","B","B","B","B","C","C"],
                'Tiempo': [64,85,76,83,81,78,68,82,89,62]}

variables = pd.DataFrame(variables)
print(variables)
print(variables.info())


# # Ejercicio 5

# In[22]:


# Punto 1

x=[24,28,29,18,95,97,90,72,87,85,74,9,40]
print(x)

x=[24,28,29,18,95,97,90,72,87,85,74,9,40]
lista_division=[]

for numero in x:
    division = numero/2
    lista_division.append(division)

print(lista_division) #aqui vemos que el unico que el residuo es 45 es el indice 6

print(x[6])


# In[23]:


# Punto 2
print(max(x))


# In[24]:


# Punto 3 
x=[24,28,29,18,95,97,90,72,87,85,74,9,40]
sumatoria= 0

for numero in x:
    sumatoria = sumatoria + numero 
    
print(sumatoria)


# In[25]:


# Punto 4

lista_cubos = []
x=[24,28,29,18,95,97,90,72,87,85,74,9,40]

for numero in x:
    cubo = numero ** 3
    lista_cubos.append(cubo)
    
print(lista_cubos)


# # Ejercicio 6 

# In[26]:


v1=[2,7,6,4,52,-2]
v2=[7,5,7,0,1,0]
v3=[2,4,3,5,6,mt.pi]

print(sum(v1))
print(sum(v2))
print(sum(v3))

### usando for 

v1=[2,7,6,4,52,-2]
sumatoria = 0

for numero in v1:
    sumatoria = sumatoria + numero 
    
print(sumatoria)

v2=[7,5,7,0,1,0]
sumatoria = 0

for numero in v2:
    sumatoria = sumatoria + numero
    
print(sumatoria)

v3=[2,4,3,5,6,mt.pi]
sumatoria = 0

for numero in v3:
    sumatoria = sumatoria + numero
    
print(sumatoria)


# # Ejercicio 7

# In[27]:


x=[24,28,29,18,95,97,90,72,87,85,74,9,40,91,87,92,-3]

import statistics as st

resumen = {"media" : st.mean(x), "moda" : st.mode(x), "maximo" : max(x), "minimo" : min(x)}
print(resumen)


# # Ejercicio 8 

# In[28]:


import numpy as np 

z=np.matrix([[9,3,4],[1,3,-1]])
o=np.matrix([[91,-3],[1,8],[-4,5]])
n=o.T
q=31

A=z+q*n
print(A)


# # Ejercicio 9 

# In[29]:


import os
import pandas as pd

direccion_actual = os.getcwd()
direccion_actual
os.chdir("/Users/heinerleivagmail.com/Machine")
os.getcwd()

datos1 = pd.read_csv('EjemploAlgoritmosRecomendacion.csv', delimiter = ';', 
                     decimal = ",", header = 0, index_col =  0)

datos1

# Punto 1

print(datos1.shape)


# In[30]:


# Punto 2

print(datos1[["VelocidadEntrega","Precio","Durabilidad"]])


# In[31]:


# Punto 3

print(datos1.iloc[:,0:3])


# In[32]:


# Punto 4 

print(datos1.loc[:,["VelocidadEntrega","Precio","Durabilidad"]])


# In[33]:


# Punto 5

print(datos1.info())


# In[34]:


# Punto 6

print(datos1.mean())


# # Ejercicio 10 

# In[35]:


datos2 = pd.read_csv('SAheart.csv', delimiter = ';', 
                     decimal = ",", header = 0, index_col =  0)
datos2

# Punto 1

print(datos2.shape)


# In[36]:


# Punto 2 

print(datos2[["tobacco","ldl","adiposity"]])


# In[37]:


# Punto 3

print(datos2.iloc[:,0:4])


# In[38]:


# Punto 4 

print(datos2.loc[:,["tobacco","ldl","adiposity"]])


# In[39]:


# Punto 5

print(datos2.info())


# In[40]:


# Punto 6 

print(datos2.sum())


# # Ejercicio 11

# In[41]:


n=100
sumatoria = 0

for valor in range (1+n):
        sumatoria+=valor**3
        
print(sumatoria)


# # Ejercicio 12

# In[42]:


n=100
sumatoria = 0

for numero in range(1+n):
    sumatoria+=numero 
print(sumatoria)


# # Ejercicio 13

# In[43]:


n=20
sumatoria = 0

for numero in range(1+n):
    if numero%2==0:
        sumatoria = sumatoria + numero
print(sumatoria)

2+4+6+8+10+12+14+16+18+20


# # Ejercicio 14

# In[44]:


n=50
sumatoria = 0

for valor in range(1+n):
    if valor%5==0:
        sumatoria = sumatoria + valor
print(sumatoria)

5+10+15+20+25+30+35+40+45+50


# # Ejercicio 15

# In[45]:


import numpy as np
lista=np.array([9,3,4,1,0,-1,4,12,-2])
matriz=lista.reshape(3,3)
matriz 

traza = 0

for columna in range (len(matriz)):
    for fila in range (len(matriz)):
        if columna == fila:
            traza=traza+matriz[columna, fila]
            
print("La traza es: " + str(traza))


# # Ejercicio 16

# In[46]:


import numpy as np
lista=np.array([1,1,1,1,1,9,9,1,1,9,9,1,1,1,1,1])
matriz=lista.reshape(4,4)
matriz

q=matriz[1,1]
r=matriz[1,2]
s=matriz[2,1]
t=matriz[2,2]

listasub=np.array([q,r,s,t])
submatriz=listasub.reshape(2,2)
print(submatriz)


# In[ ]:




