# 1

a = 0
b = 0

def comparar (a,b):
    if a > b:
        comparacion = "El numero mayor es el primer numero. Numero: " + str(a)
    else:
        comparacion = "El numero mayor es el segundo numero. Numero: " + str(b)
    return comparacion

comparar(2,90)


# 2

a = 0
b = 0
c = 0

def comprobar (a,b,c):

    if b < a > c:
        comparacion = "El numero mayor es el primer numero. Numero: "+ str(a)
    elif a < b > c:
        comparacion = "El numero mayor es el segundo numero. Numero: "  +str(b)
    else:
        comparacion = "El numero mayor es el tercer numero. Numero: " + str(c)
    # Aqui no vi la necesidad de hacer la condicion del n3, porque se
    # se sobreentiende que si no esta en alguno de los rangos puestos es el mayor
    return(comparacion)

comprobar(986,986,987)

# 3


import math

def multiplicacion (a,b,c):
  if type(a) == int and type(b) == int and type(c) == int:
    d = math.fabs(a) * math.fabs (b) * math.fabs (c)
    return(d)
  else:
    return 0

multiplicacion(1,5,4) 

# 4


numero = 0

def acumular(numero):
    if numero <= 0:
        print(str(0))
    else:
        suma = 0
        for i in range(1, numero + 1):
            suma = suma + (i**3)
        print(f"La suma del número que ha escrito es {suma}")


acumular(7)



#5

n=0

def multiplos(n):
    sumatoria = 0
    
    for i in range (1, n + 1):
            if i % 3 == 0:
                sumatoria = sumatoria + i
    return(sumatoria)
    
    
multiplos(21)

# 6


t=0

def costo(t):
    if t < 5:
        costo = 0.4
    else:
        costo = 0.4 + (t-1)/4
    return(costo)
    
costo(5)


# 7

def porcentaje_menores(numeros, x):
    menores = 0
    for numero in numeros:
        if numero < x:
            menores += 1
    porcentaje =  menores / len(numeros) * 100
    return porcentaje

porcentaje_menores([3.5, 7, 9, 3.7, 4.5, 2, 6, 2.8], 5)



#8

import math

n = 4
c = math.pi

def vector(n):
    if n == 1:
        return [2]
    elif n > 1:
        v = vector(n-1)
        v.append(v[-1]/3 + c)
        return v

vector(4)

# 9

import numpy as np 
lista  = np.array([9,3,4,1,3,-1,4,12,-2])
matriz = lista.reshape(3,3)
matriz 


def calculo(matriz):
    traza = 1
    
    for i in range (len(matriz)):
        for a in range(len(matriz)):
            if i==a:
                traza=traza+matriz[i,a]
                return(traza)
            
                
calculo(matriz)


# 10

import os 
os.getcwd()

import pandas as pd

datos_est = pd.read_csv('ejemplo_estudiantes.csv', delimiter = ';', decimal = ",", header=0, index_col = 0)


def valores(DF):
    n, m = DF.shape
    total_divisibles= 0
    for i in range(n):
        for j in range(m):
            if DF.iloc[i,j] % 2 == 0:
                total_divisibles = total_divisibles+1
                
    respuesta = {'Total_divisibles_entre_2' : total_divisibles}
    return respuesta

valores(datos_est)

# 11


import numpy as np

df = pd.read_csv('ejemplo_estudiantes.csv', delimiter = ';', decimal = ",", header=0, index_col = 0)

df.Espanol.value_counts()
df.Matematicas.value_counts()


y = df.iloc[:,0]
x = df.iloc[:,2]

def estadisticas (x,y):
    covarianza = np.cov(x,y)
    correlacion = np.corrcoef(x,y)
    return {'Variable_1'  : "Matematicas",
            'Variable_2' :  "Espanol",
            'Correlacion' : np.corrcoef(x,y),
            'Covarianza' :  np.cov(x,y)}
    
    
estadisticas(x,y)







































