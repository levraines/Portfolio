# 1


import math

class numeros:
    def __init__(self, a = 0, b = 0, c=0):
        self.__a = a
        self.__b = b
        self.__c = c
    @property  
    def a(self):
        return str(self.__a)
    @property
    def b(self):
        return str(self.__b)
    @property
    def c(self):
        return str(self.__c)
    @a.setter
    def a(self, nuevo_a):
        self.__a  = nuevo_a
    @b.setter
    def b(self, nuevo_b):
        self.__b = nuevo_b
    @c.setter
    def c(self, nuevo_c):
        self.__c = nuevo_c
    @property
    def obtener_menor_cuadrado(numeros):
        if b > a < c: 
            obtener = "El numero menor es el numero A. Numero: " + str(a**2)
        elif a > b < c:
            obtener = "El numero menor es el numero B. Numero: " + str(b**2)
        else: 
            obtener = "El numero menor es el numero C. Numero: " + str(c**2)
        return obtener
    @property
    def obtener_mayor_cubo(numeros):
        if b < a > c:
            obtener = "El numero mayor es el numero A. Numero: " + str(a**3)
        elif a < b > c: 
            obtener ="El numero mayor es el numero B. Numero: " + str(b**3)
        else:
            obtener = "El numero mayor es el numero C. Numero:" + str(c**3)
        return obtener
    @property
    def producto(self):
        return (a*b*c)
    @property
    def suma_cosenos(self):
        return math.cos(a) + math.cos(b) + math.cos(c)
    def __str__(self):
        return "Numeros originales de calculo:  (%i, %i, %i)" % (a, b, c)
            
a = 67
b = 90
c = 5

instancia = numeros()

instancia.producto
instancia.obtener_menor_cuadrado
instancia.obtener_mayor_cubo
instancia.suma_cosenos
instancia.__str__()











# 2

    
class Vuelo:
    def __init__(self, numero, hora_salida, hora_llegada):
        self.__numero = numero
        self.__hora_salida = hora_salida
        self.__hora_llegada = hora_llegada
    @property
    def numero(self):
        return (self.__numero)
    @property
    def hora_salida(self):
        return (self.__hora_salida)
    @property
    def hora_llegada(self):
        return (self.__hora_llegada)
    @numero.setter
    def numero(self, nuevo):
        self.__numero = nuevo
    @hora_llegada.setter
    def hora_llegada(self, nuevo):
        self.__hora_llegada = nuevo
    @hora_salida.setter
    def hora_salida(self, nuevo):
        self.__hora_salida = nuevo
    def __str__(self):
        print("Datos originales: Numero, Hora de Salida, Hora de LLegada ")
        return self.__numero, self.__hora_salida, self.__hora_llegada
              
    
    
prueba=Vuelo('907',13,8)

prueba.hora_llegada
prueba.numero
prueba.hora_salida
prueba.__str__()





class Pasajero:
    def __init__(self, precioTiquete, codigo, nombre, porcentaje_impuesto, descuento):
        self.__precioTiquete = precioTiquete
        self.__codigo = codigo
        self.__nombre = nombre
        self.porcentaje_impuesto = porcentaje_impuesto
        self.descuento = descuento
    @property
    def precioTiquete(self):
        return self.__precioTiquete
    @property
    def codigo(self):
        return self.__codigo
    @property
    def nombre(self):
        return self.__nombre
    @property 
    def porcentaje_impuesto(self):
        return self.__porcentaje_impuesto
    @property 
    def descuento(self):
        return self.__descuento
    @precioTiquete.setter
    def precioTiquete(self, nuevo):
        self.__precioTiquete = nuevo
    @codigo.setter
    def codigo(self, nuevo):
        self.__codigo = nuevo
    @nombre.setter
    def nombre(self, nuevo):
        self.__nombre = nuevo
    @porcentaje_impuesto.setter
    def porcentaje_impuesto(self, nuevo):
        if nuevo > 1:
            nuevo = nuevo / 100
        self.__porcentaje_impuesto = nuevo
    @descuento.setter
    def descuento(self, nuevo):
        if nuevo > 1:
            nuevo = nuevo / 100
        self.__descuento = nuevo    
    def total_pagar(self):
        precio = (self.precioTiquete + self.porcentaje_impuesto * self.precioTiquete)
        precio = precio * (1 - self.descuento)
        return precio
    def __str__(self):
        s = f"Pasajero:{self.nombre}[{self.codigo}]\nPrecio del Tiquete:{self.precioTiquete}"
        s = s + f"\nImpuesto:{self.porcentaje_impuesto*100}%\nDescuento:{self.descuento}\nTotal:{self.total_pagar()}"
        return s
    
val = Pasajero(789.90, 'AVX984', 'Katherine', 9.87, 20)
val.precioTiquete
val.codigo
val.nombre
val.porcentaje_impuesto
val.descuento
val.total_pagar()
val.__str__()




class Pasajero_Frecuente(Pasajero):
    def __init__(self, precioTiquete, codigo, nombre, porcentaje_impuesto, descuento, cant_puntos):
        super().__init__(precioTiquete, codigo, nombre, porcentaje_impuesto, descuento)
        self.__cant_puntos = cant_puntos
    @property
    def cant_puntos(self):
        return self.__cant_puntos
    @cant_puntos.setter
    def cant_puntos(self, nuevo):
        self.__cant_puntos = nuevo
    def total_pagar_frecuente(self):
        return (float(self.precioTiquete) + float(self.porcentaje_impuesto) * float(self.precioTiquete)) * - float(self.descuento) + (float(self.precioTiquete) + float(self.porcentaje_impuesto) * float(self.precioTiquete))
    def __str__(self):
        a = f"Pasajero:{self.nombre}[{self.codigo}]\nPrecio del Tiquete:{self.precioTiquete}"
        a = a + f"\nImpuesto:{self.porcentaje_impuesto*100}%\nDescuento:{self.descuento}\nTotal:{self.total_pagar_frecuente()}"
        return a
        
novo=Pasajero_Frecuente(899.8, 'COL84', 'Fabiana', 11.4, 20, 459)

novo.codigo
novo.nombre
novo.precioTiquete
novo.porcentaje_impuesto
novo.descuento
novo.cant_puntos
novo.total_pagar_frecuente()
novo.__str__()


    
class Pasajero_no_frecuente (Pasajero):
    def __init__(self, precioTiquete, codigo, nombre, porcentaje_impuesto, descuento, primer_vuelo):
        super().__init__(precioTiquete, codigo, nombre, porcentaje_impuesto, descuento)
        self.__primer_vuelo = primer_vuelo
    @property
    def primer_vuelo(self):
        return self.__primer_vuelo
    @primer_vuelo.setter
    def primer_vuelo(self, nuevo):
        self.__primer_vuelo = nuevo
    def total_pagar_nofrecuente(self):
        return (float(self.precioTiquete) + float(self.porcentaje_impuesto) * float(self.precioTiquete)) * - float(self.descuento) + (float(self.precioTiquete) + float(self.porcentaje_impuesto) * float(self.precioTiquete))
    def __str__(self):
        c = f"Pasajero:{self.nombre}[{self.codigo}]\nPrecio del Tiquete:{self.precioTiquete}"
        c = c + f"\nImpuesto:{self.porcentaje_impuesto*100}%\nDescuento:{self.descuento}\nTotal:{self.total_pagar_nofrecuente()}"
        return c


cal=Pasajero_no_frecuente(1099, 'AVX959', 'Jose', 7.89, 5, True)

cal.codigo
cal.nombre
cal.precioTiquete
cal.porcentaje_impuesto
cal.descuento
cal.primer_vuelo
cal.total_pagar_nofrecuente()
cal.__str__()  



class vuelo_carga(Vuelo):
    def __init__(self, numero, hora_salida, hora_llegada, carga):
        super().__init__(numero, hora_salida, hora_llegada)
        self.__carga = carga
    @property
    def carga (self):
        return self.__carga
    @carga.setter
    def carga (self, nuevo):
        self.__carga = nuevo
    def __str__(self):
        return "La carga de este vuelo es de: " + str(self.carga)
    
xos=vuelo_carga('AVH7',5,9, 8956.8)

xos.hora_llegada
xos.numero
xos.hora_salida
xos.carga
xos.__str__()
    
      

class VueloComercial(Vuelo):
    def __init__(self, numero, hora_salida, hora_llegada, lista_pasajeros):
        super().__init__(numero, hora_salida, hora_llegada)
        if lista_pasajeros == None:
            self.__lista_pasajeros = []
        else:
            self.__lista_pasajeros = lista_pasajeros
    @property
    def lista_pasajeros(self):
        return self.__lista_pasajeros
    def agregar(self, nuevo):
        for Pasajero in self.lista_pasajeros:
            if Pasajero.codigo == nuevo.codigo:
                return None
        self.__lista_pasajeros.append(nuevo)
    def eliminar(self, codigo):
        for Pasajero in self.lista_pasajeros:
            if Pasajero.codigo == codigo:
                self.__lista_pasajeros.pop(Pasajero)
                return None
    def monto_total_vendido(self):
        monto = 0
        for Pasajero in self.lista_pasajeros:
            monto = monto + Pasajero.total_pagar()
        return monto
    def __str__(self):
        z = f"{super().__str__()}\nMonto Vendido:{self.monto_total_vendido()}"
        z = z + "\n" + " Pasajeros ".center(25, '*') + "\n"
        for Pasajero in self.lista_pasajeros:
            z = z + "*" * 25 +"\n" + str(Pasajero) + "\n"
        return z
    
instancia = VueloComercial(758, 5, 17, ['AVX984', 'ABCH89', 'HOSE8', 'DHDF9', 'COL84', 'AVX959'])


instancia.numero
instancia.hora_salida
instancia.hora_llegada

print(VueloComercial.agregar(instancia, 'AVX959'))
print(instancia.agregar('AVX959'))
instancia.agregar("AVX959")
instancia.lista_pasajeros
instancia.monto_total_vendido()
instancia.__str__()



    
class vuelo_local(VueloComercial):
    def __init__(self, numero, hora_salida, hora_llegada, lista_pasajeros, minimo_pasajeros):
        super().__init__(numero, hora_salida, hora_llegada, lista_pasajeros)
        self.__minimo_pasajeros = minimo_pasajeros
    @property
    def minimo_pasajeros (self):
        return self.__minimo_pasajeros
    @minimo_pasajeros.setter
    def minimo_pasajeros (self, nuevo):
        self.__minimo_pasajeros = nuevo
    def __str__(self):
       return "El minimo de pasajeros de los vuelos locales es de:  " + str(self.minimo_pasajeros)
   
fal = vuelo_local(758, 5, 17, ['AVX984', 'COL84', 'AVX959'], 125)

fal.hora_llegada
fal.lista_pasajeros
fal.numero
fal.minimo_pasajeros
fal.hora_salida
fal.__str__()


 
class vuelo_internacional(VueloComercial):
    def __init__ (self, numero, hora_salida, hora_llegada, lista_pasajeros, pais_destino):
        super().__init__(numero, hora_salida, hora_llegada, lista_pasajeros)
        self.__pais_destino = pais_destino
    @property
    def pais_destino(self):
        return self.__pais_destino
    @pais_destino.setter
    def pais_destino(self, nuevo):
        self.__pais_destino = nuevo
    def __str__(self):
        return "El vuelo internacional tiene el siguiente destino: " + str(self.pais_destino)
    
veri = vuelo_internacional(758, 5, 17, ['AVX984', 'COL84', 'AVX959'], 'Japon')

veri.hora_llegada
veri.hora_salida
veri.numero
veri.lista_pasajeros
veri.__str__()
veri.pais_destino


    
    
   

# 3

import pandas as pd
import numpy as np

class mi_DF():
    def __init__(self, DF = pd.DataFrame()):
        self.__num_filas = DF.shape[0]
        self.__num_columnas = DF.shape[1]
        self.__DF = DF
    @property
    def num_filas(self):
        return self.__num_filas
    @property
    def num_columnas(self):
        return self.__num_columnas
    @property
    def DF(self):
        return self.__DF  
    def maximo(self):
        max = self.DF.iloc[0,0]
        for i in range(self.num_filas):
            for j in range(self.num_columnas):
                if self.DF.iloc[i,j] > max:
                    max = self.DF.iloc[i,j]
        return max
    def valores(self):
        min = self.DF.iloc[0,0]
        max = self.DF.iloc[0,0]
        total_ceros = 0
        total_pares = 0
        for i in range(self.num_filas):
            for j in range(self.num_columnas):
                if self.DF.iloc[i,j] > max:
                    max = self.DF.iloc[i,j]
                if self.DF.iloc[i,j] < min:
                    min = self.DF.iloc[i,j]
                if self.DF.iloc[i,j] == 0:
                    total_ceros = total_ceros+1
                if self.DF.iloc[i,j] % 2 == 0:
                    total_pares = total_pares+1
        return {'Maximo' : max, 'Minimo' : min, 'Total_Ceros' : total_ceros, 'Pares' : total_pares}
    def estadisticas(self,nc):
        media = np.mean(self.DF.iloc[:,nc])
        mediana = np.median(self.DF.iloc[:,nc])
        deviacion = np.std(self.DF.iloc[:,nc])
        varianza = np.var(self.DF.iloc[:,nc])
        maximo = np.max(self.DF.iloc[:,nc])
        minimo = np.min(self.DF.iloc[:,nc])
        return {'Variable' : self.DF.columns.values[nc],
                'Media' : media,
                'Mediana' : mediana,
                'DesEst' : deviacion,
                'Varianza' : varianza,
                'Maximo' : maximo,
                'Minimo' : minimo}
    def multiplos (self):
        total_divisibles = 0    
        for i in range (self.num_filas):
            for j in range (self.num_columnas):
                if self.DF.iloc[i,j] % 3 == 0:
                    total_divisibles = total_divisibles+1
                    respuesta = {'Total_divisibles_entre_3' : total_divisibles}
        return respuesta  
    def analitica (self, nc):
        correlacion = np.corrcoef(self.DF.iloc[:,2])
        covarianza = np.cov(self.DF.iloc[:,0])
        return {'Variable_1'  : self.DF.columns.values[2],
                'Variable_2' :  self.DF.columns.values[0],
                'Correlacion' : np.corrcoef(self.DF.iloc[:,2]),
                'Covarianza' :  np.cov(self.DF.iloc[:,0])}
        

import os 
os.getcwd()

datos_est = pd.read_csv('ejemplo_estudiantes.csv', delimiter = ';', decimal = ",", header=0, index_col = 0)

datos = mi_DF(datos_est)


print(datos.num_filas)
print(datos.num_columnas)
print(datos.DF)
print(datos.multiplos())
print(datos.analitica(2))
print(datos.analitica(0))
    

# 4

import pandas as pd
import numpy as np

class Analisis():
    def __init__(self, data = np.matrix([])):
        self.__M = data
        self.__filas = data.shape[0]
        self.__columnas = data.shape[1]
    @property
    def data (self):
        return self.__M
    @property
    def filas(self):
        return self.__filas
    @property
    def columnas(self):
       return self.__columnas
    def as_data_frame(self):
        return pd.DataFrame(self.__M)
    def des_estandar(self):
        return self.as_data_frame().std()
    def media (self):
        return self.as_data_frame().mean()
    def mediana(self):
        return self.as_data_frame().median()
    def maximo(self):
        return np.max(self.__M)
    def encontrar(self):
        numero = input("Digite un valor :")
        if numero in self.as_data_frame():
           return self.as_data_frame().index()
        else:
            print("None")
            
            
            
import pandas as pd
import numpy as np

class Analisis():
    def __init__(self, data = np.matrix([])):
        self.__M = data
        self.__filas = data.shape[0]
        self.__columnas = data.shape[1]
    @property
    def data (self):
        return self.__M
    @property
    def filas(self):
        return self.__filas
    @property
    def columnas(self):
       return self.__columnas
    def as_data_frame(self):
        return pd.DataFrame(self.__M)
    def des_estandar(self):
        return self.as_data_frame().std()
    def media (self):
        return self.as_data_frame().mean()
    def mediana(self):
        return self.as_data_frame().median()
    def maximo(self):
        return np.max(self.__M)
    def encontrar(self):
        numero = input("Digite un valor :")
        any(float(numero) in sub for sub in self.as_data_frame())
    
        
        


          
info = Analisis(np.matrix([[5,78,34],[6,2,8],[36,9,60]]))
print(info.filas)
print(info.columnas)
print(info.data)
print(info.as_data_frame())
print(info.des_estandar())
print(info.media())
print(info.mediana())
print(info.maximo())
numero = 5
print(info.encontrar())

2
3
4
5

   
       
        
        
        
        
        
   
         
   


    
    
    



        
     
        
                 
                 





   

        
        
        
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
         
        
