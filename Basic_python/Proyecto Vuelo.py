from abc import ABCMeta, abstractmethod 

# Clase Abstracta, ABC Class

class Base(metaclass= ABCMeta):
    @abstractmethod
    def __str__(self):
        pass
    @abstractmethod
    def Captura(self):
        pass


#Vuelo 

class Vuelo (Base):
    def __init__(self, numero = 0, hora_salida = 0, hora_llegada = 0):
        self.__numero = numero
        self.__hora_salida = hora_salida
        self.__hora_llegada = hora_llegada
    @property
    def numero(self):
        return self.__numero
    @property
    def hora_salida(self):
        return self.__hora_salida
    @property
    def hora_llegada(self):
        return self.__hora_llegada
    @numero.setter
    def numero(self, nuevo_numero):
        self.__numero = nuevo_numero
    @hora_llegada.setter
    def hora_llegada(self, nueva_llegada):
        self.__hora_llegada = nueva_llegada
    @hora_salida.setter
    def hora_salida(self, nueva_salida):
        self.__hora_salida = nueva_salida
    def __str__(self):
        return "Numero de vuelo:  %i\nhora salida: %i\nhora llegada: %i" % (self.numero, self.hora_salida, self.hora_llegada)
    def Captura(self):
        self.numero = int(input("Digite numero de vuelo: "))
        self.hora_salida = int(input("Digite la hora de salida del vuelo: "))
        self.hora_llegada = int(input("Digite hora de llegada del vuelo: "))
      
    
    
prueba=Vuelo(907,13,8)

prueba.hora_llegada
prueba.numero
prueba.hora_salida
prueba.__str__()


#Pasajero
class Pasajero(Base):
    def __init__(self, precioTiquete = 0, codigo = '', nombre = '', porcentaje_impuesto = 0):
        self.__precioTiquete = precioTiquete
        self.__codigo = codigo
        self.__nombre = nombre
        self.porcentaje_impuesto = porcentaje_impuesto
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
    @precioTiquete.setter
    def precioTiquete(self, nuevo_precio):
        self.__precioTiquete = nuevo_precio
    @codigo.setter
    def codigo(self, nuevo_codigo):
        self.__codigo = nuevo_codigo
    @nombre.setter
    def nombre(self, nuevo_nombre):
        self.__nombre = nuevo_nombre
    @porcentaje_impuesto.setter
    def porcentaje_impuesto(self, nuevo_porcentaje):
        if nuevo_porcentaje > 1:
            nuevo_porcentaje = nuevo_porcentaje / 100
        self.__porcentaje_impuesto = nuevo_porcentaje
    def total_pagar(self):
        precio = (self.precioTiquete + self.porcentaje_impuesto * self.precioTiquete)
        return precio
    def __str__(self):
        return "Precio tiquete: %i\ncodigo: %s\nnombre: %s\nporcentaje de impuesto: %i" % (self.precioTiquete, self.codigo, self.nombre, self.porcentaje_impuesto)
    def Captura (self):
        self.precioTiquete = int(input("Ingrese el precio del tiquete: "))
        self.codigo = input("Ingrese el codigo del vuelo: ")
        self.nombre = input("Ingrese el nombre del vuelo: ")
        self.porcentaje_impuesto  - int(input("Ingrese el porcentaje de impuesto a pagar: "))
    
val = Pasajero(789, 'AVX984', 'Lufthansa', 9)
val.precioTiquete
val.codigo
val.nombre
val.porcentaje_impuesto
val.total_pagar()
val.__str__()


#Pasajero Frecuente 

class Frecuente(Pasajero):
    def __init__(self, precioTiquete = 0, codigo = '', nombre = '', porcentaje_impuesto = 0, descuento = 0):
        super().__init__(precioTiquete, codigo, nombre, porcentaje_impuesto)
        self.__descuento = descuento
    @property
    def descuento(self):
        return self.__descuento
    @descuento.setter
    def descuento(self, nuevo_descuento):
        if nuevo_descuento > 1:
            nuevo_descuento = nuevo_descuento / 100
        self.__descuento = nuevo_descuento
    def total_pagar_frecuente(self):
        return (float(self.precioTiquete) + float(self.porcentaje_impuesto) * float(self.precioTiquete)) * - float(self.descuento) + (float(self.precioTiquete) + float(self.porcentaje_impuesto) * float(self.precioTiquete))
    def __str__(self):
        return "Precio tiquete: %i\ncodigo: %s\nnombre: %s\nporcentaje de impuesto: %i\ndescuento: %i" % (self.precioTiquete, self.codigo, self.nombre, self.porcentaje_impuesto, self.descuento)
        return
    def Captura (self):
        self.descuento = input("Ingrese el porcentaje de impuesto: ")
        
novo= Frecuente (899, 'COL84', 'Lufthansa', 11.4, 0.20)

novo.codigo
novo.nombre
novo.precioTiquete
novo.porcentaje_impuesto
novo.descuento
novo.total_pagar_frecuente()
novo.__str__()



#Vuelo Carga

class vuelo_carga(Vuelo):
    def __init__(self, numero = 0, hora_salida = 0, hora_llegada = 0, peso_maximo = 0):
        super().__init__(numero, hora_salida, hora_llegada)
        self.__peso_maximo = peso_maximo
    @property
    def peso_maximo (self):
        return self.__peso_maximo 
    @peso_maximo .setter
    def peso_maximo  (self, nuevo_peso_maximo ):
        self.__peso_maximo  = nuevo_peso_maximo 
    def __str__(self):
        return "Numero de vuelo:  %i\nhora salida: %i\nhora llegada: %i\npeso maximo: %i" % (self.numero, self.hora_salida, self.hora_llegada, self.peso_maximo)
    def Captura (self):
        self.peso_maximo  = input("Ingrese el peso maximo: ")
        
xos=vuelo_carga(899,5,9, 8956.8)

xos.hora_llegada
xos.numero
xos.hora_salida
xos.peso_maximo 
xos.__str__()
    

#Vuelo Comercial
      

class VueloComercial(Vuelo):
    def __init__(self, numero = 0, hora_salida = 0, hora_llegada = 0):
        super().__init__(numero, hora_salida, hora_llegada)
        self.__listaPasajeros = []
    def agregar_listaPasajeros (self, nuevo_pasajero):
        self.__listaPasajeros.append(nuevo_pasajero)
    def eliminar_listaPasajeros(self, codigo):
        for i in range(len(self.__listaPasajeros)):
            if codigo == self.__listaPasajeros[i].codigo:
                del self.__listaPasajeros[i]
    def monto_total_vendido(self):
        monto = 0
        for Pasajero in self.__listaPasajeros:
            monto = monto + Pasajero.total_pagar()
        return monto
    def __str__(self):
        z = f"{super().__str__()}\nMonto Vendido:{self.monto_total_vendido()}"
        z = z + "\n" + " Pasajeros ".center(25, '*') + "\n"
        for Pasajero in self.__listaPasajeros:
            z = z + "*" * 25 +"\n" + str(Pasajero) + "\n"
        return z
    
      
instancia = VueloComercial(758, 5, 17)


instancia.numero
instancia.hora_salida
instancia.hora_llegada
instancia.__str__()


#Vuelo Local


    
class Vuelo_local(VueloComercial):
    def __init__(self, numero = 0, hora_salida = 0, hora_llegada = 0, minimo_pasajeros = 0):
        super().__init__(numero, hora_salida, hora_llegada)
        self.__minimo_pasajeros = minimo_pasajeros
    @property
    def minimo_pasajeros (self):
        return self.__minimo_pasajeros
    @minimo_pasajeros.setter
    def minimo_pasajeros (self, nuevo_minimo_pasajeros):
        self.__minimo_pasajeros = nuevo_minimo_pasajeros
    def __str__(self):
       return "Numero de vuelo: %i\n Hora de salida: %i\n Hora de llegada: %i\n Minimo Pasajeros: %i" % (self.numero, self.hora_salida, self.hora_llegada, self.minimo_pasajeros)
    def Captura (self):
        self.numero = int(input("Ingrese el numero de vuelo: "))
        self.hora_salida = int(input("Ingrese la hora de salida: "))
        self.hora_llegada = int(input("Ingrese la hora de llegada: "))
        self.minimo_pasajeros = int(input("Ingrese el minimo de pasajeros para este vuelo: "))
    
       
       
       
fal = Vuelo_local(758, 7, 9, 230)
fal.hora_llegada
fal.numero
fal.minimo_pasajeros
fal.hora_salida
fal.__str__()

#Vuelo Internacional 


 
class Vuelo_internacional(VueloComercial):
    def __init__ (self, numero = 0, hora_salida = 0, hora_llegada = 0, pais_destino = ''):
        super().__init__(numero, hora_salida, hora_llegada)
        self.__pais_destino = pais_destino
    @property
    def pais_destino(self):
        return self.__pais_destino
    @pais_destino.setter
    def pais_destino(self, nuevo_pais_destino):
        self.__pais_destino = nuevo_pais_destino
    def __str__(self):
        return "Numero de vuelo: %i\n Hora de salida: %i\n Hora de llegada: %i\n Pais destino: %s" % (self.numero, self.hora_salida, self.hora_llegada, self.pais_destino)
    def Captura (self):
        self.numero = int(input("Ingrese el numero de vuelo: "))
        self.hora_salida = int(input("Ingrese la hora de salida: "))
        self.hora_llegada = int(input("Ingrese la hora de llegada: "))
        self.pais_destino = input("Ingrese el pais de destino: ")
        
    
      
veri = Vuelo_internacional(758, 5, 17, 'Japon')
veri.hora_llegada
veri.hora_salida
veri.numero
veri.__str__()
veri.pais_destino


import os 

class lista_Vuelos:
    def LeeDatosVuelo(self):
        vuelo = Vuelo()
        os.system('clear')
        vuelo.Captura()
        return vuelo
    def LeeDatosPasajero(self):
        pasajero = Pasajero()
        os.system('clear')
        pasajero.Captura()
        return pasajero
    def LeeDatosVuelo_local(self):
        vuelo_local = Vuelo_local()
        os.system('clear')
        vuelo_local.Captura()
        return vuelo_local
    def LeeDatosVuelo_internacional(self):
        vuelo_internacional = Vuelo_internacional()
        os.system('clear')
        vuelo_internacional.Captura()
        return vuelo_internacional
    

class App:
    def __init__(self):
        self.__lista = list()
        self.__lisV = lista_Vuelos()
    def __menu(self):
        print("****************************************** ")
        print(" [1] Insertar datos vuelo ")
        print(" [2] Insertar datos pasajero ")
        print(" [3] Insertar datos vuelo local ")
        print(" [4] Insertar datos vuelo internacional ")
        print(" [5] Ver la lista Polimorfica ")
        print(" [6] Borrar la lista Polimorfica ")
        print(" [7] Salir")
        print(" ============== ULTIMA LINEA ============== ")
        return input("> ")
    def __mostrarLista(self):
        for i in range (len(self.__lista)):
            print(self.__lista[i])
            print(15 * "*" + "\n")
    def principal(self):
        respuesta = " "
        while respuesta != "7":
            respuesta = self.__menu()
            if respuesta  == "1":
                self.__lista.append(self.__lisV.LeeDatosVuelo())
            elif respuesta == "2":
                self.__lista.append(self.__lisV.LeeDatosPasajero())
            elif respuesta == "3":
                self.__lista.append(self.__lisV.LeeDatosVuelo_local())
            elif respuesta == "4":
                self.__lista.append(self.__lisV.LeeDatosVuelo_internacional())
            elif respuesta == "5":
                self.__mostrarLista()
                input("Digite cualquir tecla para continuar" )
            elif respuesta == "6":
                self.__lista.clear()
            
            
test = App()
test.principal()  