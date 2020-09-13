from abc import ABCMeta, abstractmethod 

# Clase Abstracta, ABC Class

class Base(metaclass= ABCMeta):
    @abstractmethod
    def __str__(self):
        pass
    @abstractmethod
    def Captura(self):
        pass
    
    
    
    ### CLIENTE
    
class Cliente(Base):
    def __init__(self, nombre = '', direccion= ''):
        self.__nombre = nombre
        self.__direccion = direccion
    @property
    def nombre(self):
        return self.__nombre
    @property
    def direccion(self):
        return self.__direccion
    @nombre.setter
    def nombre (self, nuevo_nombre):
        self.__nombre = nuevo_nombre 
    @direccion.setter
    def direccion (self, nueva_direccion):
        self.__direccion = nueva_direccion
    def __str__(self):
        return "Nombre del cliente: %s\ndireccion: %s" % (self.nombre, self.direccion)
    def Captura(self):
        self.nombre = input("Nombre del cliente: ")
        self.direccion = input("Direccion del cliente: ")
        
x = Cliente("Romira", "San Pedro")
x.direccion
x.nombre
x.__str__()
        
    
    
### FACTURA
        
        
class Factura:
    def __init__(self, porcentaje_impuesto = 0):
        self.__porcentaje_impuesto = porcentaje_impuesto
        self.__listaCompra = []
    @property
    def porcentaje_impuesto(self):
        return self.__porcentaje_impuesto
    @porcentaje_impuesto.setter
    def porcentaje_impuesto(self, nuevo_porcentaje):
        self.__porcentaje_impuesto = nuevo_porcentaje
    def agregar_listaCompra (self, nueva_compra):
        self.__listaCompra.append(nueva_compra)
    def eliminar_listaCompra(self, codigo):
        for i in range(len(self.__listaCompra)):
            if codigo == self.__listaCompra[i].codigo:
                del self.__listaCompra[i]
    def monto_total (self, monto_compra):
        return sum(monto_compra)
    def total_pagar (self):
       s = (int(self.monto_total)+ int(self.porcentaje_impuesto) * int(self.monto_total) - int(self.porcentaje_impuesto) * int(self.monto_total)) 
       return s
    def __str__(self):
       x = "Factura: "+ str(self.__porcentaje_impuesto) + "\nlistaCompra: \n"
       for Compra  in str(self.__porcentaje_impuesto):
           x = x + "\n" + str(Compra)
           return(x)
    def Captura (self):
        self.porcentaje_impuesto = input("Ingrese porcentaje de impuesto: ")
        
        
z = Factura(3)
z.porcentaje_impuesto
z.__str__()


## Compra
       
class Compra:
    def __init__(self, codigo = '', descripcion = '', monto_compra = 0):
        self.__codigo = codigo
        self.__descripcion = descripcion
        self.__monto_compra = monto_compra
    @property
    def codigo (self):
        return self.__codigo
    @property
    def descripcion(self):
        return self.__descripcion
    @property 
    def monto_compra (self):
        return self.__monto_compra
    @codigo.setter
    def codigo (self, nuevo_codigo):
        self.__codigo = nuevo_codigo
    @descripcion.setter
    def descripcion (self, nuevo_descripcion):
        self.__descripcion = nuevo_descripcion
    @monto_compra.setter
    def monto_compra (self, nuevo_monto_compra):
        self.__monto_compra = nuevo_monto_compra
    def __str__(self):
        return "Codigo compra: %s\ndescripcion: %s\nmonto de compra: %i" % (self.codigo, self.descripcion, self.monto_compra)
    def Captura(self):
        self.codigo = input("Codigo de la compra: ")
        self.descripcion = input("Descripcion de la compra: ")
        self.monto_compra = int(input("Monto de la compra: "))
        
b = Compra("AB","Compra", 56700)        
b.codigo
b.descripcion
b.monto_compra
b.__str__()

## Factura credito 

class Factura_credito(Factura):
    def __init__(self, porcentaje_impuesto = 0, plazo_credito = 0):
        Factura.__init__(self, porcentaje_impuesto)
        self.__plazo_credito = plazo_credito
    @property
    def plazo_credito (self):
        return self.__plazo_credito
    @plazo_credito.setter
    def plazo_credito(self, nuevo_plazo_credito):
        self.__plazo_credito = nuevo_plazo_credito
    def __str__ (self):
        return "Plazo credito: %i" % (self.plazo_credito)
    def Captura(self):
        self.plazo_credito = int(input("Ingrese Plazo del credito: "))
        
c = Factura_credito(3)
c.plazo_credito
        

## Factura Contado
    
class Factura_contado(Factura):
    def __init__(self, porcentaje_impuesto = 0, porcentaje_descuento = 0):
        Factura.__init__(self, porcentaje_impuesto)
        self.__porcentaje_descuento = porcentaje_descuento
    @property
    def porcentaje_descuento(self):
        return self.__porcentaje_descuento
    @porcentaje_descuento.setter
    def porcentaje_descuento(self, nuevo_porcentaje_descuento):
        self.__porcentaje_descuento = nuevo_porcentaje_descuento
    def __str__(self):
        return "Porcentaje de descuento: %i" % (self.porcentaje_descuento)
    def Captura(self):
        self.porcentaje_descuento = float(input("Ingrese porcentaje de descuento: "))
        
v = Factura_contado(4)
v.porcentaje_descuento
        
        
import os 

class lista_facturacion:
    def LeeDatosFactura(self):
        factura = Factura()
        os.system('clear')
        factura.Captura()
        return factura
    def LeeDatosFactura_credito(self):
        factura_credito = Factura_credito()
        os.system('clear')
        factura_credito.Captura()
        return factura_credito
    def LeeDatosFactura_contado(self):
        factura_contado = Factura_contado()
        os.system('clear')
        factura_contado.Captura()
        return factura_contado
    def LeeDatosCompra(self):
        compra = Compra ()
        os.system('clear')
        compra.Captura()
        return compra 
    

class App:
    def __init__(self):
        self.__lista = list()
        self.__lis = lista_facturacion()
    def __menu(self):
        print("****************************************** ")
        print(" [1] Insertar datos compra ")
        print(" [2] Insertar Factura ")
        print(" [3] Insertar Factura Credito ")
        print(" [4] Insertar Factura Contado ")
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
                self.__lista.append(self.__lis.LeeDatosCompra())
            elif respuesta == "2":
                self.__lista.append(self.__lis.LeeDatosFactura())
            elif respuesta == "3":
                self.__lista.append(self.__lis.LeeDatosFactura_credito())
            elif respuesta == "4":
                self.__lista.append(self.__lis.LeeDatosFactura_contado())
            elif respuesta == "5":
                self.__mostrarLista()
                input("Digite cualquir tecla para continuar" )
            elif respuesta == "6":
                self.__lista.clear()
            
            
prueba = App()
prueba.principal()    
            
            

                
         





        
        
    
    
        

    
        
        
    
        
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
    
