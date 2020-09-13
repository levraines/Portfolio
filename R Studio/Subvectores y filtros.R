x = seq(3,50, by= 3.5) # Creando vector con secuencia 
x[3] # para saver la tercera entrada
x[length(x)] # la ultima entrada
x[length(x)-1] # la penultima y asi sucesivamente

x[-3] # quito la tercera entrada 

x[4:8] # solo me retorna del numero en la posicion 4 al 8

x[8:4] # del de la posicion 8 al 4

x[seq(2, length(x), by =2)] #este me retorna los numeros de las posiciones pares empezando desde el 2

x[seq(1, length(x), by = 2)] # este me retorna los numeros cada 2 en 2 pero empezando desde el primero

x[-seq(2, length(x), by = 2)] # me retorna los numeros en posiciones impares

x[(length(x)-3):length(x)] # desde los ultimos 3 de ultimo a primero. Ojo cuenta el 0. 

x[c(1,5,6)] # para filtrar los elementos en la posicion 1 5 y 6

x[x>30] # solo quiero los numeros mayores a 30

x[x>20 & x<40] # numeros entre 20 y 40

x[x!=3 & x!= 17] # quita los numeros iguales a 3 y 17

x[x<10 | x>40] # x menor o x mayor a 40

x[x>=10] # x mayor o igual a 10

x[x>10] # en esta quito el 10

x[!x<10] # x diferente de menor a 10

x[!x>10] # x doiferente de mayor a 10

x[x%%2==0] # extrayendo los numeros pares

x[x%%2==1] # extrayendo numeros impares

x>30 # ojo este retorna pero con condicion booleana
x[x>30] # este me da los numeros que son. OJO. 

x = c(1,7,4,2,4,8,9,2,0)
x
y = c(5,2,-3,-7,-1,4,-2,7,1)
y

x[y>0] # este lo que hago es que filtro con y, entonces quiero ver los numeros del vector de x
# que son analogos usando la Y, es  decir, de acuerdo a Y, muestreme los que cumplen la condicion
# de ser mayores de 0 en y, pero reflejados en X. TIENEN QUE SER DE LA MISMA LONGITUD. 

which(x>4) #elementos mayores a 4 en X, pero me da la posicion entonces:
# el 2 es el 7, el 6 el 9 y el 7 es el 9 pero de x

x[which(x>4)] # en esta si me da los numeros que son de X mayores a 4, pero usando el x adelante

x[x>4]#este es analogo al de la linea 57

which(x>2 & x< 8) # me retorna las posiciones
x[x>2 & x<8] # este si me retorna los valores

which(x<5 | x%%2 ==0) # posicion
x[which(x<5 | x%%2 == 0)] # los valores

which(x%%2==0) # retorna las posiciones

which.min(x) # la posicion del mas pequeno

x[which.min(x)] # me dice cual numero es

which(x == min(x)) # este es igual al de la linea 69

which.max(x) # posicion
which(x == max(x)) # analoga de la 75

x = c()
x # x es nulo

y = NULL
y

z = c(x, 2, y, 7)
z # z solo tiene el 2 y el 7 porque el x y el y son los nulos. 




