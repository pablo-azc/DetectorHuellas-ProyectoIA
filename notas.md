La clave para el Kmeans y el knn es tener un cumulo de puntos.
Como hacemos que estos sean numeros? vectorizando. Sacamos puntos caracteristicos (puntos en comun), y comparamos que tienen en comun esas identidades (distintas de la identidad 2)

Tenemos que encontrar algo util, que se repita para una misma identidad, pero que se diferencien de imagenes similares para distintas identidades
Tomando muestra, debemos buscar las caracteristicas mas presentes para un mismo grupo, pero tenemos que cuantificar.

Por ejemplo, para las imagenes, tenemos que agarrar los elementos (de una matriz numerica) y hacerle alguna transformacion

Lo otro que se puede hacer es talvez plantear puntos locales (como el centro, o patrones, separaciones). Una manera es notar las zonas con lazos cerrados, y otra con puntos de interseccion (cruce). Habria que buscar puntos de interes. 
Despues se podria analizar la diferencia, y buscar que la diferencia sea la mas chica posible. Si 

Deberiamos hacer algo "cretivo"? puede ser. Metodos hay muchos, pero deberiamos intentar que se asemeje. Podemos usar la media y la desviacion estandar. Para cada numero voy a tener 2 numeros, media y desviacion estandar. queremos que sea la mas baja posible para que sean lo mas parecidos posibles
Para lograr mas separacion, separo las medias 

Para el audio, podriamos analizar los espacios, para saber la longitud de la palabra. Tambien podriamos analizar los picos de frecuencias, para detectar 

Librosa: tiene los momentos septrales, que caracterizan los audios (cosas como espacios). Se puede obtener una caracterizacion con 13 componentes (podria analizar eso)