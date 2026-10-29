Se toma la captura de audio con 1s de silencio al principio (oculto como un "preparese"), despues de el silencio capturar el audio, y despues analizarlo. 
Para el silencio, se le hace la transformada de fourrier y sacar los 5 componentes principales, para despues restarlo al audio con el comando. Despues, lo que habia pensado era con eso tomar las componentes de mel creo que era? y hacer algo como tomar la cantidad de silencios para saber el comando
Averiguando con Gemini, se vio que estas funciones ya vienen en unos cuantos paquetes previos, y evita el primer camino pensado: tomar los audios por microfono, procesarlos con ffmpeg, y luego insetarlos al sistema.
Despues, se toman 3 secciones de .25s, a estas se les realiza la transformada de mel, tomando 13 datos por ventana
Se analizan las pasadas por cero, y ademas despues se realiza el analisis KMeans de toda la base de datos.
Posteriormente, con Knn, se tiene el analisis de los nuevos audios.
El programa de Analisis audio tiene a opcion de generar un grafico PCA, pero tiene el detalle de, como el vector inicial es de ~36 componenetes, no representa bien el grafico final
