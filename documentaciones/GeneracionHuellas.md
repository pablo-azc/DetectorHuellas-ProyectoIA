# Logica para la generacion de huellas

## Seleccion de programa generador

### Opciones
Investigando, las opciones pre-hechas son:
- SFinGe (Synthetic Fingerprint Generator): Modelo matemático de orientación y densidad + filtros de Gabor.Estándar académico de la Universidad de Bolonia.
    -Simula presión, humedad, elasticidad de piel y poros.
- Anguli:Algorítmico Open-Source (C++).Alternativa libre a SFinGe para generación masiva en lote de imágenes individuales o dactilogramas.
    - Coordenadas de minucias ($x, y, \theta$) y máscaras de segmentación.
- L3-SF (Level 3 Synthetic Fingerprint).Incluye poros, contornos de crestas, porosidad fina e imperfecciones de superficie/rasguños.Minucias de Nivel 1, 2 y 3 (poros).
- Modelos GAN / Diffusion (StyleGAN3, SynFinger, PrintsGAN): Generacion propia

### Elegida
- Anguli: fue elegida porque:
    - Es open-Source: la licencia es de uso libre, asi que listando la sub-licencia basta

## Generacion
### Comando inicial
Primero, se generó una base de huellas de 3 elementos, de forma que se pueda comprobar que la genreacion sea distinta. Se utilizo el comando inicial
`.\Anguli.exe -num 3 -meta -ni 1 -outdir "VariantesPermitidas" -trans 0 -rot 0 -seed 763982`
Que generó las bases. Analizadas visualmente, se notan distintas.
### Generacion de variantes
Luego, se toma la generación con detalles de imperfecciones, pero en otro archivo, para la base de datos
`.\Anguli.exe -indir "VariantesPermitidas\Fingerprints" -impr -ni 20  -scratch '0 5' -noise '0 8' -trans 15 -rot 5 -outdir "VariantesPermitidas"`
Esto es, genera huellas digitales con: cortaduras, ruido, trasladadas (+-15 pixeles), y con ligeras deformaciones por rotación (+- 5°)
### Ordenamiento
Este dataset es generado en carpetas del estilo ´Impresion_n/huella_i´. Se desea ´huella_i/impresion_n´, por lo que se hizo un pequeño script de Python para reordenarlas, con el siguiente prompt:
´Generé las huellas con Anguli; vi que el dataset es generado en carpetas del estilo "Impresion_n/huella_i" (mas específicamente, por ejemplo para la primera impresion "Impression_1\fp_1\1.jpg"), pero me gustaría "huella_i/impresion_n". Me haces un script de Python o batch que me solucione este problema? Considerá que la carpeta fp_I siempre es fp_1, lo que cambia es el numero al final (archivo.jpg)'
### Simulacion de condiciones reales
Las imagenes generadas siempre estan perfectamente recortadas, orientadas y demas.
Basado en eso, se creo un script de python que las convierta en algo mas similar a una captura real, llamado "Simular_captura.py". Este archivo lo que hace es añadir un giro y un margen a las huellas, para simular mas la condicion que se planteó: un scanner de grandes dimensiones

## Normalizado
### Recorte y Rotación
Basado en el proceso anterior, se recorta la imagen y se rota. Para lograr eso, simplemente busca el ancho minimo. Esto tiene la desventaja de que las imagenes giradas tienen una variacion importante: puede estar girada 0° o 180°
### Blur y Sharpening
La siguiente etapa para la normalizacion es aplicar un efecto de Blur para disolver el ruido "sal y pimienta", 
### Binariacion
Se consideró directamente hacer que la imagen se convierta de forma que solo queden los bordes "filosos", donde solo cuando se supere un umbral se considere "negro"