# Etiquetador-INSP

El Etiquetador es una herramienta de segmentación la cual tiene como objetivo implementar un algoritmo en el análisis de imágenes basado en objetos geográficos (GeoOBIA) que utiliza una técnica de segmentación conocida como SNIC(Simple Non-Iterative Clustering), la cual hace uso de superpíxeles.
Cada imagen se segmenta en regiones únicas para clasificar.

La consola principal del etiquetador es
<img src="https://drive.google.com/uc?id=1hZTw1g10THVHENAbOLuoENuA2A501Tug" alt="Inicio etiquetador" width="600" height="500">

En `1. Elegir archivo` se debe elegir un archivo txt con los parámetros necesarios para realizar la segmentación, así como las categorías a elegir y el prefijo con el que se guarda la imagen etiquetada y el archivo csv generado con las características, pues por default se nombran como: `prefijo + nombre_imagen`.


Se prosigue a seleccionar la imagen con `2. Elegir imagen`.

<img src="https://drive.google.com/uc?id=1r7ViV_iq32Riz38aJicdz3wJKZLyIhXt" alt="Elegir imagen" width="600" height="500">

Hasta el momento se consideran 7 categorías:

0. Árbol
1. Suelo desnudo
2. Pavimento
3. Cuerpo de agua
4. Techo de lámina
5. Techo de loza
6. Sin etiqueta

# Instrucciones para ejecutar el Etiquetador

Con el propósito de ejecutar de manera correcta el Etiqueador, se deben seguir los siguientes pasos:

1. **Crear un ambiente de Anaconda:**
   ```bash
   conda create -n env_GEOBIA python=3.9.18
   ```

2. **Activar el ambiente:**
   ```bash
    conda activate env_GEOBIA
   ```

3. **Instalar los requerimientos:**
   ```bash
   pip install -r requerimientos_etiquetador.txt
   ```

   Donde `requerimientos_etiquetador.txt` es el archivo proporcionado que contiene las paqueterías y versiones necesarias.

   Una vez completados estos pasos, el Etiquetador estará listo para ser utilizado.


Los autores del código son:
Karla Mauritania Reyes Maya, Viridiana Itzel Méndez Vásquez ([viridiana.mendez@cimat.mx](mailto:viridiana.mendez@cimat.mx)).


Un trabajo bajo la dirección de:
* Dr. Francisco Javier Hernández López (CIMAT Mérida)
* Dr. Víctor Hugo Muñíz Sánchez (CIMAT Monterrey)

En colaboración con el Instituto Nacional de Salud Pública, teniendo como contacto a la Dra. Kenya Mayela Valdez Delgado.
