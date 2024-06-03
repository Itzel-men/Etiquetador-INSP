################################################################################################
# Funciones auxiliares utilizadas para el análisis de y la clasificación de superpixeles
# 
# Autor: Viridiana Itzel Méndez Vásquez
# email: viridiana.mendez@cimat.mx
#
#################################################################################################3

from tkinter import *
from PIL import Image
from PIL import ImageTk
import cv2

import pandas as pd
import numpy as np
from scipy.stats import *
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage.feature import graycomatrix, graycoprops
from skimage import color

from skimage.morphology import disk
from skimage.filters import median

from sklearn.preprocessing import StandardScaler #Estandarización

from sklearn import metrics


def remove_rows(csv_file, columna):
    '''
    Función que elimina elementos repetidos.
    En este caso en específico se creó para eliminar índices repetidos de los archivos creados
    mediante el etiquetado manual, pues si por error se da click en un superpixel ya etiquetado
    se guarda el registro con valores 0 en algunas columnas.
    '''
    # Leer el archivo CSV
    df = pd.read_csv(csv_file)
    
    # Identificar las filas duplicadas en la columna específica
    filas_duplicadas = df.duplicated(subset=[columna], keep='first')
    
    # Filtrar el DataFrame para conservar solo las filas no duplicadas
    df_sin_duplicados = df[~filas_duplicadas]
    
    return df_sin_duplicados

def check_labels(indices,prediction, path_datos_image):
    ''' 
    Función para obtener las métricas de la clasificación, así como la matriz de confusión.

    Entrada:
        indices: Corresponde a la columna de índices de los superpixeles
                 de la imagen.
        prediction: Etiquetas proporcionadas por un método de clasificación
        path_datos_image: path del archivo .csv con el etiquetado manual
    Salida: Matriz de confusión y métricas de la clasificación.
    '''
    labels_pred = []
    labels = []
    datos_img = remove_rows(path_datos_image,'indice')
    indices_org = datos_img['indice'].values
    categorias = datos_img['categoria'].values

    for ind in indices_org:
        pos = np.where(indices == ind)
        if len(pos[0]) != 0:
            labels_pred.append(prediction[pos][0])
            labels.append(categorias[indices_org==ind][0])

    confusion_matrix = metrics.confusion_matrix(labels, labels_pred)

    print(metrics.classification_report(labels, labels_pred))

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)

    cm_display.plot(cmap=plt.cm.Blues)
    plt.title('Matriz de confusión')
    #plt.savefig('conf_cart.png',bbox_inches='tight')
    plt.show() 

def seg_SLIC(path_image):
    '''
    Esta función realiza la segmentación de las imágenes mediante el método SLIC.
    Este código fue tomado del archivo Ultima_version.py proporcionado para el etiquetado.

    Variables de entrada:
       path_image: ruta de la imagen a segmentar

    Variables de salida:
        img: Imagen segmentada
        Labels_slic: etiquetas de cada superpixel
        number_slic: Numero total de superpixeles

    '''
    ######################################
    ## S E G M E N T A C I O N
    ######################################
    #Parámetros con los cuales se realiza la segmentación
    parametro_1 = 40 #Parametro 1: region size
    parametro_2 = 10 #Parametro 2: ruler
    parametro_3 = 10  #Parametro 3: numero de iteraciones
    parametro_4 = 20 #Parametro 4: tamaño mínimo de elementos

    region_size_ = int(parametro_1)
    ruler_ = int(parametro_2)
    num_iterations = int(parametro_3)
    min_element_size = int(parametro_4)

    image = cv2.imread(path_image)
        
    img_converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    slic = cv2.ximgproc.createSuperpixelSLIC (img_converted,region_size = region_size_, ruler =  ruler_)  
    slic.iterate(num_iterations)      

    if(min_element_size > 0):
        slic.enforceLabelConnectivity(min_element_size)
            #El tamaño mínimo del elemento en porcentajes que debe absorberse en un superpíxel más grand
            #### Obteniendo los contornos
    #### Devuelve la máscara de la segmentación de superpíxeles almacenada en el objeto SuperpixelSLIC 
        #### La función devuelve los límites de la segmentación de superpíxeles.

    mask = slic.getLabelContourMask()
    dilation_size = 2

        #### Simplemente pasa la forma y el tamaño del kernel, obtiene el kernel deseado.
        #### Conde esta el ancla
    element_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,(2 * dilation_size + 1, 2 * dilation_size + 1),(dilation_size, dilation_size))
    mask = cv2.dilate(mask,element_dilate)
        ## Esta dilatando 
        
    label_slic = slic.getLabels()        # Obtener etiquetas de superpíxeles
    Labels_slic = label_slic
        
    number_slic = slic.getNumberOfSuperpixels()  # Obtenga el número de superpíxeles
    mask_inv_slic = cv2.bitwise_not(mask)  
    img_slic = cv2.bitwise_and(image,image,mask =  mask_inv_slic)
        
    imageToShowOutput = cv2.cvtColor(img_slic, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(imageToShowOutput)
    img = img.resize((600,600))

    return img, Labels_slic, number_slic

def filter_rgb(path_image, filter='median',r=None, d=None, sigmaColor = None, sigmaSpace = None):
    '''
    Función que aplica el filtro de mediana o bilateral a una imagen.

    Entradas:
        path_image: Ruta de la imagen a filtrar
        filter: Filtro a aplicar. Sólo disponible median (default) y bilateral.
        r: (Para el filtro de mediana) Radio con el que se aplica el filtro.
        d: (Para filtro bilateral) Diámetro de la vecindad de pixeles
        sigmaColor: (Para filtro bilateral) Valor de sigma en el espacio de color
        sigmaSpace: (Para filtro bilateral) Valor de sigma para el espacio de coordenadas.

    Salida:
        La función crea una carpeta llamada 'Imagenes_filtradas_filter' donde guarda la imagen filtrada.
    '''
    image = cv2.imread(path_image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    name_image = os.path.splitext(os.path.basename(path_image))[0]

    if filter == 'median':
        # Aplicar el filtrado de mediana a cada canal de color por separado
        median_filtered_r = median(image_rgb[:,:,0], disk(r))  # Filtrado de mediana para el canal rojo
        median_filtered_g = median(image_rgb[:,:,1], disk(r))  # Filtrado de mediana para el canal verde
        median_filtered_b = median(image_rgb[:,:,2], disk(r))  # Filtrado de mediana para el canal azul

        # Combinar los canales filtrados en una imagen RGB
        image_filtered = cv2.merge((median_filtered_r, median_filtered_g, median_filtered_b))

        image_filtered = cv2.cvtColor(image_filtered, cv2.COLOR_RGB2BGR)

        File_Path = 'Imagenes_filtradas_median'
        if not os.path.exists(File_Path):
            os.makedirs(File_Path)
            
        cv2.imwrite(File_Path+'\\'+'Filtered_median'+'_'+name_image+'.jpg',image_filtered)    
    elif filter == 'bilateral':
        # Aplicar el filtrado bilateral a cada canal de color por separado
        bilateral_filtered_r = cv2.bilateralFilter(image_rgb[:,:,0], d=d, sigmaColor= sigmaColor, sigmaSpace = sigmaSpace)  # Filtrado bilateral para el canal rojo
        bilateral_filtered_g = cv2.bilateralFilter(image_rgb[:,:,1], d=d, sigmaColor= sigmaColor, sigmaSpace = sigmaSpace)  # Filtrado bilateral para el canal verde
        bilateral_filtered_b = cv2.bilateralFilter(image_rgb[:,:,2], d=d, sigmaColor= sigmaColor, sigmaSpace = sigmaSpace)  # Filtrado bilateral para el canal azul

        # Combinar los canales filtrados en una imagen RGB
        image_filtered = cv2.merge((bilateral_filtered_r, bilateral_filtered_g, bilateral_filtered_b))

        image_filtered = cv2.cvtColor(image_filtered, cv2.COLOR_RGB2BGR)

        File_Path = 'Imagenes_filtradas_bilateral'
        if not os.path.exists(File_Path):
            os.makedirs(File_Path)    
        cv2.imwrite(File_Path+'\\'+'Filtered_bilateral'+'_'+name_image+'.jpg',image_filtered)

def extrac_features(path_image, rect='normal'):
    '''
    Función de segmentación y extracción de características de la imagen.
    Se realiza la estandarización de las características.

    Variables de entrada:
       path_image: ruta de la imagen a segmentar
       rect: Opción de cómo considerar el rectángulo (normal u orientado (de área mínima))

    Variables de salida:
        img: Imagen segmentada
        Labels_slic: etiquetas de cada superpixel
        number_slic: Numero total de superpixeles

    Adicionalmente crea un archivo .csv denominado 'Caract'+'_'+name_image.csv, donde name_image 
    corresponde al nombre de la imagen, este archivo contiene las características obtenidas de cada
    superpixel de la imagen.

    Parte del código también fue tomado del archivo Ultima_version.py proporcionado para el etiquetado.
    
    '''
    ######################################
    ## S E G M E N T A C I O N
    ######################################
    img_slic, Labels_slic, number_slic = seg_SLIC(path_image)
    image_seg = img_slic.copy()

    respaldo_img = img_slic.copy()
    
    
    ############################################################
    ## E X T R A C C I O N  D E  C A R A C T E R I S T I C A S
    ############################################################

    image = cv2.imread(path_image)
    img_converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Lo pasamos de BGR a grises
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sobel Detección de Border
    #Horizontal: Dx
    sobelx = cv2.Sobel(src=img_gray, ddepth=cv2.CV_32FC1, dx=1, dy=0, ksize=5) 
    #vertical:  Dy
    sobely = cv2.Sobel(src=img_gray, ddepth=cv2.CV_32FC1, dx=0, dy=1, ksize=5) 
    #Tomamos los valores absolutos 
    Dx = abs(sobelx)
    Dy = abs(sobely)

    # Suma pesada con Dx y Dy
    MG = cv2.addWeighted(Dx, 0.5, Dy, 0.5, 0.0)
    Edges = cv2.Canny(image=img_gray, threshold1=100, threshold2=200)
    
    distances=[1,2,3,4,5]
    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]
    len_dist = len(distances)
    len_ang = len(angles)

    columns = ['indice','media_r','media_g','media_b','std_r','std_g','std_b','mean_gb','std_mg','density']

    features_gray = ['dissimilarity','correlation','contrast','energy','homogeneity','asm']

    columns_gray = [f'{feature}_dist{len}_ang{ang}' for feature in features_gray for len in range(len_dist) for ang in range(len_ang)]
    columns.extend(columns_gray)

    midataframe = pd.DataFrame(columns=columns)

    for ind in range(0,number_slic):

        erosion_size = 7
        element_erode = cv2.getStructuringElement(cv2.MORPH_RECT,(2 * erosion_size + 1, 2 * erosion_size + 1),(erosion_size, erosion_size))

        mask_p = Labels_slic == ind
        mask_p = mask_p.astype(dtype=np.uint8)
        mask_p = cv2.erode(mask_p, element_erode )
        Mask_p = mask_p*255

        row_e,col_e = np.where(Mask_p[:,:] == 255)

        if (len(row_e)==0) and (len(col_e)==0):
            continue
        
        mean_rgb = (round(np.mean(image[row_e,col_e,2]),6),
                    round(np.mean(image[row_e,col_e,1]),6),
                    round(np.mean(image[row_e,col_e,0]),6))
        std_rgb = (round(np.std(image[row_e,col_e,2]),6),
                   round(np.std(image[row_e,col_e,1]),6),
                   round(np.std(image[row_e,col_e,0]),6))
        
        gray_img = color.rgb2gray(image) # Se pasa la imagen a escala de grises
        gray_img = (gray_img*255).astype(dtype=np.uint8) #Se convierte en valores enteros de 0 a 255

        if rect == 'orientado':
        
            ############################################################
            # Para obtener el rectángulo orientado de menor área que 
            # cubre al superpixel se utilizan las siguientes líneas
            ############################################################

            #Se obtiene el contorno de la máscara del superpixel
            contours, _ = cv2.findContours(Mask_p,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#find contour
            contour=contours[0]

            #Se obtiene el rectángulo de área mínima que cubre al superpixel
            rect = cv2.minAreaRect(contour) #Rectángulo
            box = cv2.boxPoints(rect) #Coordenadas de los vértices del rectángulo
            box = np.int0(box)

            # Con el fin de obtener el rectángulo orientado en la imagen, se recupera el ángulo de rotación
            centro, tamano, angulo = rect
            M = cv2.getRotationMatrix2D(centro, angulo, 1) #Matriz de rotación

            # Aplicar la matriz de transformación de rotación a la imagen para obtener la imagen y la máscara rotada
            mascara_rotada = cv2.warpAffine(Mask_p, M, (gray_img.shape[1], gray_img.shape[0])) #Máscara rotada
            imagen_rotada = cv2.warpAffine(gray_img, M, (gray_img.shape[1], gray_img.shape[0])) # Imagen en escala de grises rotada
        
            #Se obtiene el contorno de la máscara rotada
            contours_rot, _ = cv2.findContours(mascara_rotada,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_rot=contours_rot[0]

            #Rectángulo de área mínima de la imagen rotada
            rect_rot = cv2.minAreaRect(contour_rot)
            box_rot = cv2.boxPoints(rect_rot)
            box_rot = np.int0(box_rot)

            #Para cortar el rectángulo en la imagen rotada
            x, y, w, h = cv2.boundingRect(box_rot)

            # Recortar la región de interés
            gray_rect = imagen_rotada[y:y+h, x:x+w]
        elif rect=='normal':
        ############################################################
        # Para obtener el rectángulo vertical/horizontal de menor área que 
        # cubre al superpixel se utilizan las siguientes líneas
        ############################################################
            gray_rect = gray_img[min(row_e):max(row_e),min(col_e):max(col_e)]

        if (gray_rect.shape[0]==0) or (gray_rect.shape[1]==0):
            continue    

        #Se calcula la matriz de co-ocurrencia sobre el rectángulo que cubre al superpixel
        # Los ángulos son en radianes, si se ingresa una lista de ángulos se obtiene un tensor 4D
        # La tercer entrada es por distancia y la cuarta por ángulo
        glcm = graycomatrix(gray_rect, distances=distances, angles=angles, levels=256,
                        symmetric=True, normed=True)
        
        #Características obtenidas a partir de la matriz de co-ocurrencia 
        dissimilarity = np.round(graycoprops(glcm, 'dissimilarity').reshape(-1, order='F'), 6)
        correlation = np.round(graycoprops(glcm, 'correlation').reshape(-1, order='F'), 6)
        contrast = np.round(graycoprops(glcm,'contrast').reshape(-1, order='F'), 6)
        homogeneity = np.round(graycoprops(glcm,'homogeneity').reshape(-1, order='F'),6)
        energy = np.round(graycoprops(glcm,'energy').reshape(-1, order='F'),6) #Es la raíz cuadrada de ASM: Angular Second Moment
        asm = np.round(graycoprops(glcm,'ASM').reshape(-1, order='F')   ,6)

        gray_features = np.concatenate((dissimilarity,correlation,contrast,energy,homogeneity,asm))

        MG_mean = np.mean(MG[row_e,col_e])
        MG_sdt = np.mean(np.std(MG[row_e,col_e]))
           
        locations_edges = cv2.findNonZero(Edges[row_e,col_e])
        locations_mask = cv2.findNonZero(mask_p)
        edge_density = round(np.size(locations_edges)/np.size(locations_mask),4)
        
        dict_features = {'indice': ind,
                        'media_r':mean_rgb[0],
                        'media_g':mean_rgb[1],
                        'media_b':mean_rgb[2],
                        'std_r': std_rgb[0],
                        'std_g': std_rgb[1],
                        'std_b': std_rgb[2],
                        'mean_gb': MG_mean,
                        'std_mg':MG_sdt,
                        'density':edge_density
                        }
    
        features = list(dict_features.values())                      
        features.extend(list(gray_features))
    
        df_vector = pd.DataFrame(np.array(features).reshape(1,len(features)), columns=midataframe.columns)
        midataframe = pd.concat([midataframe,df_vector],ignore_index=True)
    
    features_ = midataframe.iloc[:,1:]
    ss = StandardScaler()
    scaled_features  = ss.fit_transform(features_.values)
    scaled_features = pd.DataFrame(ss.fit_transform(features_.values), columns=midataframe.columns[1:])

    scaled_features['indice'] = midataframe['indice']
        
    name_image = os.path.splitext(os.path.basename(path_image))[0]    
    scaled_features.to_csv('Caract'+'_'+name_image+'.csv',index=False) ### Guardamos los vectores

    return img_slic, Labels_slic, number_slic

def img_etiquetas(indices,prediction, Labels_slic, path_image, img_seg, save=False):
    '''
    Función que asigna el color correspondiente a cada superpixel de acuerdo a las etiquetas dadas.

    Variables de entrada:
        indices: Índice de cada superpixel
        prediction: etiquetas obtenidas por el método utilizado
        Labels_slic: Etiquetas de los superpixeles
        path_image: ruta de la imagen
        img_seg: imagen segmentada

        Labels_slic y img_seg se obtiene de la función seg_SLIC

    Variables de salida:
        img: imagen con los colores correspondientes a cada categoría.
    '''
    erosion_size = 7
    element_erode = cv2.getStructuringElement(cv2.MORPH_RECT,(2 * erosion_size + 1, 2 * erosion_size + 1),(erosion_size, erosion_size))
    image = cv2.imread(path_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    respaldo = image.copy()

    for j in range(indices.shape[0]):
        mask_p = Labels_slic == indices[j]
        mask_p = mask_p.astype(dtype=np.uint8)
        mask_p = cv2.erode(mask_p, element_erode )
        Mask_p = mask_p*255
        col,row = np.where(Mask_p[:,:] == 255)

        if prediction[j] == 0:
            #'Arbol'
            #'Verde'
            r=69
            g=139
            b=0
        if prediction[j] == 1:
            # 'Café'
            # 'Suelo Desnudo'
            #255,153,18
            r= 18
            g= 153
            b= 255
        if prediction[j] == 2:
            # Gris
            # Pavimento
            # 104,131,139
            r = 139
            g = 131
            b = 104
        if prediction[j] == 3:
            #Azul
            #Cuerpo deAgua
            # 61,89,171
            r = 171
            g = 85
            b = 61
        if prediction[j] == 4:
            #Techo de Lamina
            #205,104,137
            # rgb
            r=137
            g=104
            b=205
        if prediction[j] == 5:
            #Techo loza
            #128,0,0
            r=0
            g=0
            b=128    
        ### El orden es BGR
        respaldo[col,row,0] = b
        respaldo[col,row,1] = g
        respaldo[col,row,2] = r

    image = Image.fromarray(respaldo)
    image = image.resize((600,600))

    #Mostrar la imagen

    color_dict = {'Árbol': (0,139,69), 
           'Suelo desnudo': (255, 153, 18), 
           'Pavimento': (104, 131, 139), 
           'Cuerpo de agua': (61, 85, 171), 
           'Techo de lámina': (205, 104, 137), 
           'Techo de loza': (128,0,0)}

    # Crea los parches de color y etiquetas
    parches = [Patch(color=[r/255, g/255, b/255], label=nombre) for nombre, (r, g, b) in color_dict.items()]

    fig, ax = plt.subplots(figsize=(8,8))

    plt.imshow(image)
    plt.imshow(img_seg, alpha=0.3)
    ax.legend(handles=parches,loc='center left',bbox_to_anchor=(1, 0.5))

    # Oculta los ejes para que solo se muestre la leyenda
    ax.axis('off')

    plt.title('Imagen clasificada')
    if save == True:
        plt.tight_layout()
        name_image = os.path.splitext(os.path.basename(path_image))[0]    
        plt.savefig(name_image+'clas')
    plt.show()