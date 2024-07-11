from tkinter import *
from tkinter import filedialog
from tkinter import Tk, Button, Label
from PIL import Image
from PIL import ImageTk
import cv2
import imutils

import pandas as pd
import numpy as np
from scipy.stats import *
import os

import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage import color


def recuperacion_etiquetado():
    '''
    En esta función se recupera una imagen ya etiquetada
    '''
    path_archivo_rec = filedialog.askopenfilename(filetypes = [
        ("archivo", ".csv")])
    etiquetado_data = pd.read_csv(path_archivo_rec)
    cat_rec = etiquetado_data['categoria'] #Categoria de etiquetado
    label_rec = etiquetado_data['indice']
    tam = len(cat_rec)
    print('tamaño',tam)
    print(label_rec)
    
    global midataframe
    midataframe = etiquetado_data
    
    global Labels_slic
    labels = Labels_slic
     
    global image_seg
    imge_ = image_seg
    
    erosion_size = 7
    element_erode = cv2.getStructuringElement(cv2.MORPH_RECT,(2 * erosion_size + 1, 2 * erosion_size + 1),(erosion_size, erosion_size))
        
    for i in range(0,tam):
        
        cat = cat_rec[i]
        indice = label_rec[i]
        print(indice)
        
        mask_p = labels == indice
        mask_p = mask_p.astype(dtype=np.uint8)
        mask_p = cv2.erode(mask_p, element_erode )
        Mask_p = mask_p*255
        
        respaldo = imge_
        ## Lo vamos a necesitar para ubicar los pixeles
        if(cat == 0):
            #'Arbol'
            #'Verde'
            r=69
            g=139
            b=0
        if(cat == 1):
            # 'Café'
            # 'Suelo Desnudo'
            #255,153,18
            r= 18
            g= 153
            b= 255
        if(cat == 2):
            # Gris
            # Pavimento
            # 104,131,139
            r = 139
            g = 131
            b = 104
        if(cat == 3):
            #Azul
            #Cuerpo deAgua
            # 61,89,171
            r = 171
            g = 85
            b = 61
        if(cat == 4):
            #Techo de Lamina
            #205,104,137)
            r=137
            g=104
            b=205
        if(cat == 5):
            #Techo loza
            #128,0,0
            r=0
            g=0
            b=128
        if(cat == 6):
            #Sin etiqueta
            #Negro
            r=0
            g=0
            b=0
        if(cat == 7):
            r=0
            g=128
            b=128
        if(cat == 8):
            r = 3
            g = 3
            b = 3
        if(cat == 9):
            r = 255
            g = 253
            b = 18
            
        col,row = np.where(Mask_p[:,:]==255)
        respaldo[col,row,0] = r
        respaldo[col,row,1] = g
        respaldo[col,row,2] = b
            
        
        global respaldo_img
        respaldo_img = respaldo
    

    imageToShowOutput_ = cv2.cvtColor(respaldo, cv2.COLOR_BGR2RGB)
    img_ = Image.fromarray(imageToShowOutput_)
    img_ = img_.resize((600,600))
    img_ = ImageTk.PhotoImage(image=img_)
    lblOutputImage.configure(image=img_)
    lblOutputImage.image = img_

def elegir_archivo():
    
    path_archivo = filedialog.askopenfilename(filetypes = [
        ("archivo", ".txt")])
    
    
    if len(path_archivo) > 0:
        
        global archivo
        #archivo = open(path_archivo, 'r')
        #Lines = archivo.readlines()
        
        lista_line = []
        with open(path_archivo) as f:
            for line in f:
                line = line.partition('#')[0]
                line = line.rstrip()
                lista_line.append(line)
            print(lista_line)
        Lines = lista_line
        
        
        global parametro_1
        parametro_1 = Lines[0]
        
        global parametro_2
        parametro_2 = Lines[1]
        
        global parametro_3
        parametro_3 = Lines[2]
        
        global parametro_4
        parametro_4 = Lines[3]
        
        global categorias
        lista_c = Lines[4]
        lista_c = lista_c.split(",") 
        categorias = lista_c
        print(lista_c)
        
        global nombre_arc
        name = Lines[5]
        nombre_arc = name
        print(name)

        
        selected = IntVar()
        var = StringVar(root)
        var.set('Categorías')
        
        def display_selection(choice):
            choice = var.get()
            Label(root,text=choice).pack
            print('etiqueta',choice)
            inde = lista_c.index(choice)
            print(inde)
            
            if inde == 0:
                # Arbol
                categoria = 0
            if inde == 1:
                # Suelo desnudo
                categoria = 1
            if inde == 2:
                # Cuerpo de Agua
                categoria = 2
            if inde == 3:
                # Pavimento
                categoria = 3
            if inde == 4:
                # Techo de Lamina
                categoria = 4
            if inde == 5:
                # Techo de Loza
                categoria = 5
            if inde == 6:
                # Sin etiqueta
                categoria = 6
            if inde == 7:
                # 
                categoria = 7
            if inde == 8:
                # 
                categoria = 8
            if inde == 9:
                categoria = 9
   
                
            global categoria_et
            categoria_et =  categoria
            
        opciones = lista_c
        opcion = OptionMenu(root, var, *opciones, command= display_selection)
        opcion.config(width=15)
        opcion.place(x = 10,y = 140)
        
def elegir_imagen():
    # Especificar los tipos de archivos, para elegir solo a las imágenes
    path_image = filedialog.askopenfilename(filetypes = [
        ("image", ".tif"),
        ("image", ".png"),
        ("image", ".jpg")])
    
    global name_image
    name_image = os.path.splitext(os.path.basename(path_image))[0]
    print('name image', name_image)
    
    print('path',path_image)
    
    if len(path_image) > 0:
        

        global image
        image = cv2.imread(path_image)
        img_H = image.shape[0]
        img_W = image.shape[1]        
        global tam_x
        tam_x = img_H
        global tam_y
        tam_y = img_W
        
        print('tamaño imagen',img_H)

        
        # Para visualizar la imagen de entrada en la GUI
        imageToShow= imutils.resize(image, width=120)
        imageToShow = cv2.cvtColor(imageToShow, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(imageToShow )
        img = ImageTk.PhotoImage(image=im)
        
        lblInputImage.configure(image=img)
        lblInputImage.image = img
        # Label IMAGEN DE ENTRADA
        lblInfo1 = Label(root, text="Imagen Entrada:")
        lblInfo1.place(x=10,y=250)
        
        lblInfo4 = Label(root, text="Boton izquierdo: etiquetar")
        lblInfo4.place(x=10, y=415)
        lblInfo4 = Label(root, text="Boton derecha: borrar")
        lblInfo4.place(x=10, y=440)
        # Al momento que leemos la imagen de entrada, vaciamos
        # la iamgen de salida y se limpia la selección de los
        # radiobutton
        lblOutputImage.image = ""
        
        ###############################################################################
        ########################## FUNCIÓN DE SEGMENTACIÓN ############################
        ###############################################################################
        
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
        
        global MG_
        MG_ = MG
        
        global Edges_
        Edges_ = Edges


        global parametro_1
        global parametro_2
        global parametro_3
        global parametro_4
         
        print(parametro_1)
        region_size_ = int(parametro_1)
        print(region_size_)
        ruler_ = int(parametro_2)
        print(ruler_)
        num_iterations = int(parametro_3)
        min_element_size = int(parametro_4)
        
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
        global Labels_slic
        Labels_slic = label_slic
        
        number_slic = slic.getNumberOfSuperpixels()  # Obtenga el número de superpíxeles
        mask_inv_slic = cv2.bitwise_not(mask)  
        img_slic = cv2.bitwise_and(image,image,mask =  mask_inv_slic)

        global image_seg
        image_seg = img_slic.copy()
        
        global respaldo_img
        respaldo_img = img_slic.copy()           

       
        imageToShowOutput = cv2.cvtColor(img_slic, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(imageToShowOutput)
        img = img.resize((600,600))
        img = ImageTk.PhotoImage(image=img)
        lblOutputImage.configure(image=img)
        lblOutputImage.image = img

        
        # Label IMAGEN DE SALIDA
        lblInfo3 = Label(root, text="Imagen Salida:", font="bold")
        lblInfo3.grid(column=1, row=1, padx=5, pady=5)

###############################################################################
########################### FUNCIÓN DE BORRAR #################################
###############################################################################

def borrado(event,SLIC):

    global respaldo_img
    respaldo = respaldo_img
    
    global image_seg
    imge_ = image_seg 
    
    global midataframe
    datos = midataframe
    
    global Labels_slic
    labels = Labels_slic
    
    global tam_x
    Rate_x = tam_x/600
    global tam_y
    Rate_y = tam_y/600
    
    y_s = int(event.y)
    x_s = int(event.x)
    
    y_s = int(y_s*Rate_x)
    x_s = int(x_s*Rate_y)
    print('coordenada borrada',x_s,y_s)
    
    ### Aquí vamos a tener el indice para poder borrar y buscar donde lo estamos haciendo
    
    erosion_size = 7
    element_erode = cv2.getStructuringElement(cv2.MORPH_RECT,(2 * erosion_size + 1, 2 * erosion_size + 1),(erosion_size, erosion_size))
    mask_p = labels == labels[y_s,x_s]
    mask_p = mask_p.astype(dtype=np.uint8)
    mask_p = cv2.erode(mask_p, element_erode )
    Mask_p = mask_p*255
        

    indice = labels[y_s,x_s]
    
    indices = list(midataframe.iloc[:, 1])
    print('indice en borrado',labels[y_s,x_s]) 
    ind = indices.index(indice)
    print('borrado',ind)
    datos = datos.drop(ind,axis=0)
    datos.reset_index(inplace=True, drop=True)    
    global midaframe
    midataframe = datos
    
    print(datos)
    
    col,row = np.where(Mask_p[:,:] == 255)
    respaldo[col,row,:]= imge_[col,row,:]
        

    imageToShowOutput_ = cv2.cvtColor(respaldo, cv2.COLOR_BGR2RGB)
    img_ = Image.fromarray(imageToShowOutput_)
    img_ = img_.resize((600,600))
    img_ = ImageTk.PhotoImage(image=img_)
    lblOutputImage.configure(image=img_)
    lblOutputImage.image = img_

###############################################################################
###############################################################################
######################### Función pinta pixeles ###############################
######################### Y recupera vector     ###############################
    
def coords(event,imagen,cat):

        global image
        
        global Labels_slic
        labels = Labels_slic
        
        print (event.x,event.y)
        
        global tam_x
        Rate_x = tam_x/600
        global tam_y
        Rate_y = tam_y/600
        
        y_s = int(event.y)
        x_s = int(event.x)
        
        y_s = int(y_s*Rate_x)
        x_s = int(x_s*Rate_y)
        
        print('coor en marc',x_s,y_s)

        print("Versión de OpenCV instalada:", cv2.__version__)
        
        ### Aquí vamos a tener el indice para poder borrar y buscar donde lo estamos haciendo
        ### Variable global
        print('indice en coor', labels[y_s,x_s])
        indice = labels[y_s,x_s] 
        erosion_size = 7
        element_erode = cv2.getStructuringElement(cv2.MORPH_RECT,(2 * erosion_size + 1, 2 * erosion_size + 1),(erosion_size, erosion_size))
        
        mask_p = labels == labels[y_s,x_s]
        mask_p = mask_p.astype(dtype=np.uint8)
        mask_p = cv2.erode(mask_p, element_erode )
        Mask_p = mask_p*255

        #cv2.imshow('Mascara',mat=Mask_p)
        row_e,col_e = np.where(Mask_p[:,:] == 255)
        
        global respaldo_img
        respaldo = respaldo_img
        
        mean_rgb = (round(np.mean(respaldo[row_e,col_e,2]),6),
                    round(np.mean(respaldo[row_e,col_e,1]),6),
                    round(np.mean(respaldo[row_e,col_e,0]),6))
        std_rgb = (round(np.std(respaldo[row_e,col_e,2]),6),
                   round(np.std(respaldo[row_e,col_e,1]),6),
                   round(np.std(respaldo[row_e,col_e,0]),6))
        
        #cv2.namedWindow('Superpixel', cv2.WINDOW_NORMAL)
        #cv2.imshow('Superpixel',respaldo[(min(row_e)-20):(max(row_e)+20),(min(col_e)-20):(max(col_e)+20),:] )
        #superpixel = respaldo[(min(row_e)-20):(max(row_e)+20),(min(col_e)-20):(max(col_e)+20),:]
        #cv2.imwrite('superpixel.jpg',superpixel)
        
        gray_img = color.rgb2gray(image) # Se pasa la imagen a escala de grises
        gray_img = (gray_img*255).astype(dtype=np.uint8) #Se convierte en valores enteros de 0 a 255

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

        # Mostrar la región recortada
        #cv2.imshow('Región Recortada', gray_rect)

        ############################################################
        # Para obtener el rectángulo vertical/horizontal de menor área que 
        # cubre al superpixel se utilizan las siguientes líneas
        ############################################################

       
        #gray_rect = gray_img[min(row_e):max(row_e),min(col_e):max(col_e)]
        #cv2.line(respaldo,(min(row_e),min(col_e)),(min(row_e),max(col_e)),(255,0,0),4)
        #cv2.imshow('Gris',respaldo[(min(row_e)-30):(max(row_e)+30),(min(col_e)-30):(max(col_e)+30),:])
        
        #cv2.imshow('Gris_original',respaldo[(min(row_e)-30):(max(row_e)+30),(min(col_e)-30):(max(col_e)+30),:])
        

        distances=[1,2,3,4,5]
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]        
        
        global len_dist
        global len_ang

        len_dist = len(distances)
        len_ang = len(angles)

        
        
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
        asm = np.round(graycoprops(glcm,'ASM').reshape(-1, order='F'),6)

        gray_feature = np.concatenate((dissimilarity,correlation,contrast,energy,homogeneity,asm))

        print(mean_rgb)
        ## Lo vamos a necesitar para ubicar los pixeles
        if(cat == 0):
            #'Arbol'
            #'Verde'
            r=69
            g=139
            b=0
        if(cat == 1):
            # 'Café'
            # 'Suelo Desnudo'
            #255,153,18
            r= 18
            g= 153
            b= 255
        if(cat == 2):
            # Gris
            # Pavimento
            # 104,131,139
            r = 139
            g = 131
            b = 104
        if(cat == 3):
            #Azul
            #Cuerpo deAgua
            # 61,89,171
            r = 171
            g = 85
            b = 61
        if(cat == 4):
            #Techo de Lamina
            #205,104,137
            # rgb
            r=137
            g=104
            b=205
        if(cat == 5):
            #Techo loza
            #128,0,0
            r=0
            g=0
            b=128
        if(cat == 6):
            #Sin etiqueta
            #Negro
            r=0
            g=0
            b=0
        if(cat == 7):
            r=0
            g=128
            b=128
        if(cat == 8):
            r = 3
            g = 3
            b = 3
        if(cat == 9):
            r = 255
            g = 253
            b = 18
            
        col,row = np.where(Mask_p[:,:] == 255)
        ### El orden es BGR
        respaldo[col,row,0] = r
        respaldo[col,row,1] = g
        respaldo[col,row,2] = b
        
        #global respaldo_img
        #respaldo_img = respaldo
        
        global MG_
        MG = MG_
        MG_mean = np.mean(MG[row_e,col_e])
        MG_sdt = np.mean(np.std(MG[row_e,col_e]))
        
        global Edges_
        Edges = Edges_
        
        locations_edges = cv2.findNonZero(Edges[row_e,col_e])
        locations_mask = cv2.findNonZero(mask_p)
        edge_density = round(np.size(locations_edges)/np.size(locations_mask),4)
        
        Vector_feature = {'categoria':cat,
                          'indice':indice,
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

        features = list(Vector_feature.values())
        features.extend(list(gray_feature))
        
        global midataframe
        ### Guardamos el vector de característias
        midataframe = midataframe
        #midataframe.loc[len(midataframe.index)] = Vector_feature

        df_vector = pd.DataFrame(np.array(features).reshape(1,len(features)), columns=midataframe.columns)
        midataframe = pd.concat([midataframe,df_vector],ignore_index=True)
        
        global nombre_arc, name_image
        midataframe.to_csv(nombre_arc+'_'+name_image+'.csv',index=False) ### Guardamos los vectores
        print(midataframe)
        
        
        imageToShowOutput_ = cv2.cvtColor(respaldo, cv2.COLOR_BGR2RGB)
        img_ = Image.fromarray(imageToShowOutput_)
        img_ = img_.resize((600,600))
        img_ = ImageTk.PhotoImage(image=img_)
        lblOutputImage.configure(image=img_)
        lblOutputImage.image = img_
        cv2.imwrite(nombre_arc+'_'+name_image+'.tif', respaldo)
        ### Respaldo es la imagen con colores

###############################################################################
###############################################################################
######################### Función que segmenta ################################

def etiqueta_segmento():
    
    global categoria_et
    categoria = categoria_et
    
    global respaldo_img
    img_slic = respaldo_img
    
    
    global Labels_slic
    slic = Labels_slic
    
    imageToShowOutput = cv2.cvtColor(img_slic, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(imageToShowOutput)
    img = img.resize((600,600))
    img = ImageTk.PhotoImage(image=img)
    lblOutputImage.configure(image=img)
    lblOutputImage.image = img
    lblOutputImage.bind('<Button-1>',  lambda event, imagen = img_slic,cat=categoria: coords(event,imagen,cat))
    lblOutputImage.bind('<Button-3>',  lambda event, SLIC = slic : borrado(event,SLIC))
    
    # Label IMAGEN DE SALIDA
    lblInfo3 = Label(root, text="Imagen Salida:", font="bold")
    lblInfo3.grid(column=1, row=1, padx=5, pady=5)
    
def nuevo_proceso():
    len_dist = 5
    len_ang = 4
    columns = ['categoria','indice','media_r','media_g','media_b','std_r','std_g','std_b','mean_gb','std_mg','density']

    features_gray = ['dissimilarity','correlation','contrast','energy','homogeneity','asm']

    columns_gray = [f'{feature}_dist{len}_ang{ang}' for feature in features_gray for len in range(len_dist) for ang in range(len_ang)]
    columns.extend(columns_gray)

    datos = pd.DataFrame(columns=columns)

    global midataframe
    midataframe = datos     
    

###############################################################################
###############################################################################
#########################  VARIABLES GLOBALES #################################
image = None
archivo  = None 
respaldo_img = None 
mat_prueba_ = None
image_seg = None
categoria_et = None
categorias = None
parametro_1 = None
parametro_2 = None
parametro_3 = None
parametro_4 = None
nombre_arc = None
name_image = None
tam_y = None
tam_x = None
MG_ = None
Edges_ = None
Labels_slic = None

len_dist = 5
len_ang = 4
columns = ['categoria','indice','media_r','media_g','media_b','std_r','std_g','std_b','mean_gb','std_mg','density']

columns_gray = []
features_gray = ['dissimilarity','correlation','contrast','energy','homogeneity','asm']

columns_gray = [f'{feature}_dist{len}_ang{ang}' for feature in features_gray for len in range(len_dist) for ang in range(len_ang)]
columns.extend(columns_gray)

midataframe = pd.DataFrame(columns=columns)

                             
## Creamos la ventana
root = Tk()
root.title('Herramienta de Etiquedo.SLIC')
root.geometry('900x700') 

# Label es donde cargamos la imagen de entrada
lblInputImage = Label(root) ## Ventana donde la veremos
lblInputImage.place(x=10,y=270)

# Label donde se presentará la imagen de salida
lblOutputImage = Label(root)
lblOutputImage.grid(column=1, row=3, rowspan=8)

# Label ¿Qué categoría quieres etiquetar?
lblInfo2 = Label(root, text="¿Qué categoría vas a marcar?", width=25)
lblInfo2.place(x = 10, y = 110)

btn = Button(root, text="2.-Elegir imagen", width=25, command = elegir_imagen)
btn.place(x = 5,y = 40)

btn_a = Button(root, text = '1.-Elegir archivo', width=25, command = elegir_archivo)
btn_a.grid(column=0, row=0, padx=5, pady=5)

btn_a_r = Button(root, text = '3.-Recuperar archivo', width=25, command = recuperacion_etiquetado)
btn_a_r.place(x = 5,y = 80)


boton = Button(text="Etiquetar", command = etiqueta_segmento )
boton.place(x = 10, y = 200)


boton_nuevo_pross = Button(text="Nuevo Elemento", command = nuevo_proceso )
boton_nuevo_pross.place(x = 10, y = 500)

root-mainloop()
