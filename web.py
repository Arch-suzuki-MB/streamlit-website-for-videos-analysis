from pickletools import uint8
from tkinter.tix import COLUMN
import streamlit as st
from timeit import default_timer as timer
import cv2
import numpy as np
import threading
from matplotlib import pyplot as plt
from matplotlib import image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def approche_simple(frame,bk,th,fig_place):
    #frame=gaussian_filter(frame, sigma=2)
    if len(frame.shape)==3:
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #frame.resize((bk.shape[0],bk.shape[1]))
    if len(bk.shape)==3:
        bk=cv2.cvtColor(bk,cv2.COLOR_BGR2GRAY)
    #print(type(frame),"et type de ",type(bk))
    #print(np.subtract(frame,bk).shape)
    v=abs(frame-bk)
    #v=np.where(v>th,255,0)
    
    
    v= cv2.threshold(v, th, 1, cv2.THRESH_BINARY)[1]
    v=gaussian_filter(v, sigma=2)
    fig, ax = plt.subplots(1,1)
    plt.axis('off')
    plt.imshow(v,interpolation='nearest',cmap='gray')
    fig_place.pyplot(fig)
    plt.close()

                ##############approche 1##################""

def provide_video_back():
    back=cv2.imread("bc.jpg")
    video_tr=None
    liste_frame=[]
    liste_frame.append(back) 
    cl1,cl2=st.columns((2,2))
    with cl1:
        vid_area=st.empty()
        stop=st.checkbox("Arreter la video",value=False)
    with cl2:
        fig_place = st.empty()
        th=st.slider("chosir le seuil",1,255,value=10) 
 

    cam=cv2.VideoCapture("videos.mp4")
    while cam.isOpened():
        res,frame=cam.read()#read all the video  
        liste_frame.append(frame)   
        
        #apres sauvgarde du background frame0
        video_tr=approche_simple(liste_frame[-1],liste_frame[-2],th,fig_place)
        del liste_frame[-1]
        vid_area.image(frame, channels='BGR',width=348)
        if stop:
            del liste_frame
            break
    cam.release()
    cv2.destroyAllWindows()

               #########################approche 2: pour la differencing image #####
def provide_video_stream():
    liste_frame=[]
    cl1,cl2=st.columns((2,2))
    with cl1:
        vid_area=st.empty()
        step_time = st.empty()
        stop=st.button(label="Arreter Stream")

    with cl2:
        fig_area=st.empty()
        th=st.slider("chosir le seuil",1,255,value=10) 
    
    old_time = timer()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        time = timer()
        step_time.write(f"Step time: {time - old_time:.3f}")
        old_time = time
        ret, image = cap.read()
        vid_area.image(image, channels='BGR',width=350)
        
        
           # cv2.imwrite("frame_D%d.jpg"%count,image)
        
        liste_frame.append(image)
        if len(liste_frame)>1:
            #st.write(liste_frame[-1].shape)
            approche_simple(liste_frame[-1],liste_frame[-2],th,fig_area)
            del liste_frame[-2]
                


        if stop :
            break 
    cap.release()
    cv2.destroyAllWindows()



def provide_video():

    video_tr=None
    liste_frame=[] 
    cl1,cl2=st.columns((2,2))
    with cl1:
        vid_area=st.empty()
        stop=st.checkbox("Arreter la video",value=False)
    with cl2:
        fig_place = st.empty()
        th=st.slider("chosir le seuil",1,255,value=10) 

    cam=cv2.VideoCapture("videos.mp4")
    while cam.isOpened():
        res,frame=cam.read()#read all the video  
        liste_frame.append(frame)   
        
        if len(liste_frame)>1:
            video_tr=approche_simple(liste_frame[-1],liste_frame[-2],th,fig_place)
            vid_area.image(frame, channels='BGR',width=348)
            del liste_frame[-2]
        if stop:
            del liste_frame
            break
    cam.release()
    cv2.destroyAllWindows()

###GMM ################"
def gmm():
    cl1,cl2=st.columns((2,2))
    with cl1:
        vid_area=st.empty()
        step_time = st.empty()
        stop=st.button(label="Arreter Stream")

    with cl2:
        fig_area=st.empty()
        th=st.slider("chosir le seuil",1,255,value=10) 
    
    old_time = timer()

    cap = cv2.VideoCapture(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
    fgbg = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=th,detectShadows=True)

    #fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)

    while cap.isOpened():
        ret, frame = cap.read()
        
        if frame is None:
            break
        
        vid_area.image(frame, channels='BGR',width=350)
        fgmask = fgbg.apply(frame)

        #cv2.imshow('Frame', frame)
        #cv2.imshow('FG Mask', fgmask)
        fig, ax = plt.subplots(1,1)
        plt.axis('off')
        plt.imshow(fgmask,interpolation='nearest',cmap='gray')
        fig_area.pyplot(fig)
        plt.close()
        
        if stop :
            break
        

    cap.release()

    cv2.destroyAllWindows()



####################"moyenne glissante#########
def moyenne_glissante():
    cl1,cl2=st.columns((2,2))
    with cl1:
        vid_area=st.empty()
        step_time = st.empty()
        stop=st.button(label="Arreter Stream")

    with cl2:
        fig_area=st.empty()
        #th=st.slider("chosir le seuil",1,255,value=10) 
    
    old_time = timer()

    cap = cv2.VideoCapture(0)

    # read the frames from the camera
    _, img = cap.read()
    averageValue1 = np.float32(img)

    # modify the data type
    # setting to 32-bit floating point
    
    

    # loop runs if capturing has been initialized.
    while cap.isOpened():
    # reads frames from a camera
        _, img = cap.read()
        # using the cv2.accumulateWeighted() function
        # that updates the running average
        cv2.accumulateWeighted(img, averageValue1, 0.02)

        # converting the matrix elements to absolute values
        # and converting the result to 8-bit.
        resultingFrames1 = cv2.convertScaleAbs(averageValue1)
        
        # Show two output windows
        # the input / original frames window
        vid_area.image(img, channels='BGR',width=350)
        #cv2.imshow('InputWindow', img)

        # the window showing output of alpha value 0.02
        #cv2.imshow('averageValue1', resultingFrames1)
        fig, ax = plt.subplots(1,1)
        plt.axis('off')
        plt.imshow(resultingFrames1)
        fig_area.pyplot(fig)
        plt.close()
        
        
        # Wait for Esc key to stop the program
        if stop :
            break

        # Close the window
    cap.release()
            
        # De-allocate any associated memory usage
    cv2.destroyAllWindows()




 ########## approche filtre moyen ################
def filtre_moyen():
    liste_frame=[]
    cl1,cl2=st.columns((2,2))
    with cl1:
        vid_area=st.empty()
        step_time = st.empty()
        stop=st.checkbox(label="Arreter la video")

    with cl2:
        fig_area=st.empty()
        th=st.slider("chosir le seuil",1,255,value=10) 
    
    old_time = timer()
    cap = cv2.VideoCapture("videos.mp4")
   
    while cap.isOpened():
        time = timer()
        step_time.write(f"Step time: {time - old_time:.3f}")
        old_time = time
        ret, image = cap.read()
        vid_area.image(image, channels='BGR',width=350)
        
           # cv2.imwrite("frame_D%d.jpg"%count,image)
        #ret,image=cap.read()
        liste_frame.append(image)
        if len(liste_frame)>1:
            bac=np.sum(liste_frame[i] for i in range(len(liste_frame)-1))
            bac=bac/(len(liste_frame))
            bac=bac.astype('float32')
            bac=gaussian_filter(bac, sigma=2)
            #st.write(liste_frame[-1].shape)
            approche_simple(liste_frame[-1],bac,th,fig_area)
                #del liste_frame[-2]      
        if stop :
            del liste_frame
            break 
    cap.release()
    cv2.destroyAllWindows()

def median():

    cl1,cl2=st.columns((2,2))
    with cl1:
        vid_area=st.empty()
        step_time = st.empty()
        stop=st.checkbox(label="Arreter la video")

    with cl2:
        fig_area=st.empty()
        th=st.slider("chosir le seuil",1,255,value=10) 
    

    
    cap=cv2.VideoCapture("videos.mp4")

    m=[]
    while cap.isOpened():
        _,frame=cap.read()
        median=np.zeros((frame.shape[0],frame.shape[1]))
        liste=[]
        vid_area.image(frame, channels='BGR',width=350)
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        liste.append(frame)
        
        if len(liste)>2:
            for j in range(frame.shape[0]):#ligne    
                for k in range(frame.shape[1]):#colonne
                    #p=[liste[i][j][k] for i in range(len(liste))]
                    median[j][k]=np.median([liste[i][j][k]   for i in range(len(liste))])
                    m.append(median)
                    approche_simple(frame,median,th,fig_area)
            
        del liste
            
        
        
        
    




##### application Web ##########"
st.sidebar.title("Video Processing app")
classe=st.sidebar.selectbox("choisir le mode de traitement",['','old school','new school'])
if classe=="old school":
    approche=st.sidebar.selectbox('choix',[' ','approche simple','difference d\'image','filtre moyen','filtre median','la moyenne glissante'])
    if approche=="approche simple":
        provide_video_back()
    elif approche=="difference d'image":
        type_video=st.selectbox("Stream ou upload video",["","stream video","video_telecharger"])
        if type_video=="stream video":
            provide_video_stream()
        if type_video=="video_telecharger":
            provide_video()
    elif approche=="filtre moyen":
        filtre_moyen()
    elif approche=='filtre median' :
        median()
    elif approche=='la moyenne glissante' :
        moyenne_glissante()

if classe=="new school":
    gmm()