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


back=cv2.imread("bc.jpg")



def approche_simple(frame,bk,th,fig_place):
    #frame=gaussian_filter(frame, sigma=2)
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #frame.resize((bk.shape[0],bk.shape[1]))
    bk=cv2.cvtColor(bk,cv2.COLOR_BGR2GRAY)
    #print(type(frame),"et type de ",type(bk))
    #print(np.subtract(frame,bk).shape)
    v=abs(frame-bk)
    v=np.where(v>th,1,0)
   
    v=gaussian_filter(v, sigma=2)
    fig, ax = plt.subplots(1,1)
    plt.axis('off')
    plt.imshow(v,interpolation='nearest',cmap='gray')
    fig_place.pyplot(fig)
    plt.close()


def provide_video():
    count=1
    video_tr=None

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
        cv2.imwrite("frame%d.jpg"%count,frame)#sauvgarder
        count+=1 
        
        #apres sauvgarde du background frame0
        video_tr=approche_simple(frame,back,th,fig_place)
        vid_area.image(frame, channels='BGR',width=348)
        if stop:
            break
    cam.release()
    cv2.destroyAllWindows()



##### application Web ##########"
st.sidebar.title("Video Processing app")
classe=st.sidebar.selectbox("choisir le mode de traitement",['','old school','new school'])
if classe=="old school":
    approche=st.sidebar.selectbox('choix',['approche simple','difference d\'image','filtre moyen','filtre median','la moyenne glissante'])
    provide_video()




    
    
    
    
