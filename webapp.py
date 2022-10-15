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

def approche_simple(frame,bk,th):
    #frame=gaussian_filter(frame, sigma=2)
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #frame.resize((bk.shape[0],bk.shape[1]))
    bk=cv2.cvtColor(bk,cv2.COLOR_BGR2GRAY)
    #print(type(frame),"et type de ",type(bk))
    #print(np.subtract(frame,bk).shape)
    v=abs(frame-bk)
    v=np.where(v>th,1,0)
    v=gaussian_filter(v, sigma=2)
    plt.imshow(v,cmap="gray")
    #plt.imsave("b.jpg",v)
    plt.show()
    return v

def provide_video():
    cam=cv2.VideoCapture("videos.mp4")
    count=1
    video_tr=None
    cl1,cl2 =st.columns(2)
    with cl1:
        th=st.slider("chosir le seuil",1,255) 
    with cl2:
        stop=st.checkbox("Arreter la video",value=False)

    #cl1,cl2=st.columns(2)
    
    fig_place = st.empty()
    while cam.isOpened():
        with lock:
            res,frame=cam.read()#read all the video
            
            vid_area.image(frame, channels='BGR')
            cv2.imwrite("frame%d.jpg"%count,frame)#sauvgarder
        count+=1 
        if stop:
            break
        
        #apres sauvgarde du background frame0
        video_tr=approche_simple(frame,back,th)
        with lock: 
            fig=plt.figure(figsize=(2,1),dpi=200)
            plt.axis('off')
                #ax.cla()
            plt.imshow(video_tr,interpolation='nearest',cmap='gray')
            fig_place.pyplot(fig)
            plt.close()
        #plt.title("frame%d"%count)
        #plt.imshow(video_tr,interpolation='nearest')"""
        #print(video_tr.shape)
        
        
        #plt.imshow(video_tr,interpolation='nearest')
        
        
    cam.release()
    cv2.destroyAllWindows()

lock = threading.Lock()
vid_area=st.empty()



provide_video()


    
    
    
    
