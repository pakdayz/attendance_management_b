import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time


st.set_page_config(page_title='Prediction',layout='centered')
st.subheader('Real Time Attendance System') 


# Retrieve data from database
with st.spinner('Retriving Data from DB....'):
    redis_face_db = face_rec.retrive_data(name='academy:register')
    st.dataframe(redis_face_db)

st.success('Data succesfully retrived from DB')

#time
waitTime = 5
setTime = time.time()
realtimepred = face_rec.RealTimePred()  #real time prediction class

# Realtime Prediction

#streamlit webrtc


#callback function
def video_frame_callback(frame):
    global setTime
    img = frame.to_ndarray(format="bgr24") # 3 dimension numpy array
    #opperation  that you can perform on array
    pred_img = realtimepred.face_prediction(img,redis_face_db,
                                        'facial features',['Name','Role'],thresh=0.5)
    
    timenow = time.time()
    difftime = timenow - setTime
    if difftime >= waitTime:
        realtimepred.saveLogs_redis()
        setTime = time.time() #reset time
        print('Save data to DB')

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callback)