import streamlit as st



st.set_page_config(page_title='Attendance System',layout='wide')

st.header('Attandance System using face Recognition')

with st.spinner("Loading Models and connecting to redis db..."):
    import face_rec

st.success('model loades Sucessfully')
st.success('db Sucessfully Connected')
