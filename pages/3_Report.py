import streamlit as st
from Home import face_rec
import pandas as pd

st.set_page_config(page_title='Reporting',layout='wide')
st.subheader('Reporting') 

#retrive logs data and show report.py
#extract data from redis list 
name = 'attendance:logs'
def load_logs(name,end=-1):
    logs_list = face_rec.r.lrange(name,start=0,end=end) #extract data from redis database 
    return logs_list


#tabs to show info
tab1 , tab2, tab3 = st.tabs(['Registered_Data','Logs','Attendance Report'])

with tab1:
    if st.button ('Refresh Data'):
        with st.spinner('Retriving Data from DB....'):
            redis_face_db = face_rec.retrive_data(name='academy:register')
            st.dataframe(redis_face_db[['Name','Role']])

with tab2:
    if st.button('Refresh Logs'):
        st.write(load_logs(name=name))


with tab3:
    st.subheader('Attendance Report')
    #load logs into attribute  logs list
    logs_list = load_logs(name=name)


    #step 1 conver the logs thaa in list of bytes into list of string
    convert_byte_to_string = lambda x: x.decode('utf-8')
    logs_list_string = list(map(convert_byte_to_string, logs_list))

    #step 2 string by @ and create nested list
    split_string = lambda x: x.split('@')
    logs_nested_list = list(map(split_string, logs_list_string))

    #convert nested list info into dataframe
    logs_df = pd.DataFrame(logs_nested_list, columns= ['Name','Role','Timestamp']) 


    #step 3 time analysys or Report
    logs_df['Timestamp'] = pd.to_datetime(logs_df['Timestamp'])
    logs_df['Date'] = logs_df['Timestamp'].dt.date

    # STEP 3-1 : CAL INTIME AND OUT TIME
    # in time : At Which Person is first detected in that day (min TimeStamp of The date)
    # Out Time :  At Which Person is last detected in that day (min TimeStamp of The date)

    report_df = logs_df.groupby(by=['Date','Name','Role']).agg(
        In_time = pd.NamedAgg('Timestamp','min'), #min time
        Out_time = pd.NamedAgg('Timestamp','max') #out time
    ).reset_index()

    report_df['In_time'] = pd.to_datetime(report_df['In_time'])
    report_df['Out_time'] = pd.to_datetime(report_df['Out_time'])

    report_df['Duration'] = report_df['Out_time'] - report_df['In_time']
    
    #step 4 marking person is present or absen
    all_dates = report_df['Date'].unique()
    name_role = report_df[['Name','Role']].drop_duplicates().values.tolist()

    date_name_rol_zip = []
    for dt in all_dates:
        for name, role in name_role:
            date_name_rol_zip.append([dt, name, role])

    date_name_rol_zip_df = pd.DataFrame(date_name_rol_zip,columns=['Date','Name','Role'])

    # left join  with report DF

    date_name_rol_zip_df = pd.merge(date_name_rol_zip_df, report_df, how='left',on=['Date','Name','Role'])

    #duration
    #hours
    date_name_rol_zip_df['Duration_seconds'] = date_name_rol_zip_df['Duration'].dt.seconds
    date_name_rol_zip_df['Duration_hours'] = date_name_rol_zip_df['Duration_seconds'] / (60*60)

    def status_marker(x):
        if pd.Series(x).isnull().all():
            return 'Absent'
        elif x > 0 and x < 1:
            return 'Absent(Less than 1 hour)'
        elif x >= 1 and x < 4:
            return 'Half Day (Less than 4 hour)'
        elif x >= 4 and x < 6:
            return 'Half Day'
        elif x == 0: 
            return 'tidak absen pulang'
        elif x >= 6:
            return 'Present'
        
    date_name_rol_zip_df['Status'] = date_name_rol_zip_df['Duration_hours'].apply(status_marker)

    st.dataframe(date_name_rol_zip_df)
        
        

      
