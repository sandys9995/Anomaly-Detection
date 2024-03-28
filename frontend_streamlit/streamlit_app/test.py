import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import requests
import json
from pathlib import Path
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time



BASE_DIR = Path(__file__).resolve(strict=True).parent

############################# correct data for comparasion ###############################
corr_data_dir= f"{BASE_DIR}/2020-11-25_09-10-56-DTT-MILLP500U-X.csv"

column_indices = [0, 1, 2, 4, 6, 8, 9]
expected_columns = ['Time','PosDiff/X', 'v act/X', 's act/X','a act/X', 'I (nom)/X', 's diff/X']
initial_time = -464151000.0

df_corr = pd.read_csv(corr_data_dir,delimiter=";", usecols=column_indices)
df_corr.columns= expected_columns
df_corr = df_corr[df_corr['Time'] >= initial_time].reset_index(drop=True)



peaks, _ = find_peaks(df_corr.iloc[:,3])
#peaks_idx.append(peaks)
valleys, _ = find_peaks(-df_corr.iloc[:,3])  # Find valleys by negating the signal data
#valleys_idx.append(valleys)

a= len(df_corr.iloc[:,3][valleys][(df_corr.iloc[:,3][valleys] > 50) & (df_corr.iloc[:,3][valleys] < 600) & (df_corr.iloc[:,3][valleys].index<20000)].index)
b= len(df_corr.iloc[:,3][peaks][(df_corr.iloc[:,3][peaks] > 500) & (df_corr.iloc[:,3][peaks] < 600) & (df_corr.iloc[:,3][peaks].index>20000) & (df_corr.iloc[:,3][peaks].index<80000)].index) 

warm_len= a + b

start_point = warm_len
end_point = peaks.shape[0]

# Calculate the step size based on the number of steps (3 in this case)
step_size = 3
# Create the array with multiples of 3 till 15 times
req_segment= np.arange(warm_len, end_point,step_size)


exp_corr = []

for j in range(0,5):
    exp_corr.append(pd.DataFrame(df_corr[valleys[req_segment[j]]: valleys[req_segment[j+1]]]).reset_index(drop= True))
for j in range(5,10):
    exp_corr.append(pd.DataFrame(df_corr[valleys[req_segment[j]]: valleys[req_segment[j+1]]]).reset_index(drop=True))
for j  in range(10,15):
    exp_corr.append(pd.DataFrame(df_corr[peaks[req_segment[j]]: peaks[req_segment[j+1]]]).reset_index(drop=True))


exp_corr1_1= (exp_corr[0][exp_corr[0].iloc[:,3]> 332].reset_index(drop= True))
exp_corr1_2= (exp_corr[1][exp_corr[1].iloc[:,3]>323])
exp_corr1_3= (exp_corr[2][exp_corr[2].iloc[:,3]>310])
exp_corr1_4= (exp_corr[3])
exp_corr1_5= (exp_corr[4][exp_corr[4].iloc[:,3]>250])

exp_corr_1 = [exp_corr1_1, exp_corr1_2, exp_corr1_3, exp_corr1_4, exp_corr1_5] 

exp_corr_2= []
exp_corr_3= []

for i in range(5,15):
    if i < 10:
        exp_corr_2.append(exp_corr[i])
    else :
        exp_corr_3.append(exp_corr[i])

experiments_corr= [exp_corr_1,exp_corr_2,exp_corr_3]

#########################################################################################

def read_json_file(file_path):
    try:
        data = pd.read_json(file_path).reset_index(drop=True)
        return data
    except Exception as e:
        st.error(f"An error occurred while reading JSON data from {file_path}: {str(e)}")
        return None

rootdir = "/GF_Usecase/backend_fastapi/app/faulty_data"
rootdir1= "/GF_Usecase/backend_fastapi/app/all_data"

##################### for Faulty signals only ###########################################
all_faults = []
location = []

for root, directories, files in os.walk(rootdir):
    for file_name in files:
        location.append(root)
        all_faults.append(file_name)

def sorting_location(path):
    # Split the path by '_' and get the last part
    parts = path.split('_')
    # Convert the last part to an integer for sorting
    return int(parts[-1])

def sorting_file_name(path):
    # Split the path by '_' and get the last part
    parts = (path.replace(".","_")).split("_")
    # Convert the last part to an integer for sorting
    return int(parts[-2])

# Sort the file_paths based on the end number
sorted_loc = sorted(location, key=sorting_location)
sorted_file_name = sorted(all_faults, key=sorting_file_name)

loc_faulty_signal = [os.path.join(x, y) for x, y in zip(sorted_loc, sorted_file_name)]

defective_index = []
experiments = []

for i, file_path in enumerate(loc_faulty_signal):
    defective_index.append(sorting_location(location[i]))
    defective_index = sorted(defective_index)
    data = read_json_file(file_path)
    if data is not None:
        experiments.append(data)

########################## All Signals ###############################################

all_signals= []
location = []

for root, directories, files in os.walk(rootdir1):
    for file_name in files:
        location.append(root)
        all_signals.append(file_name)

def sorting_location(path):
    # Split the path by '_' and get the last part
    parts = path.split('_')
    # Convert the last part to an integer for sorting
    return int(parts[-1])

def sorting_file_name(path):
    # Split the path by '_' and get the last part
    parts = (path.replace(".","_")).split("_")
    # Convert the last part to an integer for sorting
    return int(parts[-2])

# Sort the file_paths based on the end number
sorted_loc = sorted(location, key=sorting_location)
sorted_file_name = sorted(all_signals, key=sorting_file_name)

loc_all_signals = [os.path.join(x, y) for x, y in zip(sorted_loc, sorted_file_name)]

all_index = []
experiments_all = []

for i, file_path in enumerate(loc_all_signals):
    all_index.append(sorting_location(location[i]))
    all_index = sorted(all_index)
    data = read_json_file(file_path)
    if data is not None:
        experiments_all.append(data)

########################################## Dashboard ###########################################
import requests

@st.cache_data
def get_predictions(uploaded_file):
    # Function to send request to FastAPI
    api_endpoint = "http://fastapi:80/predict/"  # Replace with your FastAPI endpoint
    files = {'file': uploaded_file}
    response = requests.post(api_endpoint, files=files)
    return response

def main():

    st.title('Dashboard')
    st.subheader('Anomaly Detection')

        # Initialize session state variables
    if 'run_button_state' not in st.session_state:
        st.session_state.run_button_state = False

    # File Upload Section
    uploaded_file = st.file_uploader("Upload a CSV file to process", type=["csv"])
    run_button = st.button('Run')

    if uploaded_file is not None and run_button:
        with st.spinner("Processing the uploaded file..."):
            # Simulate processing by waiting for a few seconds (Replace this with your actual processing logic)
            time.sleep(5)
            st.session_state.run_button_state = True

    # Check if file processing is done and display the appropriate message
    if uploaded_file is not None and st.session_state.run_button_state:
        processing_message = st.empty()

        if st.session_state.run_button_state:
            # Simulating completion after a delay
            time.sleep(3)
            processing_message.write("File processing done.")

            try:
                # Get predictions using the cached function
                response = get_predictions(uploaded_file)
                
                if response.status_code in {201, 200}: 
                    #st.write("Response from FastAPI:")
                    
                    # Once the file upload and response are complete, you can display your charts here.
                    # Create widgets for user interaction
                    
                    uploaded_file_name= uploaded_file.name

                    substrings_to_check = ["MILLP500", "MillP500", "Mill_P500"]
                    date = uploaded_file_name.split('_')[0]
                    axis= uploaded_file_name.replace('.',"-").split('-')[-2]

                    # Check if any of the substrings is present in the string
                    found_substring = any(substring in uploaded_file_name for substring in substrings_to_check)

                    if found_substring:
                        Machine_type= "MILL P 500 U"
                    else:
                        Machine_type= 'None'
                    
            
                    # Create widgets for user interaction in the sidebar
                    with st.sidebar:
                        experiment_selector = st.radio('Application Function Menu', ['Outlier Detection','Data Drift Detection','Model Management'])
                        selected_machine_type = st.radio('Machine Type', [Machine_type])
                        selected_date = st.radio('Date', [date])
                        selected_axis = st.radio('Axis', [axis])
                        st.write("NOTE: Version 2.0")
                        
                    # Display the selected options
                    st.markdown(f'**Feature:** {experiment_selector}')
                    
                    # Loop through defective indexes and display plots based on user selections
                    if experiment_selector == 'Outlier Detection':

                        new_title = '<p style="font-family:sans-serif; color:Black; font-size: 12px;">Please Press "RUN" to load the new results when second file will be uploaded</p>'
                        st.markdown(new_title, unsafe_allow_html=True)

                        def classify_defect(x):
                            if x <= 4:
                                return "Low"
                            elif 5 <= x < 10:
                                return "Medium"
                            else:
                                return "High"
                        st.markdown('- **Total Section:** 15')
                        st.markdown(f'- **Total Defect:** {len(defective_index)}')
                        st.markdown(f'- **Defect Indicator:** {classify_defect(len(defective_index))}')

                        columns = {'PosDiff/X': 1, 'V act/X': 2, 'S act/X': 3, 'A act/X': 4, 'I (nom)/X': 5, 'S diff/X': 6}
                        selected_signal_graph={1:'pos.png', 2:'V_act.png', 3:'S_act.png', 4:'a_act.png', 5:'I_Nom.png', 6:'S_diff.png'}

                        # Set default value to 'S act/X'
                        default_signal_type = 'S act/X'
                        selected_col = st.selectbox('Signal Type', options=list(columns.keys()), index=list(columns.keys()).index(default_signal_type))

                        analysis= ['Outliers Only','Outlier and Non-Outlier']
                        choice = st.selectbox('Analyze Mode', analysis)
            

                        if choice== 'Outliers Only' :
                            signal_name={'pos.png':'**PosDiff:** Difference between position (in a linear scale) and speed encoder', 
                                        'V_act.png':'**V act:** Actual value of the axis feed rate calculated from the position encoder', 
                                        'S_act.png':'**S act:** Actual position', 
                                        'a_act.png':'**A act:** Actual acceleration value calculated from the position encoder', 
                                        'I_Nom.png':'**I nom:** Nominal current value, which determines torque', 
                                        'S_diff.png':'**S diff:** Following error of the position controller'}

                            image= selected_signal_graph[columns[selected_col]]
                            # Your image file path
                            st.markdown(f'{signal_name[image]}')
                            image_path = f"{BASE_DIR}/{image}"
                            st.image(image_path,channels="BGR")
                            for i, indx in enumerate(defective_index):
                                try:
                                    data = experiments[i].iloc[:, columns[selected_col]]
                                    data1 = experiments_corr[indx // 5][indx % 5].iloc[:, columns[selected_col]]

                                    # Create a DataFrame for this iteration's data
                                    df = pd.DataFrame({'Test Signal': data, 'Ideal Signal': data1})

                                    # Display the plot for this iteration
                                    st.markdown(f'**Section {indx+1}: Outlier Detected**')


                                    st.line_chart(df, use_container_width=True)
                                except Exception as e:
                                    st.error(f"An error occurred while plotting: {str(e)}")
                        else :

                            for indx in (all_index):
                                try:
                                    data = experiments_all[indx].iloc[:, columns[selected_col]]
                                    data1 = experiments_corr[indx // 5][indx % 5].iloc[:, columns[selected_col]]

                                    # Create a DataFrame for this iteration's data
                                    df = pd.DataFrame({'Test Signal': data, 'Ideal Signal': data1})

                                    # Display the plot for this iteration
                                    if indx in defective_index:
                                        st.markdown(f'**Section {indx+1}: Outlier**')
                                    else:
                                        st.markdown(f'**Section {indx+1}: Non-Outlier**')

                                    st.line_chart(df, use_container_width=True)
                                except Exception as e:
                                    st.error(f"An error occurred while plotting: {str(e)}")

                    elif experiment_selector == 'Data Drift Detection':
                            
                            st.markdown('''
                                        - **Drift_Status = True**:
                                            when Error_Mse **>** Data_Drift_Threshold
                                        - **Drift_Status = False**:
                                            when Error_Mse **<** Data_Drift_Threshold
                                        ''')
                            response_dict = json.loads(response.text)

                            # Now response_dict is a Python dictionary
                            #st.write(pd.DataFrame(list(response_dict.items())[1][1]))
                            dataframe = pd.DataFrame(list(response_dict.items())[1][1])
                            dataframe['Section Index'] = range(1, 16)
                            dataframe = dataframe.set_index('Section Index')
                            st.write(dataframe)
            
                    else :

                        description = """
                        **Columns Description:**

                        These columns represent various aspects of a system's behavior and control. Below is a list of the column names and their corresponding descriptions:

                        1. **Time[us]:** Relative timestamp representing time in microseconds.
                        2. **PosDiff:** Difference between position (in a linear scale) and speed encoder.
                        3. **v act:** Actual value of the axis feed rate calculated from the position encoder.
                        4. **v act vctrl:** Shaft speed actual value calculated from the rotary speed encoder.
                        5. **s act:** Actual position.
                        6. **s nom:** Nominal position, which serves as the reference.
                        7. **a act:** Actual acceleration value calculated from the position encoder.
                        8. **M act:** Actual torque value.
                        9. **Iq nom:** Nominal current value, which determines torque.
                        10. **s diff:** Following error of the position controller.

                        """

                        st.markdown(description)

                        # Your image file path
                        st.markdown('- **Confusion Matrix**')
                        image_path = f"{BASE_DIR}/Confusion_Matrix2.png"
                        st.image(image_path,channels="BGR")

                        st.markdown('- **Correlation Matrix**')
                        image_path = f"{BASE_DIR}/correlation_matrix.png"
                        st.image(image_path,channels="BGR")
                        
                        st.markdown('''**Note: Only 6 Columns have been selected for features developement 
                                    due to highly correlation and the value of s(diff) is same in each rows.
                                    ['PosDiff/X', 'V-act/X', 'S-act/X', 'A-act/X', 'I (nom)/X', 'S-diff/X’]**''')

                    
                else:
                    st.markdown('**Press Run to see the results**')
                    #st.error(f"Failed to upload file. Status code: {response.status_code}")
            except Exception as e:
                st.error(f"An error occurred while uploading the file: {str(e)}")

if __name__ == "__main__":
    main()

