import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import iqr, kurtosis, skew
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sns
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path


BASE_DIR = Path(__file__).resolve(strict=True).parent


models= ['my_model_exp1_1','my_model_exp1_2','my_model_exp1_3','my_model_exp1_4','my_model_exp1_5',
        'my_model_exp2_1','my_model_exp2_2','my_model_exp2_3','my_model_exp2_4','my_model_exp2_5',
        'my_model_exp3_1','my_model_exp3_2','my_model_exp3_3','my_model_exp3_4','my_model_exp3_5']
loaded_model =[]
for model in models:
    loaded_model.append(tf.keras.models.load_model(f"{BASE_DIR}/{model}"))   

#csv_file= f"{BASE_DIR}/2018-12-07_09-40-17-DTT-MillP500U-0009-X.csv"

def predict_pipeline(Input_signal):
    #1. Define the columns that are useful after finding the correlation between all
    #2. read the incoming dataframe
    #3. Correct the columns name

    column_indices = [0, 1, 2, 4, 6, 8, 9]
    expected_columns = ['Time','PosDiff/X', 'v act/X', 's act/X','a act/X', 'I (nom)/X', 's diff/X']
    initial_time = -464151000.0

    df_x = pd.read_csv(Input_signal,delimiter=";", usecols=column_indices)
    df_x.columns= expected_columns
    df_x = df_x[df_x['Time'] >= initial_time].reset_index(drop=True)


    ##1. Calculate all the peaks and valleys
    ##2. Calculating start point of signal
 
    signal_data= df_x

    peaks, _ = find_peaks(signal_data.iloc[:,3])
    #peaks_idx.append(peaks)
    valleys, _ = find_peaks(-signal_data.iloc[:,3])  # Find valleys by negating the signal data
    #valleys_idx.append(valleys)

    a= len(signal_data.iloc[:,3][valleys][(signal_data.iloc[:,3][valleys] > 50) & (signal_data.iloc[:,3][valleys] < 600) & (signal_data.iloc[:,3][valleys].index<20000)].index)
    b= len(signal_data.iloc[:,3][peaks][(signal_data.iloc[:,3][peaks] > 500) & (signal_data.iloc[:,3][peaks] < 600) & (signal_data.iloc[:,3][peaks].index>20000) & (signal_data.iloc[:,3][peaks].index<80000)].index) 

    warm_len= a + b

    start_point = warm_len
    end_point = peaks.shape[0]

    # Calculate the step size based on the number of steps (3 in this case)
    step_size = 3
    # Create the array with multiples of 3 till 15 times
    req_segment= np.arange(warm_len, end_point,step_size)


    exp = []

    for j in range(0,5):
        exp.append(pd.DataFrame(signal_data[valleys[req_segment[j]]: valleys[req_segment[j+1]]]).reset_index(drop= True))
    for j in range(5,10):
        exp.append(pd.DataFrame(signal_data[valleys[req_segment[j]]: valleys[req_segment[j+1]]]).reset_index(drop=True))
    for j  in range(10,15):
        exp.append(pd.DataFrame(signal_data[peaks[req_segment[j]]: peaks[req_segment[j+1]]]).reset_index(drop=True))

    exp1_1= (exp[0][exp[0].iloc[:,3]> 332].reset_index(drop= True))
    exp1_2= (exp[1][exp[1].iloc[:,3]>323])
    exp1_3= (exp[2][exp[2].iloc[:,3]>310])
    exp1_4= (exp[3])
    exp1_5= (exp[4][exp[4].iloc[:,3]>250])

    exp_1 = [exp1_1, exp1_2, exp1_3, exp1_4, exp1_5] 

    exp_2= []
    exp_3= []

    for i in range(5,15):
        if i < 10:
            exp_2.append(exp[i])
        else :
            exp_3.append(exp[i])

    experiments= [exp_1,exp_2,exp_3]

    def fun_shallow_feature(x):
        # Raw signal features
        iqr_val = iqr(x)  # Interquartile Range
        kurt = kurtosis(x)  # Kurtosis
        skewness = skew(x)  # Skewness
        rms = sqrt(np.mean(np.square(x)))  # Root Mean Square
        variance = np.var(x)  # Variance
        mean_val = np.mean(x)  # Mean
        max_val = np.max(x)  # Maximum
        min_val = np.min(x)  # Minimum
        std_dev = np.std(x)  # Standard Deviation
        median_val = np.median(x)# Median
        summ= sum(x)/(10e+3)

        features = [iqr_val, kurt, skewness, rms, variance, mean_val, max_val, min_val, std_dev, median_val, summ]
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
        
        return features
    
    fea_i = []
    for l in range(3):
        for i in range(5):
            for j in range(1):
                fea_i_single = []
                for k in range(6):
                    fea_i_single.append(fun_shallow_feature(experiments[l][i].iloc[:,k+1])) 

                fea_i.append(pd.DataFrame(np.array(fea_i_single).reshape(1,66)))

    df_final_exp = pd.concat(fea_i, axis=0, ignore_index=True)
    df_final_exp = df_final_exp.add_prefix('c')


    ###APPLY THE MODEL TO THE TEST DATA FOR PREDICTION OF ANOMILIES!!!

    
    threshold_low = [2.0, 1, 1,0.6,200000,0.3,0.6,0.4,0.4,100000,1e+10, 1e+7 , 1e+6, 1e+5,2e+5]
    threshold_high = [3,2,2,1,300000,0.6,1,0.8,0.8,150000,2e+10,1.5e+7, 1.5e+6 , 1.5e+5, 3e+5]

    scales = ['scaler1.pkl','scaler2.pkl','scaler3.pkl','scaler4.pkl','scaler5.pkl',
            'scaler2_1.pkl','scaler2_2.pkl','scaler2_3.pkl','scaler2_4.pkl','scaler2_5.pkl',
            'scaler3_1.pkl','scaler3_2.pkl','scaler3_3.pkl','scaler3_4.pkl','scaler3_5.pkl']
    # Create an empty DataFrame to hold the results
    Errors_df = pd.DataFrame(columns=['Error_Mse', 'Outlier'])

    for i in df_final_exp.index:

        scaler= joblib.load(f"{BASE_DIR}/{scales[i]}")
        X_test = df_final_exp.iloc[i:i+1, :].values
        X_test = scaler.transform(X_test)
        
        
        # Predict using the appropriate loaded model
        reconstructed_data= (loaded_model[i].predict(X_test))
        
        mse = np.mean((X_test - reconstructed_data) ** 2, axis=1)
            # Classify outliers based on the thresholds
        if mse < threshold_low[i]:
            outlier_status = 'False'
        elif threshold_low[i] <= mse <= threshold_high[i]:
            outlier_status = 'Doubtful'
        else:
            outlier_status = 'True'
        
        # Create a DataFrame for the current index and append to the main DataFrame
        error_df = pd.DataFrame({'Error_Mse': mse, 'Outlier_Status': outlier_status})
        Errors_df = pd.concat([Errors_df, error_df], ignore_index=True)

        defective_index= np.array(Errors_df[Errors_df['Outlier_Status']=='True'].index).tolist()
        all_index= np.array(Errors_df.index).tolist()


        ####### list of all time interval of faulty

    
    faulty_dic = []
    for indx in defective_index:
        if indx <= 4:
            faulty_interval= []
            faulty_interval.append(experiments[0][indx % 5].iloc[:, 0])
            faulty_dic.append(df_x[df_x['Time'].isin(faulty_interval[0])].to_dict())
        elif 5 <= indx <= 9:
            faulty_interval= []
            faulty_interval.append(experiments[1][indx % 5].iloc[:, 0])
            faulty_dic.append(df_x[df_x['Time'].isin(faulty_interval[0])].to_dict())
        else:
            faulty_interval= []
            faulty_interval.append(experiments[2][indx % 5].iloc[:, 0])
            faulty_dic.append(df_x[df_x['Time'].isin(faulty_interval[0])].to_dict())


    all_signal_dic = []
    for indx in all_index:
        if indx <= 4:
            all_interval= []
            all_interval.append(experiments[0][indx % 5].iloc[:, 0])
            all_signal_dic.append(df_x[df_x['Time'].isin(all_interval[0])].to_dict())
        elif 5 <= indx <= 9:
            all_interval= []
            all_interval.append(experiments[1][indx % 5].iloc[:, 0])
            all_signal_dic.append(df_x[df_x['Time'].isin(all_interval[0])].to_dict())
        else:
            all_interval= []
            all_interval.append(experiments[2][indx % 5].iloc[:, 0])
            all_signal_dic.append(df_x[df_x['Time'].isin(all_interval[0])].to_dict())



    ######################## Data Drift ##########################


    data_drift_threshold = [5.2, 2.96, 3.54, 1.52, 538281, 
                        0.83, 1.46, 1.23, 1.16, 305532, 
                        4.27e+10, 2.79e+7, 2.32e+6, 2.65e+5, 4.41e+5]
    

    # Create an empty DataFrame to hold the results
    Data_Drift_df = pd.DataFrame(columns=['Data_Drift_Threshold','Error_Mse', 'Drift_Status'])

    for i in df_final_exp.index:

        scaler= joblib.load(f"{BASE_DIR}/{scales[i]}")
        X_test = df_final_exp.iloc[i:i+1, :].values
        X_test = scaler.transform(X_test)
        
        
        # Predict using the appropriate loaded model
        reconstructed_data= (loaded_model[i].predict(X_test))
        
        mse = np.mean((X_test - reconstructed_data) ** 2, axis=1)
        # Classify outliers based on the thresholds
        if mse < data_drift_threshold[i]:
            data_drift = 'False'
        else:
            data_drift = 'True'
        
        # Create a DataFrame for the current index and append to the main DataFrame
        frame = pd.DataFrame({"Data_Drift_Threshold":data_drift_threshold[i],'Error_Mse': mse, 'Drift_Status': data_drift})
        Data_Drift_df = (pd.concat([Data_Drift_df, frame], ignore_index=True))

    return [defective_index,faulty_dic,all_index,all_signal_dic,(Data_Drift_df.to_dict())]
    