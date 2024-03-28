from pydantic import BaseModel
from typing import Optional
from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI, File, UploadFile
from app.model.model import predict_pipeline
import numpy as np
import pandas as pd
import cv2
import io
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import shutil
import uvicorn

app = FastAPI()

orings = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = orings,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

@app.get("/")
def home():
    return {"health_check": "Sandy is Here"}

 

class PredictionRequest(BaseModel):
    file: UploadFile

# Define the base directory where prediction folders will be created
base_directory_faulty_data = "faulty_data"
# Define the base directory where prediction folders will be created
base_directory_all_data = "all_data"

@app.post("/predict/")
async def predict(file: UploadFile):
    try:
        file_name = file.filename
        content = await file.read()
        csv_file_name = io.StringIO(content.decode("utf-8"))

        # Perform your processing using the loaded model here
        faulty_indexes, faulty_dic_list, all_indxs, all_signal_dic, drift_data = predict_pipeline(csv_file_name)

        # Define the base directory for saving data
        base_directory_faulty_data = "/GF_Usecase/backend_fastapi/app/faulty_data"
        base_directory_all_data = "/GF_Usecase/backend_fastapi/app/all_data"

        # Ensure the base directory exists within the container
        os.makedirs(base_directory_faulty_data, exist_ok=True)
        os.makedirs(base_directory_all_data, exist_ok=True)

        # Clean up the old prediction folders (delete previous data)
        for folder_name in os.listdir(base_directory_faulty_data):
            folder_path = os.path.join(base_directory_faulty_data, folder_name)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
        
        for folder_name in os.listdir(base_directory_all_data):
            folder_path = os.path.join(base_directory_all_data, folder_name)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)

        # Iterate through each faulty index and create a folder for it
        for faulty_index, (index, data_dict) in zip(faulty_indexes, enumerate(faulty_dic_list)):
            new_folder = os.path.join(base_directory_faulty_data, f"prediction_{faulty_index}")
            os.makedirs(new_folder, exist_ok=True)

            # Save each dictionary in faulty_dic_list as a separate JSON file in the new folder
            filename = os.path.join(new_folder, f"faulty_index_{index}.json")
            with open(filename, "w") as json_file:
                json.dump(data_dict, json_file)
                
        # Iterate through each all index and create a folder for it
        for all_indx, (index, data_dict) in zip(all_indxs, enumerate(all_signal_dic)):
            new_folder = os.path.join(base_directory_all_data, f"prediction_{all_indx}")
            os.makedirs(new_folder, exist_ok=True)

            # Save each dictionary in all_signal_dic_list as a separate JSON file in the new folder
            filename = os.path.join(new_folder, f"index_{index}.json")
            with open(filename, "w") as json_file:
                json.dump(data_dict, json_file)

        return JSONResponse(content={"Uploaded File":file_name,"Drift Data": drift_data,"faulty_index":faulty_indexes}, status_code=201)
    except Exception as e:
        return JSONResponse(content={"message": f"An error occurred: {str(e)}"}, status_code=500)
