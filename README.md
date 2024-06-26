# Anomaly-Detection
Industrializing AI: A Model Development, Deployment, and Monitoring Concept for Production Environments

### Introduction

Welcome to the repository for our predictive maintenance and anomaly detection project! In this project, we have developed a comprehensive solution for predicting and preventing equipment failures in industrial settings. Leveraging advanced machine learning techniques, our system can detect anomalies in machinery data and provide timely maintenance recommendations to prevent breakdowns and optimize operational efficiency.

This repository contains all the necessary code and resources for deploying our predictive maintenance and anomaly detection system. From model implementation to frontend and backend application setups, everything is provided here to help you understand and utilize our solution effectively.

Feel free to explore the repository and learn more about our innovative approach to enhancing equipment reliability and minimizing downtime in industrial environments.

### The following readme file explains the folders in the GF_Usecase

1) backend_fastapi: It contains all the files used to build the backend docker image.
2) frontend_streamlit-app: It contains all the files used to build the frontend docker image.
3) Codes: This folder contains all the Jupyter notebooks used to create the project with explanations.
4) Docker_images: it contains both the docker image of the backend and frontend with docker-compose. One can also find a readme file to run them successfully. 



### Model Deployment Structure

#### GF_Usecase/backend Directory:
- **app/:**
  - **model/:**
    - `model.py`: Python file containing the model implementation and functions.
    - `trained_autoencoder-model * 15`: Trained autoencoder models (15 in total).
    - `scaler_files * 15`: Associated scaler files for data transformation (15 in total).
  - `main.py`: Entry point for the backend application.
- `.dockerignore`: File specifying excluded files/directories during Docker containerization.
- `Dockerfile`: Configuration for creating the backend application's Docker container.
- `requirements.txt`: List of Python dependencies for the backend application.

#### GF_Usecase/frontend Directory:
- **streamlitapp/:**
  - `Frontend.py`: Python file with code for the frontend application.
  - `other-necessary files`: Additional files essential for frontend functionality.
- `.dockerignore`: File specifying excluded files/directories during Docker containerization.
- `Dockerfile`: Configuration for creating the frontend application's Docker container.
- `requirements.txt`: List of dependencies for the frontend application.
