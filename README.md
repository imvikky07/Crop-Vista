CropVista – Crop Yield Prediction System

CropVista is a web-based crop yield prediction system built using Flask and an ensemble machine learning model (ExtraTreesRegressor).
The goal of the project is to help farmers and researchers estimate crop production based on environmental and soil parameters.

The machine learning model was trained on 85,000+ agricultural records and evaluated using a proper train–test split. The final model achieves around 85% R² on unseen data, making it reliable for practical use.

Features

Predicts crop yield based on:

Season

Crop type

Cultivated area

Rainfall

Temperature

Soil pH

Nitrogen level

Simple and clean Flask-based web interface

Graphical comparison of predicted yields for crops in the same season

Ensemble machine learning model for stable and accurate predictions

Tech Stack: Python, Flask, Scikit-learn, Pandas / NumPy, Matplotlib, HTML, CSS (Jinja templates)

Machine Learning Model

The model is built using ExtraTreesRegressor, which performed the best during experimentation.

Model Performance:

Test R²: ~85%

MAE: ~19,600

RMSE: ~168,000

The model file (model.sav) is loaded directly by the Flask app for real-time prediction.

Project Structure
CropVista/
│── app.py                 # Flask application
│── train_model.py         # ML training script (ExtraTrees model)
│── model.sav              # Saved trained model
│── Final_Dataset.csv      # Main dataset
│── Trainset.csv           # Supporting dataset
│── templates/
│     ├── home.html
│     ├── result.html
│     └── about.html
│── static/
│     └── css / js files
│── requirements.txt
└── README.md

How to Run the Project
1. Install dependencies
pip install -r requirements.txt

2. Train the model (optional)
python train_model.py

3. Start the Flask server
python app.py

4. Open in browser
http://localhost:5000/

About the Project

CropVista was built to explore how machine learning can support agriculture by predicting crop yields using real-world data. The focus was on building a practical system with a clean interface and a reliable ML pipeline.