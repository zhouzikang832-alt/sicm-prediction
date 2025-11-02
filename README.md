# Sepsis-Induced Cardiomyopathy Mortality Prediction

This Streamlit application provides an individual-level mortality risk prediction for patients with Sepsis-Induced Cardiomyopathy (SICM).  
The model was trained using CatBoost based on 20 routinely available clinical variables.

### Usage
1. Enter the patient’s clinical/lab values in the input fields.
2. Click **Predict**.
3. The app will output:
   - predicted mortality probability
   - risk category (High vs Low)

### Model
- Algorithm: CatBoost
- Input: 20 clinical features (numeric)
- Output: Probability of in-hospital mortality

### Deployment
This app is deployed on Streamlit Cloud.

### Disclaimer
This tool is intended for research and academic purposes only.  
It is **not a medical device**, and should not be used as the sole basis for clinical decisions.

### License
MIT License.
