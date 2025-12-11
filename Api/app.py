from fastapi import FastAPI,HTTPException
import joblib
import pandas as pd
from Api.Schemas import input_data

Version = "1.0.0"

app= FastAPI( 
            title="Credit Card Fraud Detection API",
            description="Predict fraud using Random Forest + Pipeline",
            version="1.0"
            )

#load the model

try:
    bundle = joblib.load(r"C:\Users\SK MIANUR RAHAMAN\OneDrive\Desktop\Credit Card Fraud\Credit_Card_Fraud_Detection\Model\Fraud_model_rf.pkl")
    model = bundle["model"]
    Threshold = bundle["threshold"]

except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")



@app.get("/")
def root():
    return {"message": "Credit Card Fraud Detection Api"}

@app.get("/health")
def health():
    return {
        "Status" : "Ok",
        "Model_Loaded" : model is not None,
        "Version" : Version
    }
#------------------------------------------------------
        #Prediction Endpoint
#-------------------------------------------------------
@app.post("/predict")
def predict(data : input_data):
    
    try:
        data_dict = data.model_dump()
        df = pd.DataFrame([data_dict])
        
        prob = model.predict_proba(df)[0]
        prob_1 = prob[1]
        prediction = 1 if prob_1 >= Threshold else 0
        confidence = float(max(prob))
        class_labels = ['0','1']
        
        class_prob = {
            label: round(float(p), 4)
            for label, p in zip(class_labels, prob)
        }
        
        return {
            "prediction": prediction,
            "Confidence" : round(confidence,4),
            "Class Probabilities" : class_prob,
            "ThresHold Used":float(Threshold)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))