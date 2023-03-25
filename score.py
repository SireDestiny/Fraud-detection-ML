import json
import joblib
import numpy as np
import os
import sklearn

# Called when the web service is loaded
def init():
    global model
    # Get and load saved model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'fraud_detection_model.pkl')
    model = joblib.load(model_path)

# Called after a request
def run(raw_data):
    # Gets input data as a numpy array
    data = json.loads(raw_data)['data']
    np_data = np.array(data)
    # using saved model to predict
    predictions = model.predict(np_data)
    
    # print data and predictions (so they will be logged!)
    log_text = 'Data:' + str(data) + ' - Predictions:' + str(predictions)
    print(log_text)
    
    # Note: category 0= (not fraud), and category 1= (fraud)
    category = ['not-fraud', 'fraud']
    predicted_category = []
    for prediction in predictions:
        predicted_category.append(category[prediction])
    # Returns predictions as JSON
    return json.dumps(predicted_category)
