#load the model and deploy in flask using request
from flask import Flask
from flask import request
from flask import jsonify
from waitress import serve



import pickle
model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file,'rb') as f_in:
    model = pickle.load(f_in)
with open(dv_file,'rb') as f1_in:
    dv = pickle.load(f1_in)
#with this file closes automatically

dv,model

app =Flask('credit_card')

@app.route('/predict',methods=['POST'])



def predict():
    client = request.get_json()
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0,1]
    decision = y_pred >=0.5

    result = {
        #bool and float are specified to avoid confusion while getting json values etc-> carify from video
        'credit card allotted probability' : float(y_pred),
        'decision': bool(decision)}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True,host= '0.0.0.0',port = 9696)


