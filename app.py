import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    predict_request = []
    credit = request.form["credit.policy"]
    predict_request.append(credit)
    interest = request.form["int.rate"]
    predict_request.append(interest)
    installment = request.form["installment"]
    predict_request.append(installment)
    log_inc = request.form["log.annual.inc"]
    predict_request.append(log_inc)
    dti = request.form["dti"]
    predict_request.append(dti)
    fico = request.form["fico"]
    predict_request.append(fico)
    cr_line = request.form["days.with.cr.line"]
    predict_request.append(cr_line)
    r_bal = request.form["revol.bal"]
    predict_request.append(r_bal)
    r_util = request.form["revol.util"]
    predict_request.append(r_util)
    inq = request.form["inq.last.6mths"]
    predict_request.append(inq)
    delinq = request.form["delinq.2yrs"]
    predict_request.append(delinq)
    p_rec = request.form["pub.rec"]
    predict_request.append(p_rec)
    
    purpose_dict = {'all_other':[1,0,0,0,0,0,0], 'credit_card':[0,1,0,0,0,0,0],
       'debt_consolidation':[0,0,1,0,0,0,0], 'educational':[0,0,0,1,0,0,0],
       'home_improvement':[0,0,0,0,1,0,0], 'major_purchase':[0,0,0,0,0,1,0],
       'small_business':[0,0,0,0,0,0,1]}
    cate = request.form["purpose"]
    predict_request.extend(purpose_dict.get(cate))
    predict_request = list(map(float,predict_request))
    predict_request = np.array(predict_request)
    prediction = model.predict([predict_request])
    if prediction[0] == 0 :
        output = 'Pay'
    else:
        output = 'Not Pay'
    return render_template('index.html', prediction_text='The Customer will {}'.format(output))


# creating predict url and only allowing post requests.
@app.route('/predict_api', methods=['POST','GET'])
def predict_api():
    # Get data from Post request
    data = request.get_json()
    results = []
    for i in data:
        predict_request = []
        credit = i["credit.policy"]
        predict_request.append(credit)
        interest = i["int.rate"]
        predict_request.append(interest)
        installment = i["installment"]
        predict_request.append(installment)
        log_inc = i["log.annual.inc"]
        predict_request.append(log_inc)
        dti = i["dti"]
        predict_request.append(dti)
        fico = i["fico"]
        predict_request.append(fico)
        cr_line = i["days.with.cr.line"]
        predict_request.append(cr_line)
        r_bal = i["revol.bal"]
        predict_request.append(r_bal)
        r_util = i["revol.util"]
        predict_request.append(r_util)
        inq = i["inq.last.6mths"]
        predict_request.append(inq)
        delinq = i["delinq.2yrs"]
        predict_request.append(delinq)
        p_rec = i["pub.rec"]
        predict_request.append(p_rec)
        purpose_dict = {'all_other':[1,0,0,0,0,0,0], 'credit_card':[0,1,0,0,0,0,0],
                        'debt_consolidation':[0,0,1,0,0,0,0], 'educational':[0,0,0,1,0,0,0],
                        'home_improvement':[0,0,0,0,1,0,0], 'major_purchase':[0,0,0,0,0,1,0],
                        'small_business':[0,0,0,0,0,0,1]}
        cate = i["purpose"]
        predict_request.extend(purpose_dict.get(cate))
        predict_request = list(map(float,predict_request))
        prediction = model.predict(np.array([predict_request]).tolist()).tolist()
        if prediction[0] == 0 :
            output = 'Pay'
        else:
            output = 'Not Pay'
    
        results.append(output)
    return jsonify(result:results)


if __name__ == "__main__":
    app.run(debug=True)
