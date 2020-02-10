import requests

url = 'http://localhost:5000/predict_api'

# Requesting Parameters:
parameters = {'credit.policy':1,'int.rate':0.1189,'installment':829.1,'log.annual.inc':11.35040654,'dti':19.48,
                            'fico':737,'days.with.cr.line':5639.958333,'revol.bal':28854,'revol.util':52.1,
                            'inq.last.6mths':0,'delinq.2yrs':0,'pub.rec':0,'purpose':'debt_consolidation'}

r = requests.post(url,json=parameters)

print(r.json()) 

