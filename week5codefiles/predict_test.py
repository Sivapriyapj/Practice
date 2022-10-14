

#get_ipython().run_line_magic('autosave', '0')



import requests



url = 'http://localhost:9696/predict'

client_id = "priya"
client = {
        "reports": 0,
        "share": 0.001694,
        "expenditure": 0.12,
        "owner": "yes"
}



response = requests.post(url,json=client).json()


print(response)
if response['decision'] == False:
    print(f'credit card has been declined to {client_id}')
else:
    print(f"Credit card has been issued to {client_id}")





