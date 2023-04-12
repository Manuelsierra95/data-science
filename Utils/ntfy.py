import requests

def send_notification(model_evaluate, metric="", model_name="", model_version="", dataset=""):    
    req = f'{metric}: {model_evaluate} \n Model: {model_name} \n Version: {model_version} \n Dataset: {dataset}'
    
    requests.post("https://ntfy.sh/training", 
        data=req.encode(encoding='utf-8'))