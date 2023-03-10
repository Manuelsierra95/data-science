import requests

def send_notification(req):
    requests.post("https://ntfy.sh/training", 
        data=req.encode(encoding='utf-8'))