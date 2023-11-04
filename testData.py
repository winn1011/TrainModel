import base64
import pickle
import cv2
import numpy
import requests
import os

def img2vec(img):
    resized = cv2.resize(img,(128,128),cv2.INTER_AREA)
    v, buffer = cv2.imencode(".jpg", resized)
    img64 = base64.b64encode(buffer).decode('utf-8')
    img642 = "data:image/jpeg;base64," + img64
    
    url = "http://localhost:8080/api/genhog"
    
    response = requests.get(url,json = {"img" : img642})
    
    if response.status_code == 200:
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError as e:
            print("Error : ",e)
    else:
        print("Status Error :", response.status_code)

data = "Dataset/test"

X = []
Y = []

for inpic in os.listdir(data):
    pic = os.path.join(data,inpic)
    if os.path.isdir(pic):
        for inname in os.listdir(pic):
            name = os.path.join(pic,inname)
            picname = cv2.imread(name)
            X.append(picname)
            Y.append(inpic)

HVT = []

for a in range(len(X)):
    try:
        res = img2vec(X[a])
        if res is not None and "HOG" in res:
            vec = res["HOG"]
            vec.append(Y[a])
            HVT.append(vec)
        else :
            print("Respone Error : ", res)
    except Exception as e:
        print("Error : ",e)
        
trdata = "test_data.pkl"

pickle.dump(HVT,open(trdata,"wb"))
print("Success")