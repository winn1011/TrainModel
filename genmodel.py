import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import xgboost as xgb
from sklearn.ensemble import VotingClassifier  

#load pickle train และ test
data_train=pickle.load(open(r'train_data.pkl','rb'))
data_test=pickle.load(open(r'test_data.pkl','rb'))

#สร้างตัวแปรมาเก็บค่า
X_data_train=[]
Y_data_train=[]
X_data_test=[]
Y_data_test=[]


X_data_train = [entry[:8100] for entry in data_train]
Y_data_train = [entry[8100] for entry in data_train]
label_encoder_train = LabelEncoder()
Y_numeric_train = label_encoder_train.fit_transform(Y_data_train)


X_data_test = [entry[:8100] for entry in data_test]
Y_data_test = [entry[8100] for entry in data_test]
label_encoder_test = LabelEncoder()
Y_numeric_test = label_encoder_test.fit_transform(Y_data_test)


model_tree = DecisionTreeClassifier(random_state=42)
model_xgb = xgb.XGBClassifier(objective="multi:softmax", num_class=len(set(Y_numeric_train)), random_state=42)

ensemble_model = VotingClassifier(estimators=[('decision_tree', model_tree), ('xgb', model_xgb)], voting='hard',weights=[1,4])
ensemble_model.fit(X_data_train, Y_numeric_train)
y_pred = ensemble_model.predict(X_data_test)

accuracy = accuracy_score(Y_numeric_test, y_pred)
print("Ensemble Accuracy:", accuracy)
confusion_mat=confusion_matrix(Y_numeric_test,y_pred)
print(confusion_mat)

write_path_model="model.pkl"
pickle.dump(ensemble_model,open(write_path_model,"wb"))
print("done")
