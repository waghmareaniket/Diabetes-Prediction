import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image

data=pd.read_csv("diabetes.csv")
data_mean=data.groupby('Outcome').mean()

x=data.drop('Outcome',axis=1)
y=data['Outcome']

sc=StandardScaler()
sc.fit(x)
x=sc.transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=16)
model=svm.SVC(kernel='linear')
model.fit(x_train,y_train)

train_y_pred= model.predict(x_train)
test_y_pred= model.predict(x_test)

train_acc=accuracy_score(train_y_pred,y_train)
test_acc=accuracy_score(test_y_pred,y_test)

def app():
    img=Image.open(r'img.jpg')
    img=img.resize((200,200))
    st.image(img,caption='Diabetes',width=200)
    st.title('Diabetes Disease Prediction')
    st.sidebar.title('Input Features')
    
    preg=st.sidebar.slider("Pregnancies",0,17,3)
    glucose=st.sidebar.slider("Glucose",0,199,117)
    bp=st.sidebar.slider("Blood Pressure",0,122,72)
    skinthick=st.sidebar.slider("Skin Thickness",0,99,23)
    insulin=st.sidebar.slider("Insulin",0,846,30)
    bmi=st.sidebar.slider("BMI",0.0,67.1,32.0)
    dpf=st.sidebar.slider("Diabetes pd Function",0.078,2.42,0.3725,0.001)
    age=st.sidebar.slider("Age",21,81,29)
    
    input=[preg,glucose,bp,skinthick,insulin,bmi,dpf,age]
    np_arr_data=np.asarray(input)
    reshape_data=np_arr_data.reshape(1,-1)
    predicter=model.predict(reshape_data)

    if predicter ==1:
        st.warning("Patient has diabetes")
    else:
        st.success("Patient has no diabetes")

if __name__=="__main__":
    app()