import streamlit as st
import pandas as pd
from PIL import Image
from load_data import load_data
from train_model import train_model
from sklearn.metrics import accuracy_score
##BIG Comment AAAAAAAAAAAAAA
image = Image.open('assets/donate.jpeg')
st.image(image,width=700)

st.markdown("<h1 style='text-align: center; color: #FF3342;'><strong><u>Predict Blood Donation for Future Expectancy</u></strong></h1>", unsafe_allow_html=True)

st.sidebar.markdown("<h1 style='text-align: center; color:#FF3342 ;'><strong><u>Specify Input Parameters</u></strong></h1>", unsafe_allow_html=True)
    
st.markdown("Forecasting blood supply is a serious and recurrent problem for blood collection managers: in January 2019, Nationwide, the Red Cross saw 27,000 fewer blood donations over the holidays than they see at other times of the year. Machine learning can be used to learn the patterns in the data to help to predict future blood donations and therefore save more lives.")
st.markdown("Understanding the Parameters -")
st.markdown("(Recency - months since the last donation)")
st.markdown("(Frequency - total number of donation)")
st.markdown("(Monetary - total blood donated in c.c.)")
st.markdown("(Time - months since the first donation)")
st.markdown("Target - (1 stands for donating blood, 0 stands for not donating blood)")

def user_input_features():
     Recency  = st.sidebar.slider('Recency', 0, 74)
     Frequency= st.sidebar.slider('Frequency', 1,43)
     Monetary = st.sidebar.slider('Monetary', 250,12500)
     Time = st.sidebar.slider('Time', 2,98)
     
     data = {'Recency': Recency,
           'Frequency': Frequency,
           'Monetary': Monetary,
           'Time':Time}
 
     features = pd.DataFrame(data, index=[0])
     return features
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    precision = accuracy_score(y_test, predictions)
    return precision

df = user_input_features()

st.write(df)

trans = load_data('dataset/transfusion.data')

model, X_test, y_test = train_model(trans.drop(columns=['target']), trans['target'].values)

prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

precision = evaluate_model(model, X_test, y_test)

st.subheader('Model Performance')
st.write(f'AUC score: {precision:.4f}')

st.subheader('Prediction - Target')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
