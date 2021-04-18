import numpy as np
import pandas as pd
import plotly.express as px
import pickle
import os
import streamlit as st
import streamlit.components.v1 as components


# EDA
my_dataset = "kolesadata.csv"

# Pipeline for predictions
my_pipeline = "pipeline"

# To Improve speed and cache data
@st.cache(persist=True)
def explore_data(dataset):
	df = pd.read_csv(os.path.join(dataset))
	return df 

@st.cache(allow_output_mutation=True)
def load_pipeline(pipeline):
	pl = pickle.load(open(os.path.join(pipeline), 'rb'))
	return pl

# Our Dataset
data = explore_data(my_dataset)
# Our Pipeline
pipeline = load_pipeline(my_pipeline)

# Exploratory data analysis page
def eda():
	st.title('Exploratory data analysis')

	# Show Dataset
	st.subheader("Preview DataFrame")	
	st.write("Head", data.head())
	st.write("Tail", data.tail())

	# Dimensions
	data_dim = st.radio('What Dimension Do You Want to Show',('Rows','Columns'))
	if data_dim == 'Rows':
		st.write("Showing Length of Rows", len(data))
	if data_dim == 'Columns':
		st.write("Showing Length of Columns", data.shape[1])

	# Distributions of features
	st.subheader("Plot distribution of feature")
	x_options = ['year', 'mileage', 'volume', 'price', 'body', 'transmission', 'wheel', 'drive', 'fuel']
	x_axis = st.selectbox('Which feature do you want to explore?', x_options)
	fig = px.histogram(data, x=x_axis)
	st.plotly_chart(fig)
	
	# Scatter plot with price
	st.subheader("Scatter plot")
	x_options = ['year', 'mileage', 'volume', 'price']
	dim = st.multiselect('Which feature do you want to explore?', x_options, default=['price', 'year'])
	fig2 = px.scatter_matrix(data, dimensions=dim)
	st.plotly_chart(fig2)


# Prediction page
def prediction():
	st.title('Car price prediction')

	# Input data
	year = st.selectbox('Год выпуска', range(1960, 2021))
	body = st.selectbox('Кузов', ['седан', 'кроссовер', 'минивэн', 'внедорожник', 'универсал', 'хэтчбек'])
	volume = st.slider('Объем двигателя', min_value=1.0, max_value=4.0, value=1.6, step=0.1)

	choose_transmission = st.radio('Коробка передач',('автомат', 'механика'))
	if choose_transmission == 'автомат':
		transmission = 'автомат'
	if choose_transmission == 'механика':
		transmission = 'механика'

	drive = st.selectbox('Привод', ['задний привод', 'передний привод', 'полный привод'])
	mileage = st.number_input('Пробег', min_value=0, max_value=None, value=10000)

	choose_wheel = st.radio('Руль', ('слева', 'справа'))
	if choose_wheel == 'слева':
		wheel = 'слева'
	if choose_wheel == 'справа':
		wheel = 'справа'

	fuel = st.selectbox('Топливо', ['бензин', 'газ-бензин', 'дизель'])

	components.html("<hr>", height=20)
	# Predict
	if st.button("Predict"):
		input_data = pd.DataFrame({'year': [year], 
              'mileage': [mileage], 
              'volume': [volume], 
              'body': [body], 
              'transmission': [transmission], 
              'wheel': [wheel], 
              'drive': [drive], 
              'fuel': [fuel]})
		result = pipeline.predict(input_data)
		st.success('Your prediction is {} tenge'.format(int(result[0])))
	components.html("<hr>", height=20)


def main():
	st.sidebar.title('What to do')
	app_mode = st.sidebar.selectbox("Choose the app mode", ["Car price prediction", "Exploratory data analysis"])

	if app_mode == "Exploratory data analysis":
		eda()
	elif app_mode == "Car price prediction":
		prediction()


if __name__ == '__main__':
	main()
