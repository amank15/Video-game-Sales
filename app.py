import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import joblib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn_pandas import CategoricalImputer


st.title("Predicting Video Games Sales using ML")
st.write("""
	Video games have become immensely popular over the past decade.\
	The global games market in 2019 was estimated at $148.8 billion.\
	In this application, we’ll implement a Machine Learning model that can predict the global sales\
	of a video game depending on certain features such as its genre, critic reviews, and user reviews in Python.\
	""")

st.write("")
st.subheader("Enter values of features for sales prediction :")
#games_list = list(["--Choose from here--"]) + list(X_train[:, 0])
genre_list = ["--Choose from here--","Racing","Simulation","Adventure","Sports","Fighting",\
"Platform","Puzzle","Misc","Action","Role-Playing","Strategy","Shooter"]
#rating_list = list(["--Choose from here--"]) + list(set(X_train[:, -1]))
rating_list = ["--Choose from here--",'RP','EC','E10+','T','K-A','AO','M','E']


col1,col2 = st.beta_columns(2)

#game = st.selectbox("Choose a game :", games_list)
genre = col1.selectbox("Select a genre :", genre_list)
rating = col1.selectbox("Choose a rating :", rating_list)
na_sales = col1.number_input("Enter North America sales (in millions) :",min_value = 0.0, value = 0.01)
eu_sales = col1.number_input("Enter Europe sales (in millions) :",min_value = 0.0, value = 0.01)
jp_sales = col1.number_input("Enter Japan sales (in millions) :",min_value = 0.0, value = 0.01)
oth_sales = col2.number_input("Enter other sales figure (in millions) :",min_value = 0.0, value = 0.01)

critic_score = col2.number_input("Enter critic score :",min_value = 0.0, value = 0.01)
critic_count = col2.number_input("Enter critic_count :",min_value = 0.0, value = 0.01)
user_score = col2.number_input("Enter user score :",min_value = 0.0, value = 0.01)
user_count = col2.number_input("Enter user_count :",min_value = 0.0, value = 0.01)

X_test = [genre,na_sales,eu_sales,jp_sales,oth_sales,critic_score,critic_count,user_score,user_count,rating]
X_test = np.array(X_test).astype(np.object).reshape((1,len(X_test)))


st.write("")
if st.button("Predict") and genre != "--Choose from here--" and rating != "--Choose from here--":

	model = joblib.load("model.joblib")
	y_pred = model.predict(X_test)
	st.write(f"The predicted gloabal sales is {np.squeeze(y_pred)} millions")
	st.write("")

	dataset = pd.read_csv('./Video_Games_Sales_as_at_22_Dec_2016.csv')
	dataset.drop(columns = ['Year_of_Release', 'Developer', 'Publisher', 'Platform'], inplace = True)
	plot_data = pd.Series(dataset.iloc[:,6].values,dataset.iloc[:,0].values)
	indices = np.random.randint(0, len(plot_data)+1,10)
	st.bar_chart(plot_data[indices],width = 500,height = 500)

elif genre == "--Choose from here--" and rating == "--Choose from here--":
	st.write("Kindly fill all the attributes to predict global sales !")
