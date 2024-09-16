# Import libraries
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels
species = iris.target_names

# Train the model
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Streamlit App
st.title("Iris Flower Species Classification")

# Create form inputs for feature values
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Collect inputs in an array for prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Prediction
prediction = clf.predict(input_data)
predicted_species = species[prediction][0]

# Display prediction
st.write(f"### Predicted Species: **{predicted_species}**")

# Display feature values used for prediction
st.write("### Input Features:")
st.write(f"**Sepal Length:** {sepal_length} cm")
st.write(f"**Sepal Width:** {sepal_width} cm")
st.write(f"**Petal Length:** {petal_length} cm")
st.write(f"**Petal Width:** {petal_width} cm")
