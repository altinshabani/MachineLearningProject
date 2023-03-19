import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('D:\Desktop\MASTER2\Machine Learning\weatherAUS.csv')

# Show the summary statistics
st.write('Summary Statistics:', data.describe())

# Create a histogram of the rainfall column
fig, ax = plt.subplots()
ax.hist(data['Rainfall'], bins=30)
ax.set_xlabel('Rainfall')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Drop rows with missing values
data.dropna(inplace=True)


# Show the summary statistics after dropping rows with missing values
st.write('Summary Statistics after Dropping Rows with Missing Values:', data.describe())

# Create a histogram of the rainfall column
fig, ax = plt.subplots()
ax.hist(data['Rainfall'], bins=30)
ax.set_xlabel('Rainfall')
ax.set_ylabel('Frequency')
st.pyplot(fig)