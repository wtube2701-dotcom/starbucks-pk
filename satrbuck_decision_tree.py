import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

st.title("Starbucks Decision Tree App")
st.write("This app visualizes a Decision Tree model.")

# Dummy data for demonstration so the app runs without external files
data = {
    'Feature1': [1, 2, 1, 2, 1, 2],
    'Feature2': [0, 0, 1, 1, 0, 1],
    'Target': [0, 0, 1, 1, 0, 1]
}
df = pd.DataFrame(data)

X = df[['Feature1', 'Feature2']]
y = df['Target']

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

fig, ax = plt.subplots(figsize=(8, 6))
plot_tree(clf, filled=True, ax=ax)
st.pyplot(fig)