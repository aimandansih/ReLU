
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="ReLU Activation Function",
    layout="wide"
)

st.title("ReLU Activation Function")

st.write(
    "Visualisation of Rectified Linear Unit (ReLU)"
)

# Mathematical formula
st.subheader("Mathematical Formula")
st.latex(r"f(x) = \max(0,x)")

# Generate input values
x = np.linspace(-6, 6, 400)

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

y = relu(x)

# Create dataframe
df = pd.DataFrame({
    "Input (x)": x,
    "ReLU Output": y
})

# Display graph
st.subheader("ReLU Activation Function Graph")

st.line_chart(
    df.set_index("Input (x)")
)

# Explanation
st.subheader("Explanation")

st.write("""
The ReLU activation function converts negative input
values into zero while maintaining positive input values.

ReLU is commonly used in hidden layers of neural networks
because it is computationally efficient.
""")
