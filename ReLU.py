import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ReLU Activation Function")

st.title("ReLU Activation Function")
st.write("Visualisation of the Rectified Linear Unit (ReLU).")

# Generate input values
x = np.linspace(-6, 6, 400)

# ReLU function
y = np.maximum(0, x)

# Plot graph
fig, ax = plt.subplots()
ax.plot(x, y, label="ReLU", linewidth=2)
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)

ax.set_title("ReLU Activation Function")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.grid(True)
ax.legend()

st.pyplot(fig)

st.write("Formula: f(x) = max(0, x)")