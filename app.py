import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # For more stylish plots

# Define the columns
columns = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
           'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
           'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']

# Load the pre-trained model
with open('loan_prediction.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app configuration
st.set_page_config(page_title="Loan Prediction App", page_icon=":money_with_wings:", layout="wide")

# Custom CSS for modern look
st.markdown("""
    <style>
    body {
        background-color: #f4f4f9;
        font-family: 'Arial', sans-serif;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .header {
        font-size: 24px;
        font-weight: bold;
        color: #1F618D;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 20px;
        color: #1F618D;
        text-align: center;
        margin-bottom: 10px;
    }
    .form-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 20px;
    }
    .input-field {
        margin-bottom: 15px;
        width: 100%;
        max-width: 500px; /* Adjust the max width as needed */
    }
    .input-field input {
        border-radius: 8px;
        border: 1px solid #d1d1d1;
        padding: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        font-size: 16px;
        transition: border-color 0.3s, box-shadow 0.3s;
    }
    .input-field input:focus {
        border-color: #2E86C1;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
    }
    .predict-btn {
        background-color: #2E86C1;
        color: white;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
        max-width: 200px; /* Adjust the max width as needed */
        padding: 10px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s, transform 0.2s;
    }
    .predict-btn:hover {
        background-color: #1F618D;
        transform: scale(1.05);
    }
    .plot-container {
        margin-top: 20px;
        text-align: center;
    }
    .error, .warning, .success {
        font-size: 16px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="title">Loan Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="header">Enter the loan details</div>', unsafe_allow_html=True)

# Create input fields for each column except 'Loan_ID' and 'Loan_Status'
with st.form(key='loan_form'):
    st.subheader("Input Details", anchor="input-details")
    st.write("Fill in the details below and click 'Predict' to get the result.")
    
    # Form container for styling
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    inputs = {}
    for column in columns[:-1]:  # Exclude 'Loan_Status'
        inputs[column] = st.text_input(column, '', key=f"{column}_input", help=f"Enter {column}", placeholder=f"Enter {column}")
    
    submit_button = st.form_submit_button(label='Predict', help="Click to get the prediction")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Make predictions
if submit_button:
    if all(v != '' for v in inputs.values()):
        try:
            input_df = pd.DataFrame([inputs])
            prediction = model.predict(input_df)
            st.success(f"Prediction: {prediction[0]}", icon="✅")
            
            # Plot the prediction with a stylish design
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=['Prediction'], y=[prediction[0]], palette='coolwarm', ax=ax)
            ax.set_title('Prediction Result', fontsize=16, fontweight='bold')
            ax.set_ylabel('Value')
            ax.set_xlabel('')
            ax.set_xticks([])
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}", icon="❌")
    else:
        st.warning("Please fill in all the fields.", icon="⚠️")
