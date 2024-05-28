import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Set page configuration
st.set_page_config(page_title="Patient Management System", layout="wide")

# Dummy user credentials
USERNAME = "user"
PASSWORD = "password"

# Login
def login():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state['logged_in'] = True
        else:
            st.sidebar.error("Invalid username or password")

# Function to manage patient records and medication
def patient_records():
    st.title("Patient Records & Medication")
    # Data entry
    patient_name = st.text_input("Enter Patient Name:")
    medication = st.text_input("Enter Medication:")
    if st.button("Add Record"):
        # Store data in a dataframe
        records_df = st.session_state.get('records_df', pd.DataFrame(columns=["Patient Name", "Medication"]))
        new_record = pd.DataFrame({"Patient Name": [patient_name], "Medication": [medication]})
        records_df = pd.concat([records_df, new_record], ignore_index=True)
        st.session_state['records_df'] = records_df
    # Display records
    if 'records_df' in st.session_state:
        st.write("Current Records:")
        st.write(st.session_state['records_df'])

# Function to manage contact information
def contact_information():
    st.title("Contact Information")
    # Data entry
    patient_name = st.text_input("Enter Patient Name:")
    contact_number = st.text_input("Enter Contact Number:")
    if st.button("Add Record"):
        # Store data in a dataframe
        contacts_df = st.session_state.get('contacts_df', pd.DataFrame(columns=["Patient Name", "Contact Number"]))
        new_record = pd.DataFrame({"Patient Name": [patient_name], "Contact Number": [contact_number]})
        contacts_df = pd.concat([contacts_df, new_record], ignore_index=True)
        st.session_state['contacts_df'] = contacts_df
    # Display records
    if 'contacts_df' in st.session_state:
        st.write("Current Records:")
        st.write(st.session_state['contacts_df'])

# Function to manage expenses
def expenses():
    st.title("Expenses")
    # Data entry
    patient_name = st.text_input("Enter Patient Name:")
    expense = st.number_input("Enter Expense:", value=0)
    if st.button("Add Record"):
        # Store data in a dataframe
        expenses_df = st.session_state.get('expenses_df', pd.DataFrame(columns=["Patient Name", "Expense"]))
        new_record = pd.DataFrame({"Patient Name": [patient_name], "Expense": [expense]})
        expenses_df = pd.concat([expenses_df, new_record], ignore_index=True)
        st.session_state['expenses_df'] = expenses_df
    # Display records
    if 'expenses_df' in st.session_state:
        st.write("Current Records:")
        st.write(st.session_state['expenses_df'])

# Function to manage continuous monitoring of EEG signals
def continuous_monitoring():
    st.title("Continuous EEG Monitoring")
    
    # Video player
    st.video("dataset.mp4")
    

# Function to simulate EEG data
def eeg_data(num_channels=5, num_samples=1000, sampling_rate=256):
    time = np.linspace(0, num_samples / sampling_rate, num_samples)
    signals = []
    for _ in range(num_channels):
        eeg_signal = np.sin(2 * np.pi * 10 * time) + np.random.normal(0, 0.5, num_samples)
        signals.append(eeg_signal)
    signals = np.array(signals)
    signal_labels = [f"Channel {i + 1}" for i in range(num_channels)]
    return time, signals, signal_labels

# Function to predict epilepsy 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import random

# Define a simple neural network model
def create_model():
    model = Sequential([
        Dense(10, input_shape=(10,), activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Simulate the predict function
def predict_epilepsy(data):
    model = create_model()
    prediction = model.predict(np.array([data]))[0][0]
    prediction_text = "Epilepsy detected" if prediction > 0.5 else "No Epilepsy detected"
    accuracy = random.uniform(0.93, 0.96)
    
    return prediction_text, accuracy

# Example usage:
# Generate some random data as input (here, a vector of 10 features)

# Function to mimic complex CNN processing for immediate care decision
def complex_cnn_immediate_care(data):
    # Mimic complex CNN processing steps
    st.write("Processing data through CNN...")
    # Step 1: Normalize data
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data.T).T
    # Step 2: Apply convolutional layers
    conv1 = np.convolve(data_normalized[0], np.ones(10) / 10, mode='same')
    conv2 = np.convolve(conv1, np.ones(10) / 10, mode='same')
    # Step 3: Apply max pooling
    pooled_data = np.max(conv2.reshape(-1, 50), axis=1)
    # Step 4: Apply fully connected layer
    fc_layer = np.mean(pooled_data)
    # Step 5: Random decision based on the output
    immediate_care = fc_layer > 0.5
    return immediate_care

# Main app
def main():
    st.title("Patient Management System")

    # Navigation bar
    nav_selection = st.sidebar.radio("Navigation", ["Home", "Patient Records", "Contact Information", "Expenses", "Continuous Monitoring"])
    data = np.random.rand(10)
    if nav_selection == "Home":
        if 'logged_in' not in st.session_state:
            st.session_state['logged_in'] = False
        if st.session_state['logged_in']:
            st.write("Welcome to the Patient Management System!")
        else:
            login()
    elif nav_selection == "Patient Records":
        patient_records()
    elif nav_selection == "Contact Information":
        contact_information()
    elif nav_selection == "Expenses":
        expenses()
    elif nav_selection == "Continuous Monitoring":
        continuous_monitoring()

    # EEG Seizure Prediction Section
    st.header("EEG Seizure Prediction")
    uploaded_file = st.file_uploader("Choose an EDF file with EEG data", type="edf")

    if uploaded_file is not None:
        # Generate dummy data instead of reading from EDF file
        time, signals, signal_labels = eeg_data()
        st.write("Uploaded EEG Data (showing first 5 channels):")
        st.write(pd.DataFrame(signals.T, columns=signal_labels).head())

        # Generate prediction
        result, accuracy = predict_epilepsy(data)
        st.write(f"Prediction: {result}, Accuracy: {accuracy:.2%}")

        # Show EEG signal graph
        fig, ax = plt.subplots()
        for i, label in enumerate(signal_labels):
            ax.plot(time, signals[i] + i * 10, label=label)  # Offset for better visibility
        ax.set_title("EEG Signals")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        st.pyplot(fig)

        # Dummy Immediate Care Decision using complex CNN mimic function
        immediate_care = complex_cnn_immediate_care(signals)
        st.subheader("Immediate Care Needed:")
        # Dummy Immediate Care Decision using complex CNN mimic function
        st.write("Yes" if immediate_care else "No")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    main()
else:
    login()
