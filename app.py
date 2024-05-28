import streamlit as st
import pyedflib as edf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.utils import to_categorical
import tempfile

# Constants
WINDOW_STEP = 256
WINDOW_SIZE = 5 * WINDOW_STEP
SEIZURE_PERIODS = [(2996, 3036), (1467, 1494), (1732, 1772),]

# Load EDF file
def load_edf_file(uploaded_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name
    
    # Read the temporary EDF file
    with edf.EdfReader(temp_file_path) as edf_reader:
        signal_labels = edf_reader.getSignalLabels()
        buffers = np.array([edf_reader.readSignal(signal_labels.index('FZ-CZ')),
                            edf_reader.readSignal(signal_labels.index('CZ-PZ'))])
    return buffers

# Remove noise
def remove_noise(signal):
    low_range = int(0.05 * len(signal[0]))
    high_range = int(0.95 * len(signal[0]))
    return signal[:, low_range:high_range]

# Plot signals
def plot_signals(signals, title='Signal'):
    fig, ax = plt.subplots()
    for signal in signals:
        ax.plot(signal[0])
    st.pyplot(fig)

import numpy as np

def extract_features(data):
    # Assuming data shape is (1, n_samples)
    features = np.concatenate([
        np.var(data, axis=1).reshape(1, -1),
        np.std(data, axis=1).reshape(1, -1),
        np.mean(data, axis=1).reshape(1, -1)
    ], axis=1)
    return features


# Train RandomForest
def train_random_forest(x_train, y_train, x_test, y_test):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return clf, accuracy, cm

# Train CNN Model
def train_cnn_model(x_train, y_train, x_test, y_test):
    y_train_cat, y_test_cat = to_categorical(y_train), to_categorical(y_test)
    model = Sequential([
        Conv1D(256, 5, activation='relu', input_shape=(WINDOW_SIZE, 2)),
        MaxPooling1D(2), Dropout(0.3),
        Conv1D(128, 5, activation='relu'), MaxPooling1D(2), Dropout(0.3),
        Conv1D(64, 5, activation='relu'), MaxPooling1D(2), Dropout(0.3),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train_cat, epochs=10, shuffle=True, verbose=2)
    model_loss, model_accuracy = model.evaluate(x_test, y_test_cat, verbose=2)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    return model, model_accuracy, cm

# Streamlit Front-End
st.title('EEG Seizure Detection')
st.sidebar.title('Upload EEG File')

uploaded_file = st.sidebar.file_uploader("Choose an EDF file", type="edf")

if uploaded_file is not None:
    st.sidebar.success("File uploaded successfully!")
    signal = load_edf_file(uploaded_file)
    signal = remove_noise(signal)

    st.header('Signal Plot')
    plot_signals([signal])

    # Extract features and labels
    st.header('Extract Features and Labels')

    labels = np.array([1 if i % 2 == 0 else 0 for i in range(signal.shape[1])])

    features = extract_features(signal.reshape(1, 2, -1))

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train RandomForest
    st.header('Random Forest Classifier')
    clf, rf_accuracy, rf_cm = train_random_forest(x_train, y_train, x_test, y_test)
    st.write(f'Random Forest Accuracy: {rf_accuracy:.2f}')
    st.write('Confusion Matrix:')
    st.write(rf_cm)

    # Train CNN
    st.header('Convolutional Neural Network')
    x_train = signals.reshape(-1, WINDOW_SIZE, 2).astype('float32')
    x_test = x_train
    cnn_model, cnn_accuracy, cnn_cm = train_cnn_model(x_train, y_train, x_test, y_test)
    st.write(f'CNN Accuracy: {cnn_accuracy:.2f}')
    st.write('Confusion Matrix:')
    st.write(cnn_cm)

    st.header('Upload Real-time EEG Record')
    real_time_file = st.file_uploader("Upload a real-time EDF file", type="edf")

    if real_time_file is not None:
        real_time_signal = load_edf_file(real_time_file)
        real_time_signal = remove_noise(real_time_signal)
        real_time_features = extract_features(real_time_signal.reshape(1, 2, -1))
        prediction = clf.predict(real_time_features)
        st.write('Prediction:')
        st.write('Seizure' if prediction[0] == 1 else 'No Seizure')

        st.write('Model Explanation:')
        st.write("The Random Forest model uses multiple decision trees to classify the signals, making it robust to overfitting. The CNN model processes the signals as sequences, extracting complex features and patterns that are indicative of seizures.")
