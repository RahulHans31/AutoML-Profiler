import streamlit as st
import os
import pandas as pd
import pickle
from pycaret.classification import setup as setup_classification, compare_model_dir as compare_model_dir_classification, pull as pull_classification
from pycaret.regression import setup as setup_regression, compare_model_dir as compare_model_dir_regression, pull as pull_regression
from ydata_profiling import ProfileReport

# Create a directory to save model_dir if it doesn't exist
if not os.path.exists('model_dir'):
    
    import tempfile

    # Temporary Directory Handling for Deployment
    temp_dir = tempfile.TemporaryDirectory()
    model_dir = temp_dir.name


# Initialize session state for recent actions
if 'recent_actions' not in st.session_state:
    st.session_state.recent_actions = []

# Sidebar and Option Selection
st.sidebar.title("Machine Learning Profiler")
option = st.sidebar.selectbox("Select an action", ("Upload Data", "Profile Data", "Model Training", "Recent Actions"))


# Upload Data Section
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if option == "Upload Data":
    st.header("Upload your dataset")
    
    if uploaded_file:
        
import tempfile
with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_data:
    data_path = temp_data.name
    with open(data_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

        with open(data_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {uploaded_file.name} successfully!")
        st.session_state.recent_actions.append(f"Uploaded dataset {uploaded_file.name}")

# Data Profiling Section
elif option == "Profile Data":
    st.header("Profile your dataset")
    
    if uploaded_file:
        
import tempfile
with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_data:
    data_path = temp_data.name
    with open(data_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

        
        if st.button("Generate Profile Report"):
            data = pd.read_csv(data_path)
            st.session_state.profile = ProfileReport(data, title="Dataset Profiling Report")
            
import tempfile
with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_report:
    output_path = temp_report.name

            st.session_state.profile.to_file(output_path)
            st.session_state.report_path = output_path  # Store the path in session state
            st.success(f"Profile report saved at: {output_path}")
            st.session_state.recent_actions.append(f"Generated profile report for {uploaded_file.name}")

        # Display the Download Report button
        if 'report_path' in st.session_state and st.session_state.report_path is not None:
            with open(st.session_state.report_path, "rb") as f:
                st.download_button("Download Report", data=f, file_name=f"{uploaded_file.name}_report.html", mime="text/html")
        
        # Display the Preview Report button
        if 'profile' in st.session_state and st.session_state.profile is not None:
            if st.button("Preview Report"):
                st.write("### Profiling Report Preview")
                st.components.v1.html(st.session_state.profile.to_html(), height=600, scrolling=True)

# Model Training Section
# Model Training Section
elif option == "Model Training":
    st.header("Train a Machine Learning Model")
    # Add slider to switch between Auto and Manual Mode
    mode = st.sidebar.radio("Choose Mode", options=["Auto Mode", "Manual Mode"])
    
    if uploaded_file:
        
import tempfile
with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_data:
    data_path = temp_data.name
    with open(data_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

        df = pd.read_csv(data_path)

        st.write("Dataset Preview:")
        st.write(df.head())

        target_variable = st.selectbox("Select the target variable", options=df.columns.tolist())

        if mode == "Auto Mode":
            st.write("Class Distribution:")
            st.write(df[target_variable].value_counts())

            is_classification = df[target_variable].dtype == 'object' or df[target_variable].nunique() < 10

            if is_classification:
                if st.button("Train Classification Model with PyCaret"):
                    setup_classification(data=df, target=target_variable, session_id=123, fold_strategy='stratifiedkfold')
                    best_model = compare_model_dir_classification(include=['xgboost', 'lightgbm', 'rf', 'dt', 'knn', 'nb', 'lr', 'svm'])
                    st.write("Training complete. Best model:", best_model)

                    model_filename = os.path.join('model_dir', f'{target_variable}_classification_model.pkl')
                    
with open(model_filename, "wb") as f:

                        pickle.dump(best_model, f)
                    st.success(f"Model saved as {model_filename}")

                    with open(model_filename, "rb") as f:
                        st.download_button("Download Model", data=f, file_name=model_filename)
            else:
                if st.button("Train Regression Model with PyCaret"):
                    setup_regression(data=df, target=target_variable, session_id=123, fold_strategy='kfold')
                    best_model = compare_model_dir_regression(include=['lr', 'rf', 'dt', 'svm', 'lasso'])
                    st.write("Training complete. Best model:", best_model)

                    model_filename = os.path.join('model_dir', f'{target_variable}_regression_model.pkl')
                    
with open(model_filename, "wb") as f:

                        pickle.dump(best_model, f)
                    st.success(f"Model saved as {model_filename}")

                    with open(model_filename, "rb") as f:
                        st.download_button("Download Model", data=f, file_name=model_filename)

        # Manual Mode
        elif mode == "Manual Mode":
            st.write("Customize your model training:")
            
            # Select columns to include or exclude
            cols_to_drop = st.multiselect("Select columns to drop", df.columns.tolist(), default=None)
            df_manual = df.drop(cols_to_drop, axis=1)

            st.write("Class Distribution:")
            st.write(df_manual[target_variable].value_counts())

            is_classification = df_manual[target_variable].dtype == 'object' or df_manual[target_variable].nunique() < 10

            # Manually select model_dir
            model_dir_to_use = st.multiselect("Select model_dir to train", options=['xgboost', 'lightgbm', 'rf', 'dt', 'knn', 'nb', 'lr', 'svm'])

            # Define evaluation metrics based on task type
            if is_classification:
                metric_options = ["Accuracy", "AUC", "F1", "Precision", "Recall"]
                metric_map = {
                    "Accuracy": "Accuracy",
                    "AUC": "AUC",
                    "F1": "F1",
                    "Precision": "Precision",
                    "Recall": "Recall"
                }
            else:
                metric_options = ["RMSE", "MAE", "R2", "MSE"]
                metric_map = {
                    "RMSE": "RMSE",
                    "MAE": "MAE",
                    "R2": "R2",
                    "MSE": "MSE"
                }

            # Model Evaluation Metric
            evaluation_metric = st.selectbox("Select Evaluation Metric", options=metric_options)

            if is_classification:
                if st.button("Train Selected Classification Model") and model_dir_to_use:
                    setup_classification(data=df_manual, target=target_variable, session_id=123, fold_strategy='stratifiedkfold')
                    best_model = compare_model_dir_classification(include=model_dir_to_use, sort=metric_map[evaluation_metric])
                    st.write("Training complete. Best model:", best_model)

                    model_filename = os.path.join('model_dir', f'{target_variable}_manual_classification_model.pkl')
                    
with open(model_filename, "wb") as f:

                        pickle.dump(best_model, f)
                    st.success(f"Model saved as {model_filename}")

                    # Download option for manual mode
                    with open(model_filename, "rb") as f:
                        st.download_button("Download Model", data=f, file_name=model_filename)

            else:
                if st.button("Train Selected Regression Model") and model_dir_to_use:
                    setup_regression(data=df_manual, target=target_variable, session_id=123, fold_strategy='kfold')
                    best_model = compare_model_dir_regression(include=model_dir_to_use, sort=metric_map[evaluation_metric])
                    st.write("Training complete. Best model:", best_model)

                    model_filename = os.path.join('model_dir', f'{target_variable}_manual_regression_model.pkl')
                    
with open(model_filename, "wb") as f:

                        pickle.dump(best_model, f)
                    st.success(f"Model saved as {model_filename}")

                    # Download option for manual mode
                    with open(model_filename, "rb") as f:
                        st.download_button("Download Model", data=f, file_name=model_filename)

# Recent Actions Section
elif option == "Recent Actions":
    st.header("Recent Actions")
    if st.session_state.recent_actions:
        for action in st.session_state.recent_actions:
            st.write(action)
    else:
        st.write("No recent actions to show.")
