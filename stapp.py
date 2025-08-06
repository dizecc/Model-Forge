import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from model_hyperparamaters import model_registry
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

from regressor_hyperparameters import regressor_registry
from ml_backend import is_regression_task


from ml_backend import (
    prepare_data,
    handle_outliers,
    build_preprocessor,
    run_pro_mode,
    evaluate_model,
)

st.markdown("""
<style>
/* Style section labels */
.radio-section [data-baseweb="radio"] > div {
    background-color: #1e293b !important;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    padding: 0.6rem 1rem;
    /* Removed width, display, gap for vertical stacking */
}

.radio-section [data-baseweb="radio"] label {
    color: #e2e8f0;
    font-size: 1.05rem;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.radio-section [data-baseweb="radio"] svg {
    fill: black !important;
    stroke: black !important;
    width: 20px;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
.option-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #a78bfa;  /* Violet */
    margin-bottom: 1.2rem;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
<style>
/* Custom label styling */
.section-label {
    font-size: 1.4rem;
    font-weight: 800;
    color: #3b82f6; /* Light blue */
    margin-bottom: 0.4rem;
    margin-top: 1.2rem;
}
</style>
""", unsafe_allow_html=True)





# -----------------SIDEBAR UI------------------------

models = {
    "Random Forest": "rf",
    "Logistic Regression": "logreg",
    "Support Vector Machine": "svm",
    "K-Nearest Neighbors": "knn",
    "MLP (Neural Network)": "mlp"
}

st.sidebar.title("Available Models")


st.sidebar.markdown("---")

for name in models:

    st.sidebar.markdown(f"- **{name}** ")
    st.sidebar.markdown("<hr style='margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)

# ------------------------------------------------------

st.set_page_config(page_title="ModelForge", layout="wide")

# -------------PAGE TITLE-----------------------------

st.markdown("""
<h1 style='
    font-size: 3rem;
    font-weight: 900;
    text-align: center;
    color: #7c3aed;
    margin-bottom: 0.2rem;
    text-shadow: 1px 1px 2px #00000050;
'>
 Model Forge
</h1>
<p style='
    text-align: center;
    font-size: 1.2rem;
    color: #cbd5e1;
    margin-top: 0.2rem;
    margin-bottom: 2rem;
'>
Where raw data meets the heat of machine learning — welcome to the forge.
</p>
""", unsafe_allow_html=True)


st.markdown("<hr style='border: 1px solid #bbb;'>", unsafe_allow_html=True)

st.markdown("""
ModelForge is an automated machine learning comparison tool where models are forged, tested, and ranked — all from a simple dataset upload.

Whether you're a data scientist, student, or ML enthusiast, ModelForge lets you:

- Upload any CSV dataset with a target column

- Forge multiple models like Logistic Regression, SVM, Random Forest, KNN, and MLP

- Evaluate them on core metrics: Accuracy, Precision, Recall, and F1-score

- Compare results side-by-side in a clean, interactive UI

- Understand which model best fits your data — instantly

Forget boilerplate code. With ModelForge, your dataset becomes a battleground of algorithms — and the best model rises from the forge.
""")
st.markdown("<hr style='border: 1px solid #bbb;'>", unsafe_allow_html=True)
# File upload
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded Successfully!")
    st.markdown("<hr style='border: 1px solid #bbb;'>", unsafe_allow_html=True)
    st.markdown("### <span style='color:#0dbd19'>Dataset view:</span>", unsafe_allow_html=True)
    st.dataframe(df.head())
    st.markdown("<hr style='border: 1px solid #bbb;'>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="option-container">', unsafe_allow_html=True)
        st.markdown('<div class="option-title">🛠️ Configure Your Model Settings</div>', unsafe_allow_html=True)

        target_column = st.selectbox("Select target column (what to predict)", df.columns)
        st.markdown('<div class="section-label">Outlier Handling Method:</div>', unsafe_allow_html=True)
        outlier_method = st.radio("", ["skip", "zscore", "iqr"])

        with st.expander("What is Z-Score and IQR ?"):
            st.markdown("""
        #### Z-Score Method
        - **Definition**: Measures how far a data point is from the mean in terms of standard deviations.
        - **Outlier Rule**: If Z > 3 or Z < -3, it's considered an outlier.
        - **When to Use**:
          - Data is **normally distributed** (bell curve).
          - **No extreme skewness**.
          - Works best with **continuous numerical features**.

        ---

        #### IQR (Interquartile Range) Method
        - **Definition**: Uses percentiles (Q1 and Q3) to find spread of the middle 50% of data.
        - **Outlier Rule**: Outliers fall below **Q1 - 1.5×IQR** or above **Q3 + 1.5×IQR**.
        - **When to Use**:
          - Data is **not normally distributed** or is **skewed**.
          - **More robust** to extreme values and non-Gaussian distributions.
          - Suitable for **small datasets** or features with long tails.

        ---

        Choosing the right method depends on the shape and spread of your data.  
        Use `Z-Score` for symmetric, clean data. Use `IQR` for skewed, noisy distributions.
            """)

        st.markdown('<div class="section-label">Choose Training Mode:</div>', unsafe_allow_html=True)
        model_mode = st.radio(" ", ["Run All Models (Pro Mode)", "Choose a Single Model"])
        selected_model_key = None
        if model_mode == "Choose a Single Model":
            selected_model_key = st.selectbox("🔍 Select a Model", list(model_registry.keys()))

        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Start Training"):
        with st.spinner("Running AutoML pipeline..."):

            st.markdown("<hr style='border: 1px solid #bbb;'>", unsafe_allow_html=True)

            # Data prep
            df_clean, outlier_cols = handle_outliers(df, method=outlier_method)
            x, y, numeric_features, categorical_features = prepare_data(df_clean, target_column)

            is_regression = is_regression_task(y)
            registry = regressor_registry if is_regression else model_registry

            preprocessor = build_preprocessor(numeric_features, categorical_features)

            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Train models

            if model_mode == "Run All Models (Pro Mode)":
                results, best_model, best_name = run_pro_mode(
                    registry, x_train, x_test, y_train, y_test, preprocessor, save_best=False,

                )
            else:
                selected_registry = {selected_model_key: model_registry[selected_model_key]}
                results, best_model, best_name = run_pro_mode(
                    selected_registry, x_train, x_test, y_train, y_test, preprocessor, save_best=False
                )

            # Model Comparison Table
            st.markdown("<hr style='border: 1px solid #bbb;'>", unsafe_allow_html=True)
            st.markdown("### Model Comparison Table")
            comparison_df = pd.DataFrame(results).set_index("model")
            comparison_df = comparison_df[["accuracy", "precision", "recall", "f1_score"]]
            comparison_df = comparison_df.round(4)
            st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'))

            # Evaluate best model
            acc, prec, rec, f1, report = evaluate_model(best_model, x_test, y_test)

            with st.expander("📘 What do these metrics mean?"):
                st.markdown("""
            - **Accuracy**  
              Measures how often the model is correct overall. It’s the ratio of correctly predicted observations to the total observations.

            - **Precision**  
              Precision tells us how many of the predicted positive cases were actually correct. Useful when **false positives** are costly.

            - **Recall**  
              Recall tells us how many of the actual positive cases we were able to correctly predict. Useful when **false negatives** are costly.

            - **F1 Score**  
              The harmonic mean of Precision and Recall. It balances both false positives and false negatives — especially useful for **imbalanced datasets**.
                """)

            st.markdown("<hr style='border: 1px solid #bbb;'>", unsafe_allow_html=True)

            st.markdown("### <span style='color:#0dbd19'>Model Evaluation: </span>", unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("Precision", f"{prec:.4f}")
            col3.metric("Recall", f"{rec:.4f}")
            col4.metric("F1 Score", f"{f1:.4f}")




            model_filename = f"{best_name}_model.pkl"
            joblib.dump(best_model, model_filename)

            with open(model_filename, "rb") as f:
                st.download_button(
                    label="📥 Download Trained Model",
                    data=f,
                    file_name=model_filename,
                    mime="application/octet-stream"
                )

            # st.write(f"**Best Model:** {best_name.upper()}")
            # st.write(f"**Accuracy:** {acc:.4f}")
            # st.write(f"**Precision:** {prec:.4f}")
            # st.write(f"**Recall:** {rec:.4f}")
            # st.write(f"**F1 Score:** {f1:.4f}")



            # -----------------------------CLASSIFICATION REPORT------------------------------

            st.markdown("### Classification Report")
            st.dataframe(pd.DataFrame(report).transpose())

            # dwnld Classification Report
            report_df = pd.DataFrame(report).transpose()
            csv_report = report_df.to_csv(index=True)

            st.download_button(
                label="📥 Download Classification Report (CSV)",
                data=csv_report,
                file_name="classification_report.csv",
                mime="text/csv"
            )

            st.markdown("<hr style='border: 1px solid #bbb;'>", unsafe_allow_html=True)

            st.markdown("""
            <div style="border-radius: 12px; border: 1px solid #ccc; padding: 1.5rem; background-color: #0c011c;">
              <h4 style="margin-top: 0;">📘 <span style="color: #600be0;">Model Summary</span></h4>

              <p>The selected <strong>best-performing model</strong> was <span style="color: #7c3aed;"><strong>Random Forest (RF)</strong></span>, chosen based on the highest test accuracy.</p>

              <h5>⚙️ Preprocessing Steps:</h5>
              <ul>
                <li>Missing value imputation</li>
                <li>Standard scaling for numerical features</li>
                <li>One-Hot Encoding for categorical features</li>
              </ul>

              <h5>📉 Other Configurations:</h5>
              <ul>
                <li><strong>Outlier Handling Method:</strong> <code>SKIP</code></li>
                <li><strong>Train/Test Split:</strong> <code>80/20 ratio</code></li>
              </ul>

              <h5>📈 Evaluation Metrics:</h5>
              <ul>
                <li><strong>Accuracy:</strong> <code>0.0380</code></li>
                <li><strong>Precision:</strong> <code>0.0353</code></li>
                <li><strong>Recall:</strong> <code>0.0260</code></li>
                <li><strong>F1 Score:</strong> <code>0.0175</code></li>
              </ul>

              <p style="margin-top: 1rem;">✅ This model is ready for <strong>deployment</strong> or further <strong>hyperparameter tuning</strong> based on your project needs.</p>
            </div>
            """, unsafe_allow_html=True)


else:
    st.info("👆 Please upload a CSV file to get started.")

st.markdown("""
<hr style="border: 0.5px solid #ccc; margin-top: 3rem;">

<div style="text-align: center; padding: 10px; font-size: 1.2rem; color: #cbd5e1;">
    Made by <strong>Prathmesh Bajpai</strong><br><br>
    <a href="https://www.linkedin.com/in/prathmesh-bajpai-8429652aa/" target="_blank" style="color: #3b82f6; text-decoration: none; margin-right: 20px;">
        LinkedIn
    </a>
    |
    <a href="https://github.com/LEADisDEAD" target="_blank" style="color: #3b82f6; text-decoration: none; margin-left: 20px;">
        GitHub
    </a>
</div>
""", unsafe_allow_html=True)

st.stop()


