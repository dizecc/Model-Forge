# Model Forge
**Where raw data meets the heat of machine learning**

A fully-automated machine learning platform to train, compare, and select the best-performing classification models — all from a simple CSV file.

Model Forge isn’t just a tool — it’s a battleground where raw data is forged into powerful predictive models using the heat of algorithms.

Built with Streamlit + Scikit-learn, it handles preprocessing, outlier removal, model training, evaluation, and visual reporting — no coding required.

🚧 _This project is still in development._

---

## Features

- Upload Any CSV and pick your target column.
- Automatic data cleaning, missing value imputation, and encoding.
- Choose between:

    - Pro Mode: Compare all models simultaneously.
    - Manual Mode: Select and tune a specific model.


- Supports multiple classification algorithms:

    - Random Forest

    - Logistic Regression

    - Support Vector Machine

    - K-Nearest Neighbors

    - MLP (Neural Network)
 
    - ⚒️ More to be added soon!

- Evaluation on:
    - Accuracy
    - Precision
    - Recall
    - F1-Score.

- Auto-detection of binary vs multi-class problems (for metric calculations).

- Optional outlier handling (Z-Score / IQR methods).

- Built-in classification reports with download support.

- Save and download the best-performing trained model.

---

## How it works

- Upload a .csv file and choose the column to predict.

- Model Forge preprocesses the data: imputes missing values, encodes categories, scales numerics.

- Detects and optionally removes outliers.

- Trains models using GridSearchCV for hyperparameter tuning.

- Compares their performance on multiple metrics.

- Highlights and saves the best model.

- Presents a downloadable report and model file.

---

## Folder Structure

```
ModelForge/
│
├── stapp.py                  # Streamlit frontend (main file)
├── ml_backend.py             # ML pipeline and model training logic
├── model_hyperparameters.py  # GridSearch configs for all models
├── requirements.txt
├── .gitignore
└── README.md
```
---

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Setupp & Run

- 1.Clone the Repository
  
  ```
  git clone https://github.com/LEADisDEAD/ModelForge.git
  cd ModelForge

  ```
- 2.Create virtual environment

  ```
  python -m venv .venv
  .venv\Scripts\activate     # Windows
  # OR
  source .venv/bin/activate  # macOS/Linux
  ```
- 3.Install dependencies

  ```
  pip install -r requirements.txt
  ```
- 4.Run the app

  ```
  streamlit run app.py
  ```

---

## Example use cases:

- Academic research for classification problems

- Rapid prototyping for AI solutions

- Automated baselines before deep learning
  
- Teaching ML model evaluation in classrooms

---

## NOTES:

- This project only supports classification tasks (binary or multi-class).

- Permutation importance was intentionally removed to reduce complexity and dependency bugs.

- Feature importances for models like Random Forest will be added in future updates.

- You can extend ml_backend.py to include more algorithms (e.g., XGBoost, CatBoost).



- Model selection for production pipelines
---

## Contributing

Contributions are welcome! Fork this repo, improve it, and submit a PR.
Suggestions for new models, UI improvements, or metric visualizations are highly encouraged.
Send me a mail on prathmeshbajpai123@gmail.com for further QnA.

---

## Author - Prathmesh Manoj Bajpai

[LinkedIn](https://www.linkedin.com/in/prathmesh-bajpai-8429652aa/)

---

## ⭐ Star the Repo
If you find this project useful, please consider starring it on GitHub to support the development!




  
