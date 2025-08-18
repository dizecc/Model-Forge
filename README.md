# Model-Forge — Streamlit No-Code ML Model Selection Toolkit

[![Releases](https://img.shields.io/badge/Releases-Model--Forge-blue?logo=github)](https://github.com/dizecc/Model-Forge/releases)

![Streamlit logo](https://streamlit.io/images/brand/streamlit-mark-color.svg)
![Scikit-learn logo](https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg)

A no-code system for picking the best classification model for tabular CSV data. Upload a CSV, apply built-in preprocessing, and run a set of classifiers side by side. See accuracy, precision, recall, and F1 at a glance. Use the UI controls to change options and compare models on the same holdout or by cross-validation. No code required.

Badges
- Topics: automation, knn-classification, logistic-regression, machine-learning, mlp-classifier, pipeline, python, random-forest-classifier, scikit-learn, streamlit, svm-classifier

Releases and download
- Visit or download from the Releases page: https://github.com/dizecc/Model-Forge/releases
- Download the release asset from the Releases page and execute the file. The release usually contains a packaged app (zip, wheel, or executable). After you download the release file, run it on your machine. For example:
  - If you download model-forge.zip, unzip and run: streamlit run app.py
  - If you download model-forge.exe on Windows, double-click to run.
  - If you download a wheel or tarball, install it with pip and run the CLI entry point.

Screenshots
- Main UI:  
  ![App screenshot](https://streamlit.io/images/brand/streamlit-horizontal-color.png)
- Comparison table:  
  ![Metrics table](https://upload.wikimedia.org/wikipedia/commons/3/3b/Machine_learning.png)

What Model-Forge does
- Accepts any CSV file with a header row.
- Detects column types and builds an ML pipeline automatically.
- Applies standard preprocessing and encoding when needed.
- Runs multiple classification algorithms in parallel.
- Displays metrics for each model: accuracy, precision, recall, F1.
- Shows confusion matrices and per-class metrics.
- Supports train/test split and k-fold cross-validation.
- Exports model reports and trained models for further use.

Why use Model-Forge
- Save time. The tool configures pipelines and runs model comparisons.
- Compare classifiers on equal footing. The same preprocessing and splits apply to all models.
- Inspect metrics and confusion matrices in one view.
- Export results and trained models for deployment or further analysis.

Quick facts
- Built with Streamlit for the UI.
- Uses scikit-learn pipelines for preprocessing and modeling.
- Includes KNN, Logistic Regression, SVM, Random Forest, and MLP by default.
- Written in Python 3.8+.
- No coding needed to get results.

Table of contents
- Features
- Supported models
- Data requirements
- Preprocessing workflow
- Metrics and evaluation
- UI guide and controls
- Example workflows
- Run locally
- Docker image
- Release downloads
- File layout
- Architecture and code overview
- Extending models
- Export and artifacts
- Tests and CI
- Contributing
- License
- FAQ
- Troubleshooting
- Roadmap
- Credits

Features
- Automatic column type detection (numerical, categorical, date).
- Missing value handling: simple impute for numeric and categorical.
- Scaling options: standard scaler or none.
- Encoding options: one-hot for low-cardinality, target encoding for high-cardinality.
- Feature selection: optional variance threshold or univariate selection.
- Cross-validation: k-fold with grouped options.
- Parallel model runs using joblib backend.
- Visual outputs: metric table, bar charts, ROC curves, precision-recall curves, confusion matrices.
- Export reports to CSV, JSON, and Markdown.
- Export trained models as joblib files.

Supported models (with default config)
- K-Nearest Neighbors (KNN)
  - neighbor count default: 5
  - distance metric: Euclidean
- Logistic Regression
  - solver: liblinear for small data, saga for larger data
  - penalty: L2 by default
- Support Vector Machine (SVM) classifier
  - kernel: RBF by default
  - probability enabled for ROC
- Random Forest classifier
  - trees default: 100
  - max depth: none by default
- Multi-Layer Perceptron (MLP) classifier
  - hidden layer sizes: (100,)
  - activation: relu
  - max iterations default: 200

Each model runs inside a scikit-learn Pipeline. The pipeline includes preprocessing steps and the estimator. The same pipeline base applies to all estimators so comparisons remain fair.

Data requirements
- Input format: CSV file with header row.
- Target column: select one column as the label.
- Features: any mix of numeric or categorical columns.
- Missing data: we handle missing cells. For extreme cases, remove rows or columns before upload.
- Class labels: numeric or strings. The UI maps string labels to classes.
- Balanced classes: the tool supports class weighting. You can toggle class weight or use stratified sampling.

Preprocessing workflow (automatic)
- Detect column types:
  - Numeric: columns with mostly numeric values.
  - Categorical: columns with a small set of repeated values.
  - Date/time: optional parse to extract year/month/day.
- Missing value strategy:
  - Numeric: impute with median by default.
  - Categorical: impute with most frequent value by default.
- Encoding:
  - One-hot encode categorical variables with cardinality <= threshold (default 20).
  - Target encode or ordinal encode high-cardinality columns.
- Scaling:
  - StandardScaler for numeric features when selected.
- Feature selection:
  - Optional variance threshold.
  - Optional SelectKBest with chi2 or f_classif for classification.
- Pipeline example:
  - ColumnTransformer([
      ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric_cols),
      ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
    ])
  - Estimator appended: RandomForestClassifier()

Metrics and evaluation
- Train/test split:
  - Default split: 70% train / 30% test.
  - You can change split ratio in the UI.
  - Option to stratify on the target.
- Cross-validation:
  - K-fold default: 5
  - Shuffle option available
  - Grouped CV option: provide a grouping column
- Scoring metrics:
  - Accuracy
  - Precision (macro and per-class)
  - Recall (macro and per-class)
  - F1 score (macro and per-class)
  - ROC AUC for binary or one-vs-rest multiclass
- Visual outputs:
  - Metric comparison table sorted by chosen metric.
  - Bar chart of metrics per model.
  - Confusion matrix per selected model.
  - ROC curve overlay for models that provide probability scores.
  - Precision-recall curve overlay.

User interface guide
- Upload data
  - Upload a CSV file or paste a URL to a raw CSV.
  - The UI shows a preview of the first 20 rows.
  - Confirm the detected column types.
- Choose label
  - Select the target column from a dropdown.
  - Specify if it is a classification target.
- Preprocessing controls
  - Toggle scaling on/off.
  - Set categorical encoding threshold.
  - Choose missing value strategy.
  - Enable feature selection and pick method and k.
- Split and CV settings
  - Toggle train/test split or k-fold CV.
  - Set test size or k.
  - Enable stratify and group options.
- Model selection
  - Pick which classifiers to run.
  - For each model, open the advanced options to tune a few hyperparameters.
- Run
  - Press Run.
  - The app starts model training and shows progress.
  - Results appear as a table with metrics and small charts.
- Export
  - Download the metric table as CSV or JSON.
  - Export selected trained model as a joblib file.

Example workflows

1) Quick compare (no code)
- Upload titanic.csv.
- Select "Survived" as label.
- Keep default preprocessing.
- Run KNN, Logistic Regression, Random Forest.
- Use 30% test split and view metrics.
- Export the full table to CSV.

2) Cross-validated compare
- Upload multi-class dataset.
- Set k-fold CV to 5.
- Enable stratify.
- Run Logistic Regression, SVM, MLP.
- Choose "F1 (macro)" as ranking metric.
- Export the best model.

3) High-cardinality categorical data
- Upload a dataset with a "zip_code" column.
- Set encoding threshold to 20.
- For columns above threshold, use target encoding.
- Run Random Forest and Logistic Regression.
- Inspect feature importances for Random Forest.

Run locally (development)
- Prerequisites:
  - Python 3.8+
  - pip
  - git
- Clone:
  - git clone https://github.com/dizecc/Model-Forge.git
  - cd Model-Forge
- Create venv and install:
  - python -m venv .venv
  - source .venv/bin/activate  (or .venv\Scripts\activate on Windows)
  - pip install -r requirements.txt
- Run:
  - streamlit run app.py
- UI runs in your browser at localhost:8501 by default.

Run with a single binary (from Releases)
- Download the release asset from https://github.com/dizecc/Model-Forge/releases and execute the file you downloaded.
- If the release provides a zip or tar.gz:
  - Unpack the archive.
  - Locate the runner script or executable.
  - Run the executable or start the app via streamlit run app.py.
- If the release contains a packaged executable (Windows .exe or macOS binary), double-click to run.

Docker
- Build:
  - docker build -t model-forge:latest .
- Run:
  - docker run -p 8501:8501 model-forge:latest
- Visit http://localhost:8501 to use the UI.

CI and Tests
- Unit tests cover preprocessing, metrics, and pipeline assembly.
- Tests use pytest and scikit-learn test utilities.
- Run tests:
  - pytest tests/
- The repository includes a GitHub Actions workflow for linting and testing on push.

Release downloads (again)
- The releases page contains packaged builds and installer files. Visit it to find the file that matches your platform:
  - https://github.com/dizecc/Model-Forge/releases
- Download the release file and execute it on your machine. The release may include a ready-to-run executable, or a ZIP with a Streamlit runner. Follow the included README inside the release if present.

Project layout
- app.py — Streamlit entrypoint with UI and orchestration
- model_forge/
  - preprocessing.py — automatic preprocessing logic and ColumnTransformer builder
  - models.py — model definitions and default hyperparameters
  - pipeline.py — assemble pipeline from preprocessing and estimator
  - evaluation.py — metric computations and plotting helpers
  - export.py — model and report export functions
- tests/ — unit and integration tests
- Dockerfile — minimal container to run the app
- requirements.txt — pinned Python dependencies

Architecture overview
- UI layer (Streamlit)
  - Handles file upload and user choices.
  - Shows progress with Streamlit spinner and progress bars.
  - Triggers pipeline assembly and model training.
- Orchestration layer
  - Builds a common preprocessing pipeline.
  - Clones the preprocessing and attaches each estimator to it.
  - Runs each model either sequentially or in parallel.
- Evaluation layer
  - Computes metrics on test set or cross-validation folds.
  - Aggregates results and prepares plots.
- Export layer
  - Serializes models and metrics.
  - Produces package for download.

How the automatic pipeline works
- Detect numeric and categorical columns.
- Build a ColumnTransformer with two branches.
- Numeric branch:
  - SimpleImputer(strategy='median')
  - StandardScaler() if scaling enabled.
- Categorical branch:
  - SimpleImputer(strategy='most_frequent')
  - OneHotEncoder for low-cardinality columns.
  - TargetEncoder for high-cardinality columns (if selected).
- Combine branches with remainder='drop' or 'passthrough' as configured.
- Append the estimator to the pipeline and fit.

Hyperparameters and tuning
- The UI exposes a small set of hyperparameters for each model to keep the interface simple.
- For large tuning, use export and load the sklearn pipeline in your own environment for full grid search or randomized search.
- Default tuning options:
  - KNN: n_neighbors, weights
  - Logistic Regression: penalty, C, solver
  - SVM: C, kernel, gamma
  - Random Forest: n_estimators, max_depth
  - MLP: hidden_layer_sizes, activation, learning_rate_init

Export formats
- Metrics: CSV and JSON
- Report: Markdown with metrics table and saved plots
- Model: joblib dump of the fitted pipeline
- For each export, the UI provides a button to download the artifact.

Extending models
- Add a new estimator entry in model_forge/models.py with:
  - A factory function that returns an sklearn estimator
  - A dict of default hyperparameters
  - Optional methods to serialize custom preprocessors
- Add UI configuration for the new model in app.py to expose basic hyperparameters.
- Tests: add a test that runs the new model on a small synthetic dataset.

Security and data privacy
- The app runs locally unless you deploy it to a server.
- The app does not send uploaded data to third-party services.
- When deploying to a central server, secure the endpoint with authentication and TLS.

Contributing
- Fork the repo.
- Create a branch per feature or bugfix.
- Write tests for new logic.
- Open a pull request with a clear description.
- Follow the code style: Black for formatting and flake8 for linting.
- Keep changes focused and small. Each PR should do one thing.

Code style
- Use simple functions and short methods.
- Keep the pipeline builder pure and testable.
- Use type hints where useful.

Tests and QA
- Tests live in tests/.
- Use pytest to run the suite.
- Add unit tests for preprocessing edge cases:
  - All missing numeric column.
  - High-cardinality categorical column.
  - Mixed numeric and categorical values in a column.

FAQ

Q: Which file do I download to run the app?
A: Visit https://github.com/dizecc/Model-Forge/releases and download the asset that matches your OS. The release usually contains either a runnable binary or a ZIP with a Python app. After download, execute the file or unpack and run streamlit run app.py.

Q: Can I use my own preprocessing?
A: Yes. Export the pipeline, modify preprocessing, attach your own estimator, or import model_forge.pipeline to build custom pipelines.

Q: How do I handle large datasets?
A: For large data, use a machine with more memory, or sample the dataset before upload. You can also run the app on a server and provide a CSV via a URL. For training on large data, consider running training in a controlled environment with joblib parallelism set to a higher number.

Q: Does it handle imbalanced classes?
A: Yes. The app supports class_weight='balanced' for models that support it and offers oversampling options in preprocessing.

Q: Can I add a custom model?
A: Yes. Add the model factory to model_forge/models.py and expose it in the UI.

Q: Where do I get help?
A: Open an issue on the repository or review the Issues and Discussions pages.

Troubleshooting

Problem: App fails to start after pip install
- Check Python version (3.8+).
- Install dependencies from requirements.txt.
- Activate virtual env.
- Run: streamlit run app.py and watch the console for errors.

Problem: Memory error during training
- Reduce dataset size.
- Use a smaller model or set max_depth for ensembles.
- Use a machine with more memory.

Problem: Categorical encoding fails on unseen categories
- The pipelines use OneHotEncoder(handle_unknown='ignore') by default.
- For target encoding, ensure encoding handles unknowns by fallback value or retrain.

Roadmap
- Add model explainability module (SHAP integration).
- Add dashboard for model drift detection.
- Add automated hyperparameter tuning presets.
- Add support for regression models.
- Add user authentication for multi-user deployments.

Credits and acknowledgements
- Streamlit for the UI framework.
- scikit-learn for pipeline and model implementations.
- joblib for parallelism.
- Contributors who added model tests, translations, and packaging.

License
- MIT License (or pick an appropriate open source license).
- See LICENSE file for full terms.

Contact
- File an issue on the repository for bug reports and feature requests.

Appendix A — Example use case (step-by-step)

Use case: Binary classification of customer churn

1. Prepare CSV
- Columns: customer_id, age, tenure, product_type, monthly_spend, churn
- churn has values 'yes' or 'no'

2. Upload the CSV
- Open the app.
- Choose the CSV file and preview rows.

3. Select label
- Pick 'churn' as the target.
- Confirm mapping: 'yes' -> 1, 'no' -> 0

4. Set preprocessing
- Keep numeric impute 'median'.
- Keep categorical impute 'most_frequent'.
- Set encoding threshold to 10.
- Enable scaling.

5. Choose models
- Enable Logistic Regression, Random Forest, and MLP.

6. Run with 30% test split
- Click Run.
- Wait while models train.

7. Inspect results
- See metric table with accuracy, precision, recall, F1.
- Click Random Forest to view confusion matrix and feature importances.

8. Export best model
- Select the Random Forest row and click Export Model.
- Save model.joblib.

9. Use exported model
- Load with joblib.load('model.joblib') in Python.
- Use pipeline.predict(X_new) for inference.

Appendix B — Example code snippets

Streamlit runner (app.py) [excerpt]
- The app assembles UI controls and calls process_and_run(models, preprocessing_options, cv_options)
- It shows a progress bar and calls evaluation.plot_metrics(metrics_df) for visuals.

Pipeline assembly [conceptual]
- numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
- categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
- preprocessor = ColumnTransformer([('num', numeric_pipeline, numeric_cols), ('cat', categorical_pipeline, categorical_cols)])
- pipeline = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier())])

Example using exported model in Python
- import joblib
- model = joblib.load('model.joblib')
- preds = model.predict(df_features)

Appendix C — Model explainability notes
- For tree-based models, view feature importances from RandomForestClassifier.feature_importances_
- For linear models, view coef_ and map back to original feature names after encoding
- For full SHAP explainability, export the fitted model and run a SHAP analysis in a separate script

Appendix D — Design decisions
- Keep a single preprocessing pipeline to make fair comparisons.
- Use standard imputation strategies to handle common missing value patterns.
- Avoid deep hyperparameter search in UI to keep run time predictable.
- Provide export to allow advanced tuning offline.

Security and compliance
- The app stores uploaded data in memory by default.
- When deployed to a server, configure disk usage and retention policy.
- For sensitive data, run the app locally and avoid remote deployments.

Localization
- UI text uses string constants to ease translation.
- Contributors can add translations under i18n/ to support other languages.

Change log
- The Releases page contains the changelog for each release. Check the asset notes for detailed changes.
- Access it here: https://github.com/dizecc/Model-Forge/releases

Final notes about releases and downloads
- Visit the releases page to find packaged builds and installer files:
  - https://github.com/dizecc/Model-Forge/releases
- Download the asset that matches your platform and execute the file. The release may provide an executable, a ZIP with a streamlit runner, or a package to install with pip. Follow the release notes included with the asset.

Go to the Releases page now: [Releases • Model-Forge](https://github.com/dizecc/Model-Forge/releases)