import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from scipy import stats
import os
import joblib


def prepare_data(df, target_column):
    df = df.dropna(axis=1, how='all')
    y = df[target_column]
    x = df.drop(columns=[target_column])
    numeric_features = x.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = x.select_dtypes(include=["object", "category"]).columns.tolist()
    return x, y, numeric_features, categorical_features

def is_regression_task(y):
    return pd.api.types.is_float_dtype(y) or y.nunique() > 10

def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor

def detect_outliers(df, method = "zscore",threshold=3):
    outlier_cols = []
    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns

    for col in numeric_cols:
        if method=="zscore":
            z = np.abs(stats.zscore(df[col]))
            if(z > threshold).sum() > 0:
                outlier_cols.append(col)

        elif method =="iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5*IQR
            upper = Q3 + 1.5*IQR

            if ((df[col]<lower)|(df[col]>upper)).sum() > 0:
                outlier_cols.append(col)

    return outlier_cols

def remove_outliers(df, method="zscore", threshold=3):
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=np.number).columns:
        if method == "zscore":
            z = np.abs(stats.zscore(df_clean[col]))
            df_clean = df_clean[z < threshold]
        elif method == "iqr":
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean

def handle_outliers(df, method="skip", threshold=3):
    outlier_cols = detect_outliers(df, method=method, threshold=threshold)

    if not outlier_cols:
        return df, []

    if method == "zscore":
        df = remove_outliers(df, method="zscore", threshold=threshold)
    elif method == "iqr":
        df = remove_outliers(df, method="iqr", threshold=threshold)

    return df, outlier_cols

def show_permutation_importance(model_pipeline, X_test, y_test, numeric_features, categorical_features, plot=True):
    try:
        preprocessor = model_pipeline.named_steps['preprocessor']
        classifier = model_pipeline.named_steps['classifier']

        X_test_transformed = preprocessor.transform(X_test)

        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
        all_feature_names = list(numeric_features) + list(cat_feature_names)

        result = permutation_importance(
            classifier,
            X_test_transformed,
            y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )

        if len(result.importances_mean) != len(all_feature_names):
            all_feature_names = [f"feature_{i}" for i in range(len(result.importances_mean))]

        importances = pd.Series(result.importances_mean, index=all_feature_names)
        importances_sorted = importances.sort_values(ascending=False)

        # Show only Top 3
        top_k = 3
        top_features = importances_sorted.head(top_k)

        st.markdown(f"### Top {top_k} Most Important Features")
        st.dataframe(top_features.to_frame("Importance"))

        # if plot:
        #     fig, ax = plt.subplots(figsize=(4, 2))
        #     top_features.plot(kind='barh', ax=ax, color="skyblue")
        #
        #     ax.set_title("Feature Importance", fontsize=10)
        #     ax.set_xlabel("Importance", fontsize=8)
        #     ax.set_ylabel("")
        #     ax.tick_params(axis='both', labelsize=8)
        #     ax.invert_yaxis()

            # fig.tight_layout()
            # st.pyplot(fig)

        return importances_sorted

    except Exception as e:
        st.error(f"⚠️ Error in calculating permutation importance: {e}")


def run_pro_mode(model_registry, x_train, x_test, y_train, y_test, preprocessor, save_best=False, is_regression=False):
    results = []
    best_model = None
    best_score = 0
    best_name = ""

    print("\n🔁 Running Pro Mode: Evaluating all models....\n")

    for name, config in model_registry.items():
        print(f"🧠 Training Model: {name}")

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', config["estimator"])
        ])

        grid = GridSearchCV(
            pipeline,
            param_grid=config["param_grid"],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )

        grid.fit(x_train, y_train)
        y_pred = grid.predict(x_test)

        # too Detect if it's binary or multiclass
        average_type = 'binary' if len(np.unique(y_test)) == 2 else 'macro'

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=average_type, zero_division=0)
        rec = recall_score(y_test, y_pred, average=average_type, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=average_type, zero_division=0)

        results.append({
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "best_params": grid.best_params_
        })

        if acc > best_score:
            best_score = acc
            best_model = grid.best_estimator_
            best_name = name

    print("\n📊 Model Comparison:")
    print("-" * 65)
    for r in results:
        print(
            f"{r['model']:<12} | Acc: {r['accuracy']:.4f} | Prec: {r['precision']:.4f} | Recall: {r['recall']:.4f} | F1: {r['f1_score']:.4f}")
    print("-" * 65)

    print(f"\n🏆 Best Model: {best_name.upper()} with Accuracy: {best_score:.4f}")

    if save_best:
        file_path = f"{best_name}_model.pkl"
        if os.path.exists(file_path):
            os.remove(file_path)
        joblib.dump(best_model, file_path)
        print(f"✅ Best model saved as: {file_path}")

    # PERMUTATION IMPORTANCE
    show_permutation_importance(
        model_pipeline=best_model,
        X_test=x_test,
        y_test=y_test,
        numeric_features=x_train.select_dtypes(include=["int64", "float64"]).columns.tolist(),
        categorical_features=x_train.select_dtypes(include=["object", "category"]).columns.tolist()
    )
    # model_filename = f"{best_name}_model.pkl"
    # joblib.dump(best_model, model_filename)

    return results, best_model,best_name



def evaluate_model(model, x_test, y_test, target_names=None):
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)


    num_classes = len(np.unique(y_test))
    average_method = 'binary' if num_classes == 2 else 'macro'

    prec = precision_score(y_test, y_pred, average=average_method, zero_division=0)
    rec = recall_score(y_test, y_pred, average=average_method, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average_method, zero_division=0)

    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    return acc, prec, rec, f1, report


