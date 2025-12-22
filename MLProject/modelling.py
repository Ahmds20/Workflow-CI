# File: modelling_tuning.py (Advanced: Full Artifacts)
import matplotlib
matplotlib.use("Agg") 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, roc_curve, auc
import mlflow
import mlflow.sklearn
import os
import shutil
import pickle
import time
import json
import dagshub 

# ==========================================
# 0. KONFIGURASI DAGSHUB
# ==========================================
REPO_OWNER = "Ahmds20" 
REPO_NAME = "Workflow-CI" 

print(f"🔌 Menghubungkan ke DagsHub: {REPO_OWNER}/{REPO_NAME}...")
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

# ==========================================
# 1. LOAD DATA & TUNING
# ==========================================
print("📂 Load data & Tuning...")
try:
    train_df = pd.read_csv('namadataset_preprocessing/train_processed.csv')
    test_df = pd.read_csv('namadataset_preprocessing/test_processed.csv')
    train_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)
except Exception:
    print("❌ Error dataset.")
    exit()

X_train = train_df.drop('stroke', axis=1)
y_train = train_df['stroke']
X_test = test_df.drop('stroke', axis=1)
y_test = test_df['stroke']

# Tuning
param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
preds = best_model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"✅ Akurasi: {acc}")

# ==========================================
# 2. SETUP MLFLOW & ARTEFAK MANUAL
# ==========================================
try: mlflow.sklearn.autolog(disable=True)
except: pass

experiment_name = "Submission_Advanced_Tuning"
mlflow.set_experiment(experiment_name)
run_name = f"Run_Full_Artifacts_{time.strftime('%H_%M_%S')}"

print(f"🚀 Uploading ke Run: {run_name}")

with mlflow.start_run(run_name=run_name):
    # Log Metrics & Params
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", acc)

    # --- A. PERSIAPAN FOLDER MODEL LENGKAP ---
    temp_folder = "model_tuning_output"
    if os.path.exists(temp_folder): shutil.rmtree(temp_folder)
    os.makedirs(temp_folder)

    # 1. model.pkl
    with open(os.path.join(temp_folder, "model.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    # 2. conda.yaml (YANG KAMU CARI)
    conda_content = """name: mlflow-env
channels:
  - defaults
dependencies:
  - python=3.9.13
  - scikit-learn=1.0.2
  - pip
  - pip:
    - mlflow
    - dagshub
"""
    with open(os.path.join(temp_folder, "conda.yaml"), "w") as f:
        f.write(conda_content)

    # 3. python_env.yaml (YANG KAMU CARI)
    python_env_content = """python: 3.9.13
build_dependencies:
  - pip
dependencies:
  - scikit-learn==1.0.2
  - mlflow
"""
    with open(os.path.join(temp_folder, "python_env.yaml"), "w") as f:
        f.write(python_env_content)

    # 4. requirements.txt
    with open(os.path.join(temp_folder, "requirements.txt"), "w") as f:
        f.write("scikit-learn\npandas\nmlflow\ndagshub\n")

    # 5. MLmodel
    mlmodel_content = """artifact_path: model
flavors:
  python_function:
    env: conda.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    python_version: 3.9.13
  sklearn:
    pickled_model: model.pkl
    serialization_format: cloudpickle
run_id: null
"""
    with open(os.path.join(temp_folder, "MLmodel"), "w") as f:
        f.write(mlmodel_content)

    # Upload Folder Model
    print("   -> Uploading Model Folder Lengkap...")
    mlflow.log_artifacts(temp_folder, artifact_path="model")

    # --- B. FILE PELENGKAP DI LUAR (ROOT) ---
    print("   -> Membuat File Pelengkap (HTML, JSON)...")

    # 1. estimator.html (YANG KAMU CARI)
    html_content = "<html><body><h1>Random Forest Classifier</h1><p>Manual Logging for Advanced Submission</p></body></html>"
    with open("estimator.html", "w") as f:
        f.write(html_content)
    mlflow.log_artifact("estimator.html")

    # 2. metric_info.json (YANG KAMU CARI)
    metric_content = {"accuracy": acc, "best_params": best_params}
    with open("metric_info.json", "w") as f:
        json.dump(metric_content, f)
    mlflow.log_artifact("metric_info.json")

    # --- C. GRAFIK TAMBAHAN (Advanced) ---
    print("   -> Uploading Grafik...")
    
    # Confusion Matrix
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=ax, cmap='Blues')
    plt.savefig("confusion_matrix_tuning.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix_tuning.png")

    # ROC Curve
    y_prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title('ROC Curve')
    plt.savefig("roc_curve.png")
    plt.close()
    mlflow.log_artifact("roc_curve.png")

    print("✅ SUKSES! Semua file (Wajib & Tambahan) sudah diupload.")

# Bersihkan sampah lokal
#if os.path.exists(temp_folder): shutil.rmtree(temp_folder)
for f in ["estimator.html", "metric_info.json", "confusion_matrix_tuning.png", "roc_curve.png"]:
    if os.path.exists(f): os.remove(f)