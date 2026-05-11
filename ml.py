import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

df = pd.read_csv(r"fraud.csv")

X = df.drop(columns=['Class', 'Time'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)


num = X.select_dtypes(include='number').columns.tolist()

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

process = ColumnTransformer([
    ('num', num_pipeline, num)
])


scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])


base_model = XGBClassifier(
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    use_label_encoder=False
)

pipeline = Pipeline([
    ('process', process),
    ('xgbm', base_model)
])

params = {
    'xgbm__learning_rate': [0.03, 0.05],
    'xgbm__n_estimators': [300, 500],
    'xgbm__max_depth': [3, 5],
    'xgbm__subsample': [0.7, 0.9],
    'xgbm__colsample_bytree': [0.7, 0.9]
}

search = RandomizedSearchCV(
    pipeline,
    params,
    cv=3,
    scoring='roc_auc',
    n_iter=8,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

search.fit(X_train, y_train)

best_model = search.best_estimator_


calibrated_model = CalibratedClassifierCV(
    best_model,
    method='sigmoid',
    cv=3
)

calibrated_model.fit(X_train, y_train)


probs = calibrated_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, probs)

print("ROC AUC:", roc_auc)

joblib.dump(calibrated_model, "model_v2.pkl")
joblib.dump(num,"features.pkl")

print('calibrateed_model min:',probs.min())
print("max:",probs.max())
print("mean:",probs.mean())

from sklearn.metrics import precision_recall_curve
import numpy as np

probs = best_model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, probs)

# Print some key threshold values
for i in range(0, len(thresholds), len(thresholds)//10):
    print(f"Threshold={thresholds[i]:.3f}, Precision={precision[i]:.3f}, Recall={recall[i]:.3f}")


print('best_model min:',probs.min())
print("max:",probs.max())
print("mean:",probs.mean())    
