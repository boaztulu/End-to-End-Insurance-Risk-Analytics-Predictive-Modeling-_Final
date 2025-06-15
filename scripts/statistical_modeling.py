import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer, RobustScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, TweedieRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
)
import shap


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'TransactionMonth' in df:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
        df['TransYear']     = df['TransactionMonth'].dt.year
        df['TransMonthNum'] = df['TransactionMonth'].dt.month
        df.drop(columns=['TransactionMonth'], inplace=True)
    if {'TotalPremium', 'PolicyTerm'} <= set(df.columns):
        df['PremiumPerTerm'] = (
            df['TotalPremium']
              .div(df['PolicyTerm'].replace(0, np.nan))
        ).fillna(df['TotalPremium'].median())
    if {'TotalClaims', 'NumberOfClaims'} <= set(df.columns):
        df['AvgClaimSeverity'] = (
            df['TotalClaims']
              .div(df['NumberOfClaims'].replace(0, np.nan))
        ).fillna(0)
    if 'VehicleYear' in df:
        df['VehicleAge'] = (2025 - df['VehicleYear']).clip(0, 60)
    return df


def preprocess_features(df: pd.DataFrame,
                        target_reg: str,
                        target_clf: str = None):
    """
    Returns:
      X            : feature DataFrame
      y_reg        : regression target Series
      y_clf        : classification target Series (or None)
      preprocessor : ColumnTransformer
    """
    df = engineer_features(df.copy())
    y_reg = df[target_reg]
    y_clf = None
    if target_clf:
        df[target_clf] = (df[target_reg] > 0).astype(int)
        y_clf = df[target_clf]

    drop_cols = [target_reg, 'UnderwrittenCoverID', 'PolicyID']
    if target_clf:
        drop_cols.append(target_clf)
    X = df.drop(columns=[c for c in drop_cols if c in df], errors='ignore')

    # clean & freq-encode categoricals
    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype(str).str.strip().replace({'': np.nan, 'nan': np.nan})
    high_card = [c for c in cat_cols if X[c].nunique() > 30]
    for c in high_card:
        freq = X[c].value_counts(normalize=True)
        X[f'{c}_freq'] = X[c].map(freq).fillna(0)
    X.drop(columns=high_card, inplace=True)
    cat_final = [c for c in cat_cols if c not in high_card]

    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()

    num_pipe = Pipeline([
        ('clip',  FunctionTransformer(lambda A: np.clip(A, -1e6, 1e6))),
        ('scale', RobustScaler())
    ])
    cat_pipe = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_final)
    ], remainder='drop')

    return X, y_reg, y_clf, preprocessor


def split_data(X, y, test_size=0.3, random_state=42):
    """Simple train/test split."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_regression_models(X_train, y_train, preprocessor):
    """
    Trains Linear, RandomForest, XGBoost regressors on log1p(y).
    Returns a dict of fitted TransformedTargetRegressor pipelines.
    """
    base_models = {
        'LinearRegression': LinearRegression(),
        'RandomForest':     RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost':          XGBRegressor(n_estimators=100,
                                         random_state=42,
                                         objective='reg:squarederror',
                                         eval_metric='rmse')
    }
    models = {}
    for name, est in base_models.items():
        pipe = Pipeline([('prep', preprocessor), ('model', est)])
        wrapped = TransformedTargetRegressor(
            regressor=pipe,
            func=np.log1p,
            inverse_func=np.expm1
        )
        wrapped.fit(X_train, y_train)
        models[name] = wrapped
    return models


def evaluate_regression(models: dict, X_test, y_test) -> pd.DataFrame:
    """Returns DataFrame with RMSE and R² for each regression model."""
    records = []
    for name, pipe in models.items():
        preds = pipe.predict(X_test)
        records.append({
            'Model': name,
            'RMSE':  np.sqrt(mean_squared_error(y_test, preds)),
            'R2':    r2_score(y_test, preds)
        })
    return pd.DataFrame(records).set_index('Model')


def train_classification_models(X_train, y_train, preprocessor):
    """
    Trains LogisticRegression, RandomForest, XGBoost classifiers.
    Returns dict of fitted pipelines.
    """
    neg, pos = np.bincount(y_train)
    scale_pos = neg / pos if pos > 0 else 1.0

    base_models = {
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'RandomForest':       RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
        'XGBoost':            XGBClassifier(n_estimators=100,
                                            random_state=42,
                                            eval_metric='logloss',
                                            use_label_encoder=False,
                                            scale_pos_weight=scale_pos)
    }
    models = {}
    for name, est in base_models.items():
        pipe = Pipeline([('prep', preprocessor), ('model', est)])
        pipe.fit(X_train, y_train)
        models[name] = pipe
    return models


def evaluate_classification(models: dict, X_test, y_test, threshold=0.5) -> pd.DataFrame:
    """Returns DataFrame with Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC."""
    records = []
    for name, pipe in models.items():
        probs = pipe.predict_proba(X_test)[:, 1]
        preds = (probs > threshold).astype(int)
        records.append({
            'Model':     name,
            'Accuracy':  accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds, zero_division=0),
            'Recall':    recall_score(y_test, preds),
            'F1':        f1_score(y_test, preds),
            'ROC-AUC':   roc_auc_score(y_test, probs),
            'PR-AUC':    average_precision_score(y_test, probs)
        })
    return pd.DataFrame(records).set_index('Model')


def compute_risk_based_premium(pipe_clf, pipe_reg, X: pd.DataFrame) -> np.ndarray:
    """
    Two-part prediction: P(claim) × E[severity].
    """
    probs = pipe_clf.predict_proba(X)[:, 1]
    preds = pipe_reg.predict(X)
    return probs * preds


def compute_shap_importance(model_pipe, X_sample: pd.DataFrame) -> pd.DataFrame:
    """
    Returns mean absolute SHAP values per feature.
    Uses TreeExplainer for tree models, LinearExplainer for linear,
    and KernelExplainer as a fallback.
    """
    # unwrap if wrapped in TransformedTargetRegressor
    pipe = getattr(model_pipe, 'regressor_', model_pipe)
    prep = pipe.named_steps['prep']
    # get the underlying estimator
    mod = pipe.named_steps.get('model') or pipe.named_steps.get('clf')

    # transform once
    Xp = prep.transform(X_sample)
    if hasattr(Xp, 'toarray'):
        Xp = Xp.toarray()

    # choose explainer
    if isinstance(mod, (RandomForestRegressor, RandomForestClassifier,
                        XGBRegressor, XGBClassifier)):
        explainer = shap.TreeExplainer(mod)
        shap_vals = explainer.shap_values(Xp)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
    elif isinstance(mod, (LinearRegression, LogisticRegression)):
        explainer = shap.LinearExplainer(
            mod,
            Xp,
            feature_perturbation="interventional"
        )
        shap_vals = explainer.shap_values(Xp)
    else:
        explainer = shap.KernelExplainer(mod.predict, shap.sample(Xp, 100))
        shap_vals = explainer.shap_values(Xp)

    # assemble feature names
    num_feats = prep.transformers_[0][2]
    cat_pipe  = prep.transformers_[1][1]
    cat_cols  = prep.transformers_[1][2]
    cat_feats = cat_pipe.named_steps['ohe'].get_feature_names_out(cat_cols)
    features  = np.concatenate([num_feats, cat_feats])

    # build and return DataFrame
    return (
        pd.DataFrame({
            'feature':       features,
            'mean_abs_shap': np.abs(shap_vals).mean(axis=0)
        })
        .sort_values('mean_abs_shap', ascending=False)
        .reset_index(drop=True)
    )


def train_two_part_model(X_train, y_clf_train, y_reg_train, preprocessor,
                         clf_model=None, reg_model=None):
    """
    Trains:
      - a classifier on all rows (y_clf_train)
      - a log-transformed regressor on positive-only rows (y_reg_train>0)
    Returns (pipe_clf, pipe_reg).
    """
    if clf_model is None:
        clf_model = RandomForestClassifier(
            n_estimators=200, class_weight='balanced', random_state=42
        )
    pipe_clf = Pipeline([('prep', preprocessor), ('clf', clf_model)])
    pipe_clf.fit(X_train, y_clf_train)

    mask = y_reg_train > 0
    X_pos = X_train[mask]
    y_pos = y_reg_train[mask]

    if reg_model is None:
        reg_model = RandomForestRegressor(n_estimators=200, random_state=42)
    pipe_reg = Pipeline([('prep', preprocessor), ('model', reg_model)])
    wrapped_reg = TransformedTargetRegressor(
        regressor=pipe_reg, func=np.log1p, inverse_func=np.expm1
    )
    wrapped_reg.fit(X_pos, y_pos)

    return pipe_clf, wrapped_reg


def evaluate_two_part_model(pipe_clf, pipe_reg,
                            X_test, y_clf_test, y_reg_test,
                            threshold=0.5):
    """
    Returns dict with:
      - classification metrics
      - severity RMSE on positives
      - combined RMSE & R² on full set
    """
    # classification
    probs = pipe_clf.predict_proba(X_test)[:, 1]
    y_pred_clf = (probs > threshold).astype(int)
    clf_metrics = {
        'Accuracy':  accuracy_score(y_clf_test, y_pred_clf),
        'Precision': precision_score(y_clf_test, y_pred_clf, zero_division=0),
        'Recall':    recall_score(y_clf_test, y_pred_clf),
        'F1':        f1_score(y_clf_test, y_pred_clf),
        'ROC-AUC':   roc_auc_score(y_clf_test, probs),
        'PR-AUC':    average_precision_score(y_clf_test, probs)
    }

    # severity on positives
    mask_pos = y_reg_test > 0
    sev_rmse = mean_squared_error(
        y_reg_test[mask_pos],
        pipe_reg.predict(X_test[mask_pos]),
        squared=False
    )

    # combined
    combined = compute_risk_based_premium(pipe_clf, pipe_reg, X_test)
    full_rmse = mean_squared_error(y_reg_test, combined, squared=False)
    full_r2   = r2_score(y_reg_test, combined)

    return {
        'classification':    clf_metrics,
        'severity_rmse_pos': sev_rmse,
        'combined_rmse_full': full_rmse,
        'combined_r2_full':   full_r2
    }


def train_tweedie_model(X_train, y_train, preprocessor,
                        power=1.5, alpha=0.5, max_iter=100):
    """
    Trains a single TweedieRegressor pipeline (handles zeros + heavy tails).
    """
    pipe = Pipeline([
        ('prep',    preprocessor),
        ('tweedie', TweedieRegressor(power=power, alpha=alpha, max_iter=max_iter))
    ])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_tweedie_model(pipe, X_test, y_test):
    preds = pipe.predict(X_test)
    return {
        'RMSE': mean_squared_error(y_test, preds, squared=False),
        'R2':   r2_score(y_test, preds)
    }
