import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import RobustScaler

MODEL_PATH = os.path.join('models', 'xgb_model.pkl')

st.set_page_config(page_title='Credit Card Fraud Detector', layout='centered')

st.title('Credit Card Fraud Detection')
st.markdown('Enter transaction features below to get a live fraud probability prediction.')
st.markdown('Dataset (Kaggle): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data')


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            return model
        except Exception:
            return None
    return None


def get_default_features():
    # Provide default zeros for V1..V28 and placeholders for Time/Amount
    features = {f'V{i}': 0.0 for i in range(1, 29)}
    features['Time'] = 0.0
    features['Amount'] = 0.0
    return features


model = load_model()

with st.form('input_form'):
    st.subheader('Transaction inputs')
    col1, col2 = st.columns(2)
    amount = col1.number_input('Amount', min_value=0.0, value=10.0, step=1.0)
    time = col2.number_input('Time (seconds since first transaction)', min_value=0.0, value=0.0, step=1.0)

    with st.expander('Advanced features (V1..V28)'):
        feat_cols = st.columns(4)
        vvals = {}
        for i in range(1, 29):
            idx = (i-1) % 4
            vvals[f'V{i}'] = feat_cols[idx].number_input(f'V{i}', value=0.0, format="%.6f")

    submit = st.form_submit_button('Predict')

if submit:
    features = get_default_features()
    features['Amount'] = amount
    features['Time'] = time
    for k, v in vvals.items():
        features[k] = v

    X = pd.DataFrame([features])

    # Basic scaling for Amount and Time like notebook
    try:
        scaler = RobustScaler()
        X[['Amount']] = scaler.fit_transform(X[['Amount']])
        X[['Time']] = scaler.fit_transform(X[['Time']])
    except Exception:
        pass

    if model is None:
        st.warning('Model not found. Train a model first or run the training script to create models/xgb_model.pkl')
        if st.button('Train a small model now (demo)'):
            if os.path.exists('creditcard.csv'):
                with st.spinner('Training sample model...'):
                    try:
                        from imblearn.over_sampling import SMOTE
                        from xgboost import XGBClassifier
                        from sklearn.model_selection import train_test_split

                        df = pd.read_csv('creditcard.csv')
                        # quick sample to speed up
                        df_sample = df.sample(n=min(20000, len(df)), random_state=42)
                        Xall = df_sample.drop('Class', axis=1)
                        yall = df_sample['Class']
                        Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.2, stratify=yall, random_state=42)
                        sm = SMOTE(sampling_strategy='minority', random_state=42)
                        Xtr_res, ytr_res = sm.fit_resample(Xtr, ytr)
                        clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, random_state=42)
                        clf.fit(Xtr_res, ytr_res)
                        os.makedirs('models', exist_ok=True)
                        joblib.dump(clf, MODEL_PATH)
                        st.success('Model trained and saved to models/xgb_model.pkl')
                        model = clf
                    except Exception as e:
                        st.error(f'Training failed: {e}')
            else:
                st.error('Dataset creditcard.csv not found in project root.')
    else:
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0,1]
            else:
                proba = float(model.predict(X))
            label = 'Fraud' if proba >= 0.5 else 'Not Fraud'
            st.metric('Fraud probability', f'{proba:.4f}')
            st.write('Prediction:', label)
        except Exception as e:
            st.error(f'Prediction failed: {e}')

st.markdown('---')
st.caption('This app is a demo — for production, host the model and apply proper input validation, scaling and security.')
