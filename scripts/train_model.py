import os
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def main():
    if not os.path.exists('creditcard.csv'):
        print('creditcard.csv not found in project root. Place the dataset in the project root and retry.')
        return

    df = pd.read_csv('creditcard.csv')
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    sm = SMOTE(sampling_strategy='minority', random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200, random_state=42)
    clf.fit(X_res, y_res)

    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, os.path.join('models', 'xgb_model.pkl'))
    print('Model trained and saved to models/xgb_model.pkl')

if __name__ == '__main__':
    main()
