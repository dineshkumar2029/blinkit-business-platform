import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def train_and_save_model():
    print("Connecting to PostgreSQL database to load orders data...")
    db_name = "blinkit"
    user = "postgres"
    password = "postgres"
    host = "localhost"
    port = 5432
    
    try:
        engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db_name}")
        df = pd.read_sql("SELECT order_date, promised_time, actual_time, region FROM orders", engine)
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['promised_time'] = pd.to_datetime(df['promised_time'])
        df['actual_time'] = pd.to_datetime(df['actual_time'])
        print(f"Successfully loaded {len(df)} orders from PostgreSQL.")
    except Exception as e:
        print(f"Error loading from PostgreSQL: {e}")
        print("Falling back to local CSV file...")
        df = pd.read_csv("d:\\DineshProjects\\blinkit-business-platform\\blinkit_orders.csv")
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['promised_time'] = pd.to_datetime(df['promised_time'])
        df['actual_time'] = pd.to_datetime(df['actual_time'])
        print(f"Loaded {len(df)} orders from CSV.")

    # 1. Feature Engineering
    print("Performing feature engineering...")
    
    # Target variable: Is_Late (1 if actual_time > promised_time, else 0)
    df['Is_Late'] = (df['actual_time'] > df['promised_time']).astype(int)
    
    # Feature columns extraction
    df['Hour_of_Day'] = df['order_date'].dt.hour
    df['Day_of_Week'] = df['order_date'].dt.weekday # 0 = Monday, 6 = Sunday
    df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)
    
    # Categorical feature: Region
    regions_list = sorted(df['region'].unique().tolist())
    print("Unique Regions in data:", regions_list)
    
    # One-hot encode regions
    df_encoded = pd.get_dummies(df, columns=['region'], prefix='region')
    
    # Select feature columns
    feature_cols = ['Hour_of_Day', 'Day_of_Week', 'Is_Weekend'] + [f'region_{r}' for r in regions_list]
    
    # Fill missing dummy columns if any
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    X = df_encoded[feature_cols].astype(float)
    y = df_encoded['Is_Late']
    
    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    print(f"Class distribution: Train (late rate: {y_train.mean():.2%}), Test (late rate: {y_test.mean():.2%})")
    
    # 3. Model Training
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # 4. Evaluation
    print("\nEvaluating model performance on test set...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    print(f"--- ROC-AUC Score: {auc:.4f} ---")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Print feature importances
    importances = clf.feature_importances_
    print("\nFeature Importances:")
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"- {col}: {imp:.4%}")
        
    # Check if target AUC is achieved
    if auc >= 0.80:
        print(f"\n[OK] SUCCESS: Model achieved AUC of {auc:.4f}, which is above the 0.80 requirement!")
    else:
        print(f"\n[WARN] WARNING: Model AUC of {auc:.4f} is below the 0.80 requirement. Retraining may be needed.")
        
    # 5. Save the trained model and feature structure
    model_data = {
        'model': clf,
        'features': feature_cols,
        'regions': regions_list,
        'auc': auc
    }
    
    model_path = "d:\\DineshProjects\\blinkit-business-platform\\src\\model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"Successfully serialized and saved model metadata to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
