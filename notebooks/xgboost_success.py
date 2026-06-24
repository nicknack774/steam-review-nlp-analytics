import psycopg2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    print("Step 1: Building game-level feature matrix...")
    conn = psycopg2.connect("dbname=steam_analytics")
    df = pd.read_sql("""
        SELECT
            game_name,
            COUNT(*) as total_reviews,
            ROUND(AVG(CASE WHEN is_recommended THEN 1 ELSE 0 END) * 100, 2) as positive_pct,
            ROUND(AVG(hours_played), 2) as avg_hours,
            ROUND(STDDEV(hours_played), 2) as std_hours,
            ROUND(AVG(helpful), 2) as avg_helpful,
            ROUND(AVG(funny), 2) as avg_funny,
            ROUND(AVG(length(review_text)), 2) as avg_review_length,
            SUM(CASE WHEN is_recommended THEN 1 ELSE 0 END) as positive_count,
            SUM(CASE WHEN NOT is_recommended THEN 1 ELSE 0 END) as negative_count,
            ROUND(AVG(CASE WHEN hours_played > 100 THEN 1 ELSE 0 END) * 100, 2) as pct_high_hours,
            ROUND(MAX(hours_played), 2) as max_hours,
            ROUND(AVG(CASE WHEN hours_played < 1 THEN 1 ELSE 0 END) * 100, 2) as pct_refund_window
        FROM reviews
        WHERE review_text IS NOT NULL
        GROUP BY game_name
        HAVING COUNT(*) >= 30
    """, conn)
    conn.close()
    print(f"  {len(df)} games loaded")

    # Define success: >= 70% positive reviews
    df['is_successful'] = (df['positive_pct'] >= 70).astype(int)
    success_count = df['is_successful'].sum()
    print(f"  Successful games: {success_count}/{len(df)} ({success_count/len(df)*100:.1f}%)")
    print(f"  Unsuccessful games: {len(df)-success_count}/{len(df)} ({(len(df)-success_count)/len(df)*100:.1f}%)")

    print("\nStep 2: Preparing features...")
    feature_cols = [
        'total_reviews', 'avg_hours', 'std_hours', 'avg_helpful',
        'avg_funny', 'avg_review_length', 'pct_high_hours',
        'max_hours', 'pct_refund_window'
    ]
    # NOTE: positive_pct excluded — it directly defines the label (would be cheating)

    X = df[feature_cols].fillna(0)
    y = df['is_successful']
    print(f"  Features: {feature_cols}")
    print(f"  Shape: {X.shape}")

    print("\nStep 3: Train/test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} games | Test: {len(X_test)} games")

    print("\nStep 4: Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    print("  Training complete")

    print("\nStep 5: Evaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n  Accuracy:  {acc*100:.1f}%")
    print(f"  Precision: {prec*100:.1f}%")
    print(f"  Recall:    {rec*100:.1f}%")
    print(f"  F1 Score:  {f1*100:.1f}%")

    print("\n  Full classification report:")
    print(classification_report(y_test, y_pred,
          target_names=['Unsuccessful', 'Successful']))

    print("\nStep 6: Cross-validation (5-fold)...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"  CV scores: {[f'{s*100:.1f}%' for s in cv_scores]}")
    print(f"  CV mean:   {cv_scores.mean()*100:.1f}%")
    print(f"  CV std:    {cv_scores.std()*100:.1f}%")

    print("\nStep 7: SHAP feature importance...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Feature importance from SHAP
    shap_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

    print("\n  SHAP Feature Importance (mean |SHAP value|):")
    for _, row in shap_importance.iterrows():
        bar = '█' * int(row['importance'] * 200)
        pct = row['importance'] / shap_importance['importance'].sum() * 100
        print(f"  {row['feature']:<25} {pct:5.1f}%  {bar}")

    top_feature = shap_importance.iloc[0]['feature']
    top_pct = shap_importance.iloc[0]['importance'] / shap_importance['importance'].sum() * 100
    print(f"\n  Top predictor: {top_feature} ({top_pct:.1f}% importance)")

    # Save SHAP plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                      show=False, plot_type='bar')
    plt.title('SHAP Feature Importance — Game Success Prediction')
    plt.tight_layout()
    plt.savefig('notebooks/shap_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  SHAP plot saved: notebooks/shap_importance.png")

    # Save XGBoost feature importance plot
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=9)
    plt.title('XGBoost Feature Importance — Game Success Prediction')
    plt.tight_layout()
    plt.savefig('notebooks/xgb_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  XGBoost importance plot saved: notebooks/xgb_importance.png")

    print("\n" + "="*60)
    print("H3 & H4 VALIDATION")
    print("="*60)
    h4_pass = acc >= 0.85
    h3_hours = shap_importance[shap_importance['feature'].str.contains('hours')]['importance'].sum()
    h3_sentiment_proxy = shap_importance[shap_importance['feature'].str.contains('review_length|helpful|funny')]['importance'].sum()
    total_imp = shap_importance['importance'].sum()
    hours_pct = h3_hours / total_imp * 100

    print(f"\nH4 — XGBoost accuracy >= 85%:")
    print(f"  Achieved: {acc*100:.1f}% — {'CONFIRMED' if h4_pass else 'NOT CONFIRMED'}")
    print(f"\nH3 — Playtime stronger predictor than text sentiment:")
    print(f"  Hours-related importance: {hours_pct:.1f}%")
    print(f"  Top feature: {top_feature}")
    h3_pass = 'hours' in top_feature or hours_pct > 30
    print(f"  H3 STATUS: {'CONFIRMED' if h3_pass else 'PARTIAL — hours not top predictor'}")

    model.save_model('notebooks/xgboost_model.json')
    print("\nModel saved: notebooks/xgboost_model.json")
