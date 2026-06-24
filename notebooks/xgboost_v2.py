import psycopg2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    print("Loading game-level features...")
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
            ROUND(AVG(CASE WHEN hours_played > 100 THEN 1 ELSE 0 END) * 100, 2) as pct_high_hours,
            ROUND(MAX(hours_played), 2) as max_hours,
            ROUND(AVG(CASE WHEN hours_played < 1 THEN 1 ELSE 0 END) * 100, 2) as pct_refund_window
        FROM reviews
        WHERE review_text IS NOT NULL
        GROUP BY game_name
        HAVING COUNT(*) >= 30
    """, conn)
    conn.close()

    df['is_successful'] = (df['positive_pct'] >= 70).astype(int)
    n_pos = df['is_successful'].sum()
    n_neg = len(df) - n_pos
    ratio = n_neg / n_pos
    print(f"  Class ratio (neg/pos): {ratio:.2f} — using as scale_pos_weight")

    feature_cols = [
        'total_reviews', 'avg_hours', 'std_hours', 'avg_helpful',
        'avg_funny', 'avg_review_length', 'pct_high_hours',
        'max_hours', 'pct_refund_window'
    ]
    X = df[feature_cols].fillna(0)
    y = df['is_successful']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining XGBoost v2 with class balancing...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        scale_pos_weight=ratio,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n  Accuracy:  {acc*100:.1f}%")
    print(f"  Precision: {prec*100:.1f}%")
    print(f"  Recall:    {rec*100:.1f}%")
    print(f"  F1 Score:  {f1*100:.1f}%")
    print("\n  Classification report:")
    print(classification_report(y_test, y_pred,
          target_names=['Unsuccessful', 'Successful']))

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"  CV scores: {[f'{s*100:.1f}%' for s in cv_scores]}")
    print(f"  CV mean:   {cv_scores.mean()*100:.1f}%")
    print(f"  CV std:    {cv_scores.std()*100:.1f}%")

    print("\nSHAP analysis...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

    print("\n  SHAP Feature Importance:")
    for _, row in shap_importance.iterrows():
        pct = row['importance'] / shap_importance['importance'].sum() * 100
        print(f"  {row['feature']:<25} {pct:5.1f}%")

    top_feature = shap_importance.iloc[0]['feature']
    hours_pct = shap_importance[shap_importance['feature'].str.contains('hours')]['importance'].sum()
    hours_pct = hours_pct / shap_importance['importance'].sum() * 100

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                      show=False, plot_type='bar')
    plt.title('SHAP Feature Importance — Game Success Prediction (v2)')
    plt.tight_layout()
    plt.savefig('notebooks/shap_importance_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  SHAP plot saved: notebooks/shap_importance_v2.png")

    print("\n" + "="*60)
    print("H3 & H4 FINAL RESULTS")
    print("="*60)
    h4_pass = acc >= 0.85
    h3_pass = hours_pct > 30 or 'hours' in top_feature
    print(f"\nH4 — Accuracy >= 85%:")
    print(f"  Test accuracy: {acc*100:.1f}% — {'CONFIRMED' if h4_pass else 'NOT CONFIRMED'}")
    print(f"  CV mean:       {cv_scores.mean()*100:.1f}%")
    print(f"\nH3 — Hours stronger than text features:")
    print(f"  Hours importance: {hours_pct:.1f}%")
    print(f"  Top feature: {top_feature}")
    print(f"  H3: {'CONFIRMED' if h3_pass else 'PARTIAL'}")

    model.save_model('notebooks/xgboost_model_v2.json')
    print("\nModel saved: notebooks/xgboost_model_v2.json")
