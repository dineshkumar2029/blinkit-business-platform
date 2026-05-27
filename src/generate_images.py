import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import pickle
from sklearn.metrics import roc_curve, auc

def generate_assets():
    print("Generating visual assets for the GitHub README...")
    
    # Establish paths
    workspace_dir = "d:\\DineshProjects\\blinkit-business-platform"
    images_dir = os.path.join(workspace_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Database connection
    db_name = "blinkit"
    user = "postgres"
    password = "postgres"
    host = "localhost"
    port = 5432
    
    try:
        engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db_name}")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return

    # Style configuration
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'figure.facecolor': '#111217',
        'axes.facecolor': '#1a1c23',
        'axes.edgecolor': '#2d313f',
        'axes.labelcolor': '#f0f2f6',
        'xtick.color': '#a3a8b4',
        'ytick.color': '#a3a8b4',
        'text.color': '#f0f2f6',
        'grid.color': '#2d313f',
        'grid.linestyle': '--',
        'grid.alpha': 0.5
    })
    
    # ----------------------------------------------------
    # Asset 1: Marketing ROI & Ad Spend vs Revenue Dual-Axis Chart
    # ----------------------------------------------------
    print("Generating Asset 1: Marketing spend vs. Revenue dual-axis chart...")
    try:
        # Load marketing and sales daily view
        query = """
        WITH Daily_Sales AS (
            SELECT CAST(order_date AS DATE) AS order_day, SUM(order_total) AS total_revenue
            FROM orders GROUP BY CAST(order_date AS DATE)
        ),
        Daily_Marketing AS (
            SELECT date AS marketing_day, SUM(spend) AS total_spend
            FROM marketing_performance GROUP BY date
        )
        SELECT 
            COALESCE(m.marketing_day, s.order_day) AS date,
            COALESCE(s.total_revenue, 0.0) AS revenue,
            COALESCE(m.total_spend, 0.0) AS spend
        FROM Daily_Marketing m
        FULL OUTER JOIN Daily_Sales s ON m.marketing_day = s.order_day
        ORDER BY date ASC;
        """
        df_roi = pd.read_sql(query, engine)
        df_roi['date'] = pd.to_datetime(df_roi['date'])
        
        # Focus on the anomaly period: Oct 5 - Oct 25, 2024 to clearly highlight the Instagram campaign failure!
        df_focus = df_roi[(df_roi['date'] >= '2024-10-01') & (df_roi['date'] <= '2024-10-31')].copy()
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot Spend as bars (Right Axis)
        ax2 = ax1.twinx()
        bars = ax2.bar(df_focus['date'], df_focus['spend'], color='#ff4d4d', alpha=0.6, width=0.6, label='Ad Spend (INR)')
        
        # Plot Revenue as line (Left Axis)
        line, = ax1.plot(df_focus['date'], df_focus['revenue'], color='#2ecc71', linewidth=3, marker='o', markersize=6, label='Revenue (INR)')
        
        ax1.set_xlabel('Date (October 2024)', color='#f0f2f6', fontsize=12, labelpad=10)
        ax1.set_ylabel('Daily Revenue (INR)', color='#2ecc71', fontsize=12)
        ax2.set_ylabel('Daily Ad Spend (INR)', color='#ff4d4d', fontsize=12)
        
        # Format axes
        ax1.tick_params(axis='y', labelcolor='#2ecc71')
        ax2.tick_params(axis='y', labelcolor='#ff4d4d')
        ax1.grid(True)
        
        # Title and highlighting the anomaly
        plt.title('Daily Marketing Spend vs. Revenue (Highlighting Instagram Campaign Failure)', color='#f0f2f6', fontsize=14, pad=20, weight='bold')
        
        # Highlight anomaly week: Oct 10 to Oct 17
        ax1.axvspan(pd.to_datetime('2024-10-10'), pd.to_datetime('2024-10-17'), color='#e74c3c', alpha=0.15, label='Failing Instagram Campaign')
        
        # Add legend
        lines = [line, bars]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', facecolor='#1a1c23', edgecolor='#2d313f')
        
        # Rotate dates
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        
        plt.savefig(os.path.join(images_dir, "marketing_roi_dual_axis.png"), dpi=150, facecolor='#111217')
        plt.close()
        print("Asset 1 generated successfully.")
    except Exception as e:
        print(f"Failed to generate Asset 1: {e}")

    # ----------------------------------------------------
    # Asset 2: ML Model ROC Curve
    # ----------------------------------------------------
    print("Generating Asset 2: Machine Learning model ROC curve...")
    try:
        # Load the saved model
        model_path = os.path.join(workspace_dir, "src", "model.pkl")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        clf = model_data['model']
        feature_cols = model_data['features']
        
        # Load test data directly from the DB to re-evaluate
        df_orders = pd.read_sql("SELECT order_date, promised_time, actual_time, region FROM orders", engine)
        df_orders['order_date'] = pd.to_datetime(df_orders['order_date'])
        df_orders['promised_time'] = pd.to_datetime(df_orders['promised_time'])
        df_orders['actual_time'] = pd.to_datetime(df_orders['actual_time'])
        
        df_orders['Is_Late'] = (df_orders['actual_time'] > df_orders['promised_time']).astype(int)
        df_orders['Hour_of_Day'] = df_orders['order_date'].dt.hour
        df_orders['Day_of_Week'] = df_orders['order_date'].dt.weekday
        df_orders['Is_Weekend'] = (df_orders['Day_of_Week'] >= 5).astype(int)
        
        df_encoded = pd.get_dummies(df_orders, columns=['region'], prefix='region')
        for col in feature_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
                
        X = df_encoded[feature_cols].astype(float)
        y = df_encoded['Is_Late']
        
        # Use simple split to get ROC curve
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        y_prob = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#00d2ff', linewidth=3, label=f'Random Forest Model (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='#e74c3c', linewidth=2, linestyle='--', label='Random Guessing (AUC = 0.5000)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', color='#f0f2f6', fontsize=12, labelpad=10)
        plt.ylabel('True Positive Rate (Sensitivity / Recall)', color='#f0f2f6', fontsize=12, labelpad=10)
        plt.title('Delivery Delay Classifier - ROC Curve', color='#f0f2f6', fontsize=14, pad=20, weight='bold')
        plt.grid(True)
        plt.legend(loc="lower right", facecolor='#1a1c23', edgecolor='#2d313f')
        
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, "roc_curve.png"), dpi=150, facecolor='#111217')
        plt.close()
        print("Asset 2 generated successfully.")
    except Exception as e:
        print(f"Failed to generate Asset 2: {e}")

    # ----------------------------------------------------
    # Asset 3: Regional Delay Heatmap
    # ----------------------------------------------------
    print("Generating Asset 3: Regional delay heatmap...")
    try:
        df_delays = pd.read_sql("""
            SELECT 
                region, 
                EXTRACT(HOUR FROM order_date) AS hour_of_day,
                AVG(CASE WHEN actual_time > promised_time THEN 1.0 ELSE 0.0 END) * 100 AS delay_rate
            FROM orders
            GROUP BY region, hour_of_day;
        """, engine)
        
        # Pivot the table for the heatmap
        pivot_df = df_delays.pivot(index='region', columns='hour_of_day', values='delay_rate')
        
        # Set up plot
        plt.figure(figsize=(12, 6))
        
        # Custom sleek color palette
        cmap = sns.dark_palette("#00d2ff", as_cmap=True)
        
        sns.heatmap(pivot_df, cmap='coolwarm', annot=False, cbar_kws={'label': 'Delay Probability (%)'})
        
        plt.title('Heatmap of Delivery Delay Probability by Region and Hour', color='#f0f2f6', fontsize=14, pad=20, weight='bold')
        plt.xlabel('Hour of Day (0 - 23)', color='#f0f2f6', fontsize=12, labelpad=10)
        plt.ylabel('Delivery Region', color='#f0f2f6', fontsize=12, labelpad=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, "regional_delays.png"), dpi=150, facecolor='#111217')
        plt.close()
        print("Asset 3 generated successfully.")
    except Exception as e:
        print(f"Failed to generate Asset 3: {e}")

    # ----------------------------------------------------
    # Asset 4: Sentiment & Ratings Breakdown
    # ----------------------------------------------------
    print("Generating Asset 4: Sentiment and ratings breakdown...")
    try:
        df_feedback = pd.read_sql("SELECT rating, sentiment FROM customer_feedback", engine)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Rating Distribution (Bar Chart)
        rating_counts = df_feedback['rating'].value_counts().sort_index()
        colors = ['#ff4d4d', '#ff944d', '#ffd14d', '#4dffd1', '#2ecc71']
        
        ax1.bar(rating_counts.index.astype(str), rating_counts.values, color=colors, edgecolor='#2d313f', width=0.6)
        ax1.set_title('Customer Feedback Ratings Distribution', color='#f0f2f6', fontsize=13, weight='bold', pad=15)
        ax1.set_xlabel('Rating Score (1 - 5 Stars)', color='#f0f2f6', labelpad=10)
        ax1.set_ylabel('Total Reviews count', color='#f0f2f6', labelpad=10)
        ax1.grid(True)
        
        # 2. Sentiment Distribution (Pie Chart)
        sentiment_counts = df_feedback['sentiment'].value_counts()
        sent_colors = ['#f0f2f6', '#ff4d4d', '#2ecc71'] # Neutral, Negative, Positive
        
        # Reorder to match colors logically if needed
        ordered_sent = []
        ordered_colors = []
        for s in ['Neutral', 'Negative', 'Positive']:
            if s in sentiment_counts:
                ordered_sent.append(s)
                if s == 'Neutral': ordered_colors.append('#34495e')
                elif s == 'Negative': ordered_colors.append('#e74c3c')
                elif s == 'Positive': ordered_colors.append('#2ecc71')
                
        counts = [sentiment_counts[s] for s in ordered_sent]
        
        wedges, texts, autotexts = ax2.pie(
            counts, 
            labels=ordered_sent, 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=ordered_colors,
            textprops=dict(color="#f0f2f6"),
            wedgeprops=dict(edgecolor='#2d313f', linewidth=1)
        )
        
        # Color percentage texts inside pie slices
        for autotext in autotexts:
            autotext.set_color('#ffffff')
            autotext.set_weight('bold')
            
        ax2.set_title('Feedback Sentiment Breakdown', color='#f0f2f6', fontsize=13, weight='bold', pad=15)
        
        plt.suptitle('Blinkit Customer Feedback Analysis', color='#f0f2f6', fontsize=16, weight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, "feedback_sentiments.png"), dpi=150, facecolor='#111217')
        plt.close()
        print("Asset 4 generated successfully.")
    except Exception as e:
        print(f"Failed to generate Asset 4: {e}")

    print("All README visual assets generated successfully!")

if __name__ == "__main__":
    generate_assets()
