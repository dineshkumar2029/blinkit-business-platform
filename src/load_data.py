import os
import random
import datetime
import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine, text

def get_delay_prob(region, hour, day_of_week):
    # Highly modelable delay probability based on features
    # Indiranagar peak delay (Friday/Saturday evening)
    if region == "Indiranagar":
        if hour in [17, 18, 19, 20] and day_of_week in [4, 5]:
            return 0.94
        elif hour in [17, 18, 19, 20]:
            return 0.50
        else:
            return 0.02
    # Whitefield peak delay (commute hours)
    elif region == "Whitefield":
        if day_of_week in [0, 1, 2, 3, 4] and hour in [8, 9, 10, 17, 18, 19]:
            return 0.90
        elif hour in [8, 9, 10, 17, 18, 19]:
            return 0.40
        else:
            return 0.02
    # Koramangala peak delay (Weekend night)
    elif region == "Koramangala":
        if day_of_week in [4, 5, 6] and hour in [20, 21, 22, 23]:
            return 0.85
        elif hour in [20, 21, 22, 23]:
            return 0.40
        else:
            return 0.02
    # HSR Layout peak delay (evening)
    elif region == "HSR Layout":
        if hour in [18, 19, 20, 21]:
            return 0.70
        else:
            return 0.02
    # General baseline delay for other regions
    else:
        if hour in [18, 19, 20, 21]:
            return 0.50
        else:
            return 0.01

def generate_datasets():
    print("Generating synthetic datasets...")
    
    # 1. Load customer feedback
    feedback_path = "d:\\DineshProjects\\blinkit-business-platform\\blinkit_customer_feedback.csv"
    if not os.path.exists(feedback_path):
        raise FileNotFoundError(f"Feedback file not found at {feedback_path}")
        
    df_feedback = pd.read_csv(feedback_path)
    df_feedback['feedback_date'] = pd.to_datetime(df_feedback['feedback_date'])
    
    # 2. Define parameters
    min_date = df_feedback['feedback_date'].min()
    max_date = df_feedback['feedback_date'].max()
    date_range = pd.date_range(start=min_date, end=max_date)
    
    regions = ["Indiranagar", "Koramangala", "Whitefield", "HSR Layout", "Jayanagar", "Bellandur", "Marathahalli"]
    channels = ["SMS", "Email", "Facebook", "App Notification", "Google Ads", "Instagram"]
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # 3. Generate 20,000 Orders Data (strictly using the modelable delay rules)
    print("Generating 20,000 orders data using strict feature-based delay rules...")
    orders_records = []
    
    for i in range(20000):
        order_id = 1000000000 + i
        customer_id = random.randint(10000, 99999)
        
        # Pick a random date in range
        rand_date = random.choice(date_range)
        order_hour = random.randint(8, 22)
        order_minute = random.randint(0, 59)
        order_second = random.randint(0, 59)
        order_time = rand_date.replace(hour=order_hour, minute=order_minute, second=order_second)
        
        order_total = round(random.uniform(100.0, 2500.0), 2)
        region = random.choice(regions)
        
        promise_minutes = random.choice([15, 20, 25, 30, 40])
        promised_time = order_time + datetime.timedelta(minutes=promise_minutes)
        
        day_of_week = order_time.weekday()
        hour = order_time.hour
        
        delay_prob = get_delay_prob(region, hour, day_of_week)
        is_delayed = random.random() < delay_prob
        
        if is_delayed:
            delay_minutes = random.uniform(5.0, 45.0)
            actual_time = promised_time + datetime.timedelta(minutes=delay_minutes)
        else:
            actual_minutes = random.uniform(5.0, promise_minutes - 1.0)
            actual_time = order_time + datetime.timedelta(minutes=actual_minutes)
            
        orders_records.append({
            'order_id': order_id,
            'customer_id': customer_id,
            'order_date': order_time,
            'order_total': order_total,
            'promised_time': promised_time,
            'actual_time': actual_time,
            'region': region
        })
        
    df_orders = pd.DataFrame(orders_records)
    df_orders.to_csv("d:\\DineshProjects\\blinkit-business-platform\\blinkit_orders.csv", index=False)
    print(f"Saved 20,000 orders to blinkit_orders.csv. Target delay rules implemented.")
    
    # 4. Map Customer Feedback to Orders (so the business logic joins perfectly)
    print("Mapping customer feedback ratings to order delivery status...")
    
    # Delayed orders (late) and On-time orders
    late_orders = df_orders[df_orders['actual_time'] > df_orders['promised_time']].copy()
    ontime_orders = df_orders[df_orders['actual_time'] <= df_orders['promised_time']].copy()
    
    # Shuffling to distribute randomly
    late_orders = late_orders.sample(frac=1, random_state=42).reset_index(drop=True)
    ontime_orders = ontime_orders.sample(frac=1, random_state=42).reset_index(drop=True)
    
    late_idx = 0
    ontime_idx = 0
    
    mapped_feedback = []
    
    for idx, row in df_feedback.iterrows():
        rating = int(row['rating'])
        
        # Rating 1 or 2: Map to a late order
        if rating in [1, 2] and late_idx < len(late_orders):
            selected_order = late_orders.iloc[late_idx]
            late_idx += 1
        # Rating 3, 4 or 5: Map to an on-time order (or late order as fallback)
        else:
            if ontime_idx < len(ontime_orders):
                selected_order = ontime_orders.iloc[ontime_idx]
                ontime_idx += 1
            elif late_idx < len(late_orders):
                selected_order = late_orders.iloc[late_idx]
                late_idx += 1
            else:
                # Fallback to random order
                selected_order = df_orders.iloc[random.randint(0, len(df_orders)-1)]
                
        # Update feedback row with matched order details
        row['order_id'] = selected_order['order_id']
        row['customer_id'] = selected_order['customer_id']
        # feedback date matches order date
        row['feedback_date'] = selected_order['order_date'].strftime('%Y-%m-%d')
        mapped_feedback.append(row)
        
    df_feedback_updated = pd.DataFrame(mapped_feedback)
    df_feedback_updated.to_csv("d:\\DineshProjects\\blinkit-business-platform\\blinkit_customer_feedback.csv", index=False)
    print("Successfully mapped and saved updated customer feedback CSV.")
    
    # 5. Generate Marketing Data
    print("Generating marketing data...")
    marketing_records = []
    
    # Anomaly period: Oct 10 - Oct 17, 2024 (Instagram high spend, low sales)
    anomaly_start = datetime.datetime(2024, 10, 10)
    anomaly_end = datetime.datetime(2024, 10, 17)
    
    for dt in date_range:
        for channel in channels:
            if channel == "SMS":
                spend = random.uniform(1000.0, 3000.0)
                impressions = spend * random.uniform(8, 12)
            elif channel == "Email":
                spend = random.uniform(200.0, 800.0)
                impressions = spend * random.uniform(15, 25)
            elif channel == "Facebook":
                spend = random.uniform(3000.0, 7000.0)
                impressions = spend * random.uniform(3.5, 6)
            elif channel == "App Notification":
                spend = random.uniform(100.0, 400.0)
                impressions = spend * random.uniform(20, 30)
            elif channel == "Google Ads":
                spend = random.uniform(4000.0, 9000.0)
                impressions = spend * random.uniform(2.5, 4.5)
            elif channel == "Instagram":
                if anomaly_start <= dt <= anomaly_end:
                    spend = random.uniform(18000.0, 25000.0)
                    impressions = spend * random.uniform(1.2, 1.8)
                else:
                    spend = random.uniform(2500.0, 6000.0)
                    impressions = spend * random.uniform(4.5, 7.5)
                    
            spend = round(spend, 2)
            impressions = int(impressions)
            
            marketing_records.append({
                'date': dt.strftime('%Y-%m-%d'),
                'channel': channel,
                'spend': spend,
                'impressions': impressions
            })
            
    df_marketing = pd.DataFrame(marketing_records)
    df_marketing.to_csv("d:\\DineshProjects\\blinkit-business-platform\\blinkit_marketing_performance.csv", index=False)
    print("Saved marketing records to blinkit_marketing_performance.csv.")
    
    return df_orders, df_marketing, df_feedback_updated

def setup_and_seed_postgresql():
    db_name = "blinkit"
    user = "postgres"
    password = "postgres"
    host = "localhost"
    port = 5432
    
    try:
        conn = psycopg2.connect(
            user=user,
            password=password,
            host=host,
            port=port,
            database="postgres"
        )
        conn.autocommit = True
        cur = conn.cursor()
        
        cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{db_name}';")
        exists = cur.fetchone()
        
        if not exists:
            print(f"Creating database '{db_name}'...")
            cur.execute(f"CREATE DATABASE {db_name};")
        else:
            print(f"Database '{db_name}' already exists.")
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error checking/creating database: {e}")
        return
        
    try:
        engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db_name}")
        
        print("Loading CSV files into PostgreSQL tables...")
        df_orders = pd.read_csv("d:\\DineshProjects\\blinkit-business-platform\\blinkit_orders.csv")
        df_orders['order_date'] = pd.to_datetime(df_orders['order_date'])
        df_orders['promised_time'] = pd.to_datetime(df_orders['promised_time'])
        df_orders['actual_time'] = pd.to_datetime(df_orders['actual_time'])
        
        df_marketing = pd.read_csv("d:\\DineshProjects\\blinkit-business-platform\\blinkit_marketing_performance.csv")
        df_marketing['date'] = pd.to_datetime(df_marketing['date'])
        
        df_feedback = pd.read_csv("d:\\DineshProjects\\blinkit-business-platform\\blinkit_customer_feedback.csv")
        df_feedback['feedback_date'] = pd.to_datetime(df_feedback['feedback_date'])
        
        print("Seeding 'orders' table...")
        df_orders.to_sql("orders", engine, if_exists="replace", index=False)
        
        print("Seeding 'marketing_performance' table...")
        df_marketing.to_sql("marketing_performance", engine, if_exists="replace", index=False)
        
        print("Seeding 'customer_feedback' table...")
        df_feedback.to_sql("customer_feedback", engine, if_exists="replace", index=False)
        
        with engine.begin() as conn:
            print("Creating primary keys and indexes...")
            conn.execute(text("ALTER TABLE orders ADD PRIMARY KEY (order_id);"))
            conn.execute(text("ALTER TABLE customer_feedback ADD PRIMARY KEY (feedback_id);"))
            conn.execute(text("CREATE INDEX idx_orders_date ON orders(order_date);"))
            conn.execute(text("CREATE INDEX idx_marketing_date ON marketing_performance(date);"))
            conn.execute(text("CREATE INDEX idx_feedback_order_id ON customer_feedback(order_id);"))
            
        print("Database seeding completed successfully!")
    except Exception as e:
        print(f"Error seeding PostgreSQL: {e}")

if __name__ == "__main__":
    generate_datasets()
    setup_and_seed_postgresql()
