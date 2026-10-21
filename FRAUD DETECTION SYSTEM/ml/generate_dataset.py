"""
Synthetic Dataset Generator for Sentinel Academic Fraud Detection System.
Team: Aniket Dubey, Raju Kumar

Generates ~50,000 realistic transactions with realistic class imbalance (~3.0% fraud),
customer baseline spending, spatio-temporal features, velocity indicators,
and realistic statistical noise to prevent trivial 100% memorization.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Fix seed for strict reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
TRANSACTIONS_CSV = os.path.join(OUTPUT_DIR, "synthetic_transactions.csv")
CUSTOMERS_CSV = os.path.join(OUTPUT_DIR, "synthetic_customers.csv")

# Constants & Categorical Dictionaries
CHANNELS = ["WEB", "MOBILE", "POS", "ATM"]
CHANNEL_WEIGHTS = [0.40, 0.45, 0.12, 0.03]

LOCATIONS = [
    "Mumbai, India", "Delhi, India", "Bengaluru, India",
    "Hyderabad, India", "Chennai, India", "Kolkata, India",
    "Pune, India", "Ahmedabad, India", "Jaipur, India",
    "London, UK", "New York, USA", "Dubai, UAE", "Singapore, SG"
]
DOMESTIC_LOCATIONS = LOCATIONS[:9]
INTERNATIONAL_LOCATIONS = LOCATIONS[9:]

MERCHANT_CATEGORIES = {
    "retail": {"risk": 0.05, "merchants": ["Amazon India", "Flipkart", "Myntra", "Reliance Digital"]},
    "grocery": {"risk": 0.02, "merchants": ["Blinkit", "BigBasket", "Zepto", "DMart"]},
    "food": {"risk": 0.03, "merchants": ["Zomato", "Swiggy", "Dominos"]},
    "travel": {"risk": 0.20, "merchants": ["MakeMyTrip", "Air India", "Booking.com", "IRCTC"]},
    "electronics": {"risk": 0.25, "merchants": ["Apple Store", "Croma", "Vijay Sales"]},
    "crypto": {"risk": 0.65, "merchants": ["CoinDCX", "WazirX", "Binance Global"]},
    "gambling": {"risk": 0.70, "merchants": ["Betway Asia", "Dream11", "PokerBaazi"]}
}
CATEGORY_KEYS = list(MERCHANT_CATEGORIES.keys())
CATEGORY_PROBS = [0.35, 0.25, 0.15, 0.10, 0.08, 0.04, 0.03]


def generate_synthetic_data(num_customers: int = 2500, num_transactions: int = 50000):
    print(f"[*] Generating {num_customers} synthetic customer profiles...")
    customers = []
    
    first_names = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Reyansh", "Muhammad", "Sai", "Arnav", "Ayaan",
                   "Krishna", "Ishaan", "Shaurya", "Atharva", "Advik", "Pranav", "Advaith", "Aaryan", "Dhruv", "Kabir",
                   "Ananya", "Diya", "Isha", "Riya", "Aadhya", "Pari", "Saanvi", "Avani", "Myra", "Sara", "Anika",
                   "John", "Jonathan", "Jennifer", "David", "Michael", "Sarah", "Emily", "Robert", "Emma"]
    last_names = ["Sharma", "Verma", "Gupta", "Malhotra", "Mehta", "Patel", "Shah", "Dubey", "Kumar", "Singh",
                  "Reddy", "Nair", "Iyer", "Choudhury", "Bose", "Banerjee", "Chatterjee", "Mishra", "Joshi", "Bhat",
                  "Doe", "Maxwell", "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson"]

    for i in range(1, num_customers + 1):
        c_code = f"{100000000000 + i}"
        fn = random.choice(first_names)
        ln = random.choice(last_names)
        dob_year = random.randint(1965, 2004)
        dob = f"{random.randint(1,28):02d}/{random.randint(1,12):02d}/{dob_year}"
        
        # Ref 1 and Ref 2 benchmark customers
        if i == 1:
            c_code = "100000000001"
            fn, ln = "John", "Doe"
            dob = "05/06/1980"
        elif i == 2:
            fn, ln = "Jonathan", "Maxwell"
            
        base_loc = random.choice(DOMESTIC_LOCATIONS)
        base_device = f"DEV-{random.randint(10000, 99999)}"
        avg_amt = float(np.random.lognormal(mean=7.5, sigma=0.6))  # mean around ~2000-3000 INR
        
        ml_risk = np.random.choice(["LOW", "MEDIUM", "HIGH"], p=[0.75, 0.20, 0.05])
        
        customers.append({
            "customer_code": c_code,
            "title": "Mr." if random.random() > 0.4 else "Ms.",
            "first_name": fn,
            "last_name": ln,
            "date_of_birth": dob,
            "gender": "Male" if random.random() > 0.4 else "Female",
            "customer_type": "1",
            "money_laundry_risk": ml_risk,
            "status": "ACTIVE",
            "registration_date": "02/05/2022",
            "id_type": "PASSPORT",
            "id_number": f"{random.randint(10000000000, 99999999999)}",
            "home_location": base_loc,
            "primary_device": base_device,
            "avg_transaction_amount": round(avg_amt, 2)
        })

    df_customers = pd.DataFrame(customers)
    df_customers.to_csv(CUSTOMERS_CSV, index=False)
    print(f"[+] Saved customer profiles to {CUSTOMERS_CSV}")

    print(f"[*] Simulating {num_transactions} chronological transactions over 90 days...")
    start_date = datetime.now() - timedelta(days=90)
    
    # Generate timestamp progression
    seconds_span = 90 * 86400
    random_seconds = np.sort(np.random.randint(0, seconds_span, size=num_transactions))
    
    transactions = []
    
    # Track per-customer sliding history for velocity calculations
    cust_last_times = {c["customer_code"]: [] for c in customers}
    
    for idx in range(num_transactions):
        c = random.choice(customers)
        c_code = c["customer_code"]
        c_avg = c["avg_transaction_amount"]
        
        txn_time = start_date + timedelta(seconds=int(random_seconds[idx]))
        hour = txn_time.hour
        is_night = int(0 <= hour <= 5)
        
        # Maintain sliding 1-hour window for customer velocity
        recent_times = [t for t in cust_last_times[c_code] if (txn_time - t).total_seconds() <= 3600]
        cust_last_times[c_code] = recent_times + [txn_time]
        
        velocity_1h = len(recent_times) + 1
        velocity_10m = len([t for t in recent_times if (txn_time - t).total_seconds() <= 600]) + 1
        
        channel = np.random.choice(CHANNELS, p=CHANNEL_WEIGHTS)
        cat = np.random.choice(CATEGORY_KEYS, p=CATEGORY_PROBS)
        merchant = random.choice(MERCHANT_CATEGORIES[cat]["merchants"])
        cat_risk = MERCHANT_CATEGORIES[cat]["risk"]
        
        # Decide fraud state with realistic multi-causal drivers (~3.0% overall fraud rate)
        # Background chance of fraud
        is_fraud = 0
        fraud_scenario = "none"
        
        # Fraud probability based on indicators
        p_fraud = 0.012  # Base spontaneous fraud rate
        
        # Simulate realistic attack vectors
        if random.random() < 0.048:
            # Staged fraud attack candidate
            scenario = random.choice(["takeover", "velocity_spike", "travel_anomaly", "high_risk_burst"])
            if scenario == "takeover":
                # High amount, new device, night
                is_fraud = 1
                fraud_scenario = "account_takeover"
            elif scenario == "velocity_spike":
                # Multiple rapid txns
                if velocity_10m >= 3:
                    is_fraud = 1
                    fraud_scenario = "card_testing"
            elif scenario == "travel_anomaly":
                # Sudden international location
                is_fraud = 1
                fraud_scenario = "impossible_travel"
            elif scenario == "high_risk_burst":
                if cat in ["crypto", "gambling"]:
                    is_fraud = 1
                    fraud_scenario = "high_risk_merchant"

        # If flagged as fraud scenario, generate parameters with realistic noise and overlap
        if is_fraud == 1:
            if random.random() < 0.20:
                # 20% subtle fraud: blends in with normal amount and no device change
                device_change = 1 if random.random() < 0.25 else 0
                location_change = 1 if random.random() < 0.20 else 0
                location = random.choice(DOMESTIC_LOCATIONS)
                device_id = c["primary_device"] if device_change == 0 else f"DEV-{random.randint(90000, 99999)}"
                amount_multiplier = float(np.random.uniform(1.2, 2.8))
                amount = float(c_avg * amount_multiplier)
            elif fraud_scenario == "card_testing":
                device_change = 1 if random.random() > 0.40 else 0
                location_change = 0
                location = random.choice(DOMESTIC_LOCATIONS)
                device_id = f"DEV-{random.randint(90000, 99999)}" if device_change else c["primary_device"]
                amount = float(np.random.uniform(15.0, 95.0))
            else:
                # Distinctive fraud
                device_change = 1 if random.random() > 0.25 else 0
                location_change = 1 if random.random() > 0.35 else 0
                location = random.choice(INTERNATIONAL_LOCATIONS if location_change else DOMESTIC_LOCATIONS)
                device_id = f"DEV-{random.randint(90000, 99999)}" if device_change else c["primary_device"]
                amount_multiplier = float(np.random.uniform(3.0, 9.5))
                amount = float(c_avg * amount_multiplier)
        else:
            # Legitimate transaction with occasional luxury spending, new phone, or travel (realistic noise)
            is_luxury = random.random() < 0.045
            device_change = 1 if random.random() < 0.09 else 0
            location_change = 1 if random.random() < 0.06 else 0
            location = random.choice(DOMESTIC_LOCATIONS if not location_change else LOCATIONS)
            device_id = f"DEV-{random.randint(10000, 99999)}" if device_change else c["primary_device"]
            
            if is_luxury:
                # Legitimate large purchase (e.g. smartphone, international flight)
                amount_multiplier = float(np.random.uniform(3.2, 7.0))
                amount = float(c_avg * amount_multiplier)
            else:
                # Standard lognormal distribution around customer baseline
                amount = float(max(50.0, np.random.normal(loc=c_avg, scale=c_avg * 0.45)))

        # Borderline noise injection (1.5% ambiguous cases)
        if is_fraud == 1 and random.random() < 0.04:
            is_fraud = 0  # True borderline legitimate
        elif is_fraud == 0 and random.random() < 0.008 and amount > c_avg * 3:
            is_fraud = 1  # Uncaught opportunistic fraud

        amount_ratio = round(amount / max(1.0, c_avg), 3)
        account_num = f"{random.randint(10000000, 99999999)}"

        transactions.append({
            "transaction_reference": f"TXN-{idx + 100001}",
            "customer_code": c_code,
            "account_number": account_num,
            "amount": round(amount, 2),
            "customer_avg_amount": round(c_avg, 2),
            "amount_ratio": amount_ratio,
            "currency": "INR",
            "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
            "hour": hour,
            "is_night": is_night,
            "channel": channel,
            "location": location,
            "merchant_name": merchant,
            "merchant_category": cat,
            "merchant_risk": cat_risk,
            "device_id": device_id,
            "device_change": device_change,
            "location_change": location_change,
            "velocity_10m": velocity_10m,
            "velocity_1h": velocity_1h,
            "is_fraud": is_fraud,
            "fraud_scenario": fraud_scenario
        })

    df_txns = pd.DataFrame(transactions)
    fraud_count = df_txns["is_fraud"].sum()
    fraud_rate = (fraud_count / len(df_txns)) * 100
    
    print(f"[+] Total transactions: {len(df_txns)}")
    print(f"[+] Fraudulent transactions: {fraud_count} ({fraud_rate:.2f}%)")
    
    df_txns.to_csv(TRANSACTIONS_CSV, index=False)
    print(f"[+] Saved synthetic transactions to {TRANSACTIONS_CSV}")
    return df_txns, df_customers


if __name__ == "__main__":
    generate_synthetic_data()
