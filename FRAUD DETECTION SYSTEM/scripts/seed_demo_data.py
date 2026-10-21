"""
Seed Demo Data Script for Sentinel Academic Fraud Detection System.
Team: Aniket Dubey, Raju Kumar

Initializes the SQLite database with benchmark entities matching Ref 1, Ref 2, and Ref 3.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy.orm import Session

from app.database import engine, Base, SessionLocal
from app.models import (
    User, Customer, PaymentInstrument, Transaction,
    Alert, Case, CaseRule, CaseDocument, Rule, Blocklist, AuditLog, SystemSetting
)
from app.security import hash_password

def seed_database():
    print("[*] Recreating database tables...")
    Base.metadata.create_all(bind=engine)
    db: Session = SessionLocal()

    try:
        # Check if users already seeded
        if db.query(User).filter(User.email == "admin@sentinel.ai").first():
            print("[!] Database already seeded. Skipping initial seed.")
            return

        print("[*] Seeding administrative and analyst users...")
        admin = User(
            email="admin@sentinel.ai",
            hashed_password=hash_password("Admin@12345"),
            name="Admin Officer",
            role="admin",
            is_active=True
        )
        analyst = User(
            email="analyst@sentinel.ai",
            hashed_password=hash_password("Analyst@12345"),
            name="Jennifer Adams",
            role="analyst",
            is_active=True
        )
        db.add_all([admin, analyst])
        db.commit()
        db.refresh(admin)
        db.refresh(analyst)

        print("[*] Seeding hybrid fraud detection rules...")
        rules = [
            Rule(
                rule_code="RULE-1792",
                name="Test rule for customer ID (Amount Spike)",
                description="Transaction amount exceeds 5x customer historical 30-day baseline",
                rule_type="AMOUNT_RATIO",
                threshold_value=5.0,
                weight=0.35,
                is_enabled=True,
                alert_count=14
            ),
            Rule(
                rule_code="RULE-2041",
                name="High Transaction Velocity (10m)",
                description="3 or more transactions executed within 10 minutes from same account",
                rule_type="VELOCITY",
                threshold_value=3.0,
                weight=0.30,
                is_enabled=True,
                alert_count=9
            ),
            Rule(
                rule_code="RULE-3108",
                name="Nocturnal High-Value Activity",
                description="Transaction initiated during off-hours (00:00 - 05:00 local time)",
                rule_type="NIGHT_TIME",
                threshold_value=1.0,
                weight=0.15,
                is_enabled=True,
                alert_count=6
            ),
            Rule(
                rule_code="RULE-4055",
                name="Impossible Travel / Geo Jump",
                description="Abrupt geographic shift without plausible flight duration",
                rule_type="IMPOSSIBLE_TRAVEL",
                threshold_value=1.0,
                weight=0.40,
                is_enabled=True,
                alert_count=11
            ),
            Rule(
                rule_code="RULE-5120",
                name="Device Fingerprint Anomaly",
                description="Unrecognized device ID or browser user-agent hash mismatch",
                rule_type="DEVICE_CHANGE",
                threshold_value=1.0,
                weight=0.20,
                is_enabled=True,
                alert_count=8
            ),
            Rule(
                rule_code="RULE-6200",
                name="High-Risk Merchant Category",
                description="Transaction routed to volatile merchant category (Crypto, Offshore Casino)",
                rule_type="MERCHANT_RISK",
                threshold_value=1.0,
                weight=0.35,
                is_enabled=True,
                alert_count=5
            ),
            Rule(
                rule_code="RULE-7019",
                name="Rapid Micro Card Testing",
                description="Sub-100 INR transaction burst indicative of automated BIN testing",
                rule_type="CARD_TESTING",
                threshold_value=100.0,
                weight=0.30,
                is_enabled=True,
                alert_count=4
            ),
            Rule(
                rule_code="RULE-8800",
                name="Administrative Blocklist Enforcement",
                description="Identifier matched against active sanctions/blocklist registry",
                rule_type="BLOCKLIST",
                threshold_value=1.0,
                weight=1.00,
                is_enabled=True,
                alert_count=3
            ),
        ]
        db.add_all(rules)
        db.commit()

        print("[*] Seeding default blocklist entries...")
        blocklist_entries = [
            Blocklist(entry_type="ACCOUNT", entry_value="44991122", reason="Confirmed card theft report", added_by_user_id=admin.id),
            Blocklist(entry_type="DEVICE", entry_value="DEV-99999", reason="Known emulator botnet cluster", added_by_user_id=admin.id),
            Blocklist(entry_type="IP", entry_value="185.220.101.5", reason="Malicious Tor exit node", added_by_user_id=admin.id),
        ]
        db.add_all(blocklist_entries)
        db.commit()

        print("[*] Seeding reference benchmark customers (Ref 1 & Ref 2)...")
        # Customer 1 (Exact match for Ref 1: John Doe, 100000000001)
        c1 = Customer(
            customer_code="100000000001",
            title="Mr.",
            first_name="John",
            last_name="Doe",
            date_of_birth="05/06/1980",
            gender="Male",
            customer_type="1",
            money_laundry_risk="MEDIUM",
            status="ACTIVE",
            registration_date="02/05/2022",
            id_type="PASSPORT",
            id_number="33488696000",
            email="john.doe@academic-demo.net",
            phone="+91 98765 43210",
            location="Mumbai, India",
            avg_transaction_amount=4200.0,
            total_transaction_count=48,
            risk_score=0.42
        )
        # Customer 2 (Exact match for Ref 2: Jonathan Maxwell)
        c2 = Customer(
            customer_code="100000000002",
            title="Mr.",
            first_name="Jonathan",
            last_name="Maxwell",
            date_of_birth="14/11/1988",
            gender="Male",
            customer_type="1",
            money_laundry_risk="HIGH",
            status="ACTIVE",
            registration_date="11/08/2021",
            id_type="NATIONAL_ID",
            id_number="99281726354",
            email="j.maxwell@academic-demo.net",
            phone="+91 91234 56789",
            location="Surulere, Lagos",
            avg_transaction_amount=25000.0,
            total_transaction_count=22,
            risk_score=0.88
        )
        db.add_all([c1, c2])
        db.commit()
        db.refresh(c1)
        db.refresh(c2)

        print("[*] Seeding benchmark payment instruments (Ref 1: 77563949 HSBC)...")
        p1 = PaymentInstrument(
            customer_id=c1.id,
            instrument_type="CARD",
            number_masked="77563949",
            entity_name="HSBC",
            status="ACTIVE",
            is_blocked=False,
            expiry_date="08/28"
        )
        p2 = PaymentInstrument(
            customer_id=c2.id,
            instrument_type="ACCOUNT",
            number_masked="25078990",
            entity_name="Standard Chartered",
            status="ACTIVE",
            is_blocked=False,
            expiry_date="12/29"
        )
        db.add_all([p1, p2])
        db.commit()
        db.refresh(p1)
        db.refresh(p2)

        print("[*] Seeding reference case (Ref 1: Case 74142)...")
        case1 = Case(
            case_reference="74142",
            customer_id=c1.id,
            assigned_user_id=analyst.id,
            locked_by_user_id=analyst.id,
            locked_at=datetime.utcnow(),
            days_until_auto_classification=20,
            alerts_count=7,
            max_alerts=10,
            state="In Review",
            classification_notes="Under active investigation. Customer contacted to verify international card authorization.",
            created_at=datetime.utcnow() - timedelta(days=2)
        )
        db.add(case1)
        db.commit()
        db.refresh(case1)

        # Rules linked to Case 74142
        case_rule1 = CaseRule(
            case_id=case1.id,
            rule_id=1,
            rule_code="1792",
            rule_name="Test rule for customer ID",
            alert_count=1
        )
        db.add(case_rule1)

        # Document linked to Case 74142
        case_doc = CaseDocument(
            case_id=case1.id,
            filename="passport_copy_verified.pdf",
            original_name="passport_scan.pdf",
            description="Verified KYC ID Document",
            file_size_bytes=428900,
            mime_type="application/pdf",
            uploaded_by_user_id=analyst.id,
            uploaded_at=datetime.utcnow() - timedelta(days=1)
        )
        db.add(case_doc)
        db.commit()

        print("[*] Seeding alerts matching Ref 2 (Jonathan Maxwell, High/Medium/Low cards)...")
        # Load small batch from synthetic data if available
        now = datetime.utcnow()
        sample_alerts = [
            {
                "customer": c2,
                "amount": 250789.90,
                "location": "Abia (First Time Login)",
                "reason": "First-time device login with high volume",
                "risk_level": "High",
                "risk_score": 0.94,
                "account": "25078990"
            },
            {
                "customer": c2,
                "amount": 250789.90,
                "location": "Surulere, Lagos",
                "reason": "Velocity surge: 4 transactions in 8 mins",
                "risk_level": "High",
                "risk_score": 0.89,
                "account": "25078990"
            },
            {
                "customer": c2,
                "amount": 250789.90,
                "location": "Surulere, Lagos",
                "reason": "Amount exceeds 4x 30-day baseline",
                "risk_level": "Medium",
                "risk_score": 0.62,
                "account": "25078990"
            },
            {
                "customer": c2,
                "amount": 250789.90,
                "location": "Surulere, Lagos",
                "reason": "Normal travel confirmation pending",
                "risk_level": "Low",
                "risk_score": 0.22,
                "account": "25078990"
            },
            {
                "customer": c2,
                "amount": 250789.90,
                "location": "Surulere, Lagos",
                "reason": "Off-hours point of sale activity",
                "risk_level": "Medium",
                "risk_score": 0.58,
                "account": "25078990"
            },
            {
                "customer": c2,
                "amount": 250789.90,
                "location": "Surulere, Lagos",
                "reason": "International wire instruction trigger",
                "risk_level": "High",
                "risk_score": 0.91,
                "account": "25078990"
            },
        ]

        for i, item in enumerate(sample_alerts, 1):
            t = Transaction(
                transaction_reference=f"TXN-BENCH-{100 + i}",
                customer_id=item["customer"].id,
                instrument_id=p2.id,
                amount=item["amount"],
                currency="INR",
                timestamp=now - timedelta(minutes=i * 12),
                channel="MOBILE",
                location=item["location"],
                merchant_name="Standard Apex Terminal",
                merchant_category="electronics",
                is_night=True if "First Time" in item["reason"] else False,
                location_change=True if "Abia" in item["location"] else False,
                device_change=True if "First Time" in item["reason"] else False,
                velocity_10m=4 if "Velocity" in item["reason"] else 1,
                velocity_1h=5 if "Velocity" in item["reason"] else 2,
                ml_probability=item["risk_score"],
                rule_score=0.85 if item["risk_level"] == "High" else 0.45,
                final_risk_score=item["risk_score"],
                risk_level=item["risk_level"].upper(),
                prediction="FRAUD" if item["risk_level"] == "High" else "LEGITIMATE",
                status="FLAGGED" if item["risk_level"] == "High" else "APPROVED",
                trigger_reasons=[item["reason"]]
            )
            db.add(t)
            db.commit()
            db.refresh(t)

            alt = Alert(
                alert_reference=f"ALT-{1000 + i}",
                transaction_id=t.id,
                customer_id=item["customer"].id,
                customer_name=f"{item['customer'].first_name} {item['customer'].last_name}",
                account_number=item["account"],
                amount=item["amount"],
                currency="INR",
                location=item["location"],
                trigger_reason=item["reason"],
                risk_score=item["risk_score"],
                risk_level=item["risk_level"],
                is_flagged=False,
                is_safe=False,
                status="OPEN",
                created_at=t.timestamp
            )
            db.add(alt)

        # Seed System Settings
        settings_list = [
            SystemSetting(key="risk_threshold_low", value="0.35", description="Lower cutoff for low-risk approvals"),
            SystemSetting(key="risk_threshold_high", value="0.70", description="Upper cutoff for high-risk manual review/block"),
            SystemSetting(key="default_currency", value="INR", description="System presentation currency"),
            SystemSetting(key="case_auto_classify_days", value="14", description="Default case auto-classification lifecycle"),
            SystemSetting(key="simulator_speed", value="4", description="Transaction generation interval in seconds"),
        ]
        db.add_all(settings_list)

        # Seed initial Audit Log
        audit = AuditLog(
            user_id=admin.id,
            user_name=admin.name,
            action="SYSTEM_INIT",
            target_type="DATABASE",
            target_id="0",
            details="Database tables created and benchmark demonstration data seeded.",
            ip_address="127.0.0.1"
        )
        db.add(audit)
        db.commit()

        print("[+] Demo data seeding complete successfully!")

    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
