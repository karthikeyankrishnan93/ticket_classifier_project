import os
import random
import pandas as pd
from fastapi import FastAPI, HTTPException
from collections import defaultdict

app = FastAPI(title="Ticket Classifier - Data Prep")

DATA_DIR = r"C:\ticket_classifier_project\data"
if os.path.exists(DATA_DIR):
    import shutil
    shutil.rmtree(DATA_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

def generate_mock_data():
    random.seed(42)
    categories = ['Technical Support', 'Billing & Invoice', 'Account Access', 'Feature Request', 'General Inquiry']
    priorities = ['Low', 'Medium', 'High', 'Urgent']

    tech_issues = ["App keeps crashing on launch.", "Cannot connect to server, 404 error.", "Latest update broke export feature."]
    billing_issues = ["Charged twice for subscription.", "Requesting refund for downtime.", "How to update credit card?"]
    account_issues = ["Locked out, reset password.", "No 2FA verification code received.", "Change my registered email."]
    feature_issues = ["Add dark mode to dashboard.", "Automated export to CSV.", "Slack integration needed."]
    general_issues = ["What are working hours?", "Where is setup documentation?", "How to cancel trial?"]

    issue_map = {'Technical Support': tech_issues, 'Billing & Invoice': billing_issues, 'Account Access': account_issues, 'Feature Request': feature_issues, 'General Inquiry': general_issues}

    data = []
    for i in range(550):
        cat = random.choice(categories)
        text = f"{random.choice(issue_map[cat])} id-{i}"
        text = f"  {text}   " if i % 3 == 0 else text
        prio = random.choice(priorities)
        data.append({'ticket_text': text, 'category': cat, 'priority': prio})

    df = pd.DataFrame(data)
    df = pd.concat([df, df.iloc[:20]], ignore_index=True)
    df.to_csv(f"{DATA_DIR}/raw_tickets.csv", index=False)
    return df

@app.post("/prepare-dataset/")
def prepare_dataset():
    try:
        df = generate_mock_data()
        
        df['ticket_text'] = df['ticket_text'].astype(str).str.lower().str.strip()
        df = df.drop_duplicates().reset_index(drop=True)

        groups = defaultdict(list)
        for idx, row in df.iterrows():
            group_key = (row['category'], row['priority'])
            groups[group_key].append(idx)

        train_indices, val_indices, test_indices = [], [], []

        for group_key, indices in groups.items():
            random.seed(42)
            random.shuffle(indices)
            n = len(indices)
            n_train = int(n * 0.70)
            n_val = int(n * 0.15)
            
            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train + n_val])
            test_indices.extend(indices[n_train + n_val:])

        train_df = df.iloc[train_indices].copy()
        val_df = df.iloc[val_indices].copy()
        test_df = df.iloc[test_indices].copy()

        train_df.to_csv(f"{DATA_DIR}/train.csv", index=False)
        val_df.to_csv(f"{DATA_DIR}/val.csv", index=False)
        test_df.to_csv(f"{DATA_DIR}/test.csv", index=False)

        return {
            "status": "Success",
            "message": "Dataset processed, stratified, and saved successfully.",
            "summary": {
                "total_cleaned_rows": len(df),
                "train_rows": len(train_df),
                "validation_rows": len(val_df),
                "test_rows": len(test_df)
            },
            "class_distribution": {
                "categories": df['category'].value_counts().to_dict(),
                "priorities": df['priority'].value_counts().to_dict()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("task1_data_prep:app", host="127.0.0.1", port=8000, reload=True)
