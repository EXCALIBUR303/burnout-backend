"""
Proves whether v2's 100% test accuracy is real signal or label leakage.

Method: a 4-node decision tree should NOT fit a noisy real-world target with
>95% accuracy. If it does, the labels are a deterministic function of the
features and our XGBoost is just memorizing that function.
"""
import argparse, json
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score

LABEL_MAP = {"Low": 0, "Moderate": 1, "Medium": 1, "High": 2}

def main(out_path: str):
    df = pd.read_csv("student_lifestyle_dataset.csv")
    df.columns = df.columns.str.strip()
    y = df["Stress_Level"].str.strip().map(LABEL_MAP).dropna().astype(int)
    X = df.loc[y.index, [
        "Study_Hours_Per_Day", "Sleep_Hours_Per_Day",
        "Social_Hours_Per_Day", "Physical_Activity_Hours_Per_Day",
    ]].astype(float)

    # 1. Tiny tree: if >0.95 accuracy, it's rule-based
    tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    cv = cross_val_score(tree, X, y, cv=5, scoring="accuracy")
    rule_acc = float(cv.mean())

    # 2. Mutual info on engineered features (the worst offender)
    X_eng = X.assign(
        study_sleep_ratio=X["Study_Hours_Per_Day"] / (X["Sleep_Hours_Per_Day"] + 0.1),
        sleep_deficit=(7.5 - X["Sleep_Hours_Per_Day"]).clip(lower=0),
    )
    mi = mutual_info_classif(X_eng, y, random_state=42)
    mi_dict = dict(zip(X_eng.columns, [round(float(v), 4) for v in mi]))

    # 3. Print the recovered rules so a human can confirm
    tree.fit(X, y)
    rules = export_text(tree, feature_names=list(X.columns))

    report = {
        "rule_recovered_accuracy": round(rule_acc, 4),
        "leakage_score": round(rule_acc, 4),
        "verdict": "LEAKAGE — labels are rule-based" if rule_acc > 0.95
                  else "Likely real signal" if rule_acc < 0.70
                  else "Mixed",
        "mutual_info_per_feature": mi_dict,
        "recovered_rules": rules,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "recovered_rules"}, indent=2))
    print("\nRecovered tree rules:\n" + rules)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="diagnostics_report.json")
    main(p.parse_args().out)
