# Import necessary libraries
import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

# HYPERPARAMETERS
DATA_DIR = "data/"
TIME = ["q1", "q2"]
PLAYERS = ["team", "player1", "player2", "player3", "player4", "player5"]
TEAMS = ["home", "away"]
PLAYER_BASIC_STATS = [
    "MP_float",
    "FGA",
    "FGPer",
    "3PA",
    "3PPer",
    "FTA",
    "FTPer",
    "ORB",
    "DRB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
    "plus_minus",
]
TEAM_BASIC_STATS = [
    "FGA",
    "FGPer",
    "3PA",
    "3PPer",
    "FTA",
    "FTPer",
    "ORB",
    "DRB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
]


def extract_features():
    features = []
    for time in TIME:
        for player in PLAYERS:
            for team in TEAMS:
                if player == "team":
                    for tbs in TEAM_BASIC_STATS:
                        features.append(f"{time}_{player}_{team}_{tbs}")
                # Uncomment the following lines if you want to include player stats
                # else:
                #     for pbs in PLAYER_BASIC_STATS:
                #         features.append(f"{time}_{player}_{team}_{pbs}")
    features.extend(
        ["team_home_wins", "team_home_losses", "team_away_wins", "team_away_losses"]
    )
    return features


def load_data():
    file_pattern = "*.csv"
    csv_files = glob.glob(os.path.join(DATA_DIR, "**", file_pattern), recursive=True)

    df_list = []
    for file in csv_files:
        df_temp = pd.read_csv(file)
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)

    # Initialize rows with missing values to 0.33
    df.fillna(0.33, inplace=True)

    # Parse 'team_home_record' into wins and losses
    home_record_split = df["team_home_record"].str.split("-", expand=True)
    home_record_split.columns = ["team_home_wins", "team_home_losses"]
    home_record_split = home_record_split.astype(int)

    # Parse 'team_away_record' into wins and losses
    away_record_split = df["team_away_record"].str.split("-", expand=True)
    away_record_split.columns = ["team_away_wins", "team_away_losses"]
    away_record_split = away_record_split.astype(int)

    # Calculate the target variable: 1 if home team wins, 0 otherwise
    y = (df["game_team_home_PTS"] > df["game_team_away_PTS"]).astype(int)

    # Update records based on the game outcome
    home_record_split["team_home_wins"] -= y
    away_record_split["team_away_losses"] -= y
    home_record_split["team_home_losses"] -= 1 - y
    away_record_split["team_away_wins"] -= 1 - y

    # Combine the modified win/loss records
    new_columns = pd.concat([home_record_split, away_record_split], axis=1)

    # Drop the original record columns
    df = df.drop(["team_home_record", "team_away_record"], axis=1)

    # Concatenate the modified new columns to the DataFrame at once
    df = pd.concat([df, new_columns], axis=1)

    # Extract features
    features = extract_features()
    X = df[features]

    return X, y


def main():
    # Load data
    X, y = load_data()

    # Get feature names
    feature_names = X.columns.tolist()

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    # Combine training and validation sets
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    # Initialize the Random Forest classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=1000, criterion="gini", max_depth=None, random_state=42, n_jobs=-1
    )

    # Train the classifier
    rf_classifier.fit(X_train_full, y_train_full)

    y_pred_rf = rf_classifier.predict(X_test)
    y_pred_rf_proba = rf_classifier.predict_proba(X_test)[:, 1]

    # Compute evaluation metrics
    accuracy_xgb = accuracy_score(y_test, y_pred_rf)
    precision_xgb = precision_score(y_test, y_pred_rf)
    recall_xgb = recall_score(y_test, y_pred_rf)
    f1_xgb = f1_score(y_test, y_pred_rf)
    auc_xgb = roc_auc_score(y_test, y_pred_rf_proba)

    print("\nXGBoost Classifier Performance:")
    print(f"Accuracy: {accuracy_xgb:.4f}")
    print(f"AUC: {auc_xgb:.4f}")
    print(f"Precision: {precision_xgb:.4f}")
    print(f"Recall: {recall_xgb:.4f}")
    print(f"F1 Score: {f1_xgb:.4f}")

    # # Make predictions on the test set
    # y_pred = rf_classifier.predict(X_test)

    # # Evaluate the model
    # test_accuracy = accuracy_score(y_test, y_pred)
    # print(f"Test Accuracy: {test_accuracy:.4f}")

    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))

    # # Confusion matrix
    # cm = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(6, 4))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.title("Confusion Matrix")
    # plt.show()

    # # Feature Importances
    # importances = rf_classifier.feature_importances_
    # feature_importances = pd.DataFrame(
    #     {"Feature": feature_names, "Importance": importances}
    # )

    # # Sort features by importance
    # feature_importances.sort_values(by="Importance", ascending=False, inplace=True)

    # # Plot top 20 features
    # top_n = 20
    # plt.figure(figsize=(12, 8))
    # sns.barplot(
    #     x="Importance",
    #     y="Feature",
    #     data=feature_importances.head(top_n),
    #     palette="viridis",
    # )
    # plt.title(f"Top {top_n} Feature Importances")
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
