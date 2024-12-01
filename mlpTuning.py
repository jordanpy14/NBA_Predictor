# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.initializers import GlorotUniform  # Xavier initialization
from tensorflow.keras.optimizers import Adam
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import keras_tuner as kt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import seaborn as sns
from sklearn.metrics import roc_curve


class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        pass  # The model is already trained

    def predict(self, X):
        y_pred_prob = self.model.predict(X)
        y_pred_class = (y_pred_prob > 0.5).astype(int).flatten()
        return y_pred_class

    def predict_proba(self, X):
        y_pred_prob = self.model.predict(X)
        y_pred_prob = np.hstack((1 - y_pred_prob, y_pred_prob))
        return y_pred_prob


# HYPERPARAMETERS
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3

tf.keras.utils.set_random_seed(42)

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
    # "PTS",
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

    # # Calculate the target variable: 1 if home team wins, 0 otherwise
    y = (df["game_team_home_PTS"] > df["game_team_away_PTS"]).astype(int)

    # # Update records based on the game outcome
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


def plot_graphs(history):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
    # summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


def build_model(hp):
    model = tf.keras.Sequential()

    max_units = 128

    num_layers = 3
    # Tune the number of layers between 1 and 3
    for i in range(num_layers):
        # Decreasing the number of units for each subsequent layer
        if max_units < 4:
            break
        units = hp.Int(f"units_{i + 1}", min_value=4, max_value=max_units, step=32)

        # Add Dense layer
        if i == 0:
            # First layer needs to specify input_shape
            model.add(
                tf.keras.layers.Dense(
                    units=units,
                    activation="relu",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                    input_shape=(num_features,),
                )
            )
        else:
            model.add(
                tf.keras.layers.Dense(
                    units=units,
                    activation="relu",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                )
            )

        # Tune dropout rate and add Dropout layer if dropout_rate > 0.0
        dropout_rate = hp.Float(f"dropout_rate_{i + 1}", 0.0, 0.5, step=0.25)
        if dropout_rate > 0.0:
            model.add(tf.keras.layers.Dropout(rate=dropout_rate))

    # Output layer
    model.add(
        tf.keras.layers.Dense(
            units=1,
            activation="sigmoid",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )
    )

    # Tune the learning rate
    # learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return model


# Custom MLP model
class MLP(tf.keras.Model):
    def __init__(self, num_features):
        super(MLP, self).__init__()
        self.dense1 = Dense(
            units=32,
            activation="relu",
            input_shape=(num_features,),
        )
        self.dense2 = Dense(
            units=16,
            activation="relu",
        )
        self.dense3 = Dense(
            units=16,
            activation="relu",
        )
        self.dropout = Dropout(0.5)
        self.output_layer = Dense(
            units=1,
            activation="sigmoid",
        )

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        x = self.dense3(x)
        x = self.dropout(x, training=training)
        x = self.output_layer(inputs)
        return x


def main():
    # Load data
    X, y = load_data()

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Build the model
    global num_features
    num_features = X_train.shape[1]

    tuner = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=60,
        executions_per_trial=1,
        directory="hyperparameter_tuning",
        project_name="mlp_tuning",
    )

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )
    tuner.search(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Retrieve the best model and hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(
        f"""
    The hyperparameter search is complete.
    Optimal number of layers: {3}
    Units per layer: {[best_hps.get(f'units_{i+1}') for i in range(3)]}
    Dropout rates: {[best_hps.get(f'dropout_rate_{i+1}') for i in range(3)]}
    """
    )

    # Build the best model
    best_model = tuner.hypermodel.build(best_hps)

    # Train the best model
    history = best_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=[early_stopping],
        verbose=1,
    )

    y_pred_proba = best_model.predict(X_test)

    # Convert probabilities to class labels
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print("Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)
    # plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        annot_kws={"size": 12},  # Adjust the font size as needed
    )
    plt.xticks([0.5, 1.5], labels=[1, 0])
    plt.yticks([0.5, 1.5], labels=[1, 0])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()

    # Evaluate the best model
    # test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
    # print(f"Test Loss: {test_loss:.4f}")
    # print(f"Test Accuracy: {test_accuracy:.4f}")
    # plot_graphs(history)

    # plot_graphs(history)
    # Optionally, save the model
    # best_model.save("models/KT_3_Layer_best_model.keras")


if __name__ == "__main__":
    main()
