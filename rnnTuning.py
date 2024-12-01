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
import matplotlib.pyplot as plt
import keras_tuner as kt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# HYPERPARAMETERS
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3

tf.keras.utils.set_random_seed(42)


DATA_DIR = "data/"
# TIME = ["q1", "q2", "h1", "game"]
TIME = ["q1", "h1"]
PLAYERS = ["team", "player1", "player2", "player3", "player4", "player5"]
TEAMS = ["home", "away"]
# PlAYER_BASIC_STATS = ["MP_float", "FG", "FGA", "FGPer", "3P", "3PA", "3PPer", "FT", "FTA", "FTPer", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "GmSc", "plus_minus", ]
PlAYER_BASIC_STATS = [
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
]


def extract_features():
    features = []
    for time in TIME:
        for player in PLAYERS:
            for team in TEAMS:
                if player == "team":
                    for tbs in TEAM_BASIC_STATS:
                        features.append(f"{time}_{player}_{team}_{tbs}")
                # else:
                #     for pbs in PlAYER_BASIC_STATS:
                #         features.append(f"{time}_{player}_{team}_{pbs}")

    return features


def calculate_nba_analytics(df):
    df["halftime_win"] = (df["h1_team_home_PTS"] > df["h1_team_away_PTS"]).astype(int)

    # Determine if the home team won the game
    df["final_win"] = (df["game_team_home_PTS"] > df["game_team_away_PTS"]).astype(int)

    print(
        sum(1 for x, y in zip(df["final_win"], df["halftime_win"]) if x == y)
        / len(df["final_win"])
    )

    # # Calculate the probability of winning the game given winning at halftime
    # # First, filter to only those games where the home team was leading at halftime
    # halftime_leads = df[df["halftime_win"] == 1]

    # halftime_losses = df[df["halftime_win"] == 0]

    # # Then, find out how often those leads turned into wins
    # probability_win_given_halftime_lead = halftime_leads["final_win"].mean()
    # home_court_win = df["final_win"].mean()

    # print(
    #     "Probability win given halftime lead AWAY: ",
    #     probability_win_given_halftime_lead,
    # )

    # print("Probability home team wins: ", 1 - home_court_win)


def load_data():
    file_pattern = "*.csv"  # Adjust the pattern if necessary
    csv_files = glob.glob(os.path.join(DATA_DIR, "**", file_pattern), recursive=True)

    df_list = []
    for file in csv_files:
        df_temp = pd.read_csv(file)
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)

    # Initialize rows with missing values to 0
    df.fillna(0, inplace=True)

    # Calculate the target variable: 1 if home team wins, 0 otherwise
    y = (df["game_team_home_PTS"] > df["game_team_away_PTS"]).astype(int)
    calculate_nba_analytics(df)

    # Extract features
    features = extract_features()
    # Defragment the DataFrame
    df = df[features].copy()

    features_q1 = [feature for feature in df.columns if feature.startswith("q1_")]
    features_h1 = [feature for feature in df.columns if feature.startswith("h1_")]

    features_q1.sort()
    features_h1.sort()

    # Extract data for each time step
    X_q1 = df[features_q1].values
    X_h1 = df[features_h1].values

    # Stack the data along a new axis to create the time steps
    X = np.stack(
        (X_q1, X_h1), axis=1
    )  # Shape: (samples, timesteps, features_per_timestep)

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

    num_lstm_layers = 2
    for i in range(num_lstm_layers):
        # Decreasing the number of units for each subsequent layer
        units = hp.Int(f"units_{i + 1}", min_value=4, max_value=max_units, step=32)

        # Add Dense layer
        if i == 0:
            # First layer needs to specify input_shape
            model.add(
                tf.keras.layers.LSTM(
                    units=units,
                    activation="relu",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                    input_shape=(timesteps, num_features),
                    return_sequences=True,
                )
            )
        else:
            model.add(
                tf.keras.layers.LSTM(
                    units=units,
                    activation="relu",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                )
            )

        # Tune dropout rate and add Dropout layer if dropout_rate > 0.0
        dropout_rate = hp.Float(f"dropout_rate_{i + 1}", 0.0, 0.5, step=0.25)
        if dropout_rate > 0.0:
            model.add(tf.keras.layers.Dropout(rate=dropout_rate))

    # Tune the number of layers between 1 and 3
    for i in range(2, 3):
        # Decreasing the number of units for each subsequent layer
        units = hp.Int(f"units_{i + 1}", min_value=4, max_value=max_units, step=32)
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


# Custom RNN model
class RNN(tf.keras.Model):
    def __init__(self, timesteps, features):
        super(RNN, self).__init__()
        self.lstm1 = LSTM(
            units=4,
            activation="relu",
            input_shape=(timesteps, features),
            return_sequences=True,
        )
        self.lstm2 = LSTM(
            units=100,
            activation="relu",
            return_sequences=False,
        )
        self.dense1 = Dense(
            units=36,
            activation="relu",
        )
        self.dropout = Dropout(0.25)
        self.output_layer = Dense(
            units=1,
            activation="sigmoid",
        )

    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.dropout(x, training=training)
        x = self.lstm2(x)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.output_layer(x)
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
    scalers = {}
    for i in range(X_train.shape[1]):  # Iterate over time steps
        scalers[i] = StandardScaler()
        X_train[:, i, :] = scalers[i].fit_transform(X_train[:, i, :])
        X_val[:, i, :] = scalers[i].transform(X_val[:, i, :])
        X_test[:, i, :] = scalers[i].transform(X_test[:, i, :])

    # Build the model
    global timesteps, num_features
    timesteps, num_features = X_train.shape[1], X_train.shape[2]

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

    # Evaluate the best model
    # test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
    # print(f"Test Loss: {test_loss:.4f}")
    # print(f"Test Accuracy: {test_accuracy:.4f}")
    # plot_graphs(history)

    # model = RNN(2, num_features)

    # # Compile the model
    # optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    # model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # # Train the model
    # history = model.fit(
    #     X_train,
    #     y_train,
    #     validation_data=(X_val, y_val),
    #     epochs=EPOCHS,
    #     batch_size=BATCH_SIZE,
    # )

    # # Evaluate the model
    # test_loss, test_accuracy = model.evaluate(X_test, y_test)
    # print(f"Test Loss: {test_loss}")
    # print(f"Test Accuracy: {test_accuracy}")

    # Optionally, save the model
    # model.save("RNN_model.keras")


if __name__ == "__main__":
    main()
