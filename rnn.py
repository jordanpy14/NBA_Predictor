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
                # else:
                #     for pbs in PlAYER_BASIC_STATS:
                #         features.append(f"{time}_{player}_{team}_{pbs}")

    return features


def calculate_nba_analytics(df):
    df["halftime_win"] = (df["h1_team_home_PTS"] > df["h1_team_away_PTS"]).astype(int)

    # Determine if the home team won the game
    df["final_win"] = (df["game_team_home_PTS"] > df["game_team_away_PTS"]).astype(int)

    # Calculate the probability of winning the game given winning at halftime
    # First, filter to only those games where the home team was leading at halftime
    halftime_leads = df[df["halftime_win"] == 1]

    # Then, find out how often those leads turned into wins
    probability_win_given_halftime_lead = halftime_leads["final_win"].mean()
    home_court_win = df["final_win"].mean()

    print("Probability win given halftime lead: ", probability_win_given_halftime_lead)
    print("Probability home team wins: ", home_court_win)


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


# Custom RNN model
class RNN(tf.keras.Model):
    def __init__(self, timesteps, features):
        super(RNN, self).__init__()
        self.lstm1 = LSTM(
            units=16,
            activation="relu",
            input_shape=(timesteps, features),
            return_sequences=False,
        )
        self.dense2 = Dense(
            units=8,
            activation="relu",
        )
        self.dense3 = Dense(
            units=4,
            activation="relu",
        )
        self.dropout = Dropout(0.5)
        self.output_layer = Dense(
            units=1,
            activation="sigmoid",
        )

    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        x = self.dense3(x)
        x = self.dropout(x, training=training)
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
    timesteps, num_features = X_train.shape[1], X_train.shape[2]
    model = RNN(2, num_features)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Optionally, save the model
    # model.save("RNN_model.keras")


if __name__ == "__main__":
    main()
