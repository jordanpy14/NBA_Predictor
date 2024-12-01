# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# HYPERPARAMETERS
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3

tf.keras.utils.set_random_seed(42)

DATA_DIR = "data/"
TIME = ["q1", "q2"]
PLAYERS = [
    "team",
    "player1",
]
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
                # Uncomment the following lines if you want to include player stats
                else:
                    for pbs in PLAYER_BASIC_STATS:
                        features.append(f"{time}_{player}_{team}_{pbs}")
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


# Custom MLP model
class MLP(tf.keras.Model):
    def __init__(self, num_features):
        super(MLP, self).__init__()
        self.dense1 = tf.keras.layers.Dense(
            units=64,
            activation="relu",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            input_shape=(num_features,),
        )
        self.dense2 = tf.keras.layers.Dense(
            units=32,
            activation="relu",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )
        self.dense3 = tf.keras.layers.Dense(
            units=16,
            activation="relu",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )
        self.dense4 = tf.keras.layers.Dense(
            units=16,
            activation="relu",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(
            units=1,
            activation="sigmoid",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
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
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Build the model
    num_features = X_train.shape[1]
    print(num_features)
    model = MLP(num_features)

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
    # model.save("MLP_model.keras")


if __name__ == "__main__":
    main()
