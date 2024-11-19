# Import necessary libraries
import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import GlorotUniform  # Xavier initialization
from tensorflow.keras.optimizers import Adam

# HYPERPARAMETERS
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3

DATA_DIR = "data/"
TIME = ["q1", "q2", "h1", "game"]
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
                if team == "team":
                    for tbs in TEAM_BASIC_STATS:
                        features.append(f"{time}_{player}_{team}_{tbs}")
                else:
                    for pbs in PlAYER_BASIC_STATS:
                        features.append(f"{time}_{player}_{team}_{pbs}")
    return features


def load_data():
    file_pattern = "*.csv"  # Adjust the pattern if necessary
    csv_files = glob.glob(os.path.join(DATA_DIR, "**", file_pattern), recursive=True)

    df_list = []
    for file in csv_files:
        df_temp = pd.read_csv(file)
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)
    features = extract_features()
    # Assume 'df' is your DataFrame
    # Adjust 'target1' and 'target2' to your actual target column names
    X = df[features].values
    y = df["WL"].values
    return X, y


# Build the model
def buildMLP(num_features, learning_rate=LR):
    model = Sequential()
    model.add(
        Dense(
            units=256,
            activation="relu",
            kernel_initializer=GlorotUniform(),
            input_shape=(num_features,),
        )
    )
    model.add(Dense(units=128, activation="relu", kernel_initializer=GlorotUniform()))
    model.add(Dropout(0.5))
    model.add(Dense(units=2, activation="sigmoid", kernel_initializer=GlorotUniform()))

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


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

model = buildMLP(X_train.shape[1], learning_rate=LR)

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

model.save("MLP_model")
