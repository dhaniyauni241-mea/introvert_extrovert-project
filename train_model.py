import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

print("Loading dataset...")
df = pd.read_csv("train.csv")

# Remove id column
df = df.drop("id", axis=1)

# Numerical columns
numerical_columns = [
    "Time_spent_Alone",
    "Social_event_attendance",
    "Going_outside",
    "Friends_circle_size",
    "Post_frequency"
]

# Fill missing numerical values with mean
for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].mean())

# Categorical columns
categorical_columns = [
    "Stage_fear",
    "Drained_after_socializing"
]

# Fill missing categorical values with mode
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Encode target column
target_encoder = LabelEncoder()
df["Personality"] = target_encoder.fit_transform(df["Personality"])

# Features and target
X = df.drop("Personality", axis=1)
y = df["Personality"]

print(X.columns)
print("Number of features:", len(X.columns))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
print("Training model...")
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Save files
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(encoders, "encoders.joblib")
joblib.dump(target_encoder, "target_encoder.joblib")

print("Done!")