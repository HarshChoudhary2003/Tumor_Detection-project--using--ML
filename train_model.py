import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Loading dataset...")
df = pd.read_csv('Tumor_Detection.csv')

print("Preprocessing...")
if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)
if "Unnamed: 32" in df.columns:
    df.drop("Unnamed: 32", axis=1, inplace=True)

df["diagnosis"] = df["diagnosis"].map({'M': 1, 'B': 0})

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Random Forest...")
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_scaled, y_train)

y_pred = rfc.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Saving model and scaler...")
joblib.dump(rfc, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("Done!")
