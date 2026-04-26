import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
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

print("Training Advanced 5-Model Neural Ensemble...")
rf = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42)
svc = SVC(probability=True, kernel='rbf', C=10, gamma='scale', random_state=42)
lr = LogisticRegression(max_iter=2000, C=1, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, alpha=0.01, random_state=42)

ensemble_model = VotingClassifier(
    estimators=[('rf', rf), ('svc', svc), ('lr', lr), ('gb', gb), ('mlp', mlp)], 
    voting='soft',
    weights=[2, 1, 1, 2, 2] # Give more weight to tree ensembles and MLP
)
ensemble_model.fit(X_train_scaled, y_train)

y_pred = ensemble_model.predict(X_test_scaled)
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

print("Saving Advanced Model and Scaler...")
joblib.dump(ensemble_model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("Done! Powerful modifications complete.")
