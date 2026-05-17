import joblib
from sklearn.ensemble import RandomForestClassifier

# Example model
model = RandomForestClassifier()

# Save model
joblib.dump(model, "fraud_model.pkl")

print("Model saved successfully")