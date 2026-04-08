from src.train import train_model, save_model
from src.load_data import load_data
from src.preprocessing import preprocess_data
from src.train import train_model

# Step 1: Load data
df = load_data('data/final.csv')

# Step 2: Preprocess
X, y = preprocess_data(df)

# Step 3: Train model
model, X_test, y_test = train_model(X, y)

print("Pipeline executed successfully!")
save_model(model)
print("Model saved successfully!")