import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def train_model(X, y):
    try:
        stratify_col = X['property_type_Condo'] if 'property_type_Condo' in X else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=stratify_col
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        print("Model trained successfully")
        return model, X_test, y_test

    except Exception as e:
        print(f"Training error: {e}")
        
def save_model(model, filename='models/model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)