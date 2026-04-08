def preprocess_data(df):
    try:
        X = df.drop('price', axis=1)
        y = df['price']
        print("Preprocessing complete")
        return X, y
    except Exception as e:
        print(f"Preprocessing error: {e}")