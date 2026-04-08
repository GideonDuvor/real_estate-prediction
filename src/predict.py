def make_prediction(model, input_data):
    try:
        prediction = model.predict(input_data)
        return prediction
    except Exception as e:
        print(f"Prediction error: {e}")