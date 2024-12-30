def preprocess_input(input_data):
    """
    Preprocess input data before prediction.
    """
    # Example scaling logic (update based on your training process)
    scaled_data = [(x - min_val) / (max_val - min_val) for x, min_val, max_val in zip(input_data, [0]*6, [100]*6)]
    return scaled_data
