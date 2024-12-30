import matplotlib.pyplot as plt

def generate_plot(input_data, prediction):
    """
    Generate a plot and save it as a PNG.
    """
    plt.figure(figsize=(6, 4))
    plt.bar([f"Sensor{i}" for i in range(1, 7)], input_data, color="skyblue")
    plt.title(f"Prediction: {prediction}")
    plt.ylabel("Sensor Values")
    plt.savefig("assets/plot.png")
    plt.close()
