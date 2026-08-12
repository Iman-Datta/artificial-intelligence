# Training Data
training_data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1)
]

# Initial Weights and Bias
w1 = 0
w2 = 0
bias = 0 # It allows the decision boundary (or threshold) to shift, making the perceptron more flexible.
learning_rate = 1 # This tells the perceptron how big a correction to make when it makes a mistake.
epochs = 5

print("=" * 60)
print("PERCEPTRON TRAINING")
print("=" * 60)


for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}")
    print("-" * 40)

    for inputs, expected in training_data:
        x1 = inputs[0]
        x2 = inputs[1]

        # Weighted Sum
        net = (x1 * w1) + (x2 * w2) + bias

        # Step Function
        if net >= 1:
            prediction = 1
        else:
            prediction = 0

        # Error
        error = expected - prediction

        # Update Weights
        w1 = w1 + (learning_rate * error * x1)
        w2 = w2 + (learning_rate * error * x2)

        # Update Bias
        bias = bias + (learning_rate * error)

        print(f"Input       : {inputs}")
        print(f"Expected    : {expected}")
        print(f"Prediction  : {prediction}")
        print(f"Error       : {error}")
        print(f"New Weights : ({w1}, {w2})")
        print(f"New Bias    : {bias}")
        print()

    print("-" * 40)
    print(f"End of Epoch {epoch+1}")
    print(f"Current Weights = ({w1}, {w2})")
    print(f"Current Bias    = {bias}")
    print("-" * 40)

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(f"Final Weight 1 = {w1}")
print(f"Final Weight 2 = {w2}")
print(f"Final Bias     = {bias}")


print("\n" + "=" * 60)
print("TESTING PHASE")
print("=" * 60)

while True:

    print("\nEnter binary inputs (0 or 1)")

    x1 = int(input("Enter x1 : "))
    x2 = int(input("Enter x2 : "))

    # Validate input
    if x1 not in [0, 1] or x2 not in [0, 1]:
        print("Please enter only 0 or 1.")
        continue

    # Weighted Sum
    net = (x1 * w1) + (x2 * w2) + bias

    # Step Function
    if net >= 1:
        prediction = 1
    else:
        prediction = 0

    print("\nPrediction")
    print("-" * 20)
    print(f"Net Value  = {net}")
    print(f"Output     = {prediction}")

    choice = input("\nDo you want to test again? (y/n): ")

    if choice.lower() != 'y':
        break

print("\nProgram Finished.")