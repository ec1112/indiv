import matplotlib.pyplot as plt


def get_epochs(filename):
	epochs = []
	with open(filename, "r") as f:
		for line in f:
			if line.startswith("Epoch: "):
				epoch_num = line.split(" ")[1]
				epochs.append(int(epoch_num))

	return epochs


def get_errors(filename):
	errors = []
	with open(filename, "r") as f:
		for line in f:
			if line.startswith("Validation loss: "):
				error = line.split(" ")[1]
				errors.append(float(error))

	return errors


def get_costs(filename):
	costs = []
	with open(filename, "r") as f:
		for line in f:
			if line.startswith("Cost: "):
				cost = line.split(" ")[1]
				costs.append(float(error))

	return costs


def get_learning_rates(filename):
	rates = []
	with open(filename, "r") as f:
		for line in f:
			if line.startswith("Learning rate: "):
				rate = line.split(" ")[1]
				rates.append(float(rate))

	return rates


filename = "metrics.txt"
epochs = get_epochs(filename)

errors = plt.plot(epochs, get_errors(filename))
plt.xlabel("Epoch (#)")
plt.ylabel("Error (%)")
plt.show()

costs = plt.plot(epochs, get_costs(filename))
plt.xlabel("Epoch (#)")
plt.ylabel("Cost (P)")
plt.show()

learning_rates = plt.plot(epochs, get_learning_rates(filename))
plt.xlabel("Epoch (#)")
plt.ylabel("Learning rate (P)")
plt.show()


