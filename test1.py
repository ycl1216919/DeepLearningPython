"""
    Testing code for different neural network configurations.
    Adapted for Python 3.5.2

    Usage in shell:
        python3.5 test.py

    Network (network.py and network2.py) parameters:
        2nd param is epochs count
        3rd param is batch size
        4th param is learning rate (eta)

    Author:
        Michał Dobrzański, 2016
        dobrzanski.michal.daniel@gmail.com
"""

# ----------------------
# - read the input data:

import mnist_loader
import numpy as np
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

# ---------------------
import matplotlib.pyplot as plt
# plot first 5 data
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(np.array(training_data[i][0]).reshape(28, 28), cmap='gray')
    plt.title(f"Label: {np.argmax(training_data[i][1])}")
plt.tight_layout()
plt.show()

# ---------------------
# - network.py example:
import network


net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
