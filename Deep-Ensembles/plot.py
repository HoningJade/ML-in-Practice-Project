import numpy as np
import matplotlib.pyplot as plt

# Accuracy plot
train_acc = np.load('train_acc.npy')
val_acc = np.load('val_acc.npy')

fig, ax = plt.subplots()
ax.plot(np.arange(1, 16), train_acc, label='Train Acc')
ax.plot(np.arange(1, 16), val_acc, label='Val Acc')
ax.legend()
ax.set_xlabel('Ensemble Size')
ax.set_ylabel('Accuracy')

plt.savefig('./acc.png')


# Calibration plot
one = np.load(f'esize_1_acc.npy')
five = np.load(f'esize_5_acc.npy')
ten = np.load(f'esize_10_acc.npy')

fig, ax = plt.subplots()
ax.plot(np.arange(0.1, 1.1, 0.1), one, label='Ensemble Size=1')
ax.plot(np.arange(0.1, 1.1, 0.1), five, label='Ensemble Size=5')
ax.plot(np.arange(0.1, 1.1, 0.1), ten, label='Ensemble Size=10')
ax.legend()
ax.set_xlabel('Confidence Threshold')
ax.set_ylabel('Training Accuracy')

plt.savefig('./calibration.png')


# Calibration plot
five_in = np.load('calibration_in.npy')
five_out = np.load('calibration_out.npy')

fig, (ax) = plt.subplots()
ax.plot(np.arange(0.0, 0.8, 0.1), five_in, label='In-Distribution')
ax.plot(np.arange(0.0, 0.8, 0.1), five_out, label='Out-of-Distribution')
ax.legend()
ax.set_xlabel('Entropy Values')
ax.set_ylabel('Count')


plt.savefig('./entropy.png')

