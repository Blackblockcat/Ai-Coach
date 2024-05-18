import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers, models
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Pad or truncate sequences to ensure all have the same length
data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Build the model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(66,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=30)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy}")


y_pred = np.round(model.predict(x_test))


conf_matrix = confusion_matrix(y_test, y_pred)


fig, axs = plt.subplots(1, 2, figsize=(12, 6))


sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axs[0])
axs[0].set_xlabel('Predicted Labels')
axs[0].set_ylabel('True Labels')
axs[0].set_title('Confusion Matrix')


def get_correct_incorrect_predictions(y_true, y_pred):
    correct = []
    incorrect = []

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct.append((y_true[i], y_pred[i]))
        else:
            incorrect.append((y_true[i], y_pred[i]))

    return correct, incorrect


correct, incorrect = get_correct_incorrect_predictions(y_test, y_pred)


correct_labels = "\n".join([f"True Label: {true_label}, Predicted Label: {pred_label}" for true_label, pred_label in correct])
axs[1].text(0.1, 0.5, f"Correct Predictions:\n{correct_labels}", fontsize=10, verticalalignment='center')
axs[1].axis('off')


incorrect_labels = "\n".join([f"True Label: {true_label}, Predicted Label: {pred_label}" for true_label, pred_label in incorrect])
axs[1].text(0.6, 0.5, f"Incorrect Predictions:\n{incorrect_labels}", fontsize=10, verticalalignment='center')
axs[1].axis('off')

plt.tight_layout()
plt.show()

# Save the model architecture as an image
model.save(r"C:\Users\Deiaa\OneDrive\Desktop\Games\American-Sign-Language\model.h5")
