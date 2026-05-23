import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train_path = "train"

categories = ["cats", "dogs"]

data = []
labels = []

IMG_SIZE = 128

for category in categories:

    folder_path = os.path.join(train_path, category)

    label = categories.index(category)

    for img in os.listdir(folder_path)[:1000]:

        img_path = os.path.join(folder_path, img)

        try:

            image = cv2.imread(img_path)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            data.append(image.flatten())

            labels.append(label)

        except:
            pass

X = np.array(data)

y = np.array(labels)

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = SVC(kernel="rbf")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========\n")

print("Accuracy :", accuracy)

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")

plt.colorbar()

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0,1], ["Cats", "Dogs"])
plt.yticks([0,1], ["Cats", "Dogs"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="black"
        )

plt.show()

plt.figure(figsize=(10,6))

for i in range(6):

    plt.subplot(2,3,i+1)

    image = scaler.inverse_transform([X_test[i]])[0]

    image = image.reshape(IMG_SIZE, IMG_SIZE)

    plt.imshow(image, cmap="gray")

    predicted = "Dog" if y_pred[i] == 1 else "Cat"

    actual = "Dog" if y_test[i] == 1 else "Cat"

    plt.title(f"P:{predicted} | A:{actual}")

    plt.axis("off")

plt.tight_layout()

plt.show()