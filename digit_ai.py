"""
Нейросеть для распознавания рукописных цифр (MNIST).
Архитектура: свёрточная нейронная сеть (CNN) на TensorFlow/Keras.

Использование:
    python digit_ai.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from PIL import Image

MODEL_FILE = "digit_model.keras"


def train_model():
    """Обучает модель на датасете MNIST и сохраняет в файл."""
    print("=" * 50)
    print("ОБУЧЕНИЕ МОДЕЛИ (займёт 3–10 минут)")
    print("=" * 50)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nТочность модели: {acc * 100:.2f}%")

    model.save(MODEL_FILE)
    print(f"Модель сохранена в файл: {MODEL_FILE}")


def recognize_image(model, image_path):
    """Распознаёт цифру на картинке с автообрезкой и центрированием."""
    img = Image.open(image_path).convert("L")
    arr = np.array(img).astype("float32") / 255.0

    # Если цифра чёрная на белом — инвертируем (MNIST: белое на чёрном)
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    # Обрезаем по границам цифры (убираем лишний фон)
    rows = np.any(arr > 0.3, axis=1)
    cols = np.any(arr > 0.3, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        arr = arr[rmin:rmax + 1, cmin:cmax + 1]

    # Вписываем в квадрат, сохраняя пропорции
    h, w = arr.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.float32)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    square[y0:y0 + h, x0:x0 + w] = arr

    # Ресайз до 28×28 с полями (как MNIST: цифра в центре, вокруг 4 px)
    img_sq = Image.fromarray((square * 255).astype(np.uint8))
    img_sq = img_sq.resize((20, 20), Image.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(img_sq, (4, 4))

    final = np.array(canvas).astype("float32") / 255.0
    final = final.reshape(1, 28, 28, 1)

    probs = model.predict(final, verbose=0)[0]
    digit = int(probs.argmax())
    confidence = probs[digit] * 100
    return digit, confidence


def main():
    print("=" * 50)
    print("НЕЙРОСЕТЬ ДЛЯ РАСПОЗНАВАНИЯ ЦИФР")
    print("=" * 50)

    if not os.path.exists(MODEL_FILE):
        print("\nМодель не найдена. Сначала нужно её обучить.")
        train_model()
    else:
        print(f"\nНайдена обученная модель: {MODEL_FILE}")

    model = load_model(MODEL_FILE)

    while True:
        print("\n" + "-" * 50)
        path = input("Введите имя картинки с цифрой (или 'выход'): ").strip().strip('"')

        if path.lower() in ("выход", "exit", "quit", "q"):
            print("До свидания!")
            break

        if not os.path.exists(path):
            print(f"Файл не найден: {path}")
            continue

        try:
            digit, confidence = recognize_image(model, path)
            print(f"\n>>> РАСПОЗНАНА ЦИФРА: {digit}")
            print(f">>> Уверенность: {confidence:.2f}%")
        except Exception as e:
            print(f"Ошибка при обработке: {e}")


if __name__ == "__main__":
    main()
    