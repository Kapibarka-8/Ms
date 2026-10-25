import numpy as np
from PIL import Image

path = "цифра для проверки.png"   # ← замените на имя вашей картинки

img = Image.open(path).convert("L")
print("Размер оригинала:", img.size)

img_resized = img.resize((28, 28))
arr = np.array(img_resized).astype("float32") / 255.0

print("Мин / макс пикселя:", arr.min(), arr.max())
print("Среднее значение:", round(float(arr.mean()), 3))
print("\nКартинка в миниатюре (что 'видит' модель):")
for row in arr[::2]:  # каждый 2-й пиксель — чтобы влезло в экран
    line = "".join("█" if v < 0.5 else " " for v in row[::1])
    print(line)
    
