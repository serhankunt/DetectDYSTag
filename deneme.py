import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Görüntü ön işleme fonksiyonu
def preprocess_image(image, img_height, img_width):
    image = cv2.resize(image, (img_width, img_height))
    image = image.astype(np.float32) / 255.0
    return image

# Veri hazırlama
# def prepare_data(data_path, img_height, img_width):
    X_train = []
    y_train = []

    if not os.path.exists(data_path):
        raise ValueError(f"Görüntü yüklenemedi: {data_path}")

    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        image = cv2.imread(file_path)

        if image is None:
            print(f"Görüntü yüklenemedi: {file_path}")
            continue

        image = preprocess_image(image, img_height, img_width)
        X_train.append(image)
        
        # Basit bir etiketleme (şu an tüm görüntüler için etiket 1 olarak kabul ediliyor)
        y_train.append(1)

    if len(X_train) == 0:
        raise ValueError("Veri yüklenemedi, model eğitimi için yeterli veri yok.")

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train
def prepare_data(positive_path, negative_path, img_height, img_width):
    X_train = []
    y_train = []

    # Pozitif sınıf için görüntüleri yükleme
    if not os.path.exists(positive_path):
        raise ValueError(f"Pozitif görüntü yolu bulunamadı: {positive_path}")

    for file_name in os.listdir(positive_path):
        file_path = os.path.join(positive_path, file_name)
        image = cv2.imread(file_path)

        if image is None:
            print(f"Pozitif görüntü yüklenemedi: {file_path}")
            continue

        # Görüntüyü işleme
        image = preprocess_image(image, img_height, img_width)
        X_train.append(image)
        y_train.append(1)  # Pozitif sınıf etiketi

    # Negatif sınıf için görüntüleri yükleme
    if not os.path.exists(negative_path):
        raise ValueError(f"Negatif görüntü yolu bulunamadı: {negative_path}")

    for file_name in os.listdir(negative_path):
        file_path = os.path.join(negative_path, file_name)
        image = cv2.imread(file_path)

        if image is None:
            print(f"Negatif görüntü yüklenemedi: {file_path}")
            continue

        # Görüntüyü işleme
        image = preprocess_image(image, img_height, img_width)
        X_train.append(image)
        y_train.append(0)  # Negatif sınıf etiketi

    if len(X_train) == 0:
        raise ValueError("Veri yüklenemedi, model eğitimi için yeterli veri yok.")

    # Listeyi numpy array'e dönüştür
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train


# Model oluşturma
def create_model(img_height, img_width):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(2, 2),
        Dropout(0.3),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Eğitim ve tahmin
# if __name__ == "__main__":
#     img_height, img_width = 128, 128
#     data_path = "C:/deneme/DYS"

#     # Veri hazırlama
#     try:
#         X_train, y_train = prepare_data(data_path, img_height, img_width)
#     except ValueError as e:
#         print(e)
#         exit(1)

if __name__ == "__main__":
    img_height, img_width = 128, 128

    positive_path = "C:/deneme/DYS/positive"
    negative_path = "C:/deneme/DYS/negative"

    # Veri hazırlama
    try:
        X_train, y_train = prepare_data(positive_path, negative_path, img_height, img_width)
    except ValueError as e:
        print(e)
        exit(1)

    # Model oluştur ve eğit
    model = create_model(img_height, img_width)

    try:
        model.fit(X_train, y_train, epochs=10)
    except ValueError as e:
        print(f"Model eğitimi sırasında hata oluştu: {e}")
        exit(1)


    # Veri artırma için ImageDataGenerator kullanımı
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Veri artırmayı hazırlanan verilere uygula
    datagen.fit(X_train)

    # Model oluştur ve eğit
    model = create_model(img_height, img_width)

    try:
        model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10)
    except ValueError as e:
        print(f"Model eğitimi sırasında hata oluştu: {e}")
        exit(1)

    # Tahmin yaparken görüntü yükle ve işleme


    image_path = "C:/deneme/DYS/beypazari1.jpg"
    #image_path = "C:/deneme/DYS/beypazarihatali1.png"
    image = cv2.imread(image_path)

    if image is not None:
        image = preprocess_image(image, img_height, img_width)
        image = np.expand_dims(image, axis=0)  # (1, 128, 128, 3)
        prediction = model.predict(image)

        if prediction[0] > 0.8:
            print(prediction[0])
            print("DYS işareti bulundu.")
        else:
            print(prediction[0])
            print("DYS işareti bulunamadı.")
    else:
        print(f"Görüntü yüklenemedi: {image_path}")
