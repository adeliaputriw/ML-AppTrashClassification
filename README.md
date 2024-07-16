# ♻️ **Waste Classification with Deep Learning** 🌍

## 🚀 **Deskripsi**

Skrip ini menggunakan TensorFlow dan Keras untuk mengklasifikasikan gambar sampah menjadi dua kategori: Organic (O) dan Recyclable (R). Dengan menggunakan dataset dari Kaggle, kita membangun dan melatih model neural network untuk klasifikasi gambar dan menilai performa model dengan metrik evaluasi.

## 🛠️ **Langkah-langkah**

1. **🔧 Setup Environment**

   Install TensorFlow dan Kaggle API:
   ```python
   import tensorflow as tf
   print(tf.__version__)  # Menampilkan versi TensorFlow
   
   !pip install kaggle
   
2. **📂 Unduh Dataset**

   Upload file kaggle.json untuk autentikasi dan unduh dataset:
   ```python
   from google.colab import files
   files.upload()  # Unggah kaggle.json
      
   # Setup Kaggle API
   !mkdir ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
      
   # Unduh dataset
   dataset_name = 'techsash/waste-classification-data'
   !kaggle datasets download -d {dataset_name}
   ```
3.  **🗂️ Ekstrak Dataset**

   Ekstrak file zip yang telah diunduh:
   ```python
   import zipfile, os

  local_zip = 'waste-classification-data.zip'
  if not os.path.exists('dataset'):
      os.mkdir('dataset')
  zip_ref = zipfile.ZipFile(local_zip, 'r')
  zip_ref.extractall('/dataset')
  zip_ref.close()
  ```
4. **📊 Visualisasi Data**
   Tampilkan distribusi kelas dan beberapa gambar acak:
   ```python
      import matplotlib.pyplot as plt
      import numpy as np
      from tqdm import tqdm
      import cv2
      
      # Load dan visualisasi data
      x_data = []
      y_data = []
      for category in glob(train_path+'/*'):
          for file in tqdm(glob(category+'/*')):
              img_array = cv2.imread(file)
              img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
              x_data.append(img_array)
              y_data.append(category.split("/")[-1])
      
      data = pd.DataFrame({'image': x_data, 'label': y_data})
      
      # Pie chart distribusi label
      colors = ['#a0d157', '#c48bb8']
      plt.pie(data.label.value_counts(), startangle=90, explode=[0.05, 0.05], autopct='%0.2f%%', labels=['Organic', 'Recyclable'], colors=colors, radius=2)
      plt.title("Distribusi Kelas Sampah ♻️", fontsize=18)
      plt.show()
      
      # Visualisasi beberapa gambar acak
      plt.figure(figsize=(20,15))
      for i in range(9):
          plt.subplot(4,3,(i%12)+1)
          index = np.random.randint(15000)
          plt.title(f'This image is of {data.label[index]}', fontdict={'size':20,'weight':'bold'})
          plt.imshow(data.image[index])
          plt.tight_layout()
      plt.suptitle("Contoh Gambar Sampah 📸", fontsize=22)
      plt.show()
   ```
4. **🧩 Bangun Model**
   Buat dan latih model neural network:
   ```python
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
    from keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import plot_model
    
    # Definisi model
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(224, 224, 3)),
        Activation("relu"),
        MaxPooling2D(),
        Conv2D(64, (3, 3)),
        Activation("relu"),
        MaxPooling2D(),
        Conv2D(128, (3, 3)),
        Activation("relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(256),
        Activation("relu"),
        Dropout(0.5),
        Dense(64),
        Activation("relu"),
        Dropout(0.5),
        Dense(numberOfClass),
        Activation("sigmoid")
    ])
    
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    # Visualisasi model
    plot_model(model, show_shapes=True, show_layer_names=True)
    plt.title("Arsitektur Model 🏗️", fontsize=22)
    plt.show()
    ```
6. **🚂 Latih Model**
   ```python
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=256,
        color_mode="rgb",
        class_mode="categorical"
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=256,
        color_mode="rgb",
        class_mode="categorical"
    )
    
    hist = model.fit_generator(
        generator=train_generator,
        epochs=10,
        validation_data=test_generator
    )
   ```
7. **🔍 Evaluasi Model**
   ```python
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    # Evaluasi model
    test_loss, test_acc = model.evaluate(test_generator, verbose=1)
    print(f"Test accuracy: {test_acc:.2f}")
    
    # Prediksi dan laporan
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = test_generator.classes
    
    # Laporan Klasifikasi
    class_labels = list(test_generator.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print("Laporan Klasifikasi 📊")
    print(report)
    
    # Matriks Kebingungan
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Matriks Kebingungan 🔍')
    plt.show()
    
    # Visualisasi Laporan Klasifikasi
    report_data = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    report_df = pd.DataFrame(report_data).transpose()
    plt.figure(figsize=(10, 8))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues')
    plt.title('Laporan Klasifikasi 🔎')
    plt.show()
    ```
   

   
