import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from keras.applications import EfficientNetB0
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_images(directory):
    images = []
    labels = []
    for label, category in enumerate(os.listdir(directory)):
        category_path = os.path.join(directory, category)
        for filename in os.listdir(category_path):
            img = image.load_img(os.path.join(category_path, filename), target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            images.append(img_array)
            labels.append(label)
    return np.vstack(images), np.array(labels)

image_directory = r"C:\Users\Y HARSHITHA\Desktop\project\myenv\brain_mri_scan_images"
images, labels = load_images(image_directory)

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape= (224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
feature_extractor = Model(inputs=base_model.input, outputs=x)

features = feature_extractor.predict(images)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test,y_pred)
print("Accuracy score of random forest model is",rf_accuracy*100)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)
nb_accuracy = accuracy_score(y_test,y_pred)
print("Accuracy score of guassianNB model is",nb_accuracy*100)

svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
sv_accuracy = accuracy_score(y_test,y_pred)
print("Accuracy score of svm model is",sv_accuracy*100)

lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)
y_pred = lr_classifier.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score of logistic regression model is", lr_accuracy*100)

accuracies = {'Random Forest': rf_accuracy, 'Naive Bayes': nb_accuracy, 'SVM': sv_accuracy, 'Logistic Regression': lr_accuracy}
plt.bar(accuracies.keys(), accuracies.values())
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Classifiers')
plt.savefig('bar_graph.png')
plt.show()


def preprocess_image(image):
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 
    return img_array

label = {
        0: 'alzheimer',
        1: "brainTumor",
        2: "epilepsy",
        3: 'normal',
        4: 'Parkinson'
    }

def predict(image):
    new_image = preprocess_image(image)
    new_features = feature_extractor.predict(new_image)
    predicted_class = rf_classifier.predict(new_features)[0]
    return predicted_class

import streamlit as st
from PIL import Image

st.title("Brain Diseases Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    predicted_class = predict(image)
    st.write("Predicted Disease Class:", label[predicted_class])