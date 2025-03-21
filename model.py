import tensorflow as tf 

import PIL

from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models
import os
#import matplotlib.pyplot as plt
import sys

# Set the path to the dataset directory
dataset_dir = "TrainingData"

# Set the path to the evaluation directory 
#eval_dataset_dir = "EvaluationData"


# Load training data and evaluation data
# 80% goes to training dataset, 20% goes to validation dataset
train_dataset = image_dataset_from_directory( # trains the model
    dataset_dir,
    validation_split = 0.2, #20% of data for validation, rest for training
    subset = "training",
    seed = 123, # shuffles the data
    image_size = (512, 512), # image resized to 512 x 512 pixels
    batch_size = 32, # processess 32 images at a time
    label_mode = 'categorical',
    class_names = ["Real", "AI"]
)

validation_dataset = image_dataset_from_directory( # evaluate model's performance during training
    dataset_dir,
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    image_size = (512, 512),
    batch_size = 32,
    label_mode = 'categorical',
    class_names = ["Real", "AI"]
)

# evaluation_dataset = image_dataset_from_directory(
#     eval_dataset_dir,
#     image_size=(512,512),
#     batch_size=32,
#     label_mode='int'


#)

# Define the model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(512, 512, 3)), # rescales from [0,255] to [0,1], expects 512 x512 with 3 channels(RGB)
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'), # 32 filters, 3 x 3 kernel
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)), # downsamples by taking in 2x2 window, focusing on important features, number of filters increase
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(), # converts 2D feature into a 1D Vector 
    layers.Dense(128, activation='relu'), # transforms learned features from before into high-level representation, produces 128 outputs representing a different high-level feature like eyes, nose, mouth
    layers.Dense(2, activation='softmax')  # Adjust output layer based on number of classes, right now 2 classes: real or fake images
])

# Compile
model.compile(optimizer='adam', # adjusts the model's weight (parameters) during training 
              loss='categorical_crossentropy', # how far prediction is from true labels
              metrics=['accuracy'])

# Train, passes through the entire dataset 10 times
model.fit(train_dataset, validation_data=validation_dataset, epochs=10)

model.save("cnn_model.keras")

#print(train_dataset.class_names)

# Evaluate the model, print out accuracy
# test_loss, test_acc = model.evaluate(validation_dataset, verbose=2)
# print(f'\nValidation accuracy: {test_acc}')

# eval_loss,eval_acc = model.evaluate(eval_dataset, verbose=2)
# print(f'\Evaluation accuracy: {eval_acc}')


# Make predictions
# for images, labels in validation_dataset.take(1):
#     predictions = model.predict(images)
#     print(f'Predicted label: {tf.argmax(predictions[0])}')
#     print(f'True label: {labels[0]}')

# for images,labels in evaluation_dataset.take(1):
#     predictions = model.predict(images)
#     print(f'Predicted label: {tf.argmax(predictions[0])}')
#     print(f'True label: {labels[0]}')

#----------------------------------------------------------

# image_path = "/Users/adhishreeviti/Dr. Davis Research Project/EvaluationData/Real/E4.jpg"

# test_image = load_img(image_path, target_size=(512, 512))
# test_image_array = img_to_array(test_image) # Convert the image to a numpy array
# test_image_array = test_image_array / 255.0 # Rescale the image (to match the model's rescaling layer)

# # Add a batch dimension (models expect a batch of inputs)
# test_image_array = tf.expand_dims(test_image_array, axis=0)

# prediction = model.predict(test_image_array)
# predicted_class = tf.argmax(prediction[0]).numpy()  # Gets the index of max value in the prediction array

# class_names = ["Real", "AI"]  
# print(f"Predicted class: {class_names[predicted_class]}") #0 is real, 1 is AI
 
