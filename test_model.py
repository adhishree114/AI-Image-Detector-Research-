import tensorflow as tf
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix

model = tf.keras.models.load_model("cnn_model.keras")

class_names = ["Real", "AI"]
eval_dir = "EvaluationData"

#labels
y_true = []
y_pred = []
misclassified_images = []
classified_images = []

output_dir = "MisclassifiedImages"
output_dir2 = "ClassifiedImages"

for category_index, category in enumerate(class_names):
    category_path = os.path.join(eval_dir, category)

    for filename in os.listdir(category_path):
        image_path = os.path.join(category_path, filename)

        # load and preprocess image
        test_image = load_img(image_path, target_size = (512, 512))
        test_image_array = img_to_array(test_image)
        test_image_array = tf.expand_dims(test_image_array, axis=0)

        #make prediction
        prediction = model.predict(test_image_array)
        #predicted_class = np.argmax(prediction[0])
        predicted_class = int(prediction[0][1] > 0.5)

        #store results
        y_true.append(category_index) #actual label (0 for Real, 1 for AI)
        y_pred.append(predicted_class) # Predicted label

        if predicted_class != category_index:
            new_filename = f"{class_names[category_index]}_as_{class_names[predicted_class]}_{filename}"
            new_path = os.path.join(output_dir, new_filename)
            shutil.copy(image_path, new_path)
            misclassified_images.append(new_path)

        if predicted_class == category_index:
            newName = f"{class_names[category_index]}_as_{class_names[predicted_class]}_{filename}"
            newPath = os.path.join(output_dir2, newName)
            shutil.copy(image_path, newPath)
            classified_images.append(newPath)


print(f"Saved {len(misclassified_images)} misclassified images to '{output_dir}' folder.")
print(f"Saved {len(classified_images)} classified images to '{output_dir2}' folder.")




#confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

#plotting it
plt.figure(figsize=(5,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()



#ROC Curve




