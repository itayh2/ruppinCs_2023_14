#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print('Final project - Agriculture Vision - presenting: Shahar Geula, Itay Hasid, Elad Shlishman')


# In[ ]:


#in case of need
# !pip install matplotlib
# !pip install tensorflow
# !pip install keras
# !pip install opencv-python-headless
# !pip install pillow
# !pip install tensorflow-datasets
# !pip install scipy
# !pip install scikit-learn
# !pip install tensorflow-gpu
# !pip install pandas
# # !pip install --upgrade notebook


# In[ ]:


#testing if there is a gpu
import tensorflow as tf
tf.config.list_physical_devices('GPU')


# In[ ]:


print("segmentation - with nir images")
import cv2 as cv
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import keras
import glob
from IPython.display import Image, display
import os
import PIL
import PIL.Image
import tensorflow_datasets as tfds


# In[ ]:


#import images 
train_img_dir = "/raid/itai.hasid/venv/Data/top4/rgb_images/rgb/"
train_nir_dir = "/raid/itai.hasid/venv/Data/top4/nir_images/nir/"
train_mask_dir = "/raid/itai.hasid/venv/Data/top4/mask_images/mask/"

img_list = os.listdir(train_img_dir)
nir_list = os.listdir(train_nir_dir)
msk_list = os.listdir(train_mask_dir)
num_images = len(os.listdir(train_img_dir))
print(num_images)
num_images_nir = len(os.listdir(train_nir_dir))
print(num_images_nir)
num_images_mask = len(os.listdir(train_mask_dir))
print(num_images_mask)

img_list.sort()
nir_list.sort()
msk_list.sort()
print(img_list[0])
print(nir_list[0])
print(msk_list[0])


# In[ ]:


#reads the the csv
labels_name = ['background','double_plant','drydown','endrow','nutrient_deficiency','planter_skip','storm_damage','water','waterway','weed_cluster']
dic = {}
counter =0
for i in labels_name:
    dic[counter] = i
    counter+=1
print(dic)

df_result = pd.read_csv('/raid/itai.hasid/venv/Data/train_merge.csv')
val_df_result = pd.read_csv('/raid/itai.hasid/venv/Data/val_merge.csv')


train_dataset_rgb = []
train_dataset_label_img = []
train_dataset_label = df_result['label']


# In[ ]:


#המחשה של כמות הדאטה של סט האימון במלואו
import numpy as np
import matplotlib.pyplot as plt
labels_number = [0,2,4,8,9]

# creating the dataset
data = {'background':0, 'drydown':0, 'nutrient_deficiency':0, 'waterway':0,'weed_cluster':0}

for i in df_result['label']:
    if i in labels_number and data[labels_name[i]]<3000:
        data[labels_name[i]] = data[labels_name[i]] +1

labels = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (18, 5))
 
for i, v in enumerate(values):
    plt.text(i, v, str(v), color='black', ha='center', fontweight='bold')

plt.bar(labels, values, color ='maroon',
        width = 0.4)
 
plt.xlabel("labels")
plt.ylabel("No. of..")
plt.title("number of images")
plt.show()


# In[ ]:


#plot the rgb, nir and mask for saniety check
img_num = random.randint(0, 800-1)

img_for_plot = cv.imread(train_img_dir+'/'+img_list[img_num], 1)
img_for_plot = cv.cvtColor(img_for_plot, cv.COLOR_BGR2RGB)

nir_img_for_plot = cv.imread(train_nir_dir+'/'+nir_list[img_num], 1)

mask_for_plot =cv.imread(train_mask_dir+'/'+msk_list[img_num], 0)
print(img_list[img_num])
print(nir_list[img_num])
print(msk_list[img_num])

plt.figure(figsize=(12, 8))
plt.subplot(131)
plt.imshow(img_for_plot)
img_list[img_num] = img_list[img_num].replace('.png', '')
number = df_result.index[df_result['image id'] == img_list[img_num]].tolist()
number = number[0]

plt.subplot(132)
plt.imshow(nir_img_for_plot, cmap='gray')
plt.title('nir image '  + dic[df_result['label'][number]])

print('label', dic[df_result['label'][number]])
plt.title(dic[df_result['label'][number]])
plt.subplot(133)
plt.imshow(mask_for_plot, cmap='gray')
plt.title('Mask '  + dic[df_result['label'][number]])
plt.show()


# In[ ]:


msk_list[35]
mask_for_plot =cv.imread(train_mask_dir+'/'+msk_list[35], 0)
mask_for_plot.max()


# In[ ]:


#create new tensor for each epoch by the batch_size
seed = 24
batch_size = 32
n_classes = 10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_data(img, nir, mask, num_class):
    # Normalize images
    img = img.astype('float32') / 255.0
    nir = nir.astype('float32') / 255.0
    
    # Concatenate images along the channel axis to be 4 channels
    input_image = np.concatenate((img, nir), axis=-1)

    # Convert mask to one-hot encoding of 10 classes 
    mask = to_categorical(mask, num_class)

    return img, nir, mask, input_image

# Define the generator.
# Apply appropriate data augmentation to images and masks.
def trainGenerator(train_img_path, train_nir_img_path, train_mask_path, num_class):
    img_data_gen_args = dict(horizontal_flip=True, vertical_flip=True)  
    mask_data_gen_args = dict(horizontal_flip=True, vertical_flip=True)

    image_datagen = ImageDataGenerator(**img_data_gen_args)
    nir_image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**mask_data_gen_args)

    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        class_mode=None,
        target_size=(512, 512),
        batch_size=batch_size,
        seed=seed)

    nir_generator = nir_image_datagen.flow_from_directory(
        train_nir_img_path,
        class_mode=None,
        target_size=(512, 512),
        color_mode='grayscale',
        batch_size=batch_size,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        class_mode=None,
        target_size=(512, 512),
        color_mode='grayscale',
        batch_size=batch_size,
        seed=seed)

    train_generator = zip(image_generator, nir_generator, mask_generator)

    for (img, nir, mask) in train_generator:
        img, nir, mask, newimg = preprocess_data(img, nir, mask, num_class)
        yield (newimg, mask)


# In[ ]:


#creat x and y of train
train_img_path = "/raid/itai.hasid/myproject/Data/top4/rgb_images/"
train_nir_path = "/raid/itai.hasid/myproject/Data/top4/nir_images/"
train_mask_path = "/raid/itai.hasid/myproject/Data/top4/mask_images/"
train_img_gen = trainGenerator(train_img_path,train_nir_path, train_mask_path, num_class=10)

#creat x and y of validation
val_img_path = "/raid/itai.hasid/myproject/Data/top4/val/val_rgb/"
val_nir_path = "/raid/itai.hasid/myproject/Data/top4/val/val_nir/"
val_mask_path = "/raid/itai.hasid/myproject/Data/top4/val/val_mask/"
val_img_gen = trainGenerator(val_img_path,val_nir_path, val_mask_path, num_class=10)


# In[ ]:


#sanity check of trainGenerator for train_img_gen
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

x, y = train_img_gen.__next__()
# print(np.unique(np.argmax(y[0],axis=2))[1])# מביא את המספר של הלייבל
print(x.shape)
for i in range(16):
    p = x[i]
    mask = np.argmax(y[i], axis=2)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(p[:, :, :3])  # RGB representation
    axes[0].set_title("RGB representation")
    axes[0].set_axis_off()  # Turn off the axis labels and ticks

    axes[1].imshow(cv.cvtColor(p[:, :, 3:], cv.COLOR_GRAY2BGR))  # NIR representation
    axes[1].set_title("NIR representation")
    axes[1].set_axis_off()  # Turn off the axis labels and ticks

    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title("Mask - "+ dic[np.unique(np.argmax(y[i],axis=2))[1]])
    axes[2].set_axis_off()  # Turn off the axis labels and ticks

    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()


# In[ ]:


#sanity check of trainGenerator for val_img_gen
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

x_val, y_val = val_img_gen.__next__()
for i in range(16):
    p = x_val[i]
    mask = np.argmax(y_val[i], axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(p[:, :, :3])  # RGB representation
    axes[0].set_title("RGB representation")
    axes[0].set_axis_off()  # Turn off the axis labels and ticks

    axes[1].imshow(cv.cvtColor(p[:, :, 3:], cv.COLOR_GRAY2BGR))  # NIR representation
    axes[1].set_title("NIR representation")
    axes[1].set_axis_off()  # Turn off the axis labels and ticks

    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title("Mask - " + dic[np.unique(np.argmax(y_val[i],axis=2))[1]])
    axes[2].set_axis_off()  # Turn off the axis labels and ticks

    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()


# In[ ]:


import tensorflow as tf

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 4

# Build the Unet model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

c1 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.BatchNormalization()(c1)
c1 = tf.keras.layers.Activation('relu')(c1)
c1 = tf.keras.layers.Dropout(0.3)(c1)
c1 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c1)
c1 = tf.keras.layers.BatchNormalization()(c1)
c1 = tf.keras.layers.Activation('relu')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.BatchNormalization()(c2)
c2 = tf.keras.layers.Activation('relu')(c2)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c2)
c2 = tf.keras.layers.BatchNormalization()(c2)
c2 = tf.keras.layers.Activation('relu')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.BatchNormalization()(c3)
c3 = tf.keras.layers.Activation('relu')(c3)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(c3)
c3 = tf.keras.layers.BatchNormalization()(c3)
c3 = tf.keras.layers.Activation('relu')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.BatchNormalization()(c4)
c4 = tf.keras.layers.Activation('relu')(c4)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same')(c4)
c4 = tf.keras.layers.BatchNormalization()(c4)
c4 = tf.keras.layers.Activation('relu')(c4)
p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

#the block 
c5 = tf.keras.layers.Conv2D(1024, (3, 3), kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.4)(c5)
c5 = tf.keras.layers.Conv2D(1024, (3, 3), kernel_initializer='he_normal', padding='same')(c5)

# Expansive path
u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.BatchNormalization()(c6)
c6 = tf.keras.layers.Activation('relu')(c6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same')(c6)
c6 = tf.keras.layers.BatchNormalization()(c6)
c6 = tf.keras.layers.Activation('relu')(c6)

u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.BatchNormalization()(c7)
c7 = tf.keras.layers.Activation('relu')(c7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(c7)
c7 = tf.keras.layers.BatchNormalization()(c7)
c7 = tf.keras.layers.Activation('relu')(c7)

u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.BatchNormalization()(c8)
c8 = tf.keras.layers.Activation('relu')(c8)
c8 = tf.keras.layers.Dropout(0.3)(c8)
c8 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c8)
c8 = tf.keras.layers.BatchNormalization()(c8)
c8 = tf.keras.layers.Activation('relu')(c8)

u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1])
c9 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.BatchNormalization()(c9)
c9 = tf.keras.layers.Activation('relu')(c9)
c9 = tf.keras.layers.Dropout(0.3)(c9)
c9 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c9)
c9 = tf.keras.layers.BatchNormalization()(c9)
c9 = tf.keras.layers.Activation('relu')(c9)

outputs = tf.keras.layers.Conv2D(10, (1, 1), activation='softmax')(c9)
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.OneHotMeanIoU(num_classes=10)])
model.summary()


# In[ ]:


#callbacks and set the steps batch
steps_per_epoch = 12000//batch_size
val_steps_per_epoch = 2400//batch_size
print(steps_per_epoch)
print(val_steps_per_epoch)
IMG_HEIGHT =512
IMG_WIDTH  = 512
IMG_CHANNELS = 3
n_classes=5

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_filepath = '/raid/itai.hasid/myproject/models/best_weights7.h5'
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)


# In[ ]:


#train the model
import time
start_time = time.time()  # Start the timer
history=model.fit(
          train_img_gen,
          steps_per_epoch=steps_per_epoch, 
          epochs=25,
          verbose=1,
          validation_data=val_img_gen,
          validation_steps=val_steps_per_epoch,
          callbacks=[checkpoint_callback])

end_time = time.time()  # Stop the timer
elapsed_time = end_time - start_time  # Calculate the elapsed time
print("Elapsed time: {:.2f} minutes".format(elapsed_time/60))


# In[ ]:


# Save the weights
model.save_weights('/raid/itai.hasid/myproject/models/top4_300x25.h5')


# In[ ]:


#upload the weights to the model without the need of train it

# Create a new model instance
#model = tf.keras.Model(inputs=[inputs],outputs=[outputs])

# Load the model architecture
model.load_weights('/raid/itai.hasid/myproject/models/top4_1500x30.h5')

# Compile the model again
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.OneHotMeanIoU(num_classes=10)])


# In[ ]:


#plot the training and validation IoU and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['one_hot_mean_io_u']
val_acc = history.history['val_one_hot_mean_io_u']

plt.plot(epochs, acc, 'y', label='Training one_hot_iou')
plt.plot(epochs, val_acc, 'r', label='Validation val_one_hot_iou')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()


# In[ ]:


#predict on the validation set
import pdb
from tensorflow.keras.metrics import OneHotMeanIoU

test_image_batch, test_mask_batch = val_img_gen.__next__()

# Convert categorical to integer for visualization and IoU calculation
test_mask_batch_argmax = np.argmax(test_mask_batch, axis=3) 
test_pred_batch = model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=3)


n_classes = 10
IOU_keras = OneHotMeanIoU(num_classes=n_classes)


for i in range(8):
    IOU_keras.update_state(test_mask_batch[i], test_pred_batch[i])  # Update the metric with one-hot encoded labels
    if IOU_keras.result().numpy() > 0.2 and len(np.unique(test_pred_batch_argmax[i])) > 1:
        # Compare predicted mask with ground truth mask
        correct_pixels = np.equal(test_pred_batch_argmax[i], test_mask_batch_argmax[i])
        num_correct = np.count_nonzero(correct_pixels)
        num_incorrect = np.prod(test_pred_batch_argmax[i].shape) - num_correct
        
        sum= num_correct+num_incorrect
        print('sum' ,sum)
        print("Number of correct pixels:", num_correct)
        print("Number of incorrect pixels:", num_incorrect)
        print("Number of correct over all pixels:", num_correct/sum)

        print("OneHotMeanIoU =", IOU_keras.result().numpy())
        
        num_true = np.count_nonzero(test_pred_batch_argmax[i])
        print("Number of pixels that are not zero:", num_true)
        print("Mean IoU =", IOU_keras.result().numpy())
        plt.figure(figsize=(12, 8))
        plt.subplot(231)
        plt.title('Testing Image')
        plt.imshow(test_image_batch[i])
        plt.subplot(232)
        plt.title('Testing Label: {0}'.format(dic[int(np.unique(np.argmax(test_mask_batch[i], axis=2))[1])]))
        plt.imshow(test_mask_batch_argmax[i])
        plt.subplot(233)
        print((np.unique(test_pred_batch_argmax[i])))
        if len(np.unique(test_pred_batch_argmax[i])) > 1:
            plt.title('Prediction on test image {0}'.format(dic[int(np.unique(test_pred_batch_argmax[i])[1])]))
            plt.imshow(test_pred_batch_argmax[i])
            plt.show()


# In[ ]:


# classification_report of each pixel
from sklearn.metrics import classification_report

label_names = ['background', 'double_plant', 'drydown', 'endrow', 'nutrient_deficiency', 'planter_skip', 'storm_damage', 'water', 'waterway', 'weed_cluster']

test_pred_batch = model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=3)
test_mask_batch_argmax = np.argmax(test_mask_batch, axis=3) 


# Flatten the predicted and ground truth masks
test_pred_flat = test_pred_batch_argmax.flatten()
test_mask_flat = test_mask_batch_argmax.flatten()

# Generate classification report
report = classification_report(test_mask_flat, test_pred_flat, labels=range(len(label_names)), zero_division=0, target_names=label_names)
print(report)


# In[ ]:


#confusion_matrix
from sklearn.metrics import confusion_matrix

label_names = ['background', 'drydown', 'nutrient_deficiency', 'waterway', 'weed_cluster']

# Flatten the predicted and ground truth masks
test_pred_flat = test_pred_batch_argmax.flatten()
test_mask_flat = test_mask_batch_argmax.flatten()

# Calculate the confusion matrix
confusion_mat = confusion_matrix(test_mask_flat, test_pred_flat)

# Calculate the true positives, false positives, and false negatives for each class
true_positives = np.diag(confusion_mat)
false_positives = np.sum(confusion_mat, axis=0) - true_positives
false_negatives = np.sum(confusion_mat, axis=1) - true_positives

# Calculate the IoU for each class
iou_scores = true_positives / (true_positives + false_positives + false_negatives)

# Print the mIoU score for each class
for i, class_name in enumerate(label_names):
    print(f"{class_name}: {iou_scores[i]}")


# In[ ]:


#calculate the Modified mIoU of all the 5 calsses (include the background)

label_names = ['background', 'drydown', 'nutrient_deficiency', 'waterway', 'weed_cluster']

# Flatten the predicted and ground truth masks
test_pred_flat = test_pred_batch_argmax.flatten()
test_mask_flat = test_mask_batch_argmax.flatten()

# Calculate the confusion matrix
confusion_mat = confusion_matrix(test_mask_flat, test_pred_flat)

# Calculate the intersection and union for each class
intersection = np.diag(confusion_mat)
union = np.sum(confusion_mat, axis=1) + np.sum(confusion_mat, axis=0) - intersection

# Calculate the IoU for each class
iou_scores = intersection / union

# Calculate the modified mIoU score by averaging the IoU scores
modified_miou = np.mean(iou_scores)

# Print the modified mIoU score
print("Modified mIoU:", modified_miou)


# In[ ]:





# In[ ]:


print('the test dataset - here we going to test the model on a test dataset ')


# In[ ]:


test_img_dir = "/raid/itai.hasid/myproject/Data/top4/test/rgb/rgb"
test_nir_dir = "/raid/itai.hasid/myproject/Data/top4/test/nir/nir"
test_mask_dir = "/raid/itai.hasid/myproject/Data/top4/test/mask/mask"

img_list = os.listdir(test_img_dir)
nir_list = os.listdir(test_nir_dir)
msk_list = os.listdir(test_mask_dir)

num_images = len(os.listdir(test_img_dir))
print(num_images)

num_images = len(os.listdir(test_nir_dir))
print(num_images)

num_images_mask = len(os.listdir(test_mask_dir))
print(num_images_mask)


img_list.sort()
nir_list.sort()
msk_list.sort()

print(img_list[0])
print(nir_list[0])
print(msk_list[0])


# In[ ]:


#creat x and y of validation - using the the above 
test_img_path = "/raid/itai.hasid/myproject/Data/top4/test/rgb"
test_nir_path = "/raid/itai.hasid/myproject/Data/top4/test/nir"
test_mask_path = "/raid/itai.hasid/myproject/Data/top4/test/mask"
test_img_gen = trainGenerator(test_img_path,test_nir_path, test_mask_path, num_class=10)


# In[ ]:


#sanity check
x, y = test_img_gen.__next__()

for i in range(3):
    p = x[i]
    mask = np.argmax(y[i], axis=2)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(p[:, :, :3])  # RGB representation
    axes[0].set_title("RGB representation")
    axes[0].set_axis_off()  # Turn off the axis labels and ticks

    axes[1].imshow(cv.cvtColor(p[:, :, 3:], cv.COLOR_GRAY2BGR))  # NIR representation
    axes[1].set_title("NIR representation")
    axes[1].set_axis_off()  # Turn off the axis labels and ticks

    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title("Mask - "+ dic[np.unique(np.argmax(y[i],axis=2))[1]])
    axes[2].set_axis_off()  # Turn off the axis labels and ticks

    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()


# In[ ]:


#predict on the test set
from tensorflow.keras.metrics import MeanIoU

test_image_batch, test_mask_batch = test_img_gen.__next__()

#Convert categorical to integer for visualization and IoU calculation
test_mask_batch_argmax = np.argmax(test_mask_batch, axis=3) 
test_pred_batch = model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=3)

n_classes = 10
IOU_keras = MeanIoU(num_classes=n_classes)  


for i in range(16):
#     print('the label',int(np.unique(np.argmax(test_mask_batch[i],axis=2))[1]))
    IOU_keras.update_state(test_pred_batch_argmax[i], test_mask_batch_argmax[i])
    if(IOU_keras.result().numpy()>0.2 and len(np.unique(test_pred_batch_argmax[i]))>1):
        print("Mean IoU =", IOU_keras.result().numpy())
        plt.figure(figsize=(12, 8))
        plt.subplot(231)
        plt.title('Testing Image')
        plt.imshow(test_image_batch[i])
        plt.subplot(232)

        plt.title('Testing Label: {0}'.format(dic[int(np.unique(np.argmax(test_mask_batch[i],axis=2))[1])]))
        plt.imshow(test_mask_batch_argmax[i])
        plt.subplot(233)
        print((np.unique(test_pred_batch_argmax[i])))
        if (len(np.unique(test_pred_batch_argmax[i]))>1):
            plt.title('Prediction on test image {0}'.format(dic[int(np.unique(test_pred_batch_argmax[i])[1])]))
            plt.imshow(test_pred_batch_argmax[i])
            plt.show()


# In[ ]:


# Save the weights
model.save_weights('/raid/itai.hasid/myproject/models/top4_300x25.h5')


# In[ ]:


print('the end!')

