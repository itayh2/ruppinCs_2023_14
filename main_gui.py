import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import tensorflow as tf

labels_name = ['background', 'double_plant', 'drydown', 'endrow', 'nutrient_deficiency', 'planter_skip',
               'storm_damage', 'water', 'waterway', 'weed_cluster']
dic = {counter: label for counter, label in enumerate(labels_name)}
# print(dic)

# Create a color map for labels
label_colors = {
    0: [0, 0, 0],       # Class 0 - Black color
    1: [255, 255, 255], # Class 1 - White color
    2: [165, 42, 42],   # Class 2 - Brown color - drydown
    3: [255, 255, 255], # Class 3 - White color
    4: [255, 255, 0],   # Class 4 - Yellow color - nutrient_deficiency
    5: [255, 255, 255], # Class 5 - White color
    6: [255, 255, 255], # Class 6 - White color
    7: [255, 255, 255], # Class 7 - White color
    8: [0, 0, 255],     # Class 8 - Blue color - waterway
    9: [0, 128, 0],     # Class 9 - Green color - weed_cluster
}

def preprocess_data(img, nir):
    print('preprocess_data')
    print(img.shape)
    print(nir.shape)

    # Normalize images
    img = img.astype('float32') / 255.0
    nir = nir.astype('float32') / 255.0
    # Stack NIR channel three times
    nir_stacked = np.stack((nir,) * 1, axis=-1)
    print(nir_stacked.shape)
    # Concatenate images along the channel axis
    input_image = np.concatenate((img, nir_stacked), axis=-1)
    print(input_image.shape)
    return input_image


def button1_click():
    global rgb_image
    image_path = filedialog.askopenfilename(initialdir=r'C:\Data\Agriculture-Vision-2021\top4\rgb_images\rgb',filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if image_path:
        image_rgb = Image.open(image_path)
        image_rgb = image_rgb.resize((512, 512), Image.LANCZOS)

        image_rgb_show = image_rgb.resize((150, 150), Image.LANCZOS)
        rgb_image = np.array(image_rgb)
        tk_image_rgb = ImageTk.PhotoImage(image_rgb_show)
        label1.config(image=tk_image_rgb)
        label1.image = tk_image_rgb


def button2_click():
    global nir_image
    image_path = filedialog.askopenfilename(initialdir=r'C:\Data\Agriculture-Vision-2021\top4\nir_images\nir' ,filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if image_path:
        image_nir = Image.open(image_path)
        image_nir = image_nir.resize((512, 512), Image.LANCZOS)

        image_nir_show = image_nir.resize((150, 150), Image.LANCZOS)
        nir_image = np.array(image_nir)
        tk_image_nir = ImageTk.PhotoImage(image_nir_show)
        label2.config(image=tk_image_nir)
        label2.image = tk_image_nir


def button3_click():
    global processed_image, rgb_image, nir_image
    processed_image = preprocess_data(rgb_image, nir_image)
    processed_image_show = Image.fromarray((processed_image * 255).astype(np.uint8))
    processed_image_show = processed_image_show.resize((150, 150), Image.LANCZOS)
    tk_processed_image = ImageTk.PhotoImage(processed_image_show)
    label3.config(image=tk_processed_image)
    label3.image = tk_processed_image


def button4_click():
    print('---------------------------------------------')
    global processed_image, model
    result = ''
    predictions = model.predict(processed_image[np.newaxis, ...])
    test_pred_argmax = np.argmax(predictions, axis=3)
    print('unique:', np.unique(test_pred_argmax[0]))
    if len(np.unique(test_pred_argmax[0])) > 1:
        result = dic[int(np.unique(test_pred_argmax[0])[1])]
        none_zero = np.count_nonzero(test_pred_argmax)
        print('none_zero pixels:', none_zero)

        # Create a color image from the predicted labels
        color_image = np.zeros((test_pred_argmax.shape[1], test_pred_argmax.shape[2], 3), dtype=np.uint8)
        for label, color in label_colors.items():
            color_image[test_pred_argmax[0] == label] = color

        # Convert the color image array to PIL Image object
        predicted_image = Image.fromarray(color_image)

        # Resize the predicted image to 200x200 pixels
        predicted_image = predicted_image.resize((250, 250), Image.LANCZOS)

        # Convert the PIL Image object to ImageTk format
        tk_predicted_image = ImageTk.PhotoImage(predicted_image)

        # Update the label to display the predicted image
        label4.config(image=tk_predicted_image)
        label4.image = tk_predicted_image

        # Update the prediction label
        prediction_label.config(text=f"Prediction: {result}")
    else:
        # Clear the labels
        label4.config(image="")
        label4.image = None
        prediction_label.config(text="Prediction: None")


# Load the model architecture
model = tf.keras.models.load_model(r"C:\Data\model\model.h5")
print(model.summary())
# Load the model weights
model.load_weights(r"C:\Data\model\top4.h5")

# Create a new Tkinter window
window = tk.Tk()
window.title("Image Boxes")
window.geometry("800x800")  # Set the window size (width x height)

# Create a Canvas widget to hold the image frame
canvas = tk.Canvas(window)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create a Scrollbar widget
scrollbar = tk.Scrollbar(window, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure the canvas scrolling with the Scrollbar
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Create a frame to contain the image boxes
image_frame = tk.Frame(canvas)
image_frame.pack(fill=tk.BOTH, padx=10, pady=10)

# Set the main title
title_label = tk.Label(image_frame, text="Agricalture Function", font=("Helvetica", 16))
title_label.pack(pady=10)

# Create a frame to hold the image labels and buttons
image_controls_frame = tk.Frame(image_frame)
image_controls_frame.pack(side=tk.TOP, pady=10)

# Create a label to display the first image
rgb_image = None
label1 = tk.Label(image_controls_frame)
label1.pack(side=tk.LEFT, padx=10)

# Create the first button to load the first image
button1 = tk.Button(image_controls_frame, text="Load RGB Image", command=button1_click)
button1.pack(side=tk.LEFT, padx=10)

# Create a label to display the second image
nir_image = None
label2 = tk.Label(image_controls_frame)
label2.pack(side=tk.LEFT, padx=10)

# Create the second button to load the second image
button2 = tk.Button(image_controls_frame, text="Load NIR Image", command=button2_click)
button2.pack(side=tk.LEFT, padx=10)

# Create the third button to preprocess the images
button3 = tk.Button(image_frame, text="Preprocess Images", command=button3_click)
button3.pack(pady=10)

# Create a label to display the processed image
label3 = tk.Label(image_frame)
label3.pack(padx=10)

# Create the fourth button to send the processed image to the prediction model
button4 = tk.Button(image_frame, text="Predict", command=button4_click)
button4.pack(pady=10)

# Create a label for the prediction result
result = ''
prediction_label = tk.Label(image_frame, text=f"Prediction: {result}", font=("Helvetica", 14))
prediction_label.pack(side=tk.TOP, padx=10, pady=10)

# Create a label to display the predicted image
label4 = tk.Label(image_frame)
label4.pack(padx=10)

# Start the Tkinter event loop
window.mainloop()
