import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from model.model import VIT
import torch
import torchvision
import glob
import os
import cv2
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#matplotlib.use('Qt5Agg')
from PIL import Image
import csv
from ast import literal_eval
import json




Actions = {1: "applauding",
           2: "blowing bubbles",
           3: "brushing teeth",
           4: "cleaning the floor",
           5: "climbing",
           6: "cooking",
           7: "cutting trees",
           8: "cutting vegetables",
           9: "drinking",
           10: "feeding a horse",
           11: "fishing",
           12: "fixing a bike",
           13: "fixing a car",
           14: "gardening",
           15: "holding an umbrella",
           16: "jumping",
           17: "looking through a microscope",
           18: "looking through a telescope",
           19: "playing guitar",
           20: "playing violin",
           21: "pouring liquid",
           22: "pushing a_cart",
           23: "reading",
           24: "phoning",
           25: "riding a bike",
           26: "riding a horse",
           27: "rowing a boat",
           28: "running",
           29: "shooting an arrow",
           30: "smoking",
           31: "taking photos",
           32: "texting message",
           33: "throwing frisby",
           34: "using a computer",
           35: "walking the dog",
           36: "washing dishes",
           37: "watching TV",
           38: "waving hands",
           39: "writing on a board",
           40: "writing on a book"}


# creating csv file that stores the sport image with its best matching csv 
# (for runtime purposes, rather than having to run model on all art inages each time)
"""with open('art_tensors.csv', 'w', newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Art Image", "Pose Probabilities"])"""

name = []

# Open the file dialog to select an image
#image_path = "angel-reese.jpeg"
#image_path = "simone_biles.jpeg"
#image_path = "swimmer.jpeg"
#image_path = "jude.jpeg"
#image_path = "hockey_player.jpeg"
#image_path = "serena_williams.jpeg"
#image_path = "dribble_bron.jpeg"
#image_path = "Angel_Reese_LSU.jpg"
#image_path = "shacarri.jpeg"

#INPUT IMAGE HERE
image_path = "shacarri2.jpeg"

name.append(image_path)

"""sports_image = plt.imread(image_path)
plt.imshow(sports_image)
plt.axis('off')
plt.show()
"""
"""try:
    image_path = "angel-reese.jpeg"
    sports_image = cv2.imread('angel-reese.jpeg')
    #cv2.imshow(sports_image)
    cv2.imshow('Image', sports_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.show()
except Exception as e:
    print("An error occurred while loading or displaying the image:", e)
"""
img_path = name[len(name) - 1]
img = Image.open(img_path)
model = VIT("ViT-B_16", 224).to('cpu')
state_dict = torch.load('model_best_correct.pth', map_location=torch.device('cpu'))["state_dict"]
model.load_state_dict(state_dict)
model.eval()
img = torchvision.transforms.functional.resize(img, (224, 224))
tensor = torchvision.transforms.functional.to_tensor(img)
tensor = tensor.unsqueeze(dim=0)
out, _ = model(tensor)
_, predicted_label = torch.max(out, 1)
print(out) # Prints out log (?) probabilities of action classes
#print(out[0])
sport_image_np = out.detach().numpy()[0]
#print(out[0].item())
#print(out[0][predicted_label].item())
action = Actions[predicted_label.item() + 1]
print(action) # Prints out the most probable action class
#label_gender.config(text=f'Action: {action}')

#the code below is the cde we used to train the model on the art images
"""art_file_dataset = "/home/asiagray/Image-Action-Recognition/dataset_updated/**/**/*"
art_files = glob.glob(art_file_dataset)
euclid_distances = []
for image in art_files:
    if image is None:
        print("Skipping None image.")
        continue
    try:
        art_img = Image.open(image)
    except Exception as e:
        print(f"Error loading image {image}: {e}")
        continue
    model = VIT("ViT-B_16", 224).to('cpu')
    try:
        model = VIT("ViT-B_16", 224).to('cpu')
        state_dict = torch.load('model_best_correct.pth', map_location=torch.device('cpu'))["state_dict"]
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        continue
    try:
        art_img = torchvision.transforms.functional.resize(art_img, (224, 224))
        tensor = torchvision.transforms.functional.to_tensor(art_img)
        tensor = tensor.unsqueeze(dim=0)
        art_out, _ = model(tensor)
        if art_out is None:
            print("Skipping None output from model.")
            continue
        _, predicted_label = torch.max(art_out, 1)
        comparable_array = art_out.detach().numpy()[0]
        art_action = Actions[predicted_label.item() + 1]

        # Write results to CSV
        with open('art_tensors.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerow([image, comparable_array])
    except Exception as e:
        print(f"Error processing image {image}: {e}")
        continue"""

min_file_name = ""
min_euclid = float('inf')

with open("/home/asiagray/Image-Action-Recognition/art_tensors.csv", 'r', newline='') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        img, probs_str = row[0], row[1]
        #probs = literal_eval(probs_str)  
        #probs_array = np.array(probs, dtype=np.float32) 
        #probs = json.loads(probs_str.replace("", ",")) 
        #probs_array = np.array(probs, dtype=np.float32)  
        """probs_str_json = probs_str.replace(" ", ",")
        probs = json.loads(probs_str_json)
        probs_array = np.array(probs, dtype=np.float32)
        euclid = np.linalg.norm(sport_image_np - probs_array)"""
        """probs = [float(val) for val in probs_str.split()]
        probs_array = np.array(probs, dtype=np.float32)
        euclid = np.linalg.norm(sport_img_np - probs_array)"""
        probs_str_cleaned = probs_str.replace("[", "").replace("]", "").replace("\n", "")
        probs_list = probs_str_cleaned.split()
        probs = [float(val) for val in probs_list]
        probs_array = np.array(probs, dtype=np.float32)
        euclid = np.linalg.norm(sport_image_np - probs_array)

        if euclid < min_euclid:
            min_euclid = euclid
            min_file_name = img
            next(reader)
print("File name belonging to the minimum Euclidean distance:", min_file_name)

"""if min_file_name is not None:
    root = tk.Tk()
    image = Image.open(min_file_name)
    tk_image = ImageTk.PhotoImage(image)
    label = tk.Label(root, image=tk_image)
    label.pack()
    root.mainloop()"""

import http.server
import socketserver

PORT = 8016
DIRECTORY = os.path.dirname(min_file_name)

class ImageServer(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

with socketserver.TCPServer(("", PORT), ImageServer) as httpd:
    print(f"Image server running at http://localhost:{PORT}")
    print(f"Access the image at http://localhost:{PORT}/{os.path.basename(min_file_name)}")
    httpd.serve_forever()

"""if min_file_name is not None:
    #image_to_show = cv2.imread(min_file_name)
    if image_to_show is not None:
        #cv2.imshow('Image match', image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #image_to_show = Image.open(min_file_name)
        #image.show()
        image_to_show.save('output_image.jpg')
        print("Image saved as 'output_image.jpg'")
        plt.imshow(image_to_show)
        plt.axis('off')
        plt.show()
    else:
        print("Failed to load image:", min_file_name)
else:
    print("Image is None")"""
    

    





#min_distance = min(euclid_distances)
#min_index = euclid_distances.index(min_distance)

#best_match = art_files[min_index]

#best_match = art_files[min_index]
#print(best_match)
#best_match_image = plt.imread(best_match)
#print(best_match)




#best_match_image = cv2.imread(best_match)
"""if min_file_name is not None:
    #cv2.imshow('Image match', best_match_image)
    plt.imshow(min_file_name)
    plt.axis('off') 
    plt.show()
else:
    print("Image is none")"""

    #print("Image loaded successfully. Shape:", best_match_image.shape)
    #print(best_match)
#plt.imshow(best_match_image)
#print("hello")
#plt.axis('off')
#plt.show()


