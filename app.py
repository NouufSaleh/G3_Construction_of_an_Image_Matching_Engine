import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import streamlit as st
from scipy.spatial.distance import cdist

# Set the Streamlit page layout to wide' for better display
st.set_page_config(layout="wide")

class app:
    def __init__(self, image_dir, similar_image_size, selected_metric, threshold):
        # Initialize the ResNet18 model for feature extraction
        self.model = torchvision.models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.image_dir = image_dir
        self.similar_image_size = similar_image_size
        self.selected_metric = selected_metric
        self.model.eval()
        self.threshold = threshold
        # Define a transformation pipeline for pre-processing images
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create a dictionary to store activation values from the model
        self.activation = {}

        # Register a forward hook to capture activation values from the 'avgpool' layer
        self.model.avgpool.register_forward_hook(self.get_activation("avgpool"))

    # Define a method to generate image embeddings
    def generate_embeds(self):
        all_names = []
        all_vecs = None
        images = os.listdir(self.image_dir)

        with torch.no_grad():
            for i, file in enumerate(images):
                try:
                    # Open and pre-process the image
                    img = Image.open(self.image_dir + file)
                    img = self.transform(img)
                    
                    # Pass the image through the model and capture the activation values
                    out = self.model(img[None, ...])
                    vec = self.activation["avgpool"].numpy().squeeze()[None, ...]

                    # Stack image embeddings and store image names
                    if all_vecs is None:
                        all_vecs = vec
                    else:
                        all_vecs = np.vstack([all_vecs, vec])
                    all_names.append(file)
                except:
                    continue

        # print("Embeddings Created Successfully..!")
        return all_vecs, all_names

    # Define a hook to capture activation values from the model
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    # Define the main application logic
    def run(self):
        # Streamlit UI elements for application layout
        st.markdown('<center><h1 style="margin-left:0">Image Search Engine</h1></center>', unsafe_allow_html=True)
        st.markdown('<br/>', unsafe_allow_html=True)

        _, ucol2, _ = st.columns(3) # Upload button columns
        _, fcol2, _ = st.columns(3) # Query image place holder

        st.markdown("<br/>", unsafe_allow_html=True)
        bcols = st.columns(11) # Show image button place holder
        uploaded_image = ucol2.file_uploader("Upload a query image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            query_img = Image.open(uploaded_image)
            query_img = query_img.resize((512, 512))
            fcol2.image(query_img, caption="Query Image", use_column_width=True)

            with torch.no_grad():
                try:
                    # Pre-process the query image and capture its activation values
                    query_img = self.transform(query_img)
                    out = self.model(query_img[None, ...])
                    target_vec = self.activation["avgpool"].numpy().squeeze()[None, ...]
                except Exception as e:
                    st.error(f"An error occurred while processing the query image: {e}")

            if bcols[5].button("Similar Images"):
                # Generate embeddings each time
                
                with bcols[5].empty(), st.spinner("Searching.."):
                    self.vecs, self.names = self.generate_embeds()
                
                st.markdown('<center><h3>Similar Images</h3></center><br/>', unsafe_allow_html=True)
                
                # Calculate the similarity scores and retrieve top 5 similar images
                sdistance = cdist(target_vec, self.vecs, metric=self.selected_metric)
                top5 = sdistance.argsort()[0][:5] # Top 5 image indices
                similar = [i for i in top5 if sdistance[0][i] < self.threshold] # Image indices less than threshold
                if len(similar) > 0:
                    columns = st.columns(len(similar))

                    for i, top_idx in enumerate(similar):
                        similar_image = Image.open(f"{self.image_dir}" + self.names[top_idx])
                        similar_image = similar_image.resize(self.similar_image_size)

                        distance = cdist(target_vec, self.vecs[top_idx][None, ...], metric=self.selected_metric)

                        caption = "Distance: {:.2f}".format(distance[0][0])

                        if i < len(columns):
                            columns[i].image(similar_image)
                            columns[i].markdown(f"<center>{caption}</center>", unsafe_allow_html=True)
                else:
                    st.warning("Sorry, No Related Images Found..!")


if __name__ == "__main__":
    # Define configuration and initialize the 'app' class
    image_dir = "./Images/"
    similar_image_size = (400, 400)
    selected_metric = "euclidean"
    threshold = 20.0

    app = app(image_dir, similar_image_size, selected_metric, threshold)
    
    # Run the application
    app.run()
