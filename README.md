# Image Search Engine

This is a Streamlit application that allows us to find similar images based on a query image. The application utilizes a pre-trained ResNet-18 model for feature extraction and computes similarity scores to identify similar images from a given directory of images.

# Pre-requisites

Before running the application, we have to install the following dependencies:

- Python
- PyTorch
- torchvision
- Pillow (PIL)
- numpy
- scipy
- streamlit

We can install these dependencies using `pip`.


# Usage

1. Download the code.

2. Organize your images in a directory and set the `image_dir` variable in the code to the path of this directory.

3. Set the desired configuration options such as `similar_image_size`, `selected_metric`, and `threshold` according to your requirements.

4. Run the application:

   ```bash
   streamlit run app.py
   ```

5. In the Streamlit app, we can upload a query image by clicking the "Upload a query image" button.

6. The application will process the query image and display the most similar images based on the chosen similarity metric and threshold.

# Configuration Options

- `image_dir`: The directory containing the images we want to search for similarities.

- `similar_image_size`: The size at which similar images are displayed in the application. Adjust this value to your preference.

- `selected_metric`: The similarity metric used to calculate the distance between embeddings. The default is "euclidean," but we can choose other metrics such as "cosine."

- `threshold`: The threshold value for considering an image as similar. Images with a distance less than this threshold are considered similar.

