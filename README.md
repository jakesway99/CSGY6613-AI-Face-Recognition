# Facial Recognition with Deep Learning and Milvus
This project is private, as it was a class project. Please reach out if you would like to see the actual repository. 

## Part 1 - Baseline Performance

### Summary
This project uses PyTorch and Milvus for facial recognition on the LFW (Labeled Faces in the Wild) dataset. The initial system uses ResNet-50 as a feature extractor, turning images into vectors for similarity search. Results from this baseline system provided insights for improvements in the second part of the project.

### Dataset
The LFW dataset contains 13233 images of 1680 unique individuals. It is a small dataset in comparison to other large-scale image datasets.

### Model
ResNet-50 was used for image feature extraction. This deep convolutional neural network uses skip connections (residual blocks) to combat issues like vanishing gradients, allowing for the creation of deeper networks for better performance.

### Preparations
Before vector search, a series of steps were performed:

1. Download the LFW dataset, and unpack the data files.
2. Implement a PyTorch dataset class to manage image extraction and file name tracking.
3. Download the pretrained ResNet50 model.
4. Start Milvus in Docker and connect to the server.
5. Create a Milvus schema named "lfw_base" for storing image vectors.

### Vector Collection & Testing
ResNet-50 was used to generate feature vectors from images which were then inserted into Milvus. Similarity search results were evaluated manually by visual inspection of the image results.

## Part 2 - Improvement Using FaceNet

### Model
In this phase, FaceNet and multi-task cascaded convolutional neural networks (MTCNNs) were used. The MTCNNs detect and crop faces from images, while FaceNet generates feature vectors from the cropped faces. FaceNet uses triplet loss for calculating loss, which helps in differentiating between same and different identities. It is an improvement over siamese networks, taking 3 inputs (anchor, positive, negative) to calculate comparative loss.

### Preparations
The same LFW dataset was used as in Part 1. Images were fed directly to FaceNet without any transformations. Each image from the dataset also returned an index for future referencing.

### Model Configuration
The MTCNN and FaceNet were initialized with FaceNet using the pretrained vggface2 dataset. A Milvus schema was created with the same specifications as in Part 1, but with a feature vector dimension of 512, matching FaceNet's embeddings.

### Embeddings
MTCNN was used first for face detection and noise reduction. The output was then fed into FaceNet to generate embeddings. These embeddings were transformed into vectors and stored in Milvus.

### Testing
Testing was carried out in the same manner as in Part 1. After getting the search results, the indexes were used to fetch the actual images from LFWDataset for visual inspection.

### Conclusion
The systems were built using PyTorch and Milvus, and tested on the LFW dataset. The second part of the project offered improvements in the search results, demonstrating the efficacy of specialized neural networks (like FaceNet) for tasks like facial recognition.

