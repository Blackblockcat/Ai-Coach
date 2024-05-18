How to Use the Model
1-Preprocessing Data:

Ensure you have a directory named data containing subdirectories for each gesture class, with images of hand gestures.
Run the code provided in the first part to preprocess the images and save the landmark data into a file named data.pickle.

2-Training the Model:
Load the data.pickle file generated in the previous step.
Run the code provided in the second part to train the neural network model using the preprocessed data.
The model will be evaluated on test data, and the test accuracy will be displayed.
The trained model will be saved as model.h5.

3-Using the Trained Model:

Ensure you have a webcam connected and functional.
Load the trained model from the model.h5 file.
Run the code provided in the third part to start real-time gesture recognition.
The webcam will capture frames, extract pose landmarks, and predict hand gestures in real-time.
