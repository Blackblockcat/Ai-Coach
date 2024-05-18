How to Use the Model
1. Preprocess the Data
Description: Preprocess the hand gesture data by extracting pose landmarks from images stored in the ./data directory using MediaPipe. The processed data will be saved in a pickle file named data.pickle.
Steps:
Ensure you have a ./data directory with subdirectories named according to different gesture classes, containing relevant images.
Run the first part of the provided code to process these images and save the landmark data.
2. Train the Model
Description: Train the neural network model using the preprocessed hand gesture data. The model will learn to recognize hand gestures based on the extracted pose landmarks.
Steps:
Load the data.pickle file containing the preprocessed data.
Run the second part of the provided code to train the model. This includes splitting the data into training and testing sets, building the model, and training it.
The model will be evaluated on the test data to assess its performance, and the test accuracy will be displayed.
The trained model will be saved to a file named model.h5.
3. Use the Trained Model
Description: Use the trained model for real-time hand gesture recognition. The webcam will capture frames, detect pose landmarks, and predict the hand gestures in real-time.
Steps:
Ensure you have a webcam connected and functional.
Load the trained model from the model.h5 file.
Run the third part of the provided code to start real-time gesture recognition. The webcam will capture frames, process them to extract pose landmarks, and use the model to predict and display the gesture.
