# Speech_emotion_recognition
## Overview
Speech Emotion Recognition (SER) identifies human emotions from speech. This project uses Librosa and an MLPClassifier to classify emotions using the RAVDESS dataset.

## Dataset
**RAVDESS Dataset**: Contains 7356 audio files rated for emotional validity. Download it [here](https://drive.google.com/file/d/1wWsrN2Ep7x6lWqOXfr4rpKGYrJhWc8z7/view).

## Libraries Used
- **Librosa**: Audio feature extraction.
- **Soundfile**: Audio file handling.
- **Scikit-learn**: MLPClassifier and evaluation.

## Features Extracted
- **MFCC**: Mel Frequency Cepstral Coefficients.
- **Chroma**: 12 different pitch classes.
- **Mel Spectrogram**: Frequency representation of sound.

## Model and Accuracy
The MLPClassifier achieved an accuracy of 68.57%.

## Google Colab
The project runs on Google Colab for efficient processing and resource management.

## Project Workflow
1. **Setup Environment**: Install required libraries.
    ```python
    !pip install librosa scikit-learn soundfile
    ```
2. **Load the Dataset**: Mount Google Drive and extract the dataset.
    ```python
    from google.colab import drive
    drive.mount('/content/drive')

    import zipfile
    with zipfile.ZipFile('/content/drive/MyDrive/RAVDESS.zip', 'r') as zip_ref:
        zip_ref.extractall('/content/RAVDESS')
    ```
3. **Extract Features**: Use Librosa to extract MFCC, chroma, and mel features.
4. **Prepare Dataset**: Load audio files, extract features, and prepare the dataset.
5. **Train Model**: Initialize and train an MLPClassifier.
6. **Evaluate Model**: Calculate the accuracy of the model.
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Train the model
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, alpha=0.01, solver='adam', random_state=42)
    model.fit(X_train, y_train_encoded)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    ```

## Conclusion
This project demonstrates the use of Librosa and MLPClassifier for SER, achieving a 68.57%
