import argparse
import pathlib
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import tkinter as tk

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Arguments
parser = argparse.ArgumentParser(description='Collect Static Gesture Samples')
parser.add_argument('--output_folder', default='./data', type=str, help='output folder to write keypoint sequences')
parser.add_argument('--name', default='pointing', type=str, help='filename: gesturename_positive/negative')

parser.add_argument('--fps',    default=30.0, type=float, help='frame per second (FPS) of webcam')
parser.add_argument('--width',  default=640, type=int, help='frame width of webcam')
parser.add_argument('--height', default=480, type=int, help='frame height of webcam')
args = parser.parse_args()

def main(args):
    keyboard = Controller()
    

    positive_samples = pd.read_csv(pathlib.Path(args.output_folder, 'swipe_positive.csv'), header=None).iloc[:,1:-1].to_numpy()
    negative_samples = pd.read_csv(pathlib.Path(args.output_folder, 'swipe_negative.csv'), header=None).iloc[:,1:-1].to_numpy()

    X = np.concatenate((positive_samples, negative_samples), axis=0)
    y = np.concatenate((np.ones(shape=positive_samples.shape[0]), 
                        -np.ones(shape=negative_samples.shape[0])), axis=0)

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X, y)
    #print('Accuracy=', accuracy_score(clf.predict(X), y))


    # Get webcam frame size and screen size
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    screen_size = (screen_height, screen_width)
    frame_size=(480, 640)
    # Create a named window
    cv2.namedWindow('Static Gesture / Test')        
    cv2.moveWindow('Static Gesture / Test', 0, 0)

    # For webcam input:
    video_capture = cv2.VideoCapture(0)


    movement = []
    recognized = False
    counter = 0


    with mp.solutions.hands.Hands() as hands:
        while video_capture.isOpened():
            success, image = video_capture.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # BGR-->RGB (models are trained on RGB frames)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # visualize
                    mp.solutions.drawing_utils.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style())

                    # deploy trained gesture recognizer
                    landmarks = np.array([[l.x,l.y,l.z] for l in hand_landmarks.landmark])
                    label = 'Right' if results.multi_handedness[index].classification[0].label=='Left' else 'Left'
                    predicted_class = int(clf.predict(landmarks.reshape(1, 63))[0])

                    # X Coordinates
                    #print(landmarks[8,0])

                    if recognized == True and counter == 10:
                        recognized = False
                        counter = 0
                    #print(f'recognized: {recognized}')
                    #print(f'counter: {counter}')

                    if recognized == False:
                        if predicted_class==1:
                            if(len(movement) == 20):
                                movement = []
                            else:
                                movement.append(landmarks[20,0])
                                #print(f'Increasing: {np.all(movement[1:] >= movement[:-1])}' )
                                if(np.all(movement[1:] >= movement[:-1]) and np.mean(movement) > 0.6):
                                    keyboard.press(Key.right)
                                    recognized = True
                                    print("Gesture recognized: swipe right")
                                if(np.all(movement[1:] <= movement[:-1]) and np.mean(movement) < 0.4):
                                    keyboard.press(Key.left)
                                    recognized = True
                                    print("Gesture recognized: swipe left")
                        else:
                            movement = []
                    else:
                        counter += 1

            # View and write video (hand keypoint visualization and frame numbers)
            cv2.imshow('Static Gesture / Test', image)

            # Break when pressed quit
            if cv2.waitKey(5) & 0xFF == 27:
                break

        video_capture.release()



if __name__ == '__main__':
    main(args)