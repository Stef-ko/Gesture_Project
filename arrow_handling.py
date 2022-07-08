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
parser.add_argument('--name', default='pointRight', type=str, help='filename: gesturename_positive/negative')

parser.add_argument('--fps', default=30.0, type=float, help='frame per second (FPS) of webcam')
parser.add_argument('--width', default=640, type=int, help='frame width of webcam')
parser.add_argument('--height', default=480, type=int, help='frame height of webcam')
args = parser.parse_args()


def main(args):
    keyboard = Controller()

    point_right_positive_samples = pd.read_csv(pathlib.Path(args.output_folder, 'pointRight_positive.csv'),
                                   header=None).iloc[:, 1:-1].to_numpy()
    point_right_negative_samples = pd.read_csv(pathlib.Path(args.output_folder, 'pointRight_negative.csv'),
                                   header=None).iloc[:, 1:-1].to_numpy()

    point_left_positive_samples = pd.read_csv(pathlib.Path(args.output_folder, 'pointLeft_positive.csv'),
                                               header=None).iloc[:, 1:-1].to_numpy()
    point_left_negative_samples = pd.read_csv(pathlib.Path(args.output_folder, 'pointLeft_negative.csv'),
                                               header=None).iloc[:, 1:-1].to_numpy()

    xr = np.concatenate((point_right_positive_samples, point_right_negative_samples), axis=0)
    yr = np.concatenate((np.ones(shape=point_right_positive_samples.shape[0]),
                        -np.ones(shape=point_right_negative_samples.shape[0])), axis=0)

    xl = np.concatenate((point_left_positive_samples, point_left_negative_samples), axis=0)
    yl = np.concatenate((np.ones(shape=point_left_positive_samples.shape[0]),
                         -np.ones(shape=point_left_negative_samples.shape[0])), axis=0)

    # Point right gesture
    clfr = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clfr.fit(xr, yr)
    print('Accuracy Right=', accuracy_score(clfr.predict(xr), yr))

    # Point left gesture
    clfl = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clfl.fit(xl, yl)
    print('Accuracy Left=', accuracy_score(clfl.predict(xl), yl))

    # Get webcam frame size and screen size
    #root = tk.Tk()
    #screen_width = root.winfo_screenwidth()
    #screen_height = root.winfo_screenheight()
    # screen_size = (screen_height, screen_width)
    # frame_size = (480, 640)

    # Create a named window
    cv2.namedWindow('Arrow Handling Gestures')
    #cv2.moveWindow('Static Gesture / Test', 0, 0)

    # For webcam input:
    video_capture = cv2.VideoCapture(0)

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
                    landmarks = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark])
                    label = 'Right' if results.multi_handedness[index].classification[0].label == 'Left' else 'Left'
                    predicted_class_right = int(clfr.predict(landmarks.reshape(1, 63))[0])
                    predicted_class_left = int(clfl.predict(landmarks.reshape(1, 63))[0])

                 #   print("predicted_class_right", predicted_class_right)
                  #  print("predicted_class_left", predicted_class_left)

                    # Count frames, when gesture recognized once to avoid multiple execution
                    if recognized == True:
                        counter += 1

                    # Allow gesture to be recognized again after 30 frames
                    if counter == 30:
                        recognized = False
                        counter = 0
                    else:
                        if label == 'Left' and predicted_class_right == 1:
                            if not recognized:
                                print("Gesture recognized: Point right")
                                keyboard.press(Key.right)
                                recognized = True

                        if label == 'Right' and predicted_class_left == 1:
                            if not recognized:
                                print("Gesture recognized: Point left")
                                keyboard.press(Key.left)
                                recognized = True


                        #index_tipm = [1280 * (1.0 - landmarks[8, 0]), 720 * landmarks[8, 1]]
                        #mouse.move(index_tipm[0], index_tipm[1])

            # View and write video (hand keypoint visualization and frame numbers)
            cv2.imshow('Static Gesture / Test', image)

            # Break when pressed quit
            if cv2.waitKey(5) & 0xFF == 27:
                break


        video_capture.release()


if __name__ == '__main__':
    main(args)