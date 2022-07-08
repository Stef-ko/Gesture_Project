import argparse
import pathlib
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp


# Arguments
parser = argparse.ArgumentParser(description='Collect Static Gesture Samples')
parser.add_argument('--output_folder', default='./data', type=str, help='output folder to write keypoint sequences')
parser.add_argument('--name', type=str, help='filename: gesturename_positive/negative')

parser.add_argument('--fps',    default=30.0, type=float, help='frame per second (FPS) of webcam')
parser.add_argument('--width',  default=640, type=int, help='frame width of webcam')
parser.add_argument('--height', default=480, type=int, help='frame height of webcam')
args = parser.parse_args()

def main(args):

    # Reading video from built-in webcam
    # Max. 1-minute is enough for static gesture samples
    video_capture = cv2.VideoCapture(0) 
    total_frame_count = int(30 * 60 ) 

    # Output
    pathlib.Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    # visualisation: frame numbers and hand keypoints
    video_writer = cv2.VideoWriter(pathlib.Path(args.output_folder, '%s_vis.avi'%args.name).as_posix(), 
                                   cv2.VideoWriter_fourcc(*'XVID'), 
                                   args.fps, (args.width, args.height), 1)
    # csv files: frame numbers, keypoint locations (x,y,z), and Left/Right
    keypoints_file = open( pathlib.Path(args.output_folder, '%s_keypoints.csv'%args.name), 'w')
    frame_num = 0

    with mp.solutions.hands.Hands(model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
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

                # save as csv files
                landmarks = np.array([[l.x,l.y,l.z] for l in hand_landmarks.landmark])
                label = 'Right' if results.multi_handedness[index].classification[0].label=='Left' else 'Left'

                # write keypoint locations: frame_num, x1, y1, z1, x2, y2, z2, ..., Left/Right
                txt = ''
                for i in range(0,21):
                    for j in range(0, 3):
                        txt = txt + '%2.3f,'%(landmarks[i,j])
                txt = '%d,'%frame_num + txt + label
                keypoints_file.write('%s\n'%txt)

            # write frame number
            image = cv2.rectangle(image, (0, 0), (175, 30), (0,0,0), -1)
            image = cv2.putText(image, 'Frame: %05d'%frame_num, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

            # View and write video (hand keypoint visualization and frame numbers)
            cv2.imshow('View Captured Gestures', image)
            video_writer.write(image.copy())

            if frame_num%300==0:
                print('%d / %d'%(frame_num, total_frame_count))
            # Break when pressed quit or reached max. sequence length
            if cv2.waitKey(5) & 0xFF == 27:
                break
            if frame_num>=total_frame_count:
                break
            frame_num += 1

        video_capture.release()
        video_writer.release()
        keypoints_file.close()

if __name__ == '__main__':
    main(args)