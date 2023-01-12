import cv2
import time
import pygame
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

"""
Authors: Kamil Rominski, Artur Jankowski
Program captures video from webcam and waits for input from user to capture photo
To capture photo create "Victoria" sign using index and middle finger
Program can delay photo for 1, 2 or 3 seconds using following gestures:
    1 second - index finger
    2 seconds - index and pinky finger
    3 seconds - index, middle and ring finger
    
To run this program, it is required to install cv2, numpy, mediapipe and matplotlib
Program runs on Python 3.10
"""

# Global variable responsible for camera delay
camera_delay = 0

# Initialize the mediapipe hands class
mp_hands = mp.solutions.hands

# Set up Hands function for video from webcam
hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize the mediapipe drawing class
mp_drawing = mp.solutions.drawing_utils


def detectHandsLandmarks(image, hands, draw=True, display=True):
    '''
    This function performs hands landmarks detection on an image.
    Args:
        image:   The input image with prominent hand(s) whose landmarks needs to be detected.
        hands:   The Hands function required to perform the hands landmarks detection.
        draw:    A boolean value that is if set to true the function draws hands landmarks on the output image.
        display: A boolean value that is if set to true the function displays the original input image, and the output
                 image with hands landmarks drawn if it was specified and returns nothing.
    Returns:
        output_image: A copy of input image with the detected hands landmarks drawn if it was specified.
        results:      The output of the hands landmarks detection on the input image.
    '''

    # Create a copy of the input image to draw landmarks on.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Hands Landmarks Detection.
    results = hands.process(imgRGB)

    # Check if landmarks are found and are specified to be drawn.
    if results.multi_hand_landmarks and draw:

        # Iterate over the found hands.
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the copy of the input image.
            mp_drawing.draw_landmarks(image=output_image, landmark_list=hand_landmarks,
                                      connections=mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                                   thickness=2, circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0),
                                                                                     thickness=2, circle_radius=2))

    # Check if the original input image and the output image are specified to be displayed.
    if display:

        # Display the original input image and the output image.
        plt.figure(figsize=[15, 15])
        plt.subplot(121);
        plt.imshow(image[:, :, ::-1]);
        plt.title("Original Image");
        plt.axis('off');
        plt.subplot(122);
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output");
        plt.axis('off');

    # Otherwise
    else:

        # Return the output image and results of hands landmarks detection.
        return output_image, results


def countFingers(image, results, draw=True, display=True):
    '''
    This function will count the number of fingers up for each hand in the image.
    Args:
        image:   The image of the hands on which the fingers counting is required to be performed.
        results: The output of the hands landmarks detection performed on the image of the hands.
        draw:    A boolean value that is if set to true the function writes the total count of fingers of the hands on the
                 output image.
        display: A boolean value that is if set to true the function displays the resultant image and returns nothing.
    Returns:
        output_image:     A copy of the input image with the fingers count written, if it was specified.
        fingers_statuses: A dictionary containing the status (i.e., open or close) of each finger of both hands.
        count:            A dictionary containing the count of the fingers that are up, of both hands.
    '''

    # Get the height and width of the input image.
    height, width, _ = image.shape

    # Create a copy of the input image to write the count of fingers on.
    output_image = image.copy()

    # Initialize a dictionary to store the count of fingers of both hands.
    count = {'RIGHT': 0, 'LEFT': 0}

    # Store the indexes of the tips landmarks of each finger of a hand in a list.
    fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]

    # Initialize a dictionary to store the status (i.e., True for open and False for close) of each finger of both hands.
    fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}

    # Iterate over the found hands in the image.
    for hand_index, hand_info in enumerate(results.multi_handedness):

        # Retrieve the label of the found hand.
        hand_label = hand_info.classification[0].label

        # Retrieve the landmarks of the found hand.
        hand_landmarks = results.multi_hand_landmarks[hand_index]

        # Iterate over the indexes of the tips landmarks of each finger of the hand.
        for tip_index in fingers_tips_ids:

            # Retrieve the label (i.e., index, middle, etc.) of the finger on which we are iterating upon.
            finger_name = tip_index.name.split("_")[0]

            # Check if the finger is up by comparing the y-coordinates of the tip and pip landmarks.
            if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):
                # Update the status of the finger in the dictionary to true.
                fingers_statuses[hand_label.upper() + "_" + finger_name] = True

                # Increment the count of the fingers up of the hand by 1.
                count[hand_label.upper()] += 1

        # Retrieve the y-coordinates of the tip and mcp landmarks of the thumb of the hand.
        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x

        # Check if the thumb is up by comparing the hand label and the x-coordinates of the retrieved landmarks.
        if (hand_label == 'Right' and (thumb_tip_x < thumb_mcp_x)) or (
                hand_label == 'Left' and (thumb_tip_x > thumb_mcp_x)):
            # Update the status of the thumb in the dictionary to true.
            fingers_statuses[hand_label.upper() + "_THUMB"] = True

            # Increment the count of the fingers up of the hand by 1.
            count[hand_label.upper()] += 1

    # Check if the total count of the fingers of both hands are specified to be written on the output image.
    if draw:
        # Write the total count of the fingers of both hands on the output image.
        cv2.putText(output_image, " Total Fingers: ", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (20, 255, 155), 2)
        cv2.putText(output_image, str(sum(count.values())), (width // 2 - 150, 240), cv2.FONT_HERSHEY_SIMPLEX,
                    8.9, (20, 255, 155), 10, 10)

    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

    # Otherwise
    else:

        # Return the output image, the status of each finger and the count of the fingers up of both hands.
        return output_image, fingers_statuses, count


def recognizeGestures(image, fingers_statuses, count, draw=True, display=True):
    '''
    This function will determine the gesture of the left and right hand in the image.
    Args:
        image:            The image of the hands on which the hand gesture recognition is required to be performed.
        fingers_statuses: A dictionary containing the status (i.e., open or close) of each finger of both hands.
        count:            A dictionary containing the count of the fingers that are up, of both hands.
        draw:             A boolean value that is if set to true the function writes the gestures of the hands on the
                          output image, after recognition.
        display:          A boolean value that is if set to true the function displays the resultant image and
                          returns nothing.
    Returns:
        output_image:   A copy of the input image with the left and right hand recognized gestures written if it was
                        specified.
        hands_gestures: A dictionary containing the recognized gestures of the right and left hand.
    '''

    global camera_delay
    # Create a copy of the input image.
    output_image = image.copy()

    # Store the labels of both hands in a list.
    hands_labels = ['RIGHT', 'LEFT']

    # Initialize a dictionary to store the gestures of both hands in the image.
    hands_gestures = {'RIGHT': "UNKNOWN", 'LEFT': "UNKNOWN"}

    # Iterate over the left and right hand.
    for hand_index, hand_label in enumerate(hands_labels):

        # Initialize a variable to store the color we will use to write the hands gestures on the image.
        # Initially it is red which represents that the gesture is not recognized.
        color = (0, 0, 255)

        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        if count[hand_label] == 2 and fingers_statuses[hand_label + '_MIDDLE'] and fingers_statuses[
            hand_label + '_INDEX']:

            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures[hand_label] = "V SIGN"

            # Update the color value to green.
            color = (0, 255, 0)

        # Check if 1 finger is up and its index
        elif count[hand_label] == 1 and fingers_statuses[hand_label + '_INDEX']:
            hands_gestures[hand_label] = "1 second"

            # change global variable of camera shuttle

            camera_delay = 1

            # Update the color value to green.
            color = (0, 255, 0)
        # Check if 2 fingers are up and they are index and pinky
        elif count[hand_label] == 2 and fingers_statuses[hand_label + '_INDEX'] and fingers_statuses[hand_label + '_PINKY']:
            hands_gestures[hand_label] = "2 seconds"

            # change global variable of camera shuttle
            camera_delay = 2

            color = (0, 255, 0)
        # Check if 3 fingers are up and they are index, middle and ring
        elif count[hand_label] == 3 and fingers_statuses[hand_label + '_INDEX'] and fingers_statuses[
            hand_label + '_MIDDLE'] and fingers_statuses[hand_label + '_RING']:
            hands_gestures[hand_label] = "3 second"

            # change global variable of camera shuttle
            camera_delay = 3

            color = (0, 255, 0)
        # Check if 5 fingers are up to reset camera shuttle delay to 0
        elif count[hand_label] == 5:
            hands_gestures[hand_label] = "reset"

            # change global variable of camera shuttle
            camera_delay = 0

        ####################################################################################################################

        # Check if the hands gestures are specified to be written.
        if draw:
            # Write the hand gesture on the output image.
            cv2.putText(output_image, hand_label + ': ' + hands_gestures[hand_label], (10, (hand_index + 1) * 60),
                        cv2.FONT_HERSHEY_PLAIN, 4, color, 5)

    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

    # Otherwise
    else:

        # Return the output image and the gestures of the both hands.
        return output_image, hands_gestures


# Initialize the camera with proper resolution
camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)
camera_video.set(4, 920)

# Create window and set its name
cv2.namedWindow('Selfie-Capturing System', cv2.WINDOW_NORMAL)

# Read the filter image with its blue, green, red, and alpha channel.
filter_imageBGRA = cv2.imread('media/filter.png', cv2.IMREAD_UNCHANGED)

# Initialize a variable to store the status of the filter (i.e., whether to apply the filter or not).
filter_on = False

# Initialize the pygame modules to play sound of camera shuttle
pygame.init()
pygame.mixer.music.load("media/cam.mp3")

# Initialize the number of consecutive frames on which we want to check the hand gestures before triggering the events.
num_of_frames = 5

# Initialize a dictionary to store the counts of the consecutive frames with the hand gestures recognized.
counter = {'V SIGN': 0}

# Initialize a variable to store the captured image.
captured_image = None

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():

    # Read a frame.
    ok, frame = camera_video.read()

    # Check if frame is not read properly then continue to the next iteration to read the next frame.
    if not ok:
        continue

    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)

    # Get the height and width of the frame of the webcam video.
    frame_height, frame_width, _ = frame.shape

    # Resize the filter image to the size of the frame.
    filter_imageBGRA = cv2.resize(filter_imageBGRA, (frame_width, frame_height))

    # Get the three-channel (BGR) image version of the filter image.
    filter_imageBGR = filter_imageBGRA[:, :, :-1]

    # Perform Hands landmarks detection on the frame.
    frame, results = detectHandsLandmarks(frame, hands_videos, draw=False, display=False)

    # Check if the hands landmarks in the frame are detected.
    if results.multi_hand_landmarks:

        # Count the number of fingers up of each hand in the frame.
        frame, fingers_statuses, count = countFingers(frame, results, draw=False, display=False)

        # Perform the hand gesture recognition on the hands in the frame.
        _, hands_gestures = recognizeGestures(frame, fingers_statuses, count, draw=True, display=False)


    # Image Capture Functionality.
    ########################################################################################################################

    # Check if the hands landmarks are detected and the gesture of any hand in the frame is V SIGN.
    if results.multi_hand_landmarks and any(hand_gesture == "V SIGN" for hand_gesture in hands_gestures.values()):

        # Increment the count of consecutive frames with V hand gesture recognized.
        counter['V SIGN'] += 1

        # Check if the counter is equal to the required number of consecutive frames.
        if counter['V SIGN'] == num_of_frames:
            # Make a border around a copy of the current frame.
            captured_image = cv2.copyMakeBorder(src=frame, top=10, bottom=10, left=10, right=10,
                                                borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
            start_timer = time.time()
            while time.time() < start_timer + camera_delay:
                cv2.imshow('Selfie-Capturing System', frame)
            # Capture an image and store it in the disk.
            cv2.imwrite('Captured_Image.png', captured_image)

            # Display a black image.
            cv2.imshow('Selfie-Capturing System', np.zeros((frame_height, frame_width)))

            # Play the image capture music to indicate the an image is captured and wait for 100 milliseconds.
            pygame.mixer.music.play()
            cv2.waitKey(100)

            # Display the captured image.
            plt.close()
            plt.figure(figsize=[10, 10])
            plt.imshow(frame[:, :, ::-1])
            plt.title("Captured Image")
            plt.axis('off')

            # Update the counter value to zero.
            counter['V SIGN'] = 0

    # Otherwise if the gesture of any hand in the frame is not V SIGN.
    else:

        # Update the counter value to zero. As we are counting the consective frames with V hand gesture.
        counter['V SIGN'] = 0

    ########################################################################################################################

    # Check if we have captured an image.
    if captured_image is not None:
        # Resize the image to the 1/5th of its current width while keeping the aspect ratio constant.
        captured_image = cv2.resize(captured_image,
                                    (frame_width // 5, int(((frame_width // 5) / frame_width) * frame_height)))

        # Get the new height and width of the image.
        img_height, img_width, _ = captured_image.shape

        # Overlay the resized captured image over the frame by updating its pixel values in the region of interest.
        frame[10: 10 + img_height, 10: 10 + img_width] = captured_image

    # Display the frame.
    cv2.imshow('Selfie-Capturing System', frame)

    # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
    k = cv2.waitKey(1) & 0xFF

    # Check if 'ESC' is pressed and break the loop.
    if (k == 27):
        break

# Release the VideoCapture Object and close the windows.
camera_video.release()
cv2.destroyAllWindows()