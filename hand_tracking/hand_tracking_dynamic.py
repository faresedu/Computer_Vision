import cv2
import mediapipe as mp
import time



cap = cv2.VideoCapture(0)

# The following are mediapipe objects
# The drawing utilities are used to draw the landmarks on the image,
# like the points, the lines, the connections, etc on fingers, hands, etc.
# The drawing styles are used to change the color and thickness of the
# drawing utilities.
mpDrawing = mp.solutions.drawing_utils  # drawing utilities
mpDrawingStyles = mp.solutions.drawing_styles  # drawing styles

# The hands object is used to detect hands in the image.
mpHands = mp.solutions.hands  # hands module
hands = mpHands.Hands()  # hands object


# # 2. Real-time video detection

with mpHands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1) # convert to RGB

        # image.flags.writeable = False
        ''' The line image.flags.writeable = False is a way to improve performance by 
        indicating that the image should be treated as read-only, which can be 
        beneficial in certain situations.
        In the context of this code, it's related to the usage of 
        the hands.process() method from the MediaPipe library. The idea is that by marking 
        the image as not writeable, you are telling the underlying processing method that it 
        doesn't need to modify the image in place, potentially allowing for optimizations.'''

        
        results = hands.process(image) # process the image and use RGB image as input

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        # convert back to BGR to display the image with OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        image_height, image_width, _ = image.shape

        if results.multi_hand_landmarks:
            # print(results.multi_hand_landmarks)
            for hand_landmarks in results.multi_hand_landmarks:
                print(hand_landmarks)
                print(
                    f'Index finger tip coordinates: (',
                    f'{hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    f'{hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                )
                mpDrawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mpHands.HAND_CONNECTIONS,
                    mpDrawingStyles.get_default_hand_landmarks_style(),
                    mpDrawingStyles.get_default_hand_connections_style())

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break


