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


# Here exist two ways to detect hands: 
# 1. Static image detection
path = '/home/eduardo/USP/Computer_Vision/hand_tracking/images/woman_hands.jpg'
path_2 = '/home/eduardo/USP/Computer_Vision/hand_tracking/images/my_hands.jpg'
IMAGE_FILES = [path, path_2]
with mpHands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:

    for idx, file in enumerate(IMAGE_FILES):
        # Read an image, flip it around y-axis for correct handedness output (see above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue

        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
            mpDrawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mpHands.HAND_CONNECTIONS,
                mpDrawingStyles.get_default_hand_landmarks_style(),
                mpDrawingStyles.get_default_hand_connections_style())
        
            cv2.imwrite(
            'hand_tracking/images/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
            
            # Draw hand world landmarks.
            if not results.multi_hand_world_landmarks:
                continue

            for hand_world_landmarks in results.multi_hand_world_landmarks:
                mpDrawing.plot_landmarks(
                    hand_world_landmarks, mpHands.HAND_CONNECTIONS, azimuth=5)