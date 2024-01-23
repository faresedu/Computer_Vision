import cv2
import mediapipe as mp
import time



class handTracking():
    def __init__(self, static_image = False, max_num_hands = 2, min_detection_confidence = 0.5, min_tracking_confidence = 0.5, images_path = None):
        self.static_image = static_image
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.images_path = images_path
        self.results = None

        # The following are mediapipe objects
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

    def dynamic_detection(self, draw = True, hand_number = 0, landmark_monitoring = None):
        cap = cv2.VideoCapture(0)
        # The hands object is used to detect hands in the image.
        with self.mp_hands.Hands(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence) as hands:

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                image = cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1)
                self.results = hands.process(image)
                image.flags.writeable = draw
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
                image_height, image_width, _ = image.shape

                if draw and self.results.multi_hand_landmarks:
                    for hand_landmarks in self.results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_draw.HAND_CONNECTIONS,
                        self.mp_draw_styles.get_default_hand_landmarks_style(),
                        self.mp_draw_styles.get_default_hand_connections_style())
                
                if landmark_monitoring is not None:
                   landmarks = self.findLandMark(image, hand_number=hand_number, draw=draw)
                   if len(landmarks) > 0: print(landmarks[landmark_monitoring])
                
                cv2.imshow('MediaPipe Hands', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        cap.release()
    
    def static_detection(self, draw = True, landmark_monitoring = None):
        with self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:

            for idx, file in enumerate(self.images_path):
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
                        f'{hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                        f'{hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                    )
                    self.mp_draw.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw_styles.get_default_hand_landmarks_style(),
                        self.mp_draw_styles.get_default_hand_connections_style())
                
                    cv2.imwrite(
                    'hand_tracking/images/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
                    
                    # Draw hand world landmarks.
                    if not results.multi_hand_world_landmarks:
                        continue

                    for hand_world_landmarks in results.multi_hand_world_landmarks:
                        self.mp_draw.plot_landmarks(
                            hand_world_landmarks, self.mp_hands.HAND_CONNECTIONS, azimuth=5)

    def findLandMark(self, img, hand_number=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lm_list



        