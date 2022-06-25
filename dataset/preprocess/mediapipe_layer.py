import cv2
import numpy as np
import mediapipe as mp


class MediaPipe(object):
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_holistic = mp.solutions.holistic
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.hand_detector = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.pose_detector = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_meshor = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.holistic_model = self.mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2
        )
        self.selfie_segmenter = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

        self.hand_activate = False
        self.face_detection_activate = False
        self.face_mesh_activate = False
        self.holistic_activate = False
        self.selfie_activate = False
        self.body_activate = False

    def holistic_process(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self.holistic_model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_contours_style())
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles
                    .get_default_pose_landmarks_style())
            self.mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style())
            self.mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style())
        return image, results

    def selfie_segmentation(self, image, bg_image=None):
        BG_COLOR = (192, 192, 192)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self.selfie_segmenter.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
        output_image = np.where(condition, image, bg_image)
        return output_image

    def face_mesh(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self.face_meshor.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
        return image

    def face_detection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self.face_detector.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                self.mp_drawing.draw_detection(image, detection)
        return image

    def hand_detection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self.hand_detector.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            results.multi_hand_landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
        return image, results.multi_hand_landmarks

    def pose_detection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self.pose_detector.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return image

    def update_setting(self, settings):
        self.hand_activate = True if settings['hand'] else False
        self.face_detection_activate = True if settings['landmark'] else False
        self.face_mesh_activate = True if settings['mesh'] else False
        self.body_activate = True if settings['body'] else False
        self.selfie_activate = True if settings['selfie'] else False
        self.holistic_activate = True if settings['holistic'] else False

    def process(self, image):
        if self.hand_activate:
            image = self.hand_detection(image)
        if self.face_detection_activate:
            image = self.face_detection(image)
        if self.face_mesh_activate:
            image = self.face_mesh(image)
        if self.body_activate:
            image = self.pose_detection(image)
        if self.selfie_activate:
            image = self.selfie_segmentation(image)
        if self.holistic_activate:
            image = self.holistic_process(image)
        return image
