import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic  # Holistic model

def mediapipe_detection(image, model):
    image = image[:, :, ::-1]  # Convert BGR to RGB
    image.flags.writeable = False
    results = model.process(image)  # Update variable name to 'results'
    image.flags.writeable = True
    image = image[:, :, ::-1]  # Convert RGB back to BGR
    return image, results

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

actions = ['aku', 'apa', 'bagaimana', 'berapa', 'di', 'dia', 'F', 'halo', 'I', 'J', 'K', 'kamu', 'kapan', 'ke', 'kita', 'makan', 'mana', 'mereka', 'minum', 'nama', 'R', 'saya', 'siapa', 'Y', 'yang', 'Z']

model = load_model('realtimeV11.h5')

sequence = []
sentence = []
predictions = []
threshold = 0.5

class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()

    def transform(self, frame):
        global sequence
        image, results = mediapipe_detection(frame.to_ndarray(format="bgr"), mp_holistic)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]

            confidence = res[np.argmax(res)]
            predictions.append(np.argmax(res))

            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

        # Display the detected sentence
        if len(sentence) > 0:
            text = ' '.join(sentence)
            text_html = f'<div class="text"><span>{text}</span></div>'
            st.markdown(text_html, unsafe_allow_html=True)

        # Display the processed image
        st.image(image, channels="BGR")


def main():
    st.title("AI Sign Language Detection")
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    webrtc_streamer(
        key="example",
        video_transformer_factory=SignLanguageTransformer,
        rtc_configuration=rtc_configuration,
    )

if __name__ == "__main__":
    main()
