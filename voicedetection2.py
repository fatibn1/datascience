import streamlit as st
import speech_recognition as sr

def transcribe_speech(language="fr-FR", pause_threshold=2):
    """
    Transcribes speech from the microphone.

    Args:
        language (str): Language code for speech recognition (e.g., "fr-FR" for French, "en-US" for English).
        pause_threshold (int): Seconds of silence to wait before ending the recording.

    Returns:
        str: Transcribed text or an error message.
    """
    r = sr.Recognizer()
    r.pause_threshold = pause_threshold  # Increase pause threshold to wait longer before ending the recording

    with sr.Microphone() as source:
        st.info("Parlez maintenant...")
        audio_text = r.listen(source)  # Listen to the microphone input
        st.info("Transcription en cours...")

        try:
            text = r.recognize_google(audio_text, language=language)  # Transcribe using Google Speech Recognition
            return text
        except sr.UnknownValueError:
            return "Désolé, je n'ai pas compris."
        except sr.RequestError:
            return "Désolé, le service de reconnaissance vocale est indisponible."

def main():
    st.title("Speech Recognition App")
    st.write("Cliquez sur le bouton pour commencer à parler:")

    # Language selection dropdown
    language_options = {
        "French": "fr-FR",
        "English (US)": "en-US",
        "Spanish": "es-ES",
        "German": "de-DE",
        "Italian": "it-IT",
    }
    selected_language = st.selectbox("Choisissez votre langue:", list(language_options.keys()))

    # Pause threshold slider
    pause_threshold = st.slider(
        "Temps d'attente avant la transcription (en secondes):",
        min_value=1,
        max_value=10,
        value=2,  # Default value
        help="Augmentez ce temps si vous avez besoin de plus de temps pour parler."
    )

    # Add a button to trigger speech recognition
    if st.button("Commencer l'enregistrement"):
        st.write("Enregistrement en cours...")
        language_code = language_options[selected_language]  # Get the language code
        text = transcribe_speech(language=language_code, pause_threshold=pause_threshold)
        st.write("Transcription : ", text)

if __name__ == "__main__":
    main()