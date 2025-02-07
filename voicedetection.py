import streamlit as st
import speech_recognition as sr

def transcribe_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Parlez maintenant...")
        audio_text = r.listen(source)
        st.info("Transcription...")

        try:
            text = r.recognize_google(audio_text, language="fr-FR")
            return text
        except sr.UnknownValueError:
            return "Désolé, je n'ai pas compris."
        except sr.RequestError:
            return "Désolé, le service de reconnaissance vocale est indisponible."

# Exemple d'utilisation dans Streamlit
if st.button("Démarrer la transcription"):
    transcription = transcribe_speech()
    st.write("Transcription :", transcription)
def main() :
    st.title("Speech Recognition App" )
    st.write("Cliquez sur le microphone pour commencer à parler:" )

    # ajouter un bouton pour déclencher la reconnaissance vocale
    if st.button("Start Recording" ):
        text = transcribe_speech()
        st.write("Transcription : ", text)
if __name__ == "__main__" :
    main()