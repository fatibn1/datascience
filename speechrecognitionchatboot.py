import nltk
import streamlit as st
import speech_recognition as sr
from nltk.chat.util import Chat, reflections
# Define chatbot pairs (questions and responses)
pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, how can I help you today?",]
    ],
    [
        r"what is your name?",
        ["I am a chatbot. You can call me ChatGPT.",]
    ],
    [
        r"how are you?",
        ["I'm doing well, thank you!", "I'm just a program, so I don't have feelings, but thanks for asking!",]
    ],
    [
        r"quit",
        ["Goodbye! It was nice talking to you.",]
    ],
]

# Create a chatbot object
assert isinstance(reflections, object)
chatbot = Chat(pairs, reflections)
def transcribe_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.write("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            st.write("Sorry, there was an issue with the speech recognition service.")
            return None
def chatbot_response(user_input):
    if user_input.lower() == "quit":
        return "Goodbye!"
    return chatbot.respond(user_input)
def main():
    st.title("Speech-Enabled Chatbot")
    st.write("Welcome to the speech-enabled chatbot! You can type or speak your message.")

    # Input method selection
    input_method = st.radio("Choose input method:", ("Text", "Speech"))

    user_input = None

    if input_method == "Text":
        user_input = st.text_input("Enter your message:")
    else:
        if st.button("Start Recording"):
            user_input = transcribe_speech()

    if user_input:
        response = chatbot_response(user_input)
        st.write(f"Chatbot: {response}")

if __name__ == "__main__":
    main()