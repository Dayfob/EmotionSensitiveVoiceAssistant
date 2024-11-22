import pyttsx3


def text_to_speech(text):
    # Initialize the TTS engine
    engine = pyttsx3.init()

    # Set properties (optional)
    engine.setProperty('voice', engine.getProperty('voices')[1].id)
    engine.setProperty('rate', 150)  # Speed (words per minute)
    engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

    # Speak the text
    engine.say(text)
    engine.runAndWait()


# Example usage
text_to_speech("Hello! This is an offline text-to-speech demonstration.")
