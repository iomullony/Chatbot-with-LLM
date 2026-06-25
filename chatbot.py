# code_02_chatbot.py
# based on code_10_02_chatbot.py from sas
# this is a streamlit page
# NLTK not used at the moment
# make sure that static files are served, see
# https://docs.streamlit.io/develop/concepts/configuration/serving-static-files
import streamlit as st
import numpy as np
import speech_recognition as sr
import base64
import pyttsx3
import io # Used to handle the audio file in memory
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Helper function to extract features (must be consistent with training)
def extract_features(words):
    features = {}
    for word in words:
        features[f'has({word.lower()})'] = True
    return features

def listen_and_transcribe():
    r = sr.Recognizer()
    st.info("Listening... Speak now.")
    
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            with open("./static/input.wav", "wb") as f:
                f.write(audio.get_wav_data())
        except sr.WaitTimeoutError:
            st.error("No speech detected within the timeout limit.")
            return None
    # Visualise the recorded audio as a spectrogram and return it as base64 HTML
    st.info("Visualising ...")
    image_html = None
    try:
        sample_rate, data = wavfile.read("./static/input.wav")
        # If stereo, use first channel
        if data.ndim > 1:
            data = data[:, 0]
        # First figure: waveform
        fig1, ax1 = plt.subplots()
        times = np.arange(len(data)) / float(sample_rate)
        ax1.plot(times, data)
        ax1.set_title("Waveform")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Amplitude")

        buf1 = io.BytesIO()
        fig1.savefig(buf1, format="png", bbox_inches='tight')
        plt.close(fig1)
        buf1.seek(0)
        img1_b64 = base64.b64encode(buf1.read()).decode('ascii')
        waveform_html = f'<img src="data:image/png;base64,{img1_b64}" alt="waveform" style="max-width:100%;">'

        # Second figure: spectrogram
        fig2, ax2 = plt.subplots()
        ax2.specgram(data, Fs=sample_rate)
        ax2.set_title("Recorded Audio Spectrogram")

        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png", bbox_inches='tight')
        plt.close(fig2)
        buf2.seek(0)
        img2_b64 = base64.b64encode(buf2.read()).decode('ascii')
        spectrogram_html = f'<img src="data:image/png;base64,{img2_b64}" alt="spectrogram" style="max-width:100%;">'

        # Combine: waveform first, then spectrogram
        image_html = waveform_html + "<br>" + spectrogram_html
    except Exception as e:
        st.warning(f"Could not generate spectrogram: {e}")

    # Transcribe audio
    text = None
    try:
        text = r.recognize_google(audio)
    except sr.UnknownValueError:
        st.error("Could not understand audio.")
    except sr.RequestError as e:
        st.error(f"Could not request results from speech recognition service; {e}")

    # Return both transcription text and image HTML so caller can persist the image
    return {"text": text, "image_html": image_html}

# --- 2. TTS/Audio Handling  (IMPLEMENTATION) ---
def text_to_audio_for_web(text):
    """
    Generates audio using pyttsx3, returns an HTML audio player.
    This method saves temporaly an audio file which is overwritten every time.
    """
    # Test pyttsx3 (simple speak)
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)
        # output will be overwritten every time
        engine.save_to_file(text, './static/output.mp3')
        # engine.say("Environment setup complete. Ready for AI assistance systems!")
        # Text-to-Speech test failed: This means you probably do not have eSpeak or eSpeak-ng installed!. PyAudio might be missing or misconfigured.
        engine.runAndWait()
        for v in voices:
            print(f"Available voices: {v.id}, gender: {v.gender}")
        print(f"Selectected voice: {voices[0].id}")
        print("Text-to-Speech output successful.")
        return f"""
        <audio controls style="width: 100%;">
            <source src="./app/static/output.mp3" type="audio/mp3">
        </audio>
        """
    except Exception as e:
        st.error(f"TTS Error: Could not generate audio. Details: {e}")
        return None


# --- 3. Assistant Core Logic ---

def assistant_action(intent):
    """Maps the recognized intent to a specific action/response."""
    actions = {
        "get_weather": "I'm checking the forecast for your location now. Expect sunny skies!",
        "control_lights": "Acknowledged. The requested light controls have been executed.",
        "set_timer": "Starting a timer. I'll let you know when time is up!",
        "tell_joke": "What do you call a fake noodle? An impasta!",
        "greet": "Hello! I am your AI assistant. How can I help you today?",
    }
    return actions.get(intent, "I'm sorry, I don't know how to handle that request yet. Try asking me about the weather or a joke.")

# --- 4. Streamlit App Layout and Logic ---

st.title("🗣️ Voice-Enabled AI Assistant Chat")
st.markdown("Use the **text input** or the **Start Voice Input** button to interact.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show spectrogram image if present
        if message.get("image_html"):
            st.markdown(message.get("image_html"), unsafe_allow_html=True)
        if message["audio_html"]:
            st.markdown(message["audio_html"], unsafe_allow_html=True) 

# Function to handle processing of any prompt (text or voice)
def handle_prompt(prompt, image_html=None):
    """Processes the prompt and updates chat history."""
    if not prompt:
        return

    # 1. User message
    st.session_state.messages.append({"role": "user", "content": prompt, "audio_html": None, "image_html": image_html})
    with st.chat_message("user"):
        st.markdown(prompt)
        if image_html:
            st.markdown(image_html, unsafe_allow_html=True)

    # 2. Recognize Intent and Generate Response
    intent = "greet"
    response_text = assistant_action(intent)
    
    # 3. Generate TTS Audio
    with st.spinner("Generating audio response..."):
        # The gTTS function is called here
        audio_html = text_to_audio_for_web(response_text) 

    # 4. Assistant response
    with st.chat_message("assistant"):
        st.markdown(response_text)
        if audio_html:
            st.markdown(audio_html, unsafe_allow_html=True)
        st.caption(f"**Intent Detected:** `{intent}`")

    # 5. Append assistant response to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text, 
        "audio_html": audio_html
    })


# --- Voice Input Trigger ---
if st.button("🎤 Start Voice Input"):
    with st.spinner("Waiting for speech..."):
        voice_result = listen_and_transcribe()

    if voice_result:
        text = voice_result.get("text")
        image_html = voice_result.get("image_html")
        if text:
            handle_prompt(text, image_html=image_html)
        else:
            # show image even if no transcription
            handle_prompt("(voice input)", image_html=image_html)
    st.rerun()
else: 
    # --- Text Input Handler ---
    if prompt := st.chat_input("Type a message or click the mic button..."):
        handle_prompt(prompt)