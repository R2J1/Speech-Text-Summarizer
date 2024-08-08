import streamlit as st
import assemblyai as aai
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model for summarization
tokenizer = AutoTokenizer.from_pretrained(".")
model = AutoModelForSeq2SeqLM.from_pretrained(".")

def generate_summary(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Generate summary
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Custom CSS for styling
st.markdown(
    """
    <style>
    .full-width-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
    }
    .input-container {
        width: 60%;
        margin-bottom: 2rem;
    }
    .text-area {
        height: 200px;
    }
    .button-container {
        width: 60%;
        display: flex;
        justify-content: center;
    }
    .file-uploader-container {
        width: 60%;
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 2rem;
    }
    .summary-container {
        width: 60%;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("Text Summarization App")

# Main container
st.markdown('<div class="full-width-container">', unsafe_allow_html=True)

# Choose input method: text area, upload text file, or upload audio file
input_method = st.radio("Choose input method:", ("Enter text", "Upload a text file", "Upload an audio file"))

if input_method == "Enter text":
    # Input text area
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    input_text = st.text_area("Enter your text here:", height=200, help="You can type or paste your text here.", key="text_area")
    st.markdown('</div>', unsafe_allow_html=True)

    # Button to generate summary
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("Generate Summary", key="generate_button"):
        if input_text.strip() != "":
            summary = generate_summary(input_text)
            st.markdown('<div class="summary-container">', unsafe_allow_html=True)
            st.subheader("Summary:")
            st.write(summary)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter some text.")
    st.markdown('</div>', unsafe_allow_html=True)

elif input_method == "Upload a text file":
    # File upload for text
    st.markdown('<div class="file-uploader-container">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a text file:", type=["txt"])

    if uploaded_file is not None:
        # Read uploaded file
        text = uploaded_file.getvalue().decode("utf-8")
        
        # Generate summary
        summary = generate_summary(text)
        st.markdown('<div class="summary-container">', unsafe_allow_html=True)
        st.subheader("Summary:")
        st.write(summary)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # File upload for audio
    st.markdown('<div class="file-uploader-container">', unsafe_allow_html=True)
    uploaded_audio = st.file_uploader("Upload an audio file:", type=["mp3", "wav", "ogg", "m4a", "webm"])

    if uploaded_audio is not None:
        # Save the uploaded audio file
        with open("uploaded_audio_file", "wb") as f:
            f.write(uploaded_audio.read())
        
        # Transcribe the audio file using AssemblyAI
        aai.settings.api_key = "133854230e3142fb8c59c19c429dd3a5"
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe("uploaded_audio_file")

        if transcript.status == aai.TranscriptStatus.error:
            st.error("Error occurred during transcription.")
        else:
            # Save transcript to a text file
            with open("transcript.txt", "w") as text_file:
                text_file.write(transcript.text)
            st.success("Transcription completed. Transcript saved to transcript.txt")

            # Read the transcript for summarization
            with open("transcript.txt", "r") as transcript_file:
                text = transcript_file.read()
            
            # Generate summary
            summary = generate_summary(text)
            st.markdown('<div class="summary-container">', unsafe_allow_html=True)
            st.subheader("Summary:")
            st.write(summary)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# End main container
st.markdown('</div>', unsafe_allow_html=True)