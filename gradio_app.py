from dotenv import load_dotenv
load_dotenv()

# VoiceBot UI with Gradio
import os
import gradio as gr

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio, transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs
from pydub import AudioSegment

# System prompt for AI doctor
system_prompt = """You have to act as a professional doctor, I know you are not but this is for learning purposes. 
            What's in this image? Do you find anything wrong with it medically? 
            If you make a differential, suggest some remedies for them. Do not add any numbers or special characters in your response. Your response should be in one long paragraph. Always answer as if you are talking to a real person.
            Do not say 'In the image I see' but say 'With what I see, I think you have ...'
            Do not respond as an AI model in markdown. Your answer should mimic that of an actual doctor, not an AI bot. 
            Keep your answer concise (max 3 sentences). No preamble, start your answer right away."""


def process_inputs(audio_filepath, image_filepath):
    # Convert speech to text
    speech_to_text_output = transcribe_with_groq(
        GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
        audio_filepath=audio_filepath,
        stt_model="whisper-large-v3"
    )

    # Analyze the image (if provided)
    if image_filepath:
        doctor_response = analyze_image_with_query(
            query=system_prompt + speech_to_text_output,
            encoded_image=encode_image(image_filepath),
            model="llama-3.2-11b-vision-preview"
        )
    else:
        doctor_response = "No image provided for me to analyze."

    # Generate text-to-speech
    audio_output_path = "final.mp3"
    text_to_speech_with_elevenlabs(input_text=doctor_response, output_filepath=audio_output_path)

    # Ensure audio file exists and is properly formatted
    if not os.path.exists(audio_output_path) or os.path.getsize(audio_output_path) == 0:
        raise Exception("Error: final.mp3 was not generated correctly.")

    # Re-encode to ensure compatibility
    audio = AudioSegment.from_file(audio_output_path, format="mp3")
    audio.export(audio_output_path, format="mp3", bitrate="128k")

    return speech_to_text_output, doctor_response, audio_output_path


# Create the Gradio interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio("final.mp3")  # Corrected to use final.mp3
    ],
    title="AI Doctor with Vision and Voice"
)

iface.launch(debug=True)
