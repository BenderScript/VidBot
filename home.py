import os
import re
import tempfile

import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import YoutubeAudioLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.schema import SystemMessage, HumanMessage
from langchain.text_splitter import TokenTextSplitter
from pydub import AudioSegment
from pytube import YouTube
from streamlit_extras.switch_page_button import switch_page

from exception_traceback import print_exception_details
from open_ai_ctx import OpenAICtx
from text_docs_processor import TextDocProcessor

st.set_page_config(page_title="GenAI Transcriber", page_icon=":rocket:", layout="wide")
st.title("GenAI - Transcriber On-Demand Chatbot App")

# Initialization
if 'openai_api_key' not in st.session_state:
    load_dotenv(override=True, dotenv_path=".env")  # take environment variables from .env.
    openai.api_key = os.getenv("OPENAI_API_KEY")
    st.session_state.openai_api_key = openai.api_key

if 'chatbot_created' not in st.session_state:
    st.session_state.chatbot_created = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp(dir=".")

temp_dir = st.session_state.temp_dir

# Initialize TextDocProcessor
if 'text_doc_processor' not in st.session_state:
    st.session_state.text_doc_processor = TextDocProcessor(temp_dir=temp_dir, env_file="./.env")

text_doc_processor = st.session_state.text_doc_processor


def save_uploaded_file(uploaded_file):
    # TODO Catch Exception if file not found
    with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return os.path.join(temp_dir, uploaded_file.name)


def replace_extension(filename, new_extension):
    root, _ = os.path.splitext(filename)
    return root + new_extension


def extract_mp3_from_video(filename):
    """
    Extract the audio from a video file and save it as an MP3 file.

    This function uses the pydub library to extract the audio from a video file and save it as an MP3 file.
    It supports .mp4 and .mkv video files. If there's an exception during this process, it catches it,
    prints its details, and returns None. Otherwise, it returns the path of the MP3 file.

    Parameters:
    filename (str): The name of the video file.

    Returns:
    str: The path of the MP3 file, or None if an exception occurred.
    """

    try:
        # Get the extension of the file
        file_extension = os.path.splitext(filename)[1]

        # Construct the paths of the input and output files
        in_file_path = os.path.join(temp_dir, filename)
        out_file_path = os.path.join(temp_dir, replace_extension(filename, ".mp3"))

        # Check if the file is a .mp4 or .mkv video file
        if file_extension.lower() in ['.mp4', '.mkv']:
            # Extract the audio from the video file and save it as an MP3 file
            audio = AudioSegment.from_file(in_file_path, file_extension.replace('.', ''))
            audio.export(out_file_path, format="mp3")
        else:
            # Raise an exception if the file is not a .mp4 or .mkv video file
            raise f"Unsupported file type: {file_extension}"

        return out_file_path  # Return the path of the MP3 file
    except Exception as e:  # Catch any exceptions that occur during execution
        print_exception_details(e)  # Print the details of the exception
        return None  # Return None in case of an exception


def sanitize_file_name(name):
    """Replaces any invalid characters in a given string with underscores to make it a safe file name."""
    # Metric names must start with a letter and may contain letters, digits, underscores, and colons.
    # They must also not end with a colon.
    regex = re.compile(r"[^a-zA-Z0-9_.]+")
    return regex.sub("_", name.strip())


def load_docs_and_transcribe_audio(audio_file_path):
    """
    Load and transcribe an audio file to text using the Langchain framework.

    This function uses a GenericLoader to load all text files in a directory recursively,
    and an OpenAIWhisperParser to parse these files. If there's an exception during this process,
    it catches it, prints its details, and returns None. Otherwise, it returns a list of documents
    representing the transcribed text.

    Parameters:
    audio_file_path (str): The path to the audio file to be transcribed.

    Returns:
    list: A list of documents representing the transcribed text, or None if an exception occurred.
    """

    try:
        # The GenericLoader is initialized with the directory of the audio file,
        # the basename of the audio file (which acts as a pattern for glob to match files),
        # and an instance of OpenAIWhisperParser which is used to parse the loaded files.
        loader = GenericLoader.from_filesystem(
            os.path.dirname(audio_file_path),  # Directory of the audio file
            glob=os.path.basename(audio_file_path),  # Basename of the audio file
            parser=OpenAIWhisperParser()  # Parser to parse the loaded files
        )

        # The load method of the GenericLoader is called to load and parse the files.
        # The parsed files are returned as a list of documents.
        docs = loader.load()

        return docs  # Return the list of documents
    except Exception as e:  # Catch any exceptions that occur during execution
        print_exception_details(e)  # Print the details of the exception
        return None  # Return None in case of an exception


def correct_audio_transcript(transcription_prompt, transcript_file):
    """
    This function corrects the audio transcript using AzureOpenAICtx.

    Parameters:
    transcription_prompt (str): The prompt for the transcription.
    transcript_file (str): The path to the file containing the transcript.

    Returns:
    str: The path to the corrected transcript file.
    """

    try:
        # Initialize AzureOpenAICtx with the given .azure.nv file
        with OpenAICtx("./.env") as llm:
            # Prepend "corrected_" to the filename of the transcript file
            corrected_transcript_file = prepend_suffix_to_filename(transcript_file, "corrected_")

            try:
                # Try to open the transcript file
                with open(transcript_file, "r") as tf:
                    # Read the content of the transcript file
                    transcript = tf.read()

                    # Initialize a TokenTextSplitter with a chunk size of 4000 and a chunk overlap of 0
                    text_splitter = TokenTextSplitter(chunk_size=4000, chunk_overlap=0)

                    # Split the text of the transcript into chunks
                    texts = text_splitter.split_text(transcript)

                    for t in texts:
                        # For each chunk, create a list of messages
                        messages = [
                            SystemMessage(
                                content=transcription_prompt
                            ),
                            HumanMessage(
                                content=t
                            ),
                        ]

                        # Get the response from LLM
                        resp = llm(messages)

                        # Append the response to the corrected transcript file
                        with open(corrected_transcript_file, "a") as ct:
                            ct.write(resp.content)

                    # Return the path to the corrected transcript file
                    return corrected_transcript_file

            except FileNotFoundError:
                # If the transcript file does not exist, print an error message
                print(f"File '{transcript_file}' does not exist.")
                st.error(f"File '{transcript_file}' does not exist.")
                return None
            except openai.RateLimitError as e:
                st.warning("OpenAI API rate limit exceeded, cannot correct transcription.")
                return None

    except FileNotFoundError as e:
        # If the .azure.nv file does not exist, print an error message
        print("File './.azure.env' does not exist.")
        st.warning("File './.azure.env' does not exist, cannot use transcription correction.")
        return None


def prepend_suffix_to_filename(path, suffix):
    directory, filename = os.path.split(path)
    new_filename = suffix + filename
    new_path = os.path.join(directory, new_filename)
    return new_path


def extract_audio_and_transcribe_from_youtube(urls, save_dir):
    """
    Extract audio from YouTube videos and transcribe it to text using the Langchain framework.

    This function uses a GenericLoader to load audio from YouTube videos and an OpenAIWhisperParser
    to parse these files. If there's an exception during this process, it catches it, prints its details,
    and returns None. Otherwise, it returns a list of documents representing the transcribed text.

    Parameters:
    urls (list): The list of URLs of the YouTube videos.
    save_dir (str): The directory where the audio files should be saved.

    Returns:
    list: A list of documents representing the transcribed text, or None if an exception occurred.
    """

    try:
        # The GenericLoader is initialized with an instance of YoutubeAudioLoader (which is initialized
        # with the list of URLs and the save directory) and an instance of OpenAIWhisperParser.

        loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser())

        # The load method of the GenericLoader is called to load and parse the files.
        # The parsed files are returned as a list of documents.
        docs = loader.load()

        return docs  # Return the list of documents
    except Exception as e:  # Catch any exceptions that occur during execution
        print_exception_details(e)  # Print the details of the exception
        return None  # Return None in case of an exception


def prepare_youtube_chatbot(url, transcription_prompt):
    with st.status("Preparing Chatbot...", expanded=True, state="running") as status:

        # Create a YouTube object
        status.write(f"Getting Youtube video info from : {url}...")
        yt = YouTube(url)
        # Get the title of the video
        title = yt.title

        status.write(f"Extracting audio from Youtube Video...this might take a while : \"{title}\"...")
        documents = extract_audio_and_transcribe_from_youtube([url], temp_dir)
        if documents is None:
            status.error(f"transcribing video: {url}.")
            url = None
            return
        status.write(f"Saving transcription...")
        transcription = ""
        transcript_file = os.path.join(temp_dir, sanitize_file_name(yt.title) + ".txt")
        with open(transcript_file, "w") as f:
            for document in documents:
                transcription += document.page_content
            f.write(transcription)

        st.header("Transcription", divider="rainbow")
        st.write(transcription)
        st.divider()

        status.write(f"Correcting transcription using OpenAI GPT-4.0...")

        corrected_transcript_file = correct_audio_transcript(transcription_prompt, transcript_file)
        # Set corrected_transcript_file to transcript_file if it is None
        if corrected_transcript_file is None:
            corrected_transcript_file = transcript_file
        status.write(f"Loading transcriptions...")
        text_doc_processor.load_text_docs(corrected_transcript_file)
        status.write(f"Splitting transcriptions...")
        text_doc_processor.split_text_docs()
        status.write(f"Inserting transcriptions into VectorDB...")
        text_doc_processor.create_text_retriever()
        status.write(f"Creating Chatbot...")
        text_doc_processor.create_text_conv_chain()
        status.update(label="Chatbot created!", state="complete", expanded=True)
        st.session_state.chatbot_created = True


def prepare_file_chatbot(uploaded_file, transcription_prompt):
    with st.status("Preparing Chatbot...this might take a while", expanded=True, state="running") as status:
        uploaded_file.name = sanitize_file_name(uploaded_file.name)
        status.write(f"Saving file: {uploaded_file.name}...")
        save_uploaded_file(uploaded_file)
        status.write(f"Extracting Audio from file...")
        output_file_name = extract_mp3_from_video(uploaded_file.name)
        if output_file_name is None:
            status.error(f"extracting audio from: {uploaded_file.name}.")
            uploaded_file = None
            return
        status.write(f"Transcribing audio using OpenAI GPT-3.5...")
        documents = load_docs_and_transcribe_audio(output_file_name)
        if documents is None:
            status.error(f"transcribing audio: {output_file_name}.")
            uploaded_file = None
            return
        transcript_file = os.path.join(temp_dir, f"transcript_{replace_extension(uploaded_file.name, '.txt')}")
        status.write(f"Saving transcription...")
        transcription = ""
        with open(transcript_file, "w") as f:
            for document in documents:
                transcription += document.page_content
            f.write(transcription)
        st.header("Transcription", divider="rainbow")
        st.write(transcription)

        status.write(f"Correcting transcription OpenAI GPT-4.0...")
        corrected_transcript_file = correct_audio_transcript(transcription_prompt, transcript_file)
        # Set corrected_transcript_file to transcript_file if it is None
        if corrected_transcript_file is None:
            corrected_transcript_file = transcript_file
        status.write(f"Loading transcriptions...")
        text_doc_processor.load_text_docs(corrected_transcript_file)
        status.write(f"Splitting transcriptions...")
        text_doc_processor.split_text_docs()
        status.write(f"Inserting transcriptions into VectorDB...")
        text_doc_processor.create_text_retriever()
        status.write(f"Creating Chatbot...")
        text_doc_processor.create_text_conv_chain()
        status.update(label="Chatbot created!", state="complete", expanded=True)
        st.session_state.chatbot_created = True
        # end of status context manager


def run():
    transcription_prompt = st.text_area(
        "Transcription correction Prompt",
        placeholder="You are a helpful assistant for the company OpenAI. Your task is to correct any spelling "
                    "discrepancies in the transcribed text. Make sure that the names of the following products are "
                    "spelled correctly: GPT-4, GPT-4V(vision), Dall-E, Dall-E 3. Only add necessary "
                    "punctuation such as periods, commas, and capitalization, and use only the context provided.",
        value="You are a helpful assistant for the company OpenAI. Your task is to correct any spelling "
                    "discrepancies in the transcribed text. Make sure that the names of the following products are "
                    "spelled correctly: GPT-4, GPT-4V(vision), Dall-E, Dall-E 3. Only add necessary "
                    "punctuation such as periods, commas, and capitalization, and use only the context provided.",
        help="Write a prompt that will be passed to GPT-3.5 to correct the transcription.",
        height=100,
    )

    url = st.text_input("Youtube URL", help="Youtube Video URL", value="https://youtu.be/h02ti0Bl6zk?si=8ekUlYXMEkNHGqBs",
                        placeholder="https://youtu.be/h02ti0Bl6zk?si=8ekUlYXMEkNHGqBs")

    uploaded_file = st.file_uploader("Choose a file", help="Upload a video file from your computer")

    if st.button('Next'):
        if url:
            prepare_youtube_chatbot(url, transcription_prompt)
            switch_page("chatbot")
        if uploaded_file:
            prepare_file_chatbot(uploaded_file, transcription_prompt)
            switch_page("chatbot")
        else:
            st.error("No file uploaded")
            switch_page("stop")


run()
