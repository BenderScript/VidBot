# VidBot

**VidBot is a chatbot that lets you ask questions about any video you want. Whether it's a movie, a lecture, or a tutorial, VidBot can help you find the information you need. You can use VidBot with videos stored on your computer or with YouTube links.**

## Application Flow

The app works as follows:

- The user sends a video to the bot or points to a YouTube URL.
- The app extracts the audio from the video.
- The app sends the audio to OpenAI and waits for transcription.
- The app collates one or more transcriptions from OpenAI into a single transcription.
- The app sends the transcription to chatGPT 4 for correction.
- The app splits and inserts the correct transcription into a vectorDB for RAG.
- Finally, a RAG chatbot is created.

## Goals

The goal of this app was to use two different chatGPT providers, **OpenAI and AzureAI**, for the creation of a video chatbot.

## Installation

**To install the app, you need to have Python 3 and pip installed on your system.** Then, install the dependencies with:

```bash
pip3 install -r requirements.txt
```

## Configuration

You should create a `.env` file in the root directory of the app with the following variables:

```bash
OPENAI_API_KEY=<your open ai key>
```

The `TEMP_DOWNLOAD_PATH` variable is used to store temporary files.

You must also provide a `.azure.env` file in the same directory with the following variables:

```bash
OPENAI_API_KEY=<your AzureAI key>
OPENAI_API_TYPE=azure
OPENAI_API_BASE=<your AzureAI base url>
OPENAI_API_DEPLOYMENT_ID=<deployment ID, some docs call it deployment name>
OPENAI_API_VERSION=2023-05-15
OPENAI_API_REGION=<AzureAI region, optional>
OPENAI_API_MAX_TOKENS=8192
OPENAI_API_TEMPERATURE=0.1
OPENAI_API_MODEL_NAME=gpt-4
OPENAI_API_EMBEDDING_MODEL_NAME=text-embedding-ada-002
OPENAI_API_EMBEDDING_MODEL_DEPLOYMENT_ID=<deployment ID, some docs call it deployment name>
OPENAI_API_EMBEDDING_API_VERSION=2023-03-15-preview
```

## Running

The app is built on top of Streamlit, but I provide a bootstrap file so you can run it simply with:

```bash
python3 bootstrap.py
```

Example URL: [Introducing GPT-4](https://youtu.be/--khbXchTeE?si=O2DtXx8fcb0H4bta)

## License

See repository.
