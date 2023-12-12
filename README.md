# VidBot

**VidBot is a chatbot that lets you ask questions about any video you want. Whether it's a movie, a lecture, or a tutorial, VidBot can help you find the information you need. You can use VidBot with videos stored on your computer or with YouTube links.**

## Application Flow

The app works as follows:

- Temp directory is created to store temporary files such as audio and transcriptions
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

## Running

The app is built on top of Streamlit, but I provide a bootstrap file so you can run it simply with:

```bash
python3 bootstrap.py
```

Example URL: [Introducing GPT-4](https://youtu.be/--khbXchTeE?si=O2DtXx8fcb0H4bta)

You can stop the application by pressing `stop` on the sidebar

## Lessons leaned

1. Langchain is poorly documented, lots of trial and error all around, specially to make custom prompts work
2. If the transcription prompt is in question format or similar, GPT-4 will answer and therefore transcription will be overwritten by that answer. LOL
3. AzureAI documentation and examples are fantastic
4. Figuring a way to stop streamlit from within for mult-page app. I ended up using a sidebar page to stop the app.

## License

See repository.
