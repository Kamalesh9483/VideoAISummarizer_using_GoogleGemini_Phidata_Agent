import gradio as gr
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
# from google.cloud import storage
from google.generativeai import upload_file, get_file
import google.generativeai as genai
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize API keys
API_KEY = os.getenv("GOOGLE_API_KEY")
# GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
# GCS_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Initialize Google Cloud Storage client
# if GCS_CREDENTIALS_PATH:
#     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCS_CREDENTIALS_PATH
#     storage_client = storage.Client()

# Initialize the agent
multimodal_Agent = Agent(
    name="Video AI Summarizer",
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGo()],
    markdown=True,
)

# def upload_to_gcs(file_path, bucket_name):
#     """Uploads a file to Google Cloud Storage."""
#     bucket = storage_client.bucket(bucket_name)
#     blob_name = os.path.basename(file_path)
#     blob = bucket.blob(blob_name)
#     blob.upload_from_filename(file_path)
#     return f"gs://{bucket_name}/{blob_name}"

def analyze_video(video_file, user_query):
    """Analyzes a video file based on user input."""
    if not user_query:
        return "Please provide a question or insight to analyze the video."

    video_path = video_file.name

    try:
        # if os.path.getsize(video_path) > 200 * 1024 * 1024:  # File size > 200MB
        #     try:
        #         gcs_uri = upload_to_gcs(video_path, GCS_BUCKET_NAME)
        #         analysis_prompt = (
        #             f"""
        #             Analyze the video located at {gcs_uri} for content and context.
        #             Respond to the following query using video insights and supplementary web research:
        #             {user_query}

        #             Provide a detailed, user-friendly, and actionable response.
        #             """
        #         )
        #         # AI agent processing with GCS URI
        #         response = multimodal_Agent.run(analysis_prompt)
        #         return response.content

        #     except Exception as e:
        #         return f"Error uploading to Google Cloud Storage: {e}"

        # Process smaller files locally
        processed_video = upload_file(video_path)
        while processed_video.state.name == "PROCESSING":
            time.sleep(1)
            processed_video = get_file(processed_video.name)

        # Prompt generation for analysis
        analysis_prompt = (
            f"""
            Analyze the uploaded video for content and context.
            Respond to the following query using video insights and supplementary web research:
            {user_query}

            Provide a detailed, user-friendly, and actionable response.
            """
        )

        # AI agent processing
        response = multimodal_Agent.run(analysis_prompt, videos=[processed_video])
        return response.content

    except Exception as error:
        return f"An error occurred during analysis: {error}"

# Gradio Interface
def gradio_interface():
    with gr.Blocks() as app:
        gr.Markdown("""#Google gemini AI Summarizer Agent using phidata""")

        with gr.Row():
            video_input = gr.File(label="Upload a video file", file_types=['.mp4', '.mov', '.avi'], interactive=True)
            user_query = gr.Textbox(label="What insights are you seeking from the video?", placeholder="Ask anything about the video content.", lines=4)

        analyze_button = gr.Button("🔍 Analyze Video")
        output = gr.Markdown()

        analyze_button.click(analyze_video, inputs=[video_input, user_query], outputs=output)

    return app

# Run Gradio App
if __name__ == "__main__":
    gradio_interface().launch()
