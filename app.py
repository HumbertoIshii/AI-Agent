from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, load_tool, tool
from transformers import pipeline
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI
from itertools import islice
from youtube_comment_downloader import *  # type: ignore

# Load the Hugging Face sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

# Custom Tool to Fetch YouTube Comments
@tool
def get_yt_comment(link: str, max_comments: int = 50) -> str:
    """A tool that fetches comments from a YouTube video and returns them as a single string with a descriptive context.
    Args:
        link: The YouTube video URL.
        max_comments: The maximum number of comments to retrieve.
    """
    try:
        downloader = YoutubeCommentDownloader()  # type: ignore
        comments = []
        for comment in downloader.get_comments_from_url(link, sort_by=SORT_BY_POPULAR):  # type: ignore
            comments.append(comment['text'])
            if len(comments) >= max_comments:
                break
        
        # Prepend a descriptive sentence and join the comments into a single string
        comment_string = "\n".join(comments)
        return f"These are the top {len(comments)} comments from the video:\n{comment_string}"
    except Exception as e:
        return f"Error fetching comments: {str(e)}"

@tool
def analyze_sentiment_of_comments(comments: str) -> str:
    """A tool that analyzes the sentiment of YouTube comments to determine the overall reception.
    Args:
        comments: A string containing the YouTube comments.
    """
    try:
        # Analyze the sentiment of each comment
        sentiments = sentiment_analyzer(comments)

        # Count the number of positive and negative sentiments
        positive_count = sum(1 for sentiment in sentiments if sentiment['label'] == 'POSITIVE')
        negative_count = sum(1 for sentiment in sentiments if sentiment['label'] == 'NEGATIVE')

        # Determine overall reception
        if positive_count > negative_count:
            return f"The overall reception is positive. Positive comments: {positive_count}, Negative comments: {negative_count}."
        elif negative_count > positive_count:
            return f"The overall reception is negative. Positive comments: {positive_count}, Negative comments: {negative_count}."
        else:
            return f"The overall reception is neutral. Positive comments: {positive_count}, Negative comments: {negative_count}."
    except Exception as e:
        return f"Error during sentiment analysis: {str(e)}"

final_answer = FinalAnswerTool()

# Hugging Face Model Setup
model = HfApiModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',  # It is possible that this model may be overloaded
    custom_role_conversions=None,
)

# Import image generation tool from Hub (though it’s not used here directly)
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

# Load prompt templates (you might want to ensure this file exists)
with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

# Define the agent with the new tool
agent = CodeAgent(
    model=model,
    tools=[final_answer, get_yt_comment, analyze_sentiment_of_comments],  # Added the new tool to the tools list
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

# Launch the Gradio UI
GradioUI(agent).launch()