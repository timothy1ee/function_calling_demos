from dotenv import load_dotenv
import chainlit as cl
from movie_functions import get_now_playing_movies
import json

load_dotenv()

# Note: If switching to LangSmith, uncomment the following, and replace @observe with @traceable
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable
# client = wrap_openai(openai.AsyncClient())

from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI
 
client = AsyncOpenAI()

gen_kwargs = {
    "model": "gpt-4o",
    "temperature": 0.2,
    "max_tokens": 500
}

SYSTEM_PROMPT = """\
You are a helpful assistant with knowledge about movies. Your primary tasks are:

1. Respond to user queries about movies and provide information.
2. Detect when a user is requesting a list of current movies.
3. When currently playing movies are requested, generate a function call instead of a direct response.

For list current movies requests, use the following function call format:
{"function": "get_current_movies", "parameters": {}}

Only use this function call when the user explicitly asks for current or now playing movies. For all other queries, respond directly to the user with your knowledge about movies.

After receiving the results of a function call, incorporate that information into your response to the user.
"""

@observe
@cl.on_chat_start
def on_chat_start():    
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)

@observe
async def generate_response(client, message_history, gen_kwargs):
    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(messages=message_history, stream=True, **gen_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)
    
    await response_message.update()

    return response_message

@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})
    
    response_message = await generate_response(client, message_history, gen_kwargs)

    # Function calls look like this: {"function": "get_current_movies", "parameters": {}}
    # Check if the response is a function call
    try:
        function_call = json.loads(response_message.content)
        if "function" in function_call and function_call["function"] == "get_current_movies":
            # Call the function from movie_functions.py
            current_movies = get_now_playing_movies()
            
            # Append the function result to the message history
            message_history.append({"role": "system", "content": f"Result of get_current_movies function call:\n\n{current_movies}"})
            
            # Generate a new response incorporating the function results
            response_message = await generate_response(client, message_history, gen_kwargs)
    except json.JSONDecodeError:
        # If it's not a valid JSON, it's not a function call, so we do nothing
        pass

    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)

if __name__ == "__main__":
    cl.main()
