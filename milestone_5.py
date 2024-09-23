from dotenv import load_dotenv
import chainlit as cl
from movie_functions import get_now_playing_movies, get_showtimes

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
You are a helpful movie chatbot that helps people explore movies that are out in \
theaters. If a user asks for recent information, output a function call and \
the system add to the context. If you need to call a function, only output the \
function call. Call functions using Python syntax in plain text, no code blocks.

You have access to the following functions:

get_now_playing()
get_showtimes(title, location)
buy_ticket(theater, movie, showtime)
confirm_ticket_purchase(theater, movie, showtime)
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

    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)

    while True:
        if "get_now_playing()" in response_message.content:
            now_playing_movies = get_now_playing_movies()
            message_history.append({"role": "system", "content": now_playing_movies})
        elif "get_showtimes(" in response_message.content:
            try:
                start = response_message.content.index("get_showtimes(") + len("get_showtimes(")
                end = response_message.content.index(")", start)
                args = response_message.content[start:end].split(",")
                print(f"Debug: Extracted args: {args}")
                
                title = args[0].strip().strip("'\"")
                location = args[1].strip().strip("'\"")
                print(f"Debug: Extracted title: {title}, location: {location}")
            except Exception as e:
                error = f"Debug: Error extracting arguments: {str(e)}"
                print(error)
                showtimes = error

            try:
                showtimes = get_showtimes(title, location)
            except Exception as e:
                error = f"Debug: Error calling get_showtimes(): {str(e)}"
                print(error)
                showtimes = error

            message_history.append({"role": "system", "content": showtimes})
            print(f"Debug: Added showtimes to message history. New length: {len(message_history)}")
        elif "buy_ticket(" in response_message.content:
            try:
                start = response_message.content.index("buy_ticket(") + len("buy_ticket(")
                end = response_message.content.index(")", start)
                args = response_message.content[start:end].split(",")
                print(f"Debug: Extracted args for buy_ticket: {args}")
                
                theater = args[0].strip().strip("'\"")
                movie = args[1].strip().strip("'\"")
                showtime = args[2].strip().strip("'\"")
                print(f"Debug: Extracted theater: {theater}, movie: {movie}, showtime: {showtime}")
                
                purchase_result = f"Ask user for purchase ticket confirmation for {movie} at {theater} for {showtime}."
                message_history.append({"role": "system", "content": purchase_result})
            except Exception as e:
                error = f"Debug: Error processing buy_ticket(): {str(e)}"
                print(error)
                message_history.append({"role": "system", "content": error})
        elif "confirm_ticket_purchase(" in response_message.content:
            try:
                start = response_message.content.index("confirm_ticket_purchase(") + len("confirm_ticket_purchase(")
                end = response_message.content.index(")", start)
                args = response_message.content[start:end].split(",")
                print(f"Debug: Extracted args for confirm_ticket_purchase: {args}")
                
                theater = args[0].strip().strip("'\"")
                movie = args[1].strip().strip("'\"")
                showtime = args[2].strip().strip("'\"")
                print(f"Debug: Extracted theater: {theater}, movie: {movie}, showtime: {showtime}")
                
                purchase_result = f"Ticket purchased for {movie} at {theater} for {showtime}."
                message_history.append({"role": "system", "content": purchase_result})
            except Exception as e:
                error = f"Debug: Error processing buy_ticket(): {str(e)}"
                print(error)
                message_history.append({"role": "system", "content": error})
        else:
            break  # Exit the loop if response_message is not one of the cases

        # Stream another response with the added system messages with the function call output
        response_message = await generate_response(client, message_history, gen_kwargs)
        message_history.append({"role": "assistant", "content": response_message.content})
        cl.user_session.set("message_history", message_history)
    
if __name__ == "__main__":
    cl.main()
