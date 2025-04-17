# imports

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import base64
from io import BytesIO
from PIL import Image
from IPython.display import Audio, display
import pygame
import time
from tools import price_function, get_ticket_price
# And this is included in a list of tools:

tools = [{"type": "function", "function": price_function}]

# System messages
system_message = "You are a helpful assistant for an Airline called FlightAI. "
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."

# Initialization

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
MODEL = "gpt-4o-mini"
openai = OpenAI()

def chat(history):
    messages = [{"role": "system", "content": system_message}] + history
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    image = None
    
    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        image = artist(city)
        response = openai.chat.completions.create(model=MODEL, messages=messages)
        
    reply = response.choices[0].message.content
    history += [{"role":"assistant", "content":reply}]

    # Comment out or delete the next line if you'd rather skip Audio for now..
    talker(reply)
    
    return history, image

# We have to write that function handle_tool_call:

def handle_tool_call(message):
    print(f"Handling tool call: {message}")
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    city = arguments.get('destination_city')
    price = get_ticket_price(city)
    response = {
        "role": "tool",
        "content": json.dumps({"destination_city": city,"price": price}),
        "tool_call_id": tool_call.id
    }
    return response, city

def artist(city):
    image_response = openai.images.generate(
            model="dall-e-3",
            prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))

def talker(message):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=message)

    audio_stream = BytesIO(response.content)
    output_filename = f"output_audio_{time.time()}.mp3"
    with open(output_filename, "wb") as f:
        f.write(audio_stream.read())

    # Play the generated audio
    # display(Audio(output_filename, autoplay=True)) # This code is suitable for Juopyter
    print(f"Created audio file at {output_filename}")

    # Using pygame
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(output_filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue



if __name__ == "__main__":
    # gr.ChatInterface(fn=chat, type="messages").launch()
    # talker("Hello, how are you?")
    # Passing in inbrowser=True in the last line will cause a Gradio window to pop up immediately.

    with gr.Blocks() as ui:
        with gr.Row():
            chatbot = gr.Chatbot(height=500, type="messages")
            image_output = gr.Image(height=500)
        with gr.Row():
            entry = gr.Textbox(label="Chat with our AI Assistant:")
        with gr.Row():
            clear = gr.Button("Clear")

        def do_entry(message, history):
            history += [{"role":"user", "content":message}]
            return "", history

        entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
            chat, inputs=chatbot, outputs=[chatbot, image_output]
        )
        clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

    ui.launch(inbrowser=True)