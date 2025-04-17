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
from tools import price_function, get_ticket_price, make_a_booking, booking_function
import ollama
# And this is included in a list of tools:

tools = [{"type": "function", "function": price_function}, {"type": "function", "function": booking_function}]
# tools = [price_function, booking_function]

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
    
    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        # Only generate image for price checks, not for bookings
        if message.tool_calls[0].function.name == "get_ticket_price":
            # image = artist(city)
            pass
        response = openai.chat.completions.create(model=MODEL, messages=messages)
        
    reply = response.choices[0].message.content
    history += [{"role":"assistant", "content":reply}]

    talker(reply)
    translated_reply = ollama_translator(reply)
    
    return history, image

# We have to write that function handle_tool_call:

def handle_tool_call(message):
    print(f"Handling tool call: {message}")
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    if function_name == "get_ticket_price":
        city = arguments.get('destination_city')
        price = get_ticket_price(city)
        response = {
            "role": "tool",
            "content": json.dumps({"destination_city": city, "price": price}),
            "tool_call_id": tool_call.id
        }
        return response, city
        
    elif function_name == "make_a_booking":
        city = arguments.get('destination_city')
        customer_name = arguments.get('customer_name')
        customer_id = arguments.get('customer_id')
        booking_result = make_a_booking(city, customer_name, customer_id)
        response = {
            "role": "tool",
            "content": json.dumps({
                "destination_city": city,
                "customer_name": customer_name,
                "customer_id": customer_id,
                "booking_result": booking_result
            }),
            "tool_call_id": tool_call.id
        }
        return response, city
        
    else:
        raise ValueError(f"Unknown function: {function_name}")

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

def ollama_translator(text, target_language="German"):
    """
    Translates text to the specified language using Ollama.
    
    Args:
        text (str): The text to translate
        target_language (str): The language to translate to (default: Arabic)
    
    Returns:
        str: The translated text
    """
    try:
        # Create a prompt that instructs the model to translate
        prompt = f"Translate the following text to {target_language}. Only output the translation, nothing else:\n\n{text}"
        
        response = ollama.chat(
            model='llama3.2:latest',  # or any other model you have installed
            messages=[
                {"role": "system", "content": "You are a professional translator. Translate the given text accurately."},
                {"role": "user", "content": prompt}
            ]
        )
        
        translated_text = response['message']['content'].strip()
        return translated_text
        
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return f"Translation failed: {str(e)}"
    
def translate_message(history):
    """
    Translates the last message in the chat history.
    
    Args:
        history (list): List of chat messages
        
    Returns:
        str: Translated text of the last message
    """
    if not history:
        return ""
    
    # Get the last message from history
    last_message = history[-1]
    
    # Extract the content from the last message
    message_content = last_message.get('content', '')
    
    if message_content:
        return ollama_translator(message_content)
    return ""

def clear_chat():
    return [], ""

if __name__ == "__main__":
    # gr.ChatInterface(fn=chat, type="messages").launch()
    # talker("Hello, how are you?")
    # Passing in inbrowser=True in the last line will cause a Gradio window to pop up immediately.

    with gr.Blocks() as ui:
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(height=500, type="messages")
                entry = gr.Textbox(label="Chat with our AI Assistant:")
                clear = gr.Button("Clear")
            with gr.Column():
                image_output = gr.Image(height=500)
            with gr.Column():
                translation_output = gr.Textbox(label="Translation (German):", lines=15)

        def do_entry(message, history):
            history += [{"role":"user", "content":message}]
            return "", history

        entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
            chat, inputs=chatbot, outputs=[chatbot, image_output]
        ).then(
            translate_message, inputs=chatbot, outputs=translation_output
        )
        
        clear.click(clear_chat, inputs=None, outputs=[chatbot, translation_output])

    ui.launch(inbrowser=False)

    # print(ollama_translator("Hello, how are you?"))