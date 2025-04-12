import os
import time
import numpy as np
import requests
import json
import base64
from flask import Flask, render_template, request, jsonify, url_for

# Initialize Flask app
app = Flask(__name__)

# Set your OpenAI API key
API_KEY = ""

# Make sure static folder exists
if not os.path.exists('static'):
    os.makedirs('static')

# Initialize conversation history
conversation_history = []

# # Direct implementation of Whisper API
def speech_to_text(audio_file_path):
    try:
        # API endpoint
        url = "https://api.openai.com/v1/audio/transcriptions"
        
        # Headers with authorization
        headers = {
            "Authorization": f"Bearer {API_KEY}"
        }
        
        # Prepare the file and form data
        with open(audio_file_path, "rb") as audio_file:
            files = {
                "file": (os.path.basename(audio_file_path), audio_file, "audio/wav")
            }
            data = {
                "model": "whisper-1",
                # "model" : "gpt-4o-transcribe",
                "language": "en",  # Specify the language if you know it
            }
            
            print(f"Sending direct API request to Whisper API: {audio_file_path}")
            
            # Make the request
            response = requests.post(url, headers=headers, files=files, data=data)
            
            # Check if successful
            if response.status_code == 200:
                result = response.json()
                transcript = result.get("text", "")
                print(f"Transcript received: {transcript}")
                return transcript
            else:
                print(f"API error: {response.status_code}, {response.text}")
                return f"Error: {response.status_code}"
    
    except Exception as e:
        print(f"Error in speech-to-text: {str(e)}")
        return f"Error: {str(e)}"


def speech_to_text_direct(audio_file_path, api_key):
    """
    Directly call the OpenAI Whisper API without relying on SDK abstractions
    """
    try:
        # API endpoint
        url = "https://api.openai.com/v1/audio/transcriptions"
        
        # Headers with authorization
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        # Prepare the file and form data
        with open(audio_file_path, "rb") as audio_file:
            files = {
                "file": (os.path.basename(audio_file_path), audio_file, "audio/wav")
            }
            data = {
                "model": "whisper-1"
            }
            
            print(f"Sending direct API request to Whisper API: {audio_file_path}")
            
            # Make the request
            response = requests.post(url, headers=headers, files=files, data=data)
            
            # Check if successful
            if response.status_code == 200:
                result = response.json()
                transcript = result.get("text", "")
                print(f"Transcript received: {transcript}")
                return transcript
            else:
                print(f"API error: {response.status_code}, {response.text}")
                return f"Error: {response.status_code}"
    
    except Exception as e:
        print(f"Error in direct speech-to-text: {str(e)}")
        return f"Error: {str(e)}"


# Simplified mood detection (since we're focusing on the API integration)
def detect_mood(audio_file_path, transcript):
    # Simplified version for testing API integration
    # In a real implementation, you would use librosa or another library
    
    # For now, we'll use a simple sentiment analysis based on the transcript
    sentiment_score = 5  # Neutral default
    
    # Very simple keyword detection (just for demonstration)
    positive_words = ["happy", "good", "great", "excellent", "wonderful", "love", "like"]
    negative_words = ["sad", "bad", "terrible", "awful", "hate", "dislike", "angry"]
    
    words = transcript.lower().split()
    
    # Adjust sentiment based on keyword presence
    for word in words:
        if word in positive_words:
            sentiment_score += 1
        if word in negative_words:
            sentiment_score -= 1
    
    # Keep within 1-10 range
    sentiment_score = max(1, min(10, sentiment_score))
    
    # Determine primary emotion (very simplified)
    if sentiment_score >= 7:
        primary_emotion = "happy"
    elif sentiment_score <= 3:
        primary_emotion = "sad"
    else:
        primary_emotion = "neutral"
    
    # Create mood object
    mood = {
        'valence': sentiment_score,  # 1-10 (negative to positive)
        'arousal': 5,  # Default to neutral arousal
        'primary_emotion': primary_emotion
    }
    
    return mood

# Direct implementation of GPT API for chat
def generate_response(transcript, mood, conversation_history):
    try:
        # API endpoint
        url = "https://api.openai.com/v1/chat/completions"
        
        # Headers
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Add detected mood to the prompt
        mood_context = f"The user's detected emotional state is: {mood['primary_emotion']} with valence (positivity) of {mood['valence']}/10."
        
        # Create a system message
        system_message = (
            "You are an empathetic AI assistant designed to help improve the user's mood through conversation. "
            f"{mood_context} "
            "If they seem negative, gradually guide them toward a more positive outlook. "
            "If they're already positive, reinforce and build on that. "
            "Keep responses concise (2-3 sentences) and conversational."
        )
        
        # Build the messages array
        messages = [
            {"role": "system", "content": system_message}
        ]
        
        # Add conversation history (limit to last 5 exchanges)
        for exchange in conversation_history[-5:]:
            messages.append({"role": "user", "content": exchange["user"]})
            if "assistant" in exchange:
                messages.append({"role": "assistant", "content": exchange["assistant"]})
        
        # Add the current user message
        messages.append({"role": "user", "content": transcript})
        
        # Prepare request data
        data = {
            "model": "gpt-4",
            "messages": messages
        }
        
        # Make the request
        response = requests.post(url, headers=headers, json=data)
        
        # Check if successful
        if response.status_code == 200:
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]
            return response_text
        else:
            print(f"GPT API error: {response.status_code}, {response.text}")
            return "I'm sorry, I'm having trouble responding right now."
    
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "I'm having trouble processing that right now. Could you please try again?"

# Direct implementation of TTS API
def text_to_speech(text, mood=None):
    try:
        # API endpoint
        url = "https://api.openai.com/v1/audio/speech"
        
        # Headers
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Select voice based on mood
        voice = "alloy"  # Default
        if mood:
            valence = mood.get('valence', 5)
            if valence < 4:  # User is negative
                voice = "nova"  # More warm and empathetic
            elif mood.get('arousal', 5) > 7:  # User is excited
                voice = "shimmer"  # More energetic
        
        # Prepare request data
        data = {
            "model": "gpt-4o-mini-tts", # "tts-1",
            "voice": voice,
            "input": text
        }
        
        # Make the request
        response = requests.post(url, headers=headers, json=data)
        
        # Check if successful
        if response.status_code == 200:
            # Save to a file in the static folder for serving
            speech_file = f"static/response_{int(time.time())}.mp3"
            with open(speech_file, "wb") as file:
                file.write(response.content)
            
            return speech_file
        else:
            print(f"TTS API error: {response.status_code}, {response.text}")
            return None
    
    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")
        return None

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    print("Audio processing request received")
    
    if 'audio' not in request.files and 'audio_data' not in request.form:
        print("No audio file or data provided")
        return jsonify({"error": "No audio file or data provided"}), 400
    
    temp_path = "temp_upload.wav"
    
    try:
        # Check if we have a file upload or base64 data
        if 'audio' in request.files:
            # Direct file upload
            audio_file = request.files['audio']
            audio_file.save(temp_path)
            print(f"Saved uploaded file to {temp_path}")
        else:
            # Base64 encoded audio data
            audio_data = request.form['audio_data']
            # Remove the data URL prefix if present
            if ',' in audio_data:
                audio_data = audio_data.split(',')[1]
            
            # Decode base64 data
            audio_bytes = base64.b64decode(audio_data)
            
            # Save to a temporary WAV file
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)
            print(f"Saved decoded base64 data to {temp_path}")
        
        # Process the audio
        # transcript = speech_to_text(temp_path)
        transcript = speech_to_text_direct(temp_path, API_KEY)
        print(f"Transcript: {transcript}")
        
        if transcript.startswith("Error:"):
            return jsonify({"error": transcript}), 500
        
        # Detect mood
        mood = detect_mood(temp_path, transcript)
        print(f"Detected mood: {mood}")
        
        # Update conversation history
        current_exchange = {"user": transcript}
        
        # Generate response
        response_text = generate_response(transcript, mood, conversation_history)
        print(f"Response: {response_text}")
        
        # Complete the exchange record
        current_exchange["assistant"] = response_text
        conversation_history.append(current_exchange)
        
        # Convert response to speech
        speech_file_path = text_to_speech(response_text, mood)
        
        # Create a URL for the audio file
        if speech_file_path:
            speech_url = url_for('static', filename=os.path.basename(speech_file_path))
        else:
            speech_url = None
        
        # Return all the processed data
        return jsonify({
            "transcript": transcript,
            "mood": mood,
            "response": response_text,
            "audio_url": speech_url
        })
        
    except Exception as e:
        print(f"Error in process_audio: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

if __name__ == '__main__':
    app.run(debug=True)