import os
import time
import numpy as np
import requests
import json
import base64
from flask import Flask, render_template, request, jsonify, url_for

from text_to_speech_openai import play_audio_file


# Import our custom modules
from mood_detector import MoodDetector
from response_generator import ResponseGenerator

# Initialize Flask app
app = Flask(__name__)

# Set your OpenAI API key
API_KEY = ""

# Initialize our mood detection and response generation systems
mood_detector = MoodDetector(API_KEY)
response_generator = ResponseGenerator(API_KEY)


# Make sure static folder exists
if not os.path.exists('static'):
    os.makedirs('static')

# Initialize conversation history
conversation_history = []
mood_history = []

# Add this audio preprocessing function before sending to Whisper
def preprocess_audio(input_file, output_file):
    try:
        # Using ffmpeg (you'll need to install it)
        os.system(f'ffmpeg -i {input_file} -ar 16000 -ac 1 -c:a pcm_s16le {output_file}')
        return output_file
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return input_file  # Return original if processing fails



def speech_to_text(audio_file_path, api_key):
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
            "model": "tts-1", # "tts-1",
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
        
        # After saving the file, verify it exists and log the size
        if os.path.exists(speech_file):
            file_size = os.path.getsize(speech_file)
            print(f"TTS file created successfully: {speech_file}, size: {file_size} bytes")
        else:
            print(f"TTS file was not created: {speech_file}")
    
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
        
        print('old temp_path', temp_path)
        ### convert the temp file for the format of interest
        output_path = './temp_upload_converted.wav'
        if os.path.exists(output_path):
            os.remove(output_path)
        temp_path = preprocess_audio( temp_path, output_path)
        print('new temp path', temp_path)

        # Process the audio
        transcript = speech_to_text(temp_path, API_KEY)
        print(f"Transcript: {transcript}")
        
        if transcript.startswith("Error:"):
            return jsonify({"error": transcript}), 500
        
        # Detect mood
        # mood = detect_mood(temp_path, transcript)
        # print(f"Detected mood: {mood}")
        
        # Detect mood using our advanced mood detector
        mood = mood_detector.detect_mood(temp_path, transcript)
        print(f"Detected mood: {mood}")
        
        # Add mood to history
        mood_history.append(mood)

        # Update conversation history
        current_exchange = {"user": transcript}
        
        # Generate response
        # response_text = generate_response(transcript, mood, conversation_history)
        # print(f"Response: {response_text}")

        # Generate personalized response
        response_text = response_generator.generate_personalized_response(transcript, mood, conversation_history)
        print(f"Response: {response_text}")
        
        # Complete the exchange record
        current_exchange["assistant"] = response_text
        conversation_history.append(current_exchange)
        
        # Convert response to speech
        speech_file_path = text_to_speech(response_text, mood)

        ### playback the speech file
        play_audio_file(speech_file_path)

        
        # Create a URL for the audio file
        if speech_file_path:
            speech_url = url_for('static', filename=os.path.basename(speech_file_path))
        else:
            speech_url = None
        

        # Generate voice improvement tips if appropriate
        voice_tips = []
        if len(mood_history) > 2:
            # Only generate tips after a few exchanges
            acoustic_features = mood_detector.extract_acoustic_features(temp_path)
            if mood['valence'] < 5 or mood['arousal'] > 7 or mood['arousal'] < 3:
                voice_tips = response_generator.get_voice_improvement_suggestions(mood, acoustic_features)
        

        # Return all the processed data
        return jsonify({
            "transcript": transcript,
            "mood": mood,
            "response": response_text,
            "audio_url": speech_url,
            "voice_tips": voice_tips if voice_tips else []
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


@app.route('/mood_history', methods=['GET'])
def get_mood_history():
    """Return the mood history for visualization/tracking"""
    history_data = []
    
    for i, mood in enumerate(mood_history):
        history_data.append({
            'exchange_number': i + 1,
            'valence': mood['valence'],
            'arousal': mood['arousal'],
            'primary_emotion': mood['primary_emotion']
        })
    
    return jsonify(history_data)

@app.route('/voice_tips', methods=['GET'])
def get_voice_tips():
    """Get specific voice improvement tips"""
    if not mood_history:
        return jsonify({"error": "No mood data available yet"}), 400
    
    # Get the most recent mood
    latest_mood = mood_history[-1]
    
    # Get the most recent audio file
    temp_path = "temp_upload.wav"
    if not os.path.exists(temp_path):
        return jsonify({"error": "No recent audio available"}), 400
    
    # Extract acoustic features
    acoustic_features = mood_detector.extract_acoustic_features(temp_path)
    
    # Generate tips
    tips = response_generator.get_voice_improvement_suggestions(latest_mood, acoustic_features)
    
    return jsonify({
        "current_mood": latest_mood,
        "voice_tips": tips
    })

if __name__ == '__main__':
    app.run(debug=True)