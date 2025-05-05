import os
import requests
import json
import time
import argparse
import sys
import platform
import subprocess

# Try to import playsound, with fallback options
try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    PLAYSOUND_AVAILABLE = False
    print("Warning: playsound module not found. Audio playback will be disabled.")
    print("To enable playback, install playsound: pip install playsound==1.2.2")

def play_audio_file(file_path):
    """
    Play an audio file using the appropriate method for the current platform
    
    Args:
        file_path (str): Path to the audio file to play
    """
    system = platform.system()
    
    try:
        if system == "Darwin":  # macOS
            # Use the 'afplay' command which is built into macOS
            subprocess.run(["afplay", file_path], check=True)
            print("Audio playback complete.")
        elif system == "Windows":
            # Use playsound on Windows
            if PLAYSOUND_AVAILABLE:
                playsound(file_path)
            else:
                # Fallback to using the 'start' command
                os.startfile(file_path)
        else:  # Linux and others
            # Try to use playsound
            if PLAYSOUND_AVAILABLE:
                playsound(file_path)
            else:
                # Try to use common Linux audio players
                for player in ["xdg-open", "aplay", "paplay"]:
                    try:
                        subprocess.run([player, file_path], check=True)
                        break
                    except (subprocess.SubprocessError, FileNotFoundError):
                        continue
    except Exception as e:
        print(f"Error playing audio: {str(e)}")
        print("You can still find the audio file at:", file_path)
        print("Try playing it manually using your system's media player.")

               

def text_to_speech(text, api_key, voice="alloy", model="tts-1", output_dir="output", play_audio=False):
    """
    Convert text to speech using OpenAI's Text-to-Speech API
    
    Args:
        text (str): The text to convert to speech
        api_key (str): Your OpenAI API key
        voice (str): The voice to use (alloy, echo, fable, onyx, nova, shimmer)
        model (str): The model to use (tts-1 or tts-1-hd)
        output_dir (str): Directory to save the audio file
        play_audio (bool): Whether to play the audio after generating it
        
    Returns:
        str: Path to the generated audio file
    """
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # API endpoint
        url = "https://api.openai.com/v1/audio/speech"
        
        # Headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Request data
        data = {
            "model": model,
            "voice": voice,
            "input": text
        }
        
        print(f"Converting text to speech using voice: {voice}")
        
        # Make the request
        response = requests.post(url, headers=headers, json=data)
        
        # Check if successful
        if response.status_code == 200:
            # Generate a filename with timestamp
            timestamp = int(time.time())
            output_file = os.path.join(output_dir, f"speech_{timestamp}.mp3")
            
            # Save the audio file
            with open(output_file, "wb") as file:
                file.write(response.content)
            
            print(f"Speech saved to: {output_file}")
            
            # # Play the audio if requested
            # if play_audio and PLAYSOUND_AVAILABLE:
            #     try:
            #         print("Playing audio...")
            #         playsound(output_file)
            #         print("Audio playback complete.")
            #     except Exception as e:
            #         print(f"Error playing audio: {str(e)}")
            #         print("You can still find the audio file at:", output_file)
            
            # Play the audio if requested
            if play_audio:
                print("Playing audio...")
                play_audio_file(output_file)

            return output_file
        else:
            print(f"API error: {response.status_code}, {response.text}")
            return None
    
    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")
        return None

def main():
    # Default API key - CHANGE THIS to your actual API key
    DEFAULT_API_KEY = ""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert text to speech using OpenAI API")
    parser.add_argument("--text", type=str, help="Text to convert to speech")
    parser.add_argument("--file", type=str, help="File containing text to convert")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="OpenAI API key")
    parser.add_argument("--voice", type=str, default="alloy", 
                        choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                        help="Voice to use")
    parser.add_argument("--model", type=str, default="tts-1", 
                        choices=["tts-1", "tts-1-hd"],
                        help="Model to use")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save the audio file")
    parser.add_argument("--play", action="store_true",
                        help="Play the audio after generating it")
    
    # Parse arguments
    args = parser.parse_args()

    args.text= "hello. I am ok. what is the plan for today?"

    args.play= True
    
    # Get text from either command line or file
    text = args.text
    if not text and args.file:
        try:
            with open(args.file, "r") as file:
                text = file.read()
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return
    
    if not text:
        print("Error: No text provided. Use --text or --file to provide text.")
        return
    
    # Convert text to speech
    output_file = text_to_speech(
        text=text,
        api_key=args.api_key,
        voice=args.voice,
        model=args.model,
        output_dir=args.output_dir,
        play_audio=args.play
    )
    
    if output_file:
        print(f"Successfully generated speech file: {output_file}")
    else:
        print("Failed to generate speech.")

if __name__ == "__main__":
    main()