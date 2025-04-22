import os
import numpy as np
import librosa
import joblib
import json
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class MoodDetector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.scaler = None
        self.model = None
        
        # Initialize or load the ML model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or load the acoustic feature model"""
        # Check if we have a saved model
        if os.path.exists('mood_model.pkl') and os.path.exists('mood_scaler.pkl'):
            print("Loading existing mood detection model...")
            self.model = joblib.load('mood_model.pkl')
            self.scaler = joblib.load('mood_scaler.pkl')
        else:
            print("No existing model found. Will train on first few samples.")
            # We'll initialize with a basic random forest that will be trained incrementally
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            self.scaler = StandardScaler()
    
    # def extract_acoustic_features(self, audio_file_path):
    #     """Extract acoustic features from audio file"""
    #     try:
    #         # Load the audio file
    #         y, sr = librosa.load(audio_file_path, sr=22050)
            
    #         # Feature set for emotion recognition:
            
    #         # 1. Pitch (Fundamental frequency)
    #         pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    #         pitch = np.mean(pitches[magnitudes > np.max(magnitudes) * 0.7]) if np.any(magnitudes > np.max(magnitudes) * 0.7) else 0
            
    #         # 2. Energy/volume
    #         energy = np.mean(librosa.feature.rms(y=y))
            
    #         # 3. Speech rate (zero-crossing rate as proxy)
    #         speech_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
            
    #         # 4. Spectral features
    #         spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    #         spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    #         spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            
    #         # 5. MFCCs (Mel-frequency cepstral coefficients)
    #         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    #         mfcc_means = np.mean(mfccs, axis=1)
            
    #         # 6. Chroma features (related to musical perception)
    #         chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    #         chroma_means = np.mean(chroma, axis=1)
            
    #         # 7. Rhythm features
    #         tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
    #         # 8. Voice quality - jitter and shimmer approximation
    #         # (using frame-to-frame variations in pitch and amplitude)
    #         if len(y) > sr // 10:  # Ensure audio is long enough
    #             frames = librosa.util.frame(y, frame_length=sr//10, hop_length=sr//20)
    #             frame_energies = np.sqrt(np.sum(frames**2, axis=0))
    #             frame_energy_variations = np.std(frame_energies) / np.mean(frame_energies) if np.mean(frame_energies) > 0 else 0
    #         else:
    #             frame_energy_variations = 0
            
    #         # Combine all features into a single vector
    #         features = np.hstack([
    #             pitch if not np.isnan(pitch) else 0,
    #             energy,
    #             speech_rate,
    #             spectral_centroid,
    #             spectral_bandwidth,
    #             spectral_rolloff,
    #             mfcc_means,
    #             chroma_means,
    #             tempo,
    #             frame_energy_variations
    #         ])
            
    #         # Log main features for debugging
    #         print(f"Key acoustic features - Pitch: {pitch:.2f}, Energy: {energy:.2f}, Speech rate: {speech_rate:.2f}, Tempo: {tempo:.2f}")
            
    #         return features
            
    #     except Exception as e:
    #         print(f"Error extracting acoustic features: {str(e)}")
    #         # Return a default feature vector if extraction fails
    #         return np.zeros(25)  # Adjust size based on your feature vector length
    
    def extract_acoustic_features(self, audio_file_path):
        """Extract acoustic features from audio file"""
        try:
            # Check if file exists and has content
            if not os.path.exists(audio_file_path):
                print(f"Audio file does not exist: {audio_file_path}")
                return np.zeros(25)
                
            file_size = os.path.getsize(audio_file_path)
            print(f"Audio file size: {file_size} bytes")
            
            if file_size == 0:
                print("Audio file is empty")
                return np.zeros(25)
            
            # Try loading with different parameters
            print("Attempting to load audio file...")
            try:
                y, sr = librosa.load(audio_file_path, sr=22050, mono=True, res_type='kaiser_fast')
                print(f"File loaded successfully. Length: {len(y)}, Sample rate: {sr}")
            except Exception as load_err:
                print(f"Error loading file with librosa: {str(load_err)}")
                # Try a different approach if loading fails
                try:
                    import soundfile as sf
                    y, sr = sf.read(audio_file_path)
                    print(f"File loaded with soundfile. Length: {len(y)}, Sample rate: {sr}")
                except Exception as sf_err:
                    print(f"Also failed with soundfile: {str(sf_err)}")
                    return np.zeros(25)
            
            # Check if the loaded audio has content
            if len(y) == 0:
                print("Loaded audio has zero length")
                return np.zeros(25)
                
            # Extract basic features that are less likely to fail
            print("Extracting basic features...")
            features = []
            
            # 1. Energy/volume (most basic feature)
            try:
                energy = np.mean(librosa.feature.rms(y=y))
                features.append(float(energy))
                print(f"Energy extracted: {energy}")
            except Exception as e:
                print(f"Energy extraction failed: {str(e)}")
                features.append(0.5)
            
            # 2. Zero crossing rate (basic feature)
            try:
                zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
                features.append(float(zcr))
                print(f"ZCR extracted: {zcr}")
            except Exception as e:
                print(f"ZCR extraction failed: {str(e)}")
                features.append(0.5)
            
            # Additional features (with individual try/except blocks)
            # ... (rest of feature extraction code, each in its own try/except block)

            # 1. Pitch (Fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch = np.mean(pitches[magnitudes > np.max(magnitudes) * 0.7]) if np.any(magnitudes > np.max(magnitudes) * 0.7) else 0
            
            # 3. Speech rate (zero-crossing rate as proxy)
            speech_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
            

            # 4. Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            
            # 5. MFCCs (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)
            
            # 6. Chroma features (related to musical perception)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_means = np.mean(chroma, axis=1)
            
            # 7. Rhythm features
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # 8. Voice quality - jitter and shimmer approximation
            # (using frame-to-frame variations in pitch and amplitude)
            if len(y) > sr // 10:  # Ensure audio is long enough
                frames = librosa.util.frame(y, frame_length=sr//10, hop_length=sr//20)
                frame_energies = np.sqrt(np.sum(frames**2, axis=0))
                frame_energy_variations = np.std(frame_energies) / np.mean(frame_energies) if np.mean(frame_energies) > 0 else 0
            else:
                frame_energy_variations = 0
            
            # Combine all features into a single vector
            features = np.hstack([
                pitch if not np.isnan(pitch) else 0,
                energy,
                speech_rate,
                spectral_centroid,
                spectral_bandwidth,
                spectral_rolloff,
                mfcc_means,
                chroma_means,
                tempo,
                frame_energy_variations
            ])
            
            # If we couldn't extract enough features, pad with zeros
            while len(features) < 25:
                features.append(0.0)
                
            return np.array(features)
                
        except Exception as e:
            import traceback
            print(f"Error extracting acoustic features: {str(e)}")
            print(traceback.format_exc())  # Print full stack trace
            return np.zeros(25)  # Return zeros as fallback
        

    def analyze_text_sentiment(self, text):
        """Analyze sentiment from text using OpenAI API"""
        try:
            url = "https://api.openai.com/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Create a prompt that guides the model to return JSON format
            system_content = """
            Analyze the emotional content of the following text and return a JSON object with these fields:
            - valence: numeric score from 1-10 where 1 is extremely negative and 10 is extremely positive
            - arousal: numeric score from 1-10 where 1 is very calm/passive and 10 is very excited/active
            - dominance: numeric score from 1-10 where 1 is feeling controlled/submissive and 10 is feeling in control/dominant
            - primary_emotion: one of [anger, fear, sadness, disgust, surprise, joy, neutral, anxiety, confusion]
            - secondary_emotion: another emotion from the list above that may be present
            - confidence: your confidence in this analysis from 0-1
            
            Format your response as a valid JSON object with no explanation or additional text.
            Example: {"valence": 7, "arousal": 4, "dominance": 6, "primary_emotion": "joy", "secondary_emotion": "surprise", "confidence": 0.8}
            """

            data = {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": text}
                ],
                # "response_format": {"type": "json_object"}
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            # if response.status_code == 200:
            #     result = response.json()
            #     sentiment = json.loads(result["choices"][0]["message"]["content"])
                
            #     print(f"Text sentiment analysis: {sentiment}")
            #     return sentiment

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse the JSON from the response content
                # Find the JSON part if there's any text explanation
                try:
                    # Try to find JSON in the response
                    import re
                    json_match = re.search(r'({.*})', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        sentiment = json.loads(json_str)
                    else:
                        # If no JSON pattern found, try the entire content
                        sentiment = json.loads(content)
                    
                    print(f"Text sentiment analysis: {sentiment}")
                    return sentiment
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON from: {content}")
                    # Return default values
                    return {
                        'valence': 5, 'arousal': 5, 'dominance': 5,
                        'primary_emotion': 'neutral', 'secondary_emotion': 'neutral',
                        'confidence': 0.5
                    }

            else:
                print(f"Sentiment API error: {response.status_code}, {response.text}")
                return {
                    'valence': 5,
                    'arousal': 5,
                    'dominance': 5,
                    'primary_emotion': 'neutral',
                    'secondary_emotion': 'neutral',
                    'confidence': 0.5
                }
                
        except Exception as e:
            print(f"Error analyzing text sentiment: {str(e)}")
            return {
                'valence': 5,
                'arousal': 5,
                'dominance': 5,
                'primary_emotion': 'neutral',
                'secondary_emotion': 'neutral',
                'confidence': 0.5
            }
    
    def predict_acoustic_emotion(self, features):
        """Predict emotion from acoustic features using ML model"""
        # This is where we'd use the ML model for prediction
        # If we don't have enough training data yet, return a neutral prediction
        
        if len(getattr(self.model, 'classes_', [])) < 3:
            # Not enough training data yet
            return {
                'predicted_emotion': 'neutral',
                'confidence': 0.5
            }
        
        try:
            # Scale features
            scaled_features = self.scaler.transform(features.reshape(1, -1))
            
            # Predict emotion
            emotion = self.model.predict(scaled_features)[0]
            
            # Get prediction probabilities
            probs = self.model.predict_proba(scaled_features)[0]
            confidence = np.max(probs)
            
            return {
                'predicted_emotion': emotion,
                'confidence': float(confidence)
            }
        except Exception as e:
            print(f"Error predicting acoustic emotion: {str(e)}")
            return {
                'predicted_emotion': 'neutral',
                'confidence': 0.5
            }
    
    def update_model(self, features, emotion_label):
        """Update the model with new training data"""
        # This would be called if the user provides feedback on the emotion detection
        # or if we want to train on the text-based emotion detection
        
        if not hasattr(self.model, 'classes_'):
            # First sample, initialize with some basic emotion classes
            self.model.classes_ = np.array(['neutral', 'happy', 'sad', 'angry', 'anxious'])
            
            # Create a synthetic dataset to start with
            X_synthetic = np.random.rand(5, features.shape[0])
            y_synthetic = np.array(['neutral', 'happy', 'sad', 'angry', 'anxious'])
            
            # Fit scaler on synthetic data plus the new sample
            X_combined = np.vstack([X_synthetic, features.reshape(1, -1)])
            self.scaler.fit(X_combined)
            
            # Scale the data
            X_scaled = self.scaler.transform(X_combined)
            
            # Add the new sample label
            y_combined = np.append(y_synthetic, emotion_label)
            
            # Fit the model
            self.model.fit(X_scaled, y_combined)
        else:
            # Update existing model with new data
            # First check if we need to add a new class
            if emotion_label not in self.model.classes_:
                self.model.classes_ = np.append(self.model.classes_, emotion_label)
            
            # Scale the features
            scaled_features = self.scaler.transform(features.reshape(1, -1))
            
            # Call partial_fit method (if available) or retrain
            if hasattr(self.model, 'partial_fit'):
                self.model.partial_fit(scaled_features, np.array([emotion_label]), 
                                     classes=self.model.classes_)
            else:
                # Get the current training data (this is a simplified approach)
                # In practice, you'd maintain a dataset of training examples
                X_train = scaled_features
                y_train = np.array([emotion_label])
                self.model.fit(X_train, y_train)
        
        # Save the updated model and scaler
        joblib.dump(self.model, 'mood_model.pkl')
        joblib.dump(self.scaler, 'mood_scaler.pkl')
        
        print(f"Model updated with new emotion data: {emotion_label}")
    
    def detect_mood(self, audio_file_path, transcript):
        """Main method to detect mood from both audio and text"""
        # 1. Extract acoustic features
        acoustic_features = self.extract_acoustic_features(audio_file_path)
        
        # 2. Analyze text sentiment
        text_sentiment = self.analyze_text_sentiment(transcript)
        
        # 3. Predict acoustic emotion (if we have a trained model)
        acoustic_prediction = self.predict_acoustic_emotion(acoustic_features)
        
        # 4. Combine the analyses
        # Weight the predictions based on confidence
        text_confidence = text_sentiment.get('confidence', 0.5)
        acoustic_confidence = acoustic_prediction.get('confidence', 0.5)
        
        # Normalize weights to sum to 1
        total_confidence = text_confidence + acoustic_confidence
        if total_confidence > 0:
            text_weight = text_confidence / total_confidence
            acoustic_weight = acoustic_confidence / total_confidence
        else:
            text_weight = 0.5
            acoustic_weight = 0.5
        
        # Combine valence and arousal with weighted average
        valence = (text_sentiment.get('valence', 5) * text_weight + 
                  (5 + 2 * (1 if acoustic_prediction.get('predicted_emotion') == 'happy' else 
                           -1 if acoustic_prediction.get('predicted_emotion') == 'sad' else 0)) * acoustic_weight)
        
        arousal = (text_sentiment.get('arousal', 5) * text_weight + 
                  (5 + 2 * (1 if acoustic_prediction.get('predicted_emotion') in ['angry', 'excited'] else 
                           -1 if acoustic_prediction.get('predicted_emotion') in ['sad', 'neutral'] else 0)) * acoustic_weight)
        
        # Determine primary emotion
        # If acoustic and text emotions match, use that
        if acoustic_prediction.get('predicted_emotion') == text_sentiment.get('primary_emotion'):
            primary_emotion = acoustic_prediction.get('predicted_emotion')
        else:
            # Otherwise, use the one with higher confidence
            if text_confidence >= acoustic_confidence:
                primary_emotion = text_sentiment.get('primary_emotion')
            else:
                primary_emotion = acoustic_prediction.get('predicted_emotion')
        
        # If the model is not yet well-trained, rely more on text sentiment
        if len(getattr(self.model, 'classes_', [])) < 3:
            primary_emotion = text_sentiment.get('primary_emotion')
            secondary_emotion = text_sentiment.get('secondary_emotion', 'neutral')
        else:
            secondary_emotion = (text_sentiment.get('secondary_emotion') if primary_emotion == acoustic_prediction.get('predicted_emotion')
                              else text_sentiment.get('primary_emotion') if primary_emotion == acoustic_prediction.get('predicted_emotion')
                              else text_sentiment.get('secondary_emotion', 'neutral'))
        
        # 5. Update the model with the text-based emotion for future improvement
        # This uses the text sentiment as "ground truth" for training
        if text_confidence > 0.7:
            self.update_model(acoustic_features, text_sentiment.get('primary_emotion'))
        
        # 6. Create the final mood assessment
        mood = {
            'valence': max(1, min(10, valence)),  # 1-10 scale
            'arousal': max(1, min(10, arousal)),  # 1-10 scale
            'dominance': text_sentiment.get('dominance', 5),  # Use text analysis for dominance
            'primary_emotion': primary_emotion,
            'secondary_emotion': secondary_emotion,
            'text_confidence': text_confidence,
            'acoustic_confidence': acoustic_confidence,
            'combined_confidence': (text_confidence + acoustic_confidence) / 2
        }
        
        print(f"Final mood assessment: {mood}")
        return mood
