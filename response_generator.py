import json
import requests

class ResponseGenerator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.response_strategies = {
            # Strategies for low valence emotions
            'sad': {
                'approach': 'empathetic validation followed by gentle redirection',
                'tone': 'warm and supportive',
                'techniques': ['active listening', 'validation', 'gentle reframing', 'hope instillation']
            },
            'angry': {
                'approach': 'validation of feelings with de-escalation',
                'tone': 'calm and steady',
                'techniques': ['acknowledgment', 'reflective listening', 'emotion diffusion', 'perspective offering']
            },
            'anxious': {
                'approach': 'containment and grounding',
                'tone': 'reassuring and measured',
                'techniques': ['normalization', 'grounding techniques', 'cognitive reframing', 'solution exploration']
            },
            'fearful': {
                'approach': 'safety building and reality testing',
                'tone': 'gentle and reassuring',
                'techniques': ['safety reinforcement', 'reality checking', 'coping strategy suggestion', 'incremental exposure']
            },
            
            # Strategies for neutral/mixed valence emotions
            'neutral': {
                'approach': 'engagement and curious exploration',
                'tone': 'friendly and conversational',
                'techniques': ['open questions', 'gentle encouragement', 'interest building', 'value exploration']
            },
            'surprised': {
                'approach': 'curious exploration and processing',
                'tone': 'interested and engaged',
                'techniques': ['curious questioning', 'meaning making', 'integration', 'new perspective offering']
            },
            'confused': {
                'approach': 'clarification and organization',
                'tone': 'clear and methodical',
                'techniques': ['summarizing', 'clarifying', 'step-by-step thinking', 'insight building']
            },
            
            # Strategies for high valence emotions
            'happy': {
                'approach': 'celebration and reinforcement',
                'tone': 'cheerful and enthusiastic',
                'techniques': ['positive reinforcement', 'joy magnification', 'value connection', 'future positive projection']
            },
            'excited': {
                'approach': 'enthusiasm matching and channeling',
                'tone': 'energetic and encouraging',
                'techniques': ['mirroring enthusiasm', 'positive channeling', 'constructive focusing', 'opportunity exploration']
            }
        }
    
    def get_voice_parameters(self, mood):
        """Determine TTS voice parameters based on mood assessment"""
        # Base voice selection
        voice_type = "alloy"  # Default neutral voice
        
        # Adjust based on emotional state
        valence = mood.get('valence', 5)
        arousal = mood.get('arousal', 5)
        primary_emotion = mood.get('primary_emotion', 'neutral')
        
        if valence < 4:  # Negative emotions
            voice_type = "nova"  # Warmer, more empathetic voice
        elif valence > 7 and arousal > 6:  # Very positive and energetic
            voice_type = "shimmer"  # More energetic voice
            
        # Create voice parameters object
        voice_params = {
            "voice": voice_type,
            # You could add more TTS parameters here if the API supports them
            # such as speaking rate, pitch adjustments, etc.
        }
        
        return voice_params
    
    def generate_personalized_response(self, transcript, mood, conversation_history):
        """Generate a personalized response based on user's mood and conversation history"""
        # Get the appropriate response strategy based on primary emotion
        primary_emotion = mood.get('primary_emotion', 'neutral')
        strategy = self.response_strategies.get(primary_emotion, self.response_strategies['neutral'])
        
        # Create the system message with personalized guidance
        system_message = f"""
        You are an empathetic AI assistant designed to improve the user's mood through conversation.
        
        Current emotional assessment:
        - Primary emotion: {mood.get('primary_emotion', 'neutral')}
        - Secondary emotion: {mood.get('secondary_emotion', 'neutral')}
        - Valence (positivity): {mood.get('valence', 5)}/10
        - Arousal (energy level): {mood.get('arousal', 5)}/10
        - Dominance (feeling of control): {mood.get('dominance', 5)}/10
        
        Your response approach: {strategy['approach']}
        Your tone should be: {strategy['tone']}
        
        Techniques to use in your response:
        - {strategy['techniques'][0]}
        - {strategy['techniques'][1]}
        - {strategy['techniques'][2]}
        - {strategy['techniques'][3]}
        
        Keep your response conversational and natural. Respond in 2-3 sentences unless a more detailed response is appropriate.
        If the user seems to be in significant distress, recommend professional help.
        """
        
        # Add voice modulation guidance if appropriate
        if mood.get('arousal', 5) > 7:
            system_message += "\nThe user seems highly energetic, match their energy but aim to gradually guide them to a more balanced state."
        elif mood.get('arousal', 5) < 3:
            system_message += "\nThe user seems low energy, show warmth while gradually increasing energy in your responses."
        
        # Add guidance for specific negative emotions
        if primary_emotion in ['sad', 'angry', 'fearful', 'anxious'] and mood.get('valence', 5) < 3:
            system_message += "\nThe user is experiencing significant negative emotions. Show empathy first, validate their feelings, and then gently suggest perspective shifts or coping strategies."
        
        # Build the messages array for the API call
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
        
        try:
            # Make the API call
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "gpt-4",
                "messages": messages
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                return response_text
            else:
                print(f"API error: {response.status_code}, {response.text}")
                return "I'm having trouble processing that right now. How are you feeling today?"
        
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize for the technical difficulty. Would you like to continue our conversation?"
    
    def get_voice_improvement_suggestions(self, mood, acoustic_features):
        """Generate suggestions for improving voice tone based on mood and acoustic features"""
        # This would be called if the user wants vocal tone improvement tips
        
        suggestions = []
        
        # Extract key voice features
        energy = acoustic_features[1] if len(acoustic_features) > 1 else 0.5  # Normalized energy
        speech_rate = acoustic_features[2] if len(acoustic_features) > 2 else 0.5  # Normalized speech rate
        pitch = acoustic_features[0] if len(acoustic_features) > 0 else 220  # Pitch in Hz
        
        # Normalize the values for easier comparison
        energy_norm = min(1.0, energy * 10)  # 0-1 scale
        speech_rate_norm = min(1.0, speech_rate * 20)  # 0-1 scale
        
        # Generate suggestions based on current emotional state and voice characteristics
        primary_emotion = mood.get('primary_emotion', 'neutral')
        valence = mood.get('valence', 5)
        arousal = mood.get('arousal', 5)
        
        # Suggestions for negative emotions with vocal issues
        if valence < 4:
            if energy_norm < 0.3:
                suggestions.append("Try increasing your vocal energy slightly. Speaking with a bit more projection can help convey confidence.")
            
            if speech_rate_norm < 0.3:
                suggestions.append("Consider speaking at a slightly faster pace. Very slow speech can sometimes reinforce negative emotions.")
            elif speech_rate_norm > 0.8 and primary_emotion in ['anxious', 'fearful']:
                suggestions.append("Try slowing your speech rate slightly. Taking time between phrases can help reduce anxiety in your voice.")
                
            if primary_emotion == 'sad':
                suggestions.append("Slightly raising the pitch of your voice and adding more tonal variation might help shift your emotional state.")
                
        # Suggestions for overly intense emotions
        elif primary_emotion in ['angry', 'excited'] and arousal > 7:
            if energy_norm > 0.8:
                suggestions.append("Try softening your voice slightly. A more measured tone can help you communicate more effectively.")
                
            if speech_rate_norm > 0.8:
                suggestions.append("Consider slowing your speech rate. Taking brief pauses between thoughts can help organize your message clearly.")
        
        # General improvement suggestions
        if len(suggestions) == 0:
            if energy_norm < 0.4 and speech_rate_norm < 0.4:
                suggestions.append("Adding more energy and slightly increasing your speaking pace can make your voice more engaging.")
            elif energy_norm > 0.8 and speech_rate_norm > 0.8:
                suggestions.append("Speaking slightly slower and with a more measured tone can help your message land more effectively.")
            else:
                suggestions.append("Your vocal tone sounds well-balanced. Maintaining consistent eye contact while speaking can further enhance your communication.")
        
        # Add a positive reinforcement
        suggestions.append("Remember that authentic communication is most important - these are just small adjustments that might help align your voice with your intentions.")
        
        return suggestions
