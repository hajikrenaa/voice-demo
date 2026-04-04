from openai import AsyncOpenAI, OpenAIError
from config import Config
from typing import AsyncIterator, List, Dict
import logging

logger = logging.getLogger(__name__)


class LLMService:
    """
    GPT-4o conversation service with streaming support
    """

    def __init__(self):
        """Initialize LLM service with OpenAI client"""
        self.client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.LLM_MODEL
        self.temperature = Config.LLM_TEMPERATURE
        self.max_tokens = Config.LLM_MAX_TOKENS
        self.system_prompt = Config.SYSTEM_PROMPT
        self.conversation_history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """
        Add a message to conversation history

        Args:
            role: "user" or "assistant"
            content: Message content
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })

        # Trim history if too long
        if len(self.conversation_history) > Config.MAX_HISTORY_MESSAGES:
            # Keep first message (often contains important context)
            # and trim from the middle
            self.conversation_history = [
                self.conversation_history[0],
                *self.conversation_history[-(Config.MAX_HISTORY_MESSAGES - 1):]
            ]

    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """
        Get formatted messages for OpenAI API

        Returns:
            List of message dicts with system prompt
        """
        return [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history
        ]

    async def generate_response_stream(self, user_message: str) -> AsyncIterator[str]:
        """
        Generate streaming response from GPT-4o

        Yields sentences as they complete for parallel TTS processing

        Args:
            user_message: User's message

        Yields:
            str: Complete sentences as they finish
        """
        # Add user message to history
        self.add_message("user", user_message)

        try:
            # Call GPT-4o with streaming
            logger.info(f"Generating response for: {user_message[:50]}...")

            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=self.get_messages_for_api(),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )

            # Stream tokens and buffer into sentences
            full_response = ""
            sentence_buffer = ""

            # Sentence ending punctuation
            sentence_endings = {'.', '!', '?'}
            # Also break on commas and semicolons for faster response
            quick_breaks = {',', ';', ':', '-'}

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    sentence_buffer += token

                    # Check if the token ends with sentence-ending punctuation
                    if any(token.rstrip().endswith(p) for p in sentence_endings):
                        sentence = sentence_buffer.strip()
                        if sentence and len(sentence) > 3:  # Avoid single-word yields
                            logger.debug(f"Yielding sentence: {sentence}")
                            yield sentence
                            sentence_buffer = ""

                    # Yield on commas for faster starts (but require min length)
                    elif any(token.rstrip().endswith(p) for p in quick_breaks) and len(sentence_buffer.strip()) > 20:
                        sentence = sentence_buffer.strip()
                        if sentence:
                            logger.debug(f"Yielding on pause: {sentence}")
                            yield sentence
                            sentence_buffer = ""

                    # Also yield on newlines (paragraph breaks)
                    elif '\n' in token and sentence_buffer.strip():
                        sentence = sentence_buffer.strip()
                        if sentence:
                            logger.debug(f"Yielding on newline: {sentence}")
                            yield sentence
                            sentence_buffer = ""

            # Yield any remaining text
            if sentence_buffer.strip():
                logger.debug(f"Yielding final buffer: {sentence_buffer.strip()}")
                yield sentence_buffer.strip()

            # Add complete response to history
            self.add_message("assistant", full_response)
            logger.info(f"Response completed: {full_response[:100]}...")

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            error_message = "I apologize, I'm having trouble processing that right now."
            self.add_message("assistant", error_message)
            yield error_message

        except Exception as e:
            logger.error(f"Unexpected error in LLM service: {str(e)}")
            error_message = "I encountered an unexpected error."
            self.add_message("assistant", error_message)
            yield error_message

    async def generate_response(self, user_message: str) -> str:
        """
        Generate complete response (non-streaming)

        Args:
            user_message: User's message

        Returns:
            str: Complete response
        """
        # Collect all sentences from streaming
        sentences = []
        async for sentence in self.generate_response_stream(user_message):
            sentences.append(sentence)

        return " ".join(sentences)

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history

        Returns:
            List of message dicts
        """
        return self.conversation_history.copy()

    def set_system_prompt(self, prompt: str):
        """
        Update system prompt

        Args:
            prompt: New system prompt
        """
        self.system_prompt = prompt
        logger.info("System prompt updated")
