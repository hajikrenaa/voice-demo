from datetime import datetime
from typing import List, Dict
from openai import AsyncOpenAI, OpenAIError
from config import Config
import logging
import json

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation history and generates summaries
    """

    def __init__(self):
        """Initialize conversation manager"""
        self.client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        self.messages: List[Dict[str, str]] = []
        self.metadata = {
            "start_time": None,
            "end_time": None,
            "turn_count": 0,
            "total_user_words": 0,
            "total_assistant_words": 0
        }

    def start_conversation(self):
        """Mark conversation start"""
        self.metadata["start_time"] = datetime.now()
        logger.info("Conversation started")

    def end_conversation(self):
        """Mark conversation end"""
        self.metadata["end_time"] = datetime.now()
        logger.info("Conversation ended")

    def add_message(self, role: str, content: str):
        """
        Add a message to the conversation

        Args:
            role: "user" or "assistant"
            content: Message content
        """
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # Update metadata
        if role == "user":
            self.metadata["turn_count"] += 1
            self.metadata["total_user_words"] += len(content.split())
        elif role == "assistant":
            self.metadata["total_assistant_words"] += len(content.split())

        logger.debug(f"Added {role} message: {content[:50]}...")

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages"""
        return self.messages.copy()

    def get_conversation_duration(self) -> float:
        """
        Get conversation duration in seconds

        Returns:
            float: Duration in seconds, or 0 if not ended
        """
        if not self.metadata["start_time"]:
            return 0.0

        end_time = self.metadata["end_time"] or datetime.now()
        duration = (end_time - self.metadata["start_time"]).total_seconds()
        return duration

    def build_transcript(self) -> str:
        """
        Build a formatted transcript of the conversation

        Returns:
            str: Formatted transcript
        """
        transcript_lines = []

        for msg in self.messages:
            role_label = "USER" if msg["role"] == "user" else "ASSISTANT"
            transcript_lines.append(f"{role_label}: {msg['content']}")

        return "\n\n".join(transcript_lines)

    async def generate_summary(self) -> Dict:
        """
        Generate comprehensive conversation summary using GPT-4o

        Returns:
            dict: {
                "overview": "Brief summary...",
                "key_points": ["Point 1", "Point 2", ...],
                "topics": ["topic1", "topic2", ...],
                "action_items": ["Action 1", ...],
                "sentiment": "positive/neutral/negative",
                "metadata": {...}
            }
        """
        if not self.messages:
            return {
                "overview": "No conversation to summarize.",
                "key_points": [],
                "topics": [],
                "action_items": [],
                "sentiment": "neutral",
                "metadata": self.get_metadata()
            }

        # Build transcript
        transcript = self.build_transcript()

        try:
            logger.info("Generating conversation summary...")

            # Create summary prompt (compact to save tokens)
            summary_prompt = f"""Summarize this conversation as JSON with keys: overview (1 sentence), key_points (array, max 3), topics (array), action_items (array), sentiment (positive/neutral/negative).

{transcript}"""

            # Use gpt-4o-mini for cheaper summaries
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize conversations as concise JSON."
                    },
                    {
                        "role": "user",
                        "content": summary_prompt
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=300,
                temperature=0.3
            )

            # Parse JSON response
            summary_data = json.loads(response.choices[0].message.content)

            # Add metadata
            summary_data["metadata"] = self.get_metadata()

            logger.info("Summary generated successfully")

            return summary_data

        except OpenAIError as e:
            logger.error(f"OpenAI API error during summary generation: {str(e)}")
            return self._get_fallback_summary()

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse summary JSON: {str(e)}")
            return self._get_fallback_summary()

        except Exception as e:
            logger.error(f"Unexpected error generating summary: {str(e)}")
            return self._get_fallback_summary()

    def _get_fallback_summary(self) -> Dict:
        """
        Get a basic fallback summary when AI generation fails

        Returns:
            dict: Basic summary with metadata
        """
        # Extract basic info from messages
        topics = set()
        for msg in self.messages:
            # Simple topic extraction (first few words)
            words = msg["content"].split()[:5]
            topics.update(words)

        return {
            "overview": f"Conversation with {self.metadata['turn_count']} exchanges.",
            "key_points": [
                f"User asked {self.metadata['turn_count']} questions or made statements",
                f"Conversation lasted {int(self.get_conversation_duration())} seconds"
            ],
            "topics": list(topics)[:5],  # Top 5 words as topics
            "action_items": [],
            "sentiment": "neutral",
            "metadata": self.get_metadata()
        }

    def get_metadata(self) -> Dict:
        """
        Get conversation metadata

        Returns:
            dict: Metadata including duration, turn count, word counts
        """
        return {
            "turn_count": self.metadata["turn_count"],
            "total_user_words": self.metadata["total_user_words"],
            "total_assistant_words": self.metadata["total_assistant_words"],
            "duration_seconds": self.get_conversation_duration(),
            "total_messages": len(self.messages),
            "conversation_started": self.metadata["start_time"].isoformat() if self.metadata["start_time"] else None,
            "conversation_ended": self.metadata["end_time"].isoformat() if self.metadata["end_time"] else None
        }

    def export_transcript(self, format: str = "text") -> str:
        """
        Export conversation transcript in various formats

        Args:
            format: "text", "json", or "markdown"

        Returns:
            str: Formatted transcript
        """
        if format == "json":
            return json.dumps({
                "messages": self.messages,
                "metadata": self.get_metadata()
            }, indent=2)

        elif format == "markdown":
            lines = ["# Conversation Transcript\n"]
            lines.append(f"**Duration**: {int(self.get_conversation_duration())} seconds\n")
            lines.append(f"**Turns**: {self.metadata['turn_count']}\n")
            lines.append("\n---\n")

            for msg in self.messages:
                role = "**User**" if msg["role"] == "user" else "*Assistant*"
                lines.append(f"{role}: {msg['content']}\n")

            return "\n".join(lines)

        else:  # text
            return self.build_transcript()

    def clear(self):
        """Clear conversation history and reset metadata"""
        self.messages = []
        self.metadata = {
            "start_time": None,
            "end_time": None,
            "turn_count": 0,
            "total_user_words": 0,
            "total_assistant_words": 0
        }
        logger.info("Conversation cleared")
