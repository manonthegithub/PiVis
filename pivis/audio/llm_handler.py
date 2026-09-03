"""LLM integration for processing transcribed text."""

import asyncio
import logging
from typing import Optional, Dict, AsyncGenerator
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """Abstract base for LLM backends."""

    @abstractmethod
    async def process_text(self, text: str, context: Optional[str] = None) -> Dict:
        """Process text through LLM.

        Returns:
            Dict with keys: response, tokens_used, error (optional)
        """
        pass


class OpenAILLM(LLMBackend):
    """OpenAI GPT-based LLM."""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        """Initialize OpenAI LLM.

        Args:
            api_key: OpenAI API key
            model: Model name
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"

    async def process_text(self, text: str, context: Optional[str] = None) -> Dict:
        """Process text through GPT.

        Args:
            text: Transcribed text to process
            context: Optional system context/instructions

        Returns:
            LLM response dict
        """
        import aiohttp

        system_prompt = context or "You are a helpful assistant."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.7,
                    },
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            "response": data["choices"][0]["message"]["content"],
                            "tokens_used": data.get("usage", {}).get("total_tokens", 0),
                            "model": self.model,
                        }
                    else:
                        error = await resp.text()
                        logger.error(f"LLM API error: {error}")
                        return {
                            "response": "",
                            "tokens_used": 0,
                            "error": f"API error: {resp.status}",
                        }
        except asyncio.TimeoutError:
            return {
                "response": "",
                "tokens_used": 0,
                "error": "LLM API timeout",
            }
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return {
                "response": "",
                "tokens_used": 0,
                "error": str(e),
            }


class LocalLLM(LLMBackend):
    """Local LLM via Ollama or similar."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "neural-chat"):
        """Initialize local LLM.

        Args:
            base_url: Ollama/local API base URL
            model: Model name
        """
        self.base_url = base_url
        self.model = model
        self.api_endpoint = f"{base_url}/api/generate"

    async def process_text(self, text: str, context: Optional[str] = None) -> Dict:
        """Process text through local LLM.

        Args:
            text: Transcribed text
            context: Optional system context

        Returns:
            LLM response dict
        """
        import aiohttp

        prompt = f"{context or ''}\n\n{text}".strip()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_endpoint,
                    json={"model": self.model, "prompt": prompt, "stream": False},
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            "response": data.get("response", ""),
                            "tokens_used": 0,  # Local model doesn't track
                            "model": self.model,
                        }
                    else:
                        logger.error(f"Local LLM error: {resp.status}")
                        return {
                            "response": "",
                            "tokens_used": 0,
                            "error": f"API error: {resp.status}",
                        }
        except asyncio.TimeoutError:
            return {
                "response": "",
                "tokens_used": 0,
                "error": "LLM timeout",
            }
        except Exception as e:
            logger.error(f"Local LLM call failed: {e}")
            return {
                "response": "",
                "tokens_used": 0,
                "error": str(e),
            }


class LLMService:
    """Manages LLM processing for transcribed text."""

    def __init__(self, backend: Optional[LLMBackend] = None):
        """Initialize LLM service.

        Args:
            backend: LLM backend (defaults to local Ollama)
        """
        self.backend = backend or LocalLLM()
        self.max_retries = 3
        self.retry_delay_ms = 200

    async def process_transcription(
        self, transcription: Dict, system_context: Optional[str] = None
    ) -> Dict:
        """Process transcribed text through LLM with retry logic.

        Args:
            transcription: STT result dict (text, confidence, language)
            system_context: Optional system prompt

        Returns:
            LLM response with metadata
        """
        text = transcription.get("text", "").strip()

        if not text:
            logger.warning("Empty text for LLM processing")
            return {
                "response": "",
                "tokens_used": 0,
                "error": "Empty input text",
                "input": transcription,
            }

        for attempt in range(self.max_retries):
            try:
                result = await self.backend.process_text(text, system_context)

                if not result.get("error"):
                    return {
                        **result,
                        "input": transcription,
                        "attempt": attempt + 1,
                    }

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = (self.retry_delay_ms * (2 ** attempt)) / 1000.0
                    await asyncio.sleep(delay)
                    logger.debug(f"LLM retry {attempt + 1}/{self.max_retries}")

            except Exception as e:
                logger.error(f"LLM processing error: {e}")
                if attempt == self.max_retries - 1:
                    return {
                        "response": "",
                        "tokens_used": 0,
                        "error": str(e),
                        "input": transcription,
                        "attempt": attempt + 1,
                    }

        return result
