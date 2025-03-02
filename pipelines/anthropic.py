"""
title: Anthropic Pipeline
author: OpenWebUI Community
date: 2024-01-13
version: 1.0
license: MIT
description: A pipeline for integrating Anthropic's Claude models into OpenWebUI
requirements: requests, sseclient-py, pydantic
environment_variables: ANTHROPIC_API_KEY
"""

import os
import json
import requests
from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel
import sseclient
import logging

logger = logging.getLogger(__name__)

def pop_system_message(messages: List[dict]) -> tuple[Optional[str], List[dict]]:
    """Extract system message from the messages list."""
    if not messages:
        return None, messages
    
    if messages[0]["role"] == "system":
        return messages[0]["content"], messages[1:]
    return None, messages

class Pipeline:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = ""
        MAX_IMAGES: int = 5
        MAX_IMAGE_SIZE_MB: int = 100
        DEFAULT_MAX_TOKENS: int = 4096
        DEFAULT_TEMPERATURE: float = 0.8
        DEFAULT_TOP_K: int = 40
        DEFAULT_TOP_P: float = 0.9

    def __init__(self):
        self.type = "manifold"
        self.id = "anthropic"
        self.name = "anthropic/"
        
        # Initialize valves with environment variables
        self.valves = self.Valves(
            ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY", "")
        )
        
        self.base_url = 'https://api.anthropic.com/v1/messages'
        self.update_headers()

    def update_headers(self):
        """Update API headers with current configuration."""
        self.headers = {
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
            'x-api-key': self.valves.ANTHROPIC_API_KEY
        }

    def get_anthropic_models(self) -> List[dict]:
        """Return available Anthropic models."""
        return [
            {"id": "claude-3-haiku-20240307", "name": "claude-3-haiku"},
            {"id": "claude-3-opus-20240229", "name": "claude-3-opus"},
            {"id": "claude-3-sonnet-20240229", "name": "claude-3-sonnet"},
            {"id": "claude-3-5-haiku-20241022", "name": "claude-3.5-haiku"},
            {"id": "claude-3-5-sonnet-20241022", "name": "claude-3.5-sonnet"},
        ]

    async def on_startup(self):
        """Startup hook for pipeline initialization."""
        logger.info(f"Starting Anthropic pipeline")
        if not self.valves.ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not set")

    async def on_shutdown(self):
        """Shutdown hook for cleanup."""
        logger.info(f"Shutting down Anthropic pipeline")

    async def on_valves_updated(self):
        """Handler for valve updates."""
        self.update_headers()

    def pipelines(self) -> List[dict]:
        """Return available pipelines (models)."""
        return self.get_anthropic_models()

    def process_image(self, image_data: dict) -> dict:
        """Process image data for API submission."""
        try:
            if image_data["url"].startswith("data:image"):
                mime_type, base64_data = image_data["url"].split(",", 1)
                media_type = mime_type.split(":")[1].split(";")[0]
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    },
                }
            return {
                "type": "image",
                "source": {"type": "url", "url": image_data["url"]},
            }
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise ValueError(f"Invalid image data: {e}")

    def pipe(
        self, 
        user_message: str, 
        model_id: str, 
        messages: List[dict], 
        body: dict
    ) -> Union[str, Generator, Iterator]:
        """Main pipeline processing method."""
        try:
            # Clean body of unnecessary keys
            for key in ['user', 'chat_id', 'title']:
                body.pop(key, None)

            # Extract system message
            system_message, messages = pop_system_message(messages)

            # Process messages and images
            processed_messages = []
            image_count = 0
            total_image_size = 0

            for message in messages:
                processed_content = []
                
                if isinstance(message.get("content"), list):
                    for item in message["content"]:
                        if item["type"] == "text":
                            processed_content.append({
                                "type": "text", 
                                "text": item["text"]
                            })
                        elif item["type"] == "image_url":
                            if image_count >= self.valves.MAX_IMAGES:
                                raise ValueError(
                                    f"Maximum of {self.valves.MAX_IMAGES} images per API call exceeded"
                                )

                            processed_image = self.process_image(item["image_url"])
                            processed_content.append(processed_image)

                            # Calculate image size for base64 images
                            if processed_image["source"]["type"] == "base64":
                                image_size = len(processed_image["source"]["data"]) * 3 / 4
                                total_image_size += image_size
                                
                                if total_image_size > self.valves.MAX_IMAGE_SIZE_MB * 1024 * 1024:
                                    raise ValueError(
                                        f"Total size of images exceeds {self.valves.MAX_IMAGE_SIZE_MB}MB limit"
                                    )

                            image_count += 1
                else:
                    processed_content = [{
                        "type": "text", 
                        "text": message.get("content", "")
                    }]

                processed_messages.append({
                    "role": message["role"], 
                    "content": processed_content
                })

            # Prepare API payload
            payload = {
                "model": model_id,
                "messages": processed_messages,
                "max_tokens": body.get("max_tokens", self.valves.DEFAULT_MAX_TOKENS),
                "temperature": body.get("temperature", self.valves.DEFAULT_TEMPERATURE),
                "top_k": body.get("top_k", self.valves.DEFAULT_TOP_K),
                "top_p": body.get("top_p", self.valves.DEFAULT_TOP_P),
                "stop_sequences": body.get("stop", []),
                "stream": body.get("stream", False)
            }

            # Add system message if present
            if system_message:
                payload["system"] = str(system_message)

            # Handle streaming vs non-streaming responses
            if payload["stream"]:
                return self.stream_response(payload)
            return self.get_completion(payload)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return f"Error: {str(e)}"

    def stream_response(self, payload: dict) -> Generator:
        """Handle streaming responses from the API."""
        try:
            response = requests.post(
                self.base_url, 
                headers=self.headers, 
                json=payload, 
                stream=True
            )

            if response.status_code != 200:
                raise Exception(f"API Error: {response.status_code} - {response.text}")

            client = sseclient.SSEClient(response)
            for event in client.events():
                try:
                    data = json.loads(event.data)
                    
                    if data["type"] == "content_block_start":
                        yield data["content_block"]["text"]
                    elif data["type"] == "content_block_delta":
                        yield data["delta"]["text"]
                    elif data["type"] == "message_stop":
                        break
                        
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON: {event.data}")
                except KeyError as e:
                    logger.error(f"Unexpected data structure: {e}")
                    logger.debug(f"Full data: {data}")
                    
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"Error during streaming: {str(e)}"

    def get_completion(self, payload: dict) -> str:
        """Handle non-streaming responses from the API."""
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            
            if response.status_code != 200:
                raise Exception(f"API Error: {response.status_code} - {response.text}")

            result = response.json()
            return result["content"][0]["text"] if "content" in result and result["content"] else ""
            
        except Exception as e:
            logger.error(f"Completion error: {e}")
            return f"Error getting completion: {str(e)}"