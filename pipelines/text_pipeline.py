from typing import List, Union, Generator, Iterator
from pydantic import BaseModel

class Pipeline:
    class Valves(BaseModel):
        UPPERCASE_ENABLED: bool = True
        ADD_PREFIX: bool = True
        PREFIX_TEXT: str = "Processed: "
        
    def __init__(self):
        self.name = "Simple Text Pipeline"
        
        # Initialize default valve settings
        self.valves = self.Valves(
            **{
                "UPPERCASE_ENABLED": True,
                "ADD_PREFIX": True,
                "PREFIX_TEXT": "Processed: "
            }
        )

    async def on_startup(self):
        # This function is called when the server is started
        print(f"Starting {self.name}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped
        print(f"Shutting down {self.name}")
        pass

    def process_text(self, text: str) -> str:
        """Process the input text based on valve settings"""
        result = text
        
        if self.valves.UPPERCASE_ENABLED:
            result = result.upper()
            
        if self.valves.ADD_PREFIX:
            result = f"{self.valves.PREFIX_TEXT}{result}"
            
        return result

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """Main pipeline processing function"""
        
        # Log incoming message
        print(f"Processing message: {user_message}")
        
        # Process the text
        processed_text = self.process_text(user_message)
        
        return processed_text