from typing import List, Union, Generator, Iterator
import subprocess
import sys
import logging


class Pipeline:
    def __init__(self):
        self.name = "Text-to-Art Pipeline"
        logging.basicConfig(level=logging.INFO)

    async def on_startup(self):
        """
        Lifecycle method called when the pipeline starts.
        """
        logging.info(f"{self.name} is starting up...")

    async def on_shutdown(self):
        """
        Lifecycle method called when the pipeline shuts down.
        """
        logging.info(f"{self.name} is shutting down...")

    def execute_art_command(self, text: str, font: str) -> Union[str, None]:
        """
        Generate ASCII art for the given text using the 'art' library.

        Args:
            text (str): The text to convert into ASCII art.
            font (str): The font style to use.

        Returns:
            str: The ASCII art if successful, or an error message if not.
        """
        try:
            # Dynamically generate the Python code
            code = f"""
from art import text2art
try:
    print(text2art("{text}", font="{font}"))
except Exception as e:
    print(f"Error: {{str(e)}}")
"""
            # Use the current Python executable to run the subprocess
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                check=True,
            )

            # Check for "Error:" in the output
            if "Error:" in result.stdout:
                return result.stdout.strip()
            return result.stdout.strip()

        except subprocess.CalledProcessError as e:
            logging.error(f"Subprocess failed: {e}")
            return f"Error generating ASCII art: {e.output.strip()}"
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return f"Unexpected error: {e}"

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Process the user's text and generate ASCII art.

        Args:
            user_message (str): The user's input text.
            model_id (str): Unused in this pipeline but reserved for future use.
            messages (List[dict]): Unused in this pipeline but reserved for future use.
            body (dict): Additional options such as font style.

        Returns:
            str: The generated ASCII art or an error message.
        """
        logging.info(f"Processing user message: {user_message}")

        # Extract font from the body or use a default font
        font = body.get("font", "block")
        if not user_message.strip():
            return "Error: Please provide some text to generate ASCII art."

        # Generate ASCII art
        ascii_art = self.execute_art_command(user_message.strip(), font)
        logging.info(f"Generated ASCII Art: {ascii_art}")
        return ascii_art or "Failed to generate ASCII art."


# Example Usage
if __name__ == "__main__":
    import asyncio

    # Initialize the pipeline
    pipeline = Pipeline()

    # Run the startup process
    asyncio.run(pipeline.on_startup())

    # Simulate user input
    try:
        user_message = "Hello, World!"
        body = {"font": "block"}  # Try different fonts like "block", "random", "thin", etc.
        output = pipeline.pipe(user_message, "", [], body)
        print(output)
    finally:
        # Ensure shutdown process runs
        asyncio.run(pipeline.on_shutdown())
