import asyncio
from typing import List, Dict, Optional
from rich.console import Console
from rich.table import Table
import logging

# Import the Pipeline class from the implemented academic RAG filter
from academic_rag_pipeline import Pipeline

# Initialize logging and rich console for output
console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenWebUITest:
    def __init__(self):
        self.pipeline = Pipeline()
        self.test_queries = [
            "What are the latest advancements in quantum computing?",
            "Explain the current understanding of dark matter.",
            "How has AI impacted climate change research?",
            "What are the applications of blockchain in academia?",
            "Discuss the role of CRISPR in modern genetics."
        ]

    async def simulate_openwebui_interaction(self):
        """Simulate the lifecycle of an OpenWebUI pipeline interaction."""
        # Simulate OpenWebUI starting up the pipeline
        console.print("\n[bold cyan]Starting OpenWebUI Simulation[/bold cyan]")
        await self.pipeline.on_startup()

        try:
            # Process each query as a separate user interaction
            for query in self.test_queries:
                console.print(f"\n[bold yellow]Processing query:[/bold yellow] {query}")

                # Simulate a user message payload as received by OpenWebUI
                body = {
                    "messages": [
                        {"role": "user", "content": query}
                    ]
                }

                # Simulate the `inlet` step where the user's query is processed
                processed_body = await self.pipeline.inlet(body)

                # Display the modified system message added to the body
                system_message = processed_body["messages"][0]
                console.print(f"\n[bold green]System Message Added:[/bold green] {system_message['content']}")

                # Simulate generating an assistant response
                assistant_response = "Here is my response based on the provided context."

                # Simulate the `outlet` step where the assistant's response is processed
                processed_response = await self.pipeline.outlet(assistant_response)
                console.print(f"[bold blue]Final Response Sent to User:[/bold blue] {processed_response}")

        except Exception as e:
            console.print(f"[red]Error during interaction simulation: {str(e)}[/red]", highlight=True)
        finally:
            # Simulate OpenWebUI shutting down the pipeline
            await self.pipeline.on_shutdown()
            console.print("\n[bold cyan]Simulation Completed[/bold cyan]")

if __name__ == "__main__":
    console.print("[bold cyan]Testing OpenWebUI Pipeline Integration[/bold cyan]")
    tester = OpenWebUITest()
    asyncio.run(tester.simulate_openwebui_interaction())
