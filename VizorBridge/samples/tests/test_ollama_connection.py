import asyncio
import json
import ollama

# Rich imports for better console output
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

console = Console()

# Global model parameters
LOCAL_MODEL = 'gemma3:1b'


async def test_ollama_connection():
    """Test basic Ollama connection"""
    console.print("[cyan]Testing Ollama connection...[/cyan]")
    try:
        # List available models
        models = ollama.list()
        
        # Extract model names from the response
        if hasattr(models, 'models'):
            # If models is an object with a models attribute
            model_list = models.models
        elif 'models' in models:
            # If models is a dict with 'models' key
            model_list = models['models']
        else:
            model_list = models
        
        model_names = []
        for model in model_list:
            if isinstance(model, tuple):
                # Handle tuple case
                model_names.append(str(model[0]))
            elif hasattr(model, 'model'):
                # If it's a Model object with a 'model' attribute
                model_names.append(model.model)
            elif isinstance(model, dict):
                # If it's a dictionary
                model_names.append(model.get('name', model.get('model', str(model))))
            else:
                # If it's a string or other type
                model_names.append(str(model))        
        console.print(f"[green]✓[/green] Available models: {model_names}")
        return True
    except Exception as e:
        console.print(f"[red]✗[/red] Ollama connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_ollama_chat():
    """Test Ollama chat functionality"""
    console.print("[cyan]Testing Ollama chat...[/cyan]")
    try:
        response = ollama.chat(model=LOCAL_MODEL, messages=[
            {
                'role': 'user',
                'content': 'Hello! Can you respond with a simple greeting?',
            },
        ])
        console.print("[green]✓[/green] Ollama chat response:", response['message']['content'])
        return True
    except Exception as e:
        console.print(f"[red]✗[/red] Ollama chat test failed: {e}")
        return False


async def test_ollama_robot_commands():
    """Test Ollama generating robot movement commands"""
    console.print("[cyan]Testing Ollama robot command generation...[/cyan]")
    try:
        response = ollama.chat(model=LOCAL_MODEL, messages=[
            {
                'role': 'user',
                'content': 'Generate a simple robot movement command for testing. Respond with just a JSON object containing tcp coordinates like {"x": 0.1, "y": 0.2, "z": 0.3, "rx": 0, "ry": 0, "rz": 0}.',
            },
        ])
        console.print("[blue]Robot command response:[/blue]", response['message']['content'])
        
        # Try to parse JSON from response
        import re
        json_match = re.search(r'\{.*\}', response['message']['content'])
        if json_match:
            try:
                movement_command = json.loads(json_match.group())
                console.print(f"[green]✓[/green] Parsed movement command: {movement_command}")
                return True
            except json.JSONDecodeError:
                console.print("[red]✗[/red] Failed to parse JSON from response")
                return False
        else:
            console.print("[red]✗[/red] No JSON found in response")
            return False
            
    except Exception as e:
        console.print(f"[red]✗[/red] Robot command generation test failed: {e}")
        return False


async def run_ollama_tests():
    """Run all Ollama tests"""
    console.print(Panel("[bold cyan]Ollama Test Suite[/bold cyan]", expand=False))
    
    tests = [
        ("Connection Test", test_ollama_connection()),
        ("Chat Test", test_ollama_chat()),
        ("Robot Command Generation Test", test_ollama_robot_commands())
    ]
    
    results = []
    for test_name, test_coro in tests:
        console.print(f"\n[bold yellow]--- {test_name} ---[/bold yellow]")
        result = await test_coro
        results.append((test_name, result))    
    console.print(Panel("[bold cyan]Test Results[/bold cyan]", expand=False))
    for test_name, result in results:
        status = "[green]PASS[/green]" if result else "[red]FAIL[/red]"
        console.print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    overall_status = "[green]ALL TESTS PASSED[/green]" if all_passed else "[red]SOME TESTS FAILED[/red]"
    console.print(f"\n[bold]Overall: {overall_status}[/bold]")


if __name__ == "__main__":
    asyncio.run(run_ollama_tests())
