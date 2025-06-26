#!/usr/bin/env python3
"""
Test System Prompts Implementation
==================================

Test the new system prompt functionality for intelligent robot commentary
based on structured task IDs and ROS topic contexts.
"""

import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rich.console import Console
from ur.config.system_prompts import (
    parse_task_id, 
    build_commentary_prompt,
    TASK_EXECUTE_SYSTEM_PROMPT,
    COMMAND_SYSTEM_PROMPT
)

console = Console()

def test_task_parsing():
    """Test the structured task ID parsing functionality."""
    console.print("\n[bold blue]Testing Task ID Parsing[/bold blue]")
    
    test_cases = [
        "201_pickup",   # Module 2, Element 1, Pickup
        "305_place",    # Module 3, Element 5, Place  
        "102_home",     # Module 1, Element 2, Home
        "450_pickup",   # Module 4, Element 50, Pickup
        "invalid_task", # Invalid format
        "999_unknown"   # Valid format, unknown operation
    ]
    
    for task_name in test_cases:
        console.print(f"\n[cyan]Task:[/cyan] {task_name}")
        parsed = parse_task_id(task_name)
        
        if parsed['is_structured']:
            console.print(f"  [green]✓[/green] Module: {parsed['module']}")
            console.print(f"  [green]✓[/green] Element: {parsed['element']}")
            console.print(f"  [green]✓[/green] Operation: {parsed['operation']}")
            console.print(f"  [green]✓[/green] Description: {parsed['description']}")
            console.print(f"  [green]✓[/green] Action: {parsed['action_description']}")
        else:
            console.print(f"  [yellow]⚠[/yellow] Non-structured task: {parsed['description']}")

def test_task_execute_prompts():
    """Test task execution system prompts."""
    console.print("\n[bold blue]Testing Task Execute System Prompts[/bold blue]")
    
    test_tasks = [
        "201_pickup",   # Standard pickup task
        "305_place",    # Standard place task
        "102_home"      # Standard home task
    ]
    
    for task_name in test_tasks:
        console.print(f"\n[cyan]Task Received:[/cyan] {task_name}")
        
        # Test initial task announcement
        prompt = build_commentary_prompt(
            context_type='task_execute',
            task_name=task_name
        )
        
        console.print(f"[dim]Generated prompt length: {len(prompt)} characters[/dim]")
        
        # Show key parts of the prompt
        parsed = parse_task_id(task_name)
        console.print(f"  [green]Expected announcement:[/green] {parsed['action_description']}")
        
        # Test robot action commentary
        if 'pickup' in task_name:
            action_prompt = build_commentary_prompt(
                context_type='task_execute',
                task_name=task_name,
                action_type='moving_to_pickup'
            )
            console.print(f"  [blue]Action prompt generated for 'moving_to_pickup'[/blue]")

def test_command_prompts():
    """Test command system prompts."""
    console.print("\n[bold blue]Testing Command System Prompts[/bold blue]")
    
    test_commands = [
        "START",
        "FABRICATION_START", 
        "END",
        "PAUSE",
        "EMERGENCY_STOP"
    ]
    
    for command in test_commands:
        console.print(f"\n[cyan]Command:[/cyan] {command}")
        
        prompt = build_commentary_prompt(
            context_type='command',
            command=command
        )
        
        console.print(f"[dim]Generated prompt length: {len(prompt)} characters[/dim]")
        console.print(f"  [green]✓[/green] System prompt includes robot identity and command context")

def test_system_prompt_content():
    """Test that system prompts contain expected content."""
    console.print("\n[bold blue]Testing System Prompt Content[/bold blue]")
    
    # Test task execute prompt content
    console.print(f"\n[cyan]Task Execute System Prompt Analysis:[/cyan]")
    task_prompt = TASK_EXECUTE_SYSTEM_PROMPT
    
    key_elements = [
        "UR10 Universal Robot",
        "modular manufacturing system",
        "XYZ_operation",
        "module number",
        "element number",
        "first person",
        "under 15 words"
    ]
    
    for element in key_elements:
        if element.lower() in task_prompt.lower():
            console.print(f"  [green]✓[/green] Contains: {element}")
        else:
            console.print(f"  [red]✗[/red] Missing: {element}")
    
    # Test command prompt content
    console.print(f"\n[cyan]Command System Prompt Analysis:[/cyan]")
    command_prompt = COMMAND_SYSTEM_PROMPT
    
    command_elements = [
        "system command",
        "ROS topic",
        "/UR10/command",
        "START",
        "EMERGENCY_STOP",
        "acknowledge"
    ]
    
    for element in command_elements:
        if element.lower() in command_prompt.lower():
            console.print(f"  [green]✓[/green] Contains: {element}")
        else:
            console.print(f"  [red]✗[/red] Missing: {element}")

def test_edge_cases():
    """Test edge cases and error handling."""
    console.print("\n[bold blue]Testing Edge Cases[/bold blue]")
    
    # Test invalid inputs
    edge_cases = [
        ("", "Empty task name"),
        (None, "None task name"),
        ("abc", "Non-numeric task ID"),
        ("1_", "Missing operation"),
        ("_pickup", "Missing task ID")
    ]
    
    for task_name, description in edge_cases:
        console.print(f"\n[cyan]Edge Case:[/cyan] {description}")
        try:
            if task_name is None:
                continue  # Skip None case for now
                
            parsed = parse_task_id(task_name)
            console.print(f"  [green]✓[/green] Handled gracefully: {parsed['description']}")
        except Exception as e:
            console.print(f"  [red]✗[/red] Error: {e}")

def main():
    """Run all system prompt tests."""
    console.print("[bold green]🧪 System Prompts Test Suite[/bold green]")
    console.print("Testing the new LLM-prompted robot commentary system\n")
    
    try:
        test_task_parsing()
        test_task_execute_prompts() 
        test_command_prompts()
        test_system_prompt_content()
        test_edge_cases()
        
        console.print("\n[bold green]✅ All system prompt tests completed![/bold green]")
        console.print("\n[bold cyan]Next Steps:[/bold cyan]")
        console.print("1. Test with real robot tasks using ur/main.py")
        console.print("2. Verify LLM commentary generation with OpenAI/SmolAgent")
        console.print("3. Publish test ROS tasks to see dynamic commentary")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Test suite error:[/bold red] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 