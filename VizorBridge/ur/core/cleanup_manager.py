"""
Centralized Cleanup Manager for UR Voice System

This module provides a centralized way to manage cleanup of all system components
with proper dependency handling, timeout management, and error resilience.
"""

import asyncio
import threading
import time
from typing import Optional, List, Callable, Dict, Any
from enum import Enum
from dataclasses import dataclass
from rich.console import Console

console = Console()


class CleanupPhase(Enum):
    """Cleanup phases in dependency order."""
    VOICE_AGENT = "voice_agent"          # Stop voice agent first
    BRIDGE_FABRICATION = "bridge_fab"    # Stop fabrication mode
    BRIDGE_CONNECTIONS = "bridge_conn"   # Close ROS connections
    ROBOT_CONNECTION = "robot_conn"      # Close robot connections
    GLOBAL_CLEANUP = "global_cleanup"    # Clean up global state
    UI_CLEANUP = "ui_cleanup"            # Stop UI displays


@dataclass
class CleanupTask:
    """Individual cleanup task definition."""
    name: str
    phase: CleanupPhase
    cleanup_func: Callable
    timeout: float = 5.0
    async_task: bool = False
    critical: bool = True  # If False, failures won't stop the cleanup process


class SystemCleanupManager:
    """Centralized cleanup manager for the entire UR Voice System."""
    
    def __init__(self):
        self.cleanup_tasks: Dict[CleanupPhase, List[CleanupTask]] = {
            phase: [] for phase in CleanupPhase
        }
        self.cleanup_lock = threading.Lock()
        self.is_cleaning_up = False
        
    def register_cleanup_task(self, task: CleanupTask):
        """Register a cleanup task to be executed during shutdown."""
        with self.cleanup_lock:
            self.cleanup_tasks[task.phase].append(task)
            console.print(f"[blue]📋[/blue] Registered cleanup task: {task.name} ({task.phase.value})")
    
    def register_component(self, 
                          component_name: str,
                          voice_agent=None,
                          bridge=None, 
                          robot_connection=None,
                          ui=None):
        """
        Register a component's cleanup methods, ensuring no duplicates.
        This method is now idempotent.
        """
        
        with self.cleanup_lock:
            # First, remove any existing cleanup tasks for this component to prevent duplicates
            for phase in self.cleanup_tasks:
                self.cleanup_tasks[phase] = [
                    task for task in self.cleanup_tasks[phase] if not task.name.startswith(f"{component_name}_")
                ]

        # Voice Agent Cleanup
        if voice_agent:
            # Use synchronous stop method if available to avoid event loop conflicts
            if hasattr(voice_agent, 'stop_sync') and callable(voice_agent.stop_sync):
                self.register_cleanup_task(CleanupTask(
                    name=f"{component_name}_voice_agent",
                    phase=CleanupPhase.VOICE_AGENT,
                    cleanup_func=voice_agent.stop_sync,
                    timeout=3.0,
                    async_task=False  # Synchronous method
                ))
            elif hasattr(voice_agent, 'stop') and callable(voice_agent.stop):
                # Fallback to async method if sync not available
                self.register_cleanup_task(CleanupTask(
                    name=f"{component_name}_voice_agent",
                    phase=CleanupPhase.VOICE_AGENT,
                    cleanup_func=voice_agent.stop,
                    timeout=3.0,
                    async_task=True
                ))
        
        # Bridge Fabrication Cleanup
        if bridge:
            if hasattr(bridge, '_end_fabrication'):
                self.register_cleanup_task(CleanupTask(
                    name=f"{component_name}_end_fabrication",
                    phase=CleanupPhase.BRIDGE_FABRICATION,
                    cleanup_func=lambda: bridge._end_fabrication() if bridge.fabrication_active else None,
                    timeout=2.0
                ))
            
            # Bridge Connection Cleanup
            if hasattr(bridge, '_cleanup_connections'):
                self.register_cleanup_task(CleanupTask(
                    name=f"{component_name}_bridge_connections",
                    phase=CleanupPhase.BRIDGE_CONNECTIONS,
                    cleanup_func=bridge._cleanup_connections,
                    timeout=3.0
                ))
        
        # Robot Connection Cleanup
        if robot_connection:
            if hasattr(robot_connection, 'cleanup'):
                self.register_cleanup_task(CleanupTask(
                    name=f"{component_name}_robot_connection",
                    phase=CleanupPhase.ROBOT_CONNECTION,
                    cleanup_func=robot_connection.cleanup,
                    timeout=2.0
                ))
        
        # UI Cleanup
        if ui:
            if hasattr(ui, 'stop_live_display'):
                self.register_cleanup_task(CleanupTask(
                    name=f"{component_name}_ui",
                    phase=CleanupPhase.UI_CLEANUP,
                    cleanup_func=ui.stop_live_display,
                    timeout=1.0,
                    critical=False  # UI cleanup failures are not critical
                ))
    
    async def cleanup_all(self, timeout: float = 30.0) -> bool:
        """Execute all cleanup tasks in dependency order."""
        if self.is_cleaning_up:
            console.print("[yellow]⚠[/yellow] Cleanup already in progress...")
            return True
            
        with self.cleanup_lock:
            self.is_cleaning_up = True
        
        try:
            console.print("[yellow]🧹[/yellow] Starting system cleanup...")
            start_time = time.time()
            success = True
            
            # Execute cleanup phases in order
            for phase in CleanupPhase:
                phase_tasks = self.cleanup_tasks[phase]
                if not phase_tasks:
                    continue
                    
                console.print(f"[blue]🔧[/blue] Cleanup phase: {phase.value}")
                
                # Execute tasks in parallel within each phase
                try:
                    phase_timeout = timeout / len(list(CleanupPhase))
                    phase_success = await asyncio.wait_for(
                        self._execute_phase_tasks(phase_tasks),
                        timeout=phase_timeout
                    )
                    if not phase_success:
                        success = False
                except asyncio.TimeoutError:
                    console.print(f"[red]⏰[/red] Phase {phase.value} timeout")
                    success = False
            
            elapsed = time.time() - start_time
            status = "✅" if success else "⚠️"
            console.print(f"[green]{status}[/green] System cleanup completed in {elapsed:.2f}s")
            return success
            
        except Exception as e:
            console.print(f"[red]❌[/red] Cleanup error: {e}")
            return False
        finally:
            self.is_cleaning_up = False
    
    async def _execute_phase_tasks(self, tasks: List[CleanupTask]) -> bool:
        """Execute all tasks in a phase concurrently."""
        if not tasks:
            return True
        
        # Group tasks by sync/async
        sync_tasks = [task for task in tasks if not task.async_task]
        async_tasks = [task for task in tasks if task.async_task]
        
        results = []
        
        # Execute sync tasks in thread pool
        if sync_tasks:
            sync_futures = [
                asyncio.get_event_loop().run_in_executor(
                    None, self._execute_sync_task, task
                ) for task in sync_tasks
            ]
            results.extend(await asyncio.gather(*sync_futures, return_exceptions=True))
        
        # Execute async tasks
        if async_tasks:
            async_futures = [
                self._execute_async_task(task) for task in async_tasks
            ]
            results.extend(await asyncio.gather(*async_futures, return_exceptions=True))
        
        # Check results
        success = True
        for i, result in enumerate(results):
            task = (sync_tasks + async_tasks)[i]
            if isinstance(result, Exception):
                if task.critical:
                    console.print(f"[red]❌[/red] Critical task failed: {task.name} - {result}")
                    success = False
                else:
                    console.print(f"[yellow]⚠[/yellow] Non-critical task failed: {task.name} - {result}")
            else:
                console.print(f"[green]✅[/green] Task completed: {task.name}")
        
        return success
    
    def _execute_sync_task(self, task: CleanupTask) -> bool:
        """Execute a synchronous cleanup task."""
        try:
            result = task.cleanup_func()
            return True
        except Exception as e:
            if task.critical:
                raise e
            else:
                console.print(f"[yellow]⚠[/yellow] Non-critical sync task error: {task.name} - {e}")
                return False
    
    async def _execute_async_task(self, task: CleanupTask) -> bool:
        """Execute an asynchronous cleanup task."""
        try:
            await asyncio.wait_for(task.cleanup_func(), timeout=task.timeout)
            return True
        except asyncio.TimeoutError:
            error_msg = f"Task timeout after {task.timeout}s: {task.name}"
            if task.critical:
                raise Exception(error_msg)
            else:
                console.print(f"[yellow]⚠[/yellow] {error_msg}")
                return False
        except Exception as e:
            if task.critical:
                raise e
            else:
                console.print(f"[yellow]⚠[/yellow] Non-critical async task error: {task.name} - {e}")
                return False
    
    def add_global_cleanup_tasks(self):
        """Add cleanup tasks for global system state."""
        try:
            from ur.core.connection import cleanup_all
            
            self.register_cleanup_task(CleanupTask(
                name="global_connections",
                phase=CleanupPhase.GLOBAL_CLEANUP,
                cleanup_func=cleanup_all,
                timeout=2.0,
                critical=False
            ))
        except ImportError:
            console.print("[yellow]⚠[/yellow] Could not import global cleanup functions")


# Global cleanup manager instance
_cleanup_manager: Optional[SystemCleanupManager] = None

def get_cleanup_manager() -> SystemCleanupManager:
    """Get or create the global cleanup manager."""
    global _cleanup_manager
    if _cleanup_manager is None:
        _cleanup_manager = SystemCleanupManager()
        _cleanup_manager.add_global_cleanup_tasks()
    return _cleanup_manager

def register_for_cleanup(component_name: str, **components):
    """Convenience function to register components for cleanup."""
    manager = get_cleanup_manager()
    manager.register_component(component_name, **components)

async def cleanup_system(timeout: float = 30.0) -> bool:
    """Convenience function to cleanup the entire system."""
    manager = get_cleanup_manager()
    return await manager.cleanup_all(timeout)

def cleanup_system_sync(timeout: float = 30.0) -> bool:
    """Synchronous wrapper for cleanup_system."""
    try:
        import asyncio
        return asyncio.run(cleanup_system(timeout))
    except Exception as e:
        console.print(f"[red]❌[/red] Sync cleanup error: {e}")
        return False

def reset_cleanup_manager():
    """Reset the cleanup manager (useful for testing)."""
    global _cleanup_manager
    _cleanup_manager = None 