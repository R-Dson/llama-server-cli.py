#!/usr/bin/env python3

import os
import sys
import json
import signal
import subprocess
import time
import glob
import threading
import queue
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import print as rprint
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.status import Status
import questionary
from prompt_toolkit.keys import Keys
from prompt_toolkit.shortcuts import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import get_app


# Main app
app = typer.Typer(help="LLama Server CLI Tool")
profile_app = typer.Typer(help="Profile management")
server_app = typer.Typer(help="Server management")
config_app = typer.Typer(help="Interactive configuration")
app.add_typer(profile_app, name="profile")
app.add_typer(server_app, name="server")
app.add_typer(config_app, name="config")

# Rich console for pretty output
console = Console()


def numbered_choice(message, choices):
    """
    Display a menu with numbered choices using questionary with number key shortcuts
    """
    # Add numbers only to the first 9 choices for display
    numbered_choices = []
    for i, choice in enumerate(choices):
        if i < 9:  # Only first 9 choices get number prefixes
            numbered_choices.append(f"{i+1}. {choice}")
        else:
            # For options beyond 9, don't show a number prefix
            numbered_choices.append(f"   {choice}")
    numbered_choices = choices
    # Display the menu with custom instruction
    console.print(f"\n[cyan]{message}[/cyan]")
    console.print("[dim](Use arrow keys to navigate or press number keys 1-9 to select)[/dim]")
    
    try:
        # Use questionary's built-in shortcut support
        selected = questionary.select(
            "",  # Empty prompt since we already printed one
            choices=numbered_choices,
            qmark="",
            use_shortcuts=True,  # Enable shortcut keys
            use_indicator=True,
            style=questionary.Style([
                ('selected', 'bg:#268bd2 #ffffff'),
                ('highlighted', 'bg:#268bd2 #ffffff'),
            ]),
        ).ask()
    except Exception as e:
        console.print(f"[red]Menu error: {e}[/red]")
        return None
    
    # If selection was made, extract the actual choice
    if selected:
        # For options with number prefixes (1-9), remove the prefix
        if selected.startswith(tuple("123456789")):
            # Find the first dot+space and extract everything after it
            idx = selected.find('. ')
            if idx >= 0:
                return selected[idx+2:]
        else:
            # For options beyond 9, just remove the leading spaces
            return selected.strip()
    
    return None


class LlamaServerCLI:
    def __init__(self):
        self.config_path = "config.json"
        self.llama_server_process = None
        # Define default_config before using it in load_config
        self.default_config = {
            "model": None,
            "ctx_size": 2048,
            "n_gpu_layers": 99,
            "threads": -1,
            "temp": 0.8,
            "top_k": 40,
            "top_p": 0.9,
            "batch_size": 512,
            "host": "127.0.0.1",
            "port": 8080,
            "flash_attn": True,
            "mlock": False,
            "no_mmap": False,
            "ignore_eos": False,
            "embedding": False,
            "continuous_batching": True,
            "profiles": {
                "default": {}
            }
        }
        # Now load config which may use default_config
        self.config = self.load_config()
        # Create server monitor
        self.server_monitor = ServerMonitor(self)

    def load_config(self) -> Dict:
        """Load configuration from config file or create default if not exists"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                return self.default_config
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")
            return self.default_config

    def save_config(self) -> None:
        """Save current configuration to config file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            console.print(f"[green]Configuration saved to {self.config_path}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving config: {e}[/red]")

    def list_profiles(self) -> None:
        """List all available profiles"""
        profiles = self.config.get("profiles", {})
        if not profiles:
            console.print("[yellow]No profiles found.[/yellow]")
            return

        table = Table(title="Available Profiles")
        table.add_column("Profile", style="cyan")
        table.add_column("Status", style="green")

        active_profile = self.config.get("active_profile", "default")
        for name in profiles:
            status = "ACTIVE" if name == active_profile else ""
            table.add_row(name, status)

        console.print(table)

    def show_profile(self, name: str) -> None:
        """Show details of a specific profile"""
        profiles = self.config.get("profiles", {})
        if name not in profiles:
            console.print(f"[red]Profile '{name}' not found[/red]")
            return

        profile_settings = profiles[name]
        active_settings = self.get_active_settings(name)
        
        table = Table(title=f"Profile: {name}")
        table.add_column("Setting", style="cyan")
        table.add_column("Value")  # Remove fixed style, will set per row
        table.add_column("Source", style="yellow")
        
        for key, value in active_settings.items():
            if key != "profiles" and key != "active_profile":
                source = "profile" if key in profile_settings else "default"
                # Use red for None values, green for others
                value_style = "red" if value is None else "green"
                table.add_row(key, f"[{value_style}]{value}[/{value_style}]", source)
                
        console.print(table)

    def create_profile(self, name: str) -> None:
        """Create a new profile"""
        if "profiles" not in self.config:
            self.config["profiles"] = {}
            
        if name in self.config["profiles"]:
            console.print(f"[yellow]Profile '{name}' already exists[/yellow]")
            return
            
        self.config["profiles"][name] = {}
        self.save_config()
        console.print(f"[green]Profile '{name}' created[/green]")

    def delete_profile(self, name: str) -> None:
        """Delete an existing profile"""
        if name == "default":
            console.print("[red]Cannot delete default profile[/red]")
            return
            
        if name not in self.config.get("profiles", {}):
            console.print(f"[red]Profile '{name}' not found[/red]")
            return
            
        del self.config["profiles"][name]
        
        # If we deleted the active profile, switch to default
        if self.config.get("active_profile") == name:
            self.config["active_profile"] = "default"
            
        self.save_config()
        console.print(f"[green]Profile '{name}' deleted[/green]")

    def set_active_profile(self, name: str) -> None:
        """Set the active profile"""
        if name not in self.config.get("profiles", {}):
            console.print(f"[red]Profile '{name}' not found[/red]")
            return
        
        # Check if this is actually a change
        current_profile = self.config.get("active_profile", "default")
        is_change = current_profile != name
            
        self.config["active_profile"] = name
        self.save_config()
        console.print(f"[green]Active profile set to '{name}'[/green]")
        
        # If the server is running and we changed profiles, restart with new profile
        if is_change and self.llama_server_process and self.llama_server_process.poll() is None:
            self.restart_server_seamless(name)

    def set_setting(self, profile: str, key: str, value: Any) -> None:
        """Set a setting for a specific profile"""
        if profile not in self.config.get("profiles", {}):
            console.print(f"[red]Profile '{profile}' not found[/red]")
            return
            
        # Convert value to appropriate type
        try:
            # Try to convert to number if possible
            if isinstance(value, str):
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                else:
                    try:
                        if "." in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        # Keep as string if not a number
                        pass
        except:
            # Keep as is if not a string
            pass
        
        # Check if this is actually a change
        current_value = self.config["profiles"][profile].get(key)
        is_change = current_value != value
            
        self.config["profiles"][profile][key] = value
        self.save_config()
        console.print(f"[green]Set {key}={value} for profile '{profile}'[/green]")
        
        # If the server is running and this is the active profile, restart it
        active_profile = self.config.get("active_profile", "default")
        if is_change and profile == active_profile and self.llama_server_process and self.llama_server_process.poll() is None:
            self.restart_server_seamless(active_profile)

    def clear_setting(self, profile: str, key: str) -> None:
        """Clear a setting from a specific profile"""
        if profile not in self.config.get("profiles", {}):
            console.print(f"[red]Profile '{profile}' not found[/red]")
            return
        
        if key == "Back to settings menu":
            return

        if key not in self.config["profiles"][profile]:
            console.print(f"[yellow]Setting '{key}' not found in profile '{profile}'[/yellow]")
            return
            
        del self.config["profiles"][profile][key]
        self.save_config()
        console.print(f"[green]Cleared {key} from profile '{profile}'[/green]")
        
        # If the server is running and this is the active profile, restart it
        active_profile = self.config.get("active_profile", "default")
        if profile == active_profile and self.llama_server_process and self.llama_server_process.poll() is None:
            self.restart_server_seamless(active_profile)

    def get_active_settings(self, profile_name: Optional[str] = None) -> Dict:
        """Get active settings by combining defaults with profile settings"""
        if profile_name is None:
            profile_name = self.config.get("active_profile", "default")
            
        # Start with a copy of the default config
        settings = {k: v for k, v in self.config.items() if k != "profiles"}
        
        # Override with profile-specific settings
        profile = self.config.get("profiles", {}).get(profile_name, {})
        settings.update(profile)
        
        return settings

    def build_llama_args(self, profile_name: Optional[str] = None) -> List[str]:
        """Build command line arguments for llama-server based on active settings"""
        settings = self.get_active_settings(profile_name)
        
        args = ["./llama-server"]
        
        # Model is required
        if not settings.get("model"):
            console.print("[red]Error: No model specified in configuration[/red]")
            return []
            
        args.extend(["-m", settings["model"]])
        
        # Add other settings
        mapping = {
            "context_size": ("-c", str),
            "threads": ("-t", str),
            "temp": ("--temp", str),
            "top_k": ("--top-k", str),
            "top_p": ("--top-p", str),
            "batch_size": ("-b", str),
            "host": ("--host", str),
            "port": ("--port", str),
            "n_predict": ("-n", str),
            "mirostat": ("--mirostat", str),
            "mirostat_eta": ("--mirostat-eta", str),
            "mirostat_tau": ("--mirostat-tau", str),
            "repeat_penalty": ("--repeat-penalty", str),
            "n_gpu_layers": ("-ngl", str),
            "seed": ("-s", str),
            "ctx_size": ("-c", str),
        }
        
        for key, (arg, converter) in mapping.items():
            if key in settings and settings[key] is not None:
                args.extend([arg, converter(settings[key])])
                
        # Handle boolean flags
        bool_flags = {
            "ignore_eos": "--ignore-eos",
            "no_mmap": "--no-mmap",
            "mlock": "--mlock",
            "embedding": "--embedding",
            "flash_attn": "--flash-attn",
            "continuous_batching": "--cont-batching",
            "no_continuous_batching": "--no-cont-batching",
        }
        
        for key, arg in bool_flags.items():
            # Only add the flag if it's set to True
            # For no_continuous_batching, it's the inverse of continuous_batching
            if key == "no_continuous_batching":
                if settings.get("continuous_batching") is False:
                    args.append(arg)
            elif key in settings and settings[key]:
                args.append(arg)
                
        return args

    def start_server(self, profile_name: Optional[str] = None, background: bool = False, output_queue: Optional[queue.Queue] = None) -> bool:
        """Start the llama-server with the specified profile"""
        # Get settings to validate before starting
        settings = self.get_active_settings(profile_name)
        
        # Validate essential settings
        validation_errors = []
        
        # Check for model (most critical setting)
        if not settings.get("model"):
            validation_errors.append("[bold red]ERROR:[/bold red] No model specified in configuration")
        elif not os.path.exists(settings["model"]):
            validation_errors.append(f"[bold red]ERROR:[/bold red] Model file not found: {settings['model']}")
        
        # Check for other important settings
        if "ctx_size" in settings and (not isinstance(settings["ctx_size"], int) or settings["ctx_size"] <= 0):
            validation_errors.append(f"[bold yellow]WARNING:[/bold yellow] Invalid context size: {settings['ctx_size']}")
        
        if "threads" in settings and not isinstance(settings["threads"], int):
            validation_errors.append(f"[bold yellow]WARNING:[/bold yellow] Invalid thread count: {settings['threads']}")
        
        # Display validation errors
        if validation_errors:
            console.clear()  # Clear screen first to make error message prominent
            console.print("\n[red]Cannot start server due to configuration issues:[/red]")
            for error in validation_errors:
                console.print(f"  {error}")
            
            # Suggest next steps
            console.print("\n[cyan]What to do next:[/cyan]")
            console.print("  1. Edit your profile settings to fix these issues")
            console.print(f"  2. Set a model path with: ./llama-server-cli.py profile set {profile_name or settings.get('active_profile', 'default')} model /path/to/model.gguf")
            console.print("  3. Or choose a different profile with valid settings")
            
            # Wait for user acknowledgment before returning to menu
            console.print("\n[yellow]Press Enter to return to the menu...[/yellow]")
            input()
            return False
        
        args = self.build_llama_args(profile_name)
        if not args:
            return False
            
        console.print(f"[green]Starting llama-server with command:[/green] {' '.join(args)}")
        
        try:
            # Use Popen to keep a reference to the process
            self.llama_server_process = subprocess.Popen(
                args, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Register a signal handler to kill the server on exit
            def cleanup(signum, frame):
                self.stop_server()
                sys.exit(0)
                
            signal.signal(signal.SIGINT, cleanup)
            signal.signal(signal.SIGTERM, cleanup)
            
            # In background mode, start a thread to read output and put it in the queue
            if background and output_queue:
                def output_reader():
                    for line in iter(self.llama_server_process.stdout.readline, ''):
                        if line:
                            output_queue.put(line.rstrip())
                    # When the process ends, put a status message in the queue
                    output_queue.put(("STATUS", "Server process ended"))
                
                reader_thread = threading.Thread(target=output_reader)
                reader_thread.daemon = True
                reader_thread.start()
                
                return True
            
            # In foreground mode, print the output directly
            else:
                # Print server output with rich
                with console.status("[bold green]Server started. Press Ctrl+C to stop.") as status:
                    console.print("[bold green]Server started. Press Ctrl+C to stop.[/bold green]")
                    while self.llama_server_process.poll() is None:
                        line = self.llama_server_process.stdout.readline()
                        if line:
                            console.print(line.rstrip())
                        
                # Check exit code
                exit_code = self.llama_server_process.returncode
                if exit_code != 0:
                    console.print(f"[red]Server exited with code {exit_code}[/red]")
                    
        except Exception as e:
            console.print(f"[red]Error starting server: {e}[/red]")
            self.stop_server()
            return False
            
        return True

    def stop_server(self) -> None:
        """Stop the running llama-server"""
        if self.llama_server_process and self.llama_server_process.poll() is None:
            console.print("[yellow]Stopping llama-server...[/yellow]")
            try:
                self.llama_server_process.terminate()
                # Wait up to 5 seconds for graceful termination
                try:
                    self.llama_server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    console.print("[red]Server not responding, forcing kill...[/red]")
                    self.llama_server_process.kill()
            except Exception as e:
                console.print(f"[red]Error stopping server: {e}[/red]")
                
            self.llama_server_process = None
            console.print("[green]Server stopped[/green]")
        else:
            console.print("[yellow]No running server to stop[/yellow]")

    def interactive_config(self) -> None:
        """Interactive configuration mode"""
        try:
            while True:
                console.clear()
                rprint(Panel.fit("LLama Server CLI - Interactive Configuration", style="bold cyan"))
                
                # Get or select active profile
                active_profile = self.config.get("active_profile", "default")
                profiles = list(self.config.get("profiles", {}).keys())
                
                # Display current profile info
                self.show_profile(active_profile)
                
                # Display server status if running
                if self.llama_server_process and self.llama_server_process.poll() is None:
                    console.print("[bold green]Server is running in the background[/bold green]")
                
                # Add appropriate server option based on state
                if self.llama_server_process and self.llama_server_process.poll() is None:
                    menu_options = ["Stop background server", "Restart server with current profile"]
                else:
                    menu_options = ["Start server with current profile"]
                
                # Main menu with numbered options
                menu_options.extend([
                    "Edit current profile settings",
                    "Change active profile",
                    "Create new profile",
                    "Delete a profile",
                    "Exit"
                ])
                
                choice = numbered_choice("What would you like to do?", menu_options)
                
                if not choice:  # Handle None return (Ctrl+C or similar)
                    break
                
                if choice == "Change active profile":
                    if len(profiles) <= 1:
                        console.print("[yellow]Only one profile exists.[/yellow]")
                        time.sleep(1)
                        continue
                        
                    selected_profile = numbered_choice("Select profile:", profiles)
                    
                    if selected_profile:
                        self.set_active_profile(selected_profile)
                        time.sleep(1)
                    
                elif choice == "Edit current profile settings":
                    self._interactive_edit_profile(active_profile)
                    
                elif choice == "Create new profile":
                    name = questionary.text("Enter new profile name:").ask()
                    if name and name.strip():
                        self.create_profile(name.strip())
                        if questionary.confirm(f"Set {name} as active profile?").ask():
                            self.set_active_profile(name.strip())
                    time.sleep(1)
                    
                elif choice == "Delete a profile":
                    if len(profiles) <= 1:
                        console.print("[yellow]Cannot delete the only profile.[/yellow]")
                        time.sleep(1)
                        continue
                        
                    to_delete = numbered_choice("Select profile to delete:",
                        [p for p in profiles if p != "default"]
                    )
                    
                    if to_delete:
                        if questionary.confirm(f"Are you sure you want to delete {to_delete}?").ask():
                            self.delete_profile(to_delete)
                    time.sleep(1)
                    
                elif choice == "Start server with current profile":
                    console.clear()
                    # Create a background queue even though we won't display it in the main menu
                    bg_queue = queue.Queue()
                    start_success = self.start_server(active_profile, background=True, output_queue=bg_queue)
                    if start_success:
                        console.print("[green]Server started in background mode[/green]")
                    time.sleep(1)
                
                elif choice == "Stop background server":
                    self.stop_server()
                    time.sleep(1)
                
                elif choice == "Restart server with current profile":
                    # Stop current server
                    self.stop_server()
                    time.sleep(1)
                    # Start new one
                    bg_queue = queue.Queue()
                    start_success = self.start_server(active_profile, background=True, output_queue=bg_queue)
                    if start_success:
                        console.print("[green]Server restarted in background mode[/green]")
                    time.sleep(1)
                    
                elif choice == "Exit":
                    # Make sure to stop the server before exiting
                    if self.llama_server_process and self.llama_server_process.poll() is None:
                        console.print("[yellow]Stopping server before exit...[/yellow]")
                        self.stop_server()
                    break
        except KeyboardInterrupt:
            # Make sure to stop the server on Ctrl+C as well
            if self.llama_server_process and self.llama_server_process.poll() is None:
                console.print("[yellow]Stopping server before exit...[/yellow]")
                self.stop_server()
            print("\nExiting...")
            sys.exit(0)

    def _interactive_edit_profile(self, profile_name: str) -> None:
        """Interactive profile editor"""
        try:
            while True:
                console.clear()
                rprint(Panel.fit(f"Editing Profile: {profile_name}", style="bold cyan"))
                
                # Show current profile settings
                self.show_profile(profile_name)
                
                # Build settings menu
                profile_settings = self.config.get("profiles", {}).get(profile_name, {})
                all_settings = self.get_active_settings(profile_name)
                
                # Common settings to configure
                common_settings = [
                    "model",
                    "ctx_size",
                    "threads",
                    "temp",
                    "top_k",
                    "top_p",
                    "batch_size",
                    "host",
                    "port",
                    "n_gpu_layers",
                    "flash_attn",
                    "mlock",
                    "no_mmap",
                    "continuous_batching",
                ]
                
                # Add existing settings that might not be in common_settings
                for key in profile_settings:
                    if key not in common_settings and key != "profiles" and key != "active_profile":
                        common_settings.append(key)
                
                # Menu options
                choices = [
                    f"Set {key} (current: {all_settings.get(key, 'Not set')})" 
                    for key in common_settings
                ] + ["Add custom setting", "Clear a setting", "Back to main menu"]
                
                choice = numbered_choice("Choose setting to edit:", choices)
                
                if not choice:  # Handle None return (Ctrl+C or similar)
                    break
                
                if choice == "Back to main menu":
                    break
                elif choice == "Add custom setting":
                    key = questionary.text("Enter setting name:").ask()
                    if key and key.strip():
                        value = questionary.text(f"Enter value for {key}:").ask()
                        if value is not None:  # Allow empty string
                            self.set_setting(profile_name, key.strip(), value)
                            time.sleep(1)
                elif choice == "Clear a setting":
                    if not profile_settings:
                        console.print("[yellow]No settings to clear in this profile.[/yellow]")
                        time.sleep(1)
                        continue
                    choices_list = list(profile_settings.keys())
                    choices_list.append("Back to settings menu")
                    setting_to_clear = numbered_choice("Select setting to clear:",
                        choices_list
                    )
                    
                    if setting_to_clear:
                        self.clear_setting(profile_name, setting_to_clear)
                        time.sleep(1)
                else:
                    # Extract the setting name from the choice string
                    setting = choice.split(" ")[1]
                    
                    # Get appropriate input based on setting type
                    if setting == "model":
                        # Find available models
                        models = find_gguf_models()
                        
                        if not models:
                            console.print("[yellow]No GGUF models found in the gguf directory or current directory.[/yellow]")
                            console.print("[yellow]Please place your models in the gguf directory and try again.[/yellow]")
                            console.print("[yellow]Alternatively, you can enter a custom path:[/yellow]")
                            value = questionary.text(
                                f"Enter path to model file:",
                                default=str(all_settings.get(setting, ""))
                            ).ask()
                        else:
                            # Add option for manual entry
                            models.append("Enter custom path manually")
                            
                            # Show model selection menu
                            console.print("[green]Available models:[/green]")
                            selected_model = numbered_choice("Select a model to use:", models)
                            
                            if selected_model == "Enter custom path manually":
                                value = questionary.text(
                                    f"Enter path to model file:",
                                    default=str(all_settings.get(setting, ""))
                                ).ask()
                            else:
                                value = selected_model
                    elif setting in ["temp", "top_p"]:
                        value = questionary.text(
                            f"Enter value for {setting} (0.0-1.0):",
                            default=str(all_settings.get(setting, ""))
                        ).ask()
                    elif setting in ["ctx_size", "threads", "top_k", "batch_size", "port", "n_gpu_layers"]:
                        value = questionary.text(
                            f"Enter value for {setting} (number):",
                            default=str(all_settings.get(setting, ""))
                        ).ask()
                    elif setting in ["ignore_eos", "no_mmap", "mlock", "embedding", "flash_attn", "continuous_batching"]:
                        value = questionary.confirm(f"Enable {setting}?", default=bool(all_settings.get(setting, False))).ask()
                    else:
                        value = questionary.text(
                            f"Enter value for {setting}:",
                            default=str(all_settings.get(setting, ""))
                        ).ask()
                    
                    if value is not None:  # Allow empty string and False
                        self.set_setting(profile_name, setting, value)
                        time.sleep(0.5)
        except KeyboardInterrupt:
            # Return to previous menu on Ctrl+C
            pass

    def restart_server_seamless(self, profile_name: str) -> None:
        """Restart the server seamlessly in the background without visible messages or delays"""
        if self.llama_server_process and self.llama_server_process.poll() is None:
            # Stop current server silently
            self.llama_server_process.terminate()
            try:
                self.llama_server_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.llama_server_process.kill()
                
            # Start new server silently
            bg_queue = queue.Queue()
            args = self.build_llama_args(profile_name)
            if not args:
                return False
                
            try:
                # Start new process
                self.llama_server_process = subprocess.Popen(
                    args, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Set up background output reader
                def output_reader():
                    for line in iter(self.llama_server_process.stdout.readline, ''):
                        if line:
                            bg_queue.put(line.rstrip())
                    bg_queue.put(("STATUS", "Server process ended"))
                
                reader_thread = threading.Thread(target=output_reader)
                reader_thread.daemon = True
                reader_thread.start()
                
                return True
            except Exception:
                return False
        return False


class ServerMonitor:
    """Class to monitor a running llama-server and handle live updates"""
    
    def __init__(self, cli):
        self.cli = cli
        self.output_queue = queue.Queue()
        self.running = False
        self.monitor_thread = None
        self.current_profile = None
        self.current_settings = {}
        self.server_output = []
        self.max_log_lines = 20  # Maximum number of log lines to display
        
    def start_monitoring(self, profile_name: str) -> None:
        """Start monitoring the server with the specified profile"""
        if self.running:
            console.print("[yellow]Server is already running and being monitored[/yellow]")
            return
        
        self.current_profile = profile_name
        self.current_settings = self.cli.get_active_settings(profile_name)
        
        # Start the server
        success = self.cli.start_server(profile_name, background=True, output_queue=self.output_queue)
        if not success:
            return
            
        self.running = True
        
        # Start the monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Enter monitoring UI
        self._monitoring_ui()
    
    def _monitor_loop(self) -> None:
        """Background thread to monitor server output"""
        while self.running and self.cli.llama_server_process:
            try:
                # Check if server is still running
                if self.cli.llama_server_process.poll() is not None:
                    self.running = False
                    self.output_queue.put(("STATUS", "Server stopped with exit code " + 
                                         str(self.cli.llama_server_process.returncode)))
                    break
                
                # Read output from queue
                try:
                    line = self.output_queue.get(timeout=0.1)
                    if isinstance(line, tuple) and line[0] == "STATUS":
                        # Status update
                        self.server_output.append(f"[yellow]{line[1]}[/yellow]")
                    else:
                        # Log line
                        self.server_output.append(str(line))
                    
                    # Trim log to max lines
                    if len(self.server_output) > self.max_log_lines:
                        self.server_output = self.server_output[-self.max_log_lines:]
                        
                    self.output_queue.task_done()
                except queue.Empty:
                    pass
                    
            except Exception as e:
                self.server_output.append(f"[red]Monitor error: {e}[/red]")
                
            time.sleep(0.1)
    
    def _monitoring_ui(self) -> None:
        """Interactive UI for monitoring the server"""
        try:
            while self.running:
                console.clear()
                
                # Display current status
                if self.cli.llama_server_process and self.cli.llama_server_process.poll() is None:
                    status_text = f"[green]Server running with profile: {self.current_profile}[/green]"
                else:
                    status_text = "[red]Server stopped[/red]"
                
                rprint(Panel.fit(f"Llama Server Monitor - {status_text}", style="bold cyan"))
                
                # Show last X lines of output
                console.print("[bold]Server Output:[/bold]")
                for line in self.server_output:
                    console.print(line)
                
                # Menu options
                console.print("\n[bold]Options:[/bold]")
                options = [
                    "Edit settings and restart server",
                    "Restart server with current settings",
                    "Stop server and return to main menu",
                    "View current settings"
                ]
                
                choice = numbered_choice("What would you like to do?", options)
                
                if choice == "Edit settings and restart server":
                    # Stop the server
                    if self.cli.llama_server_process and self.cli.llama_server_process.poll() is None:
                        self.cli.stop_server()
                        self.running = False
                        time.sleep(1)  # Give the server time to stop
                    
                    # Edit settings
                    self.cli._interactive_edit_profile(self.current_profile)
                    
                    # Restart the server and monitoring
                    self.start_monitoring(self.current_profile)
                    return
                    
                elif choice == "Stop server and return to main menu":
                    if self.cli.llama_server_process and self.cli.llama_server_process.poll() is None:
                        self.cli.stop_server()
                    self.running = False
                    return
                    
                elif choice == "Restart server with current settings":
                    # Stop the server
                    if self.cli.llama_server_process and self.cli.llama_server_process.poll() is None:
                        self.cli.stop_server()
                        time.sleep(1)  # Give the server time to stop
                    
                    # Restart it
                    success = self.cli.start_server(self.current_profile, background=True, output_queue=self.output_queue)
                    if success:
                        self.running = True
                        self.server_output.append("[green]Server restarted[/green]")
                    
                elif choice == "View current settings":
                    console.clear()
                    self.cli.show_profile(self.current_profile)
                    console.print("\n[yellow]Press Enter to continue...[/yellow]")
                    input()
                
        except KeyboardInterrupt:
            # Handle Ctrl+C
            console.print("[yellow]Stopping monitoring...[/yellow]")
            self.running = False
            if self.cli.llama_server_process and self.cli.llama_server_process.poll() is None:
                self.cli.stop_server()
                
        finally:
            # Make sure the server is properly cleaned up
            self.running = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2)


# Create singleton instance
cli = LlamaServerCLI()


@profile_app.command("list")
def list_profiles():
    """List all available profiles"""
    cli.list_profiles()


@profile_app.command("show")
def show_profile(name: str = typer.Argument(..., help="Profile name")):
    """Show details of a specific profile"""
    cli.show_profile(name)


@profile_app.command("create")
def create_profile(name: str = typer.Argument(..., help="Profile name")):
    """Create a new profile"""
    cli.create_profile(name)


@profile_app.command("delete")
def delete_profile(name: str = typer.Argument(..., help="Profile name")):
    """Delete an existing profile"""
    cli.delete_profile(name)


@profile_app.command("use")
def set_active_profile(name: str = typer.Argument(..., help="Profile name")):
    """Set active profile"""
    cli.set_active_profile(name)


@profile_app.command("set")
def set_setting(
    profile: str = typer.Argument(..., help="Profile name"),
    key: str = typer.Argument(..., help="Setting key"),
    value: str = typer.Argument(..., help="Setting value")
):
    """Set a profile setting"""
    cli.set_setting(profile, key, value)


@profile_app.command("clear")
def clear_setting(
    profile: str = typer.Argument(..., help="Profile name"),
    key: str = typer.Argument(..., help="Setting key")
):
    """Clear a profile setting"""
    cli.clear_setting(profile, key)


@server_app.command("start")
def start_server(profile: Optional[str] = typer.Option(None, help="Profile name to use")):
    """Start the server with specified profile (or active profile if none specified)"""
    cli.start_server(profile)


@server_app.command("stop")
def stop_server():
    """Stop the running server"""
    cli.stop_server()


@config_app.command("interactive")
def interactive_config():
    """Start interactive configuration mode"""
    cli.interactive_config()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """LLama Server CLI Tool"""
    try:
        # If no subcommand is provided, launch interactive mode
        if ctx.invoked_subcommand is None:
            cli.interactive_config()
    finally:
        # Always ensure server is stopped when application exits
        if cli.llama_server_process and cli.llama_server_process.poll() is None:
            console.print("[yellow]Stopping server before exit...[/yellow]")
            cli.stop_server()


# Function to find GGUF models in gguf directory and subdirectories
def find_gguf_models() -> List[str]:
    """Find all GGUF model files in the gguf directory and subdirectories"""
    # Check if gguf directory exists, create it if it doesn't
    if not os.path.exists('gguf'):
        os.makedirs('gguf')
        console.print("[yellow]Created gguf directory for models[/yellow]")
        return []
    
    # Find all .gguf files in the gguf directory and subdirectories
    models = []
    for root, _, files in os.walk('gguf'):
        for file in files:
            if file.endswith('.gguf'):
                model_path = os.path.join(root, file)
                models.append(model_path)
    
    # Also look for .gguf files in the current directory
    for file in glob.glob('*.gguf'):
        models.append(file)
    
    # Sort models by name
    models.sort()
    
    return models


if __name__ == "__main__":
    app() 