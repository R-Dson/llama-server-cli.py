#!/usr/bin/env python3

import glob
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from threading import Thread
from typing import Any, Dict, List, Optional

import questionary
import requests
import typer
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Main app
app = typer.Typer(help="LLama Server CLI Tool")
profile_app = typer.Typer(help="Profile management")
server_app = typer.Typer(help="Server management")
config_app = typer.Typer(help="Interactive configuration")
api_app = typer.Typer(help="API server management")  
app.add_typer(profile_app, name="profile")
app.add_typer(server_app, name="server")
app.add_typer(config_app, name="config")
app.add_typer(api_app, name="api") 

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
            numbered_choices.append(f"{i + 1}. {choice}")
        else:
            # For options beyond 9, don't show a number prefix
            numbered_choices.append(f"   {choice}")
    numbered_choices = choices
    # Display the menu with custom instruction
    console.print(f"\n[cyan]{message}[/cyan]")
    console.print(
        "[dim](Use arrow keys to navigate or press number keys 1-9 to select)[/dim]"
    )

    try:
        # Use questionary's built-in shortcut support
        selected = questionary.select(
            "",  # Empty prompt since we already printed one
            choices=numbered_choices,
            qmark="",
            use_shortcuts=True,  # Enable shortcut keys
            use_indicator=True,
            style=questionary.Style(
                [
                    ("selected", "bg:#268bd2 #ffffff"),
                    ("highlighted", "bg:#268bd2 #ffffff"),
                ]
            ),
        ).ask()
    except Exception as e:
        console.print(f"[red]Menu error: {e}[/red]")
        return None

    # If selection was made, extract the actual choice
    if selected:
        # For options with number prefixes (1-9), remove the prefix
        if selected.startswith(tuple("123456789")):
            # Find the first dot+space and extract everything after it
            idx = selected.find(". ")
            if idx >= 0:
                return selected[idx + 2 :]
        else:
            # For options beyond 9, just remove the leading spaces
            return selected.strip()

    return None


class LlamaServerCLI:
    def __init__(self):
        self.console = console
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
            "port": 8999,
            "flash_attn": True,
            "mlock": False,
            "no_mmap": False,
            "ignore_eos": False,
            "embedding": False,
            "continuous_batching": True,
            "api_host": "0.0.0.0",
            "api_port": 8000,
            "profiles": {"default": {}},
        }
        # Now load config which may use default_config
        self.config = self.load_config()
        # Create server monitor
        self.server_monitor = ServerMonitor(self)
        # Create API server
        self.api_server = APIServer(self)
        self.api_thread = None
        
    def _output_reader(self, process, queue, handle_ending=True):
        """Common output reading function for process streams"""
        try:
            for line in iter(process.stdout.readline, ""):
                if line:
                    queue.put(line.rstrip())
        except Exception as e:
            queue.put(f"Error reading output: {e}")
        finally:
            # Add status message when the process ends
            if handle_ending:
                queue.put(("STATUS", "Server process ended"))

    def _handle_error(self, operation, error, console_only=False):
        """Common error handling function"""
        error_msg = f"Error {operation}: {error}"
        self.console.print(f"[red]{error_msg}[/red]")
        if not console_only:
            return error_msg

    def load_config(self) -> Dict:
        """Load configuration from config file or create default if not exists"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    return json.load(f)
            else:
                return self.default_config
        except Exception as e:
            self._handle_error("loading config", e, True)
            return self.default_config

    def save_config(self) -> None:
        """Save current configuration to config file"""
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)
            console.print(f"[green]Configuration saved to {self.config_path}[/green]")
        except Exception as e:
            self._handle_error("saving config", e, True)

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
        if (
            is_change
            and self.llama_server_process
            and self.llama_server_process.poll() is None
        ):
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
        if (
            is_change
            and profile == active_profile
            and self.llama_server_process
            and self.llama_server_process.poll() is None
        ):
            self.restart_server_seamless(active_profile)

    def clear_setting(self, profile: str, key: str) -> None:
        """Clear a setting from a specific profile"""
        if profile not in self.config.get("profiles", {}):
            console.print(f"[red]Profile '{profile}' not found[/red]")
            return

        if key == "Back to settings menu":
            return

        if key not in self.config["profiles"][profile]:
            console.print(
                f"[yellow]Setting '{key}' not found in profile '{profile}'[/yellow]"
            )
            return

        del self.config["profiles"][profile][key]
        self.save_config()
        console.print(f"[green]Cleared {key} from profile '{profile}'[/green]")

        # If the server is running and this is the active profile, restart it
        active_profile = self.config.get("active_profile", "default")
        if (
            profile == active_profile
            and self.llama_server_process
            and self.llama_server_process.poll() is None
        ):
            self.restart_server_seamless(active_profile)

    def _get_setting(self, profile_name, key, default=None):
        """Safely get setting value with fallback to default profile or global default"""
        # Get the profile settings
        profiles = self.config.get("profiles", {})
        profile = profiles.get(profile_name, {})
        
        # First try profile-specific setting
        if key in profile:
            return profile[key]
        
        # Then try global setting
        if key in self.config:
            return self.config[key]
            
        # Finally fallback to provided default
        return default
        
    def get_active_settings(self, profile_name: Optional[str] = None) -> Dict:
        """Get active settings by combining defaults with profile settings"""
        if profile_name is None:
            profile_name = self.config.get("active_profile", "default")

        # Start with all global settings except profiles
        settings = {k: v for k, v in self.config.items() if k != "profiles" and k != "active_profile"}

        # Add all profile-specific settings that override globals
        profile = self.config.get("profiles", {}).get(profile_name, {})
        settings.update(profile)

        return settings

    def build_llama_args(self, profile_name: Optional[str] = None) -> List[str]:
        """Build command line arguments for llama-server based on active settings"""
        settings = self.get_active_settings(profile_name)

        args = ["./llama-server"]

        # Model is required
        if not settings.get("model"):
            self._handle_error("building args", "No model specified in configuration", True)
            return []

        args.extend(["-m", settings["model"]])

        # --- Settings with Values ---
        # Map setting names to their command line args and conversion functions
        value_args = {
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

        # Add all parameters with values
        for key, (arg, converter) in value_args.items():
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
        }

        # Add boolean flags if set to True
        for key, arg in bool_flags.items():
            if settings.get(key, False):
                args.append(arg)
                
        # Special case for no_continuous_batching (inverse of continuous_batching)
        if settings.get("continuous_batching") is False:
            args.append("--no-cont-batching")

        return args

    def start_server(
        self,
        profile_name: Optional[str] = None,
        background: bool = False,
        output_queue: Optional[queue.Queue] = None,
    ) -> bool:
        """Start the llama-server with the specified profile"""
        # Get settings to validate before starting
        settings = self.get_active_settings(profile_name)

        # Validate essential settings
        validation_errors = []

        # Check for model (most critical setting)
        if not settings.get("model"):
            validation_errors.append(
                "[bold red]ERROR:[/bold red] No model specified in configuration"
            )
        elif not os.path.exists(settings["model"]):
            validation_errors.append(
                f"[bold red]ERROR:[/bold red] Model file not found: {settings['model']}"
            )

        # Check for other important settings
        if "ctx_size" in settings and (
            not isinstance(settings["ctx_size"], int) or settings["ctx_size"] <= 0
        ):
            validation_errors.append(
                f"[bold yellow]WARNING:[/bold yellow] Invalid context size: {settings['ctx_size']}"
            )

        if "threads" in settings and not isinstance(settings["threads"], int):
            validation_errors.append(
                f"[bold yellow]WARNING:[/bold yellow] Invalid thread count: {settings['threads']}"
            )

        # Display validation errors
        if validation_errors:
            console.clear()  # Clear screen first to make error message prominent
            console.print(
                "\n[red]Cannot start server due to configuration issues:[/red]"
            )
            for error in validation_errors:
                console.print(f"  {error}")

            # Suggest next steps
            console.print("\n[cyan]What to do next:[/cyan]")
            console.print("  1. Edit your profile settings to fix these issues")
            console.print(
                f"  2. Set a model path with: ./llama-server-cli.py profile set {profile_name or settings.get('active_profile', 'default')} model /path/to/model.gguf"
            )
            console.print("  3. Or choose a different profile with valid settings")

            # Wait for user acknowledgment before returning to menu
            if not background:
                console.print("\n[yellow]Press Enter to return to the menu...[/yellow]")
                input()
            return False

        args = self.build_llama_args(profile_name)
        if not args:
            return False

        console.print(
            f"[green]Starting llama-server with command:[/green] {' '.join(args)}"
        )

        try:
            # Use Popen to keep a reference to the process
            self.llama_server_process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )

            # In background mode, start a thread to read output and put it in the queue
            if background and output_queue:
                reader_thread = threading.Thread(
                    target=lambda: self._output_reader(self.llama_server_process, output_queue)
                )
                reader_thread.daemon = (
                    True  # Make thread daemon so it doesn't block program exit
                )
                reader_thread.start()

                return True

            # In foreground mode, print the output directly
            else:
                # Print server output with rich
                with console.status(
                    "[bold green]Server started. Press Ctrl+C to stop."
                ) as status:
                    console.print(
                        "[bold green]Server started. Press Ctrl+C to stop.[/bold green]"
                    )
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
            if self.llama_server_process:
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
                rprint(
                    Panel.fit(
                        "LLama Server CLI - Interactive Configuration",
                        style="bold cyan",
                    )
                )

                # Show status of both servers
                llama_status = (
                    "[green]Running[/green]"
                    if (
                        self.llama_server_process
                        and self.llama_server_process.poll() is None
                    )
                    else "[red]Stopped[/red]"
                )
                api_status = (
                    "[green]Running[/green]"
                    if self.api_server.running
                    else "[red]Stopped[/red]"
                )
                console.print(f"\n[bold]Llama Server Status:[/bold] {llama_status}")
                console.print(f"[bold]API Server Status:[/bold] {api_status}")

                # Get or select active profile
                active_profile = self.config.get("active_profile", "default")
                profiles = list(self.config.get("profiles", {}).keys())

                # Display current profile info
                self.show_profile(active_profile)

                # Build server options based on current state
                server_options = []
                if (
                    self.llama_server_process
                    and self.llama_server_process.poll() is None
                ):
                    server_options.extend(["Stop Llama Server", "Restart Llama Server"])
                else:
                    server_options.append("Start Llama Server")

                if self.api_server.running:
                    server_options.extend(["Stop API Server", "Restart API Server"])
                else:
                    server_options.append("Start API Server")

                # Main menu with numbered options
                menu_options = [
                    *server_options,
                    "Edit current profile settings",
                    "Edit API server settings",  # Add API server settings option
                    "Change active profile",
                    "Create new profile",
                    "Delete a profile",
                    "Exit",
                ]

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
                    
                elif choice == "Edit API server settings":
                    # Add API server configuration
                    api_host = questionary.text(
                        "API Server Host:", default=str(self.config.get("api_host", "0.0.0.0"))
                    ).ask()
                    
                    api_port = questionary.text(
                        "API Server Port:", default=str(self.config.get("api_port", 8000))
                    ).ask()
                    
                    try:
                        api_port = int(api_port)
                        self.config["api_host"] = api_host
                        self.config["api_port"] = api_port
                        self.save_config()
                        console.print("[green]API server settings updated[/green]")
                        
                        # Update API server settings if it's running
                        if self.api_server.running:
                            restart = questionary.confirm("Restart API server with new settings?").ask()
                            if restart:
                                self.api_server.stop()
                                # Update the server object with new settings
                                self.api_server.host = api_host
                                self.api_server.port = api_port
                                self.api_server.start()
                                console.print("[green]API server restarted with new settings[/green]")
                    except ValueError:
                        self._handle_error("updating API settings", "Port must be a number", True)
                    time.sleep(1)

                elif choice == "Create new profile":
                    name = questionary.text("Enter new profile name:").ask()
                    if name and name.strip():
                        self.create_profile(name.strip())
                        if questionary.confirm(f"Set {name} as active profile?").ask():
                            self.set_active_profile(name.strip())
                    time.sleep(1)

                elif choice == "Delete a profile":
                    if len(profiles) <= 1:
                        console.print(
                            "[yellow]Cannot delete the only profile.[/yellow]"
                        )
                        time.sleep(1)
                        continue

                    to_delete = numbered_choice(
                        "Select profile to delete:",
                        [p for p in profiles if p != "default"],
                    )

                    if to_delete:
                        if questionary.confirm(
                            f"Are you sure you want to delete {to_delete}?"
                        ).ask():
                            self.delete_profile(to_delete)
                    time.sleep(1)

                elif choice == "Start Llama Server":
                    console.clear()
                    bg_queue = queue.Queue()
                    start_success = self.start_server(
                        active_profile, background=True, output_queue=bg_queue
                    )
                    if start_success:
                        console.print(
                            "[green]Llama Server started in background mode[/green]"
                        )
                    time.sleep(1)

                elif choice == "Stop Llama Server":
                    self.stop_server()
                    time.sleep(1)

                elif choice == "Restart Llama Server":
                    self.stop_server()
                    time.sleep(1)
                    bg_queue = queue.Queue()
                    start_success = self.start_server(
                        active_profile, background=True, output_queue=bg_queue
                    )
                    if start_success:
                        console.print(
                            "[green]Llama Server restarted in background mode[/green]"
                        )
                    time.sleep(1)

                elif choice == "Start API Server":
                    self.api_server.start()
                    console.print("[green]API Server started[/green]")
                    time.sleep(1)

                elif choice == "Stop API Server":
                    self.api_server.stop()
                    console.print("[yellow]API Server stopped[/yellow]")
                    time.sleep(1)

                elif choice == "Restart API Server":
                    self.api_server.stop()
                    self.api_server.start()
                    console.print("[green]API Server restarted[/green]")
                    time.sleep(1)

                elif choice == "Exit":
                    # Make sure to stop both servers before exiting
                    if (
                        self.llama_server_process
                        and self.llama_server_process.poll() is None
                    ):
                        console.print(
                            "[yellow]Stopping Llama Server before exit...[/yellow]"
                        )
                        self.stop_server()
                    if self.api_server.running:
                        console.print(
                            "[yellow]Stopping API Server before exit...[/yellow]"
                        )
                        self.api_server.stop()
                    break

        except KeyboardInterrupt:
            # Make sure to stop both servers on Ctrl+C as well
            if self.llama_server_process and self.llama_server_process.poll() is None:
                console.print("[yellow]Stopping Llama Server before exit...[/yellow]")
                self.stop_server()
            if self.api_server.running:
                console.print("[yellow]Stopping API Server before exit...[/yellow]")
                self.api_server.stop()
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
                    if (
                        key not in common_settings
                        and key != "profiles"
                        and key != "active_profile"
                    ):
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
                        console.print(
                            "[yellow]No settings to clear in this profile.[/yellow]"
                        )
                        time.sleep(1)
                        continue
                    choices_list = list(profile_settings.keys())
                    choices_list.append("Back to settings menu")
                    setting_to_clear = numbered_choice(
                        "Select setting to clear:", choices_list
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
                            console.print(
                                "[yellow]No GGUF models found in the gguf directory or current directory.[/yellow]"
                            )
                            console.print(
                                "[yellow]Please place your models in the gguf directory and try again.[/yellow]"
                            )
                            console.print(
                                "[yellow]Alternatively, you can enter a custom path:[/yellow]"
                            )
                            value = questionary.text(
                                "Enter path to model file:",
                                default=str(all_settings.get(setting, "")),
                            ).ask()
                        else:
                            # Add option for manual entry
                            models.append("Enter custom path manually")

                            # Show model selection menu
                            console.print("[green]Available models:[/green]")
                            selected_model = numbered_choice(
                                "Select a model to use:", models
                            )

                            if selected_model == "Enter custom path manually":
                                value = questionary.text(
                                    "Enter path to model file:",
                                    default=str(all_settings.get(setting, "")),
                                ).ask()
                            else:
                                value = selected_model
                    elif setting in ["temp", "top_p"]:
                        value = questionary.text(
                            f"Enter value for {setting} (0.0-1.0):",
                            default=str(all_settings.get(setting, "")),
                        ).ask()
                    elif setting in [
                        "ctx_size",
                        "threads",
                        "top_k",
                        "batch_size",
                        "port",
                        "n_gpu_layers",
                    ]:
                        value = questionary.text(
                            f"Enter value for {setting} (number):",
                            default=str(all_settings.get(setting, "")),
                        ).ask()
                    elif setting in [
                        "ignore_eos",
                        "no_mmap",
                        "mlock",
                        "embedding",
                        "flash_attn",
                        "continuous_batching",
                    ]:
                        value = questionary.confirm(
                            f"Enable {setting}?",
                            default=bool(all_settings.get(setting, False)),
                        ).ask()
                    else:
                        value = questionary.text(
                            f"Enter value for {setting}:",
                            default=str(all_settings.get(setting, "")),
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
                    universal_newlines=True,
                )

                # Set up background output reader
                reader_thread = threading.Thread(
                    target=lambda: self._output_reader(self.llama_server_process, bg_queue, False)
                )
                reader_thread.daemon = True
                reader_thread.start()

                return True
            except Exception:
                return False
        return False


class APIServer:
    def __init__(self, cli):
        self.cli = cli
        self.app = FastAPI()
        self.router = APIRouter()
        self.setup_routes()
        self.server = None
        self.running = False
        # Use values from configuration or fallback to defaults
        self.host = self.cli.config.get("api_host", "0.0.0.0")
        self.port = self.cli.config.get("api_port", 8000)  # Default to port 8000 if not specified
        self.lock = threading.Lock()  # Add a lock for model switching
        self.timeout = 30  # Timeout for server requests

    def setup_routes(self):
        self.router.add_api_route("/v1/models", self.list_models, methods=["GET"])
        self.router.add_api_route(
            "/v1/models/{model}", self.model_info, methods=["GET"]
        )
        self.router.add_api_route(
            "/v1/chat/completions", self.create_chat_completion, methods=["POST"]
        )
        self.app.include_router(self.router)

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    async def list_models(self):
        profiles = self.cli.config.get("profiles", {})
        return {
            "object": "list",
            "data": [
                {
                    "id": profile_name,
                    "object": "profile",
                    "owned_by": "custom",
                    "permissions": [],
                    "profile_settings": self.cli.get_active_settings(profile_name),
                }
                for profile_name in profiles
            ],
        }

    async def model_info(self, model: str):
        if model not in self.cli.config.get("profiles", {}):
            raise HTTPException(status_code=404, detail="Profile not found")
        return {
            "id": model,
            "object": "profile",
            "owned_by": "custom",
            "permissions": [],
            "settings": self.cli.get_active_settings(model),
        }

    async def create_chat_completion(self, request: dict):
        model = request.get("model")
        if not model:
            raise HTTPException(400, "No profile specified")

        if model not in self.cli.config.get("profiles", {}):
            raise HTTPException(404, "Profile not found")

        with self.lock:
            current_profile = self.cli.config.get("active_profile", "default")
            if model != current_profile:
                console.print(
                    f"[yellow]Switching from profile {current_profile} to {model}[/yellow]"
                )

                # First stop the current llama-server if it's running
                if (
                    self.cli.llama_server_process
                    and self.cli.llama_server_process.poll() is None
                ):
                    console.print("[yellow]Stopping current server...[/yellow]")
                    try:
                        self.cli.llama_server_process.terminate()
                        try:
                            self.cli.llama_server_process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            self.cli.llama_server_process.kill()
                            self.cli.llama_server_process.wait()
                    except Exception as e:
                        raise HTTPException(
                            500, f"Failed to stop current server: {str(e)}"
                        )
                    finally:
                        self.cli.llama_server_process = None

                # Update the active profile in config
                self.cli.config["active_profile"] = model
                self.cli.save_config()

                # Start server with new profile
                console.print(
                    f"[yellow]Starting server with profile {model}...[/yellow]"
                )
                bg_queue = queue.Queue()
                if not self.cli.start_server(
                    model, background=True, output_queue=bg_queue
                ):
                    raise HTTPException(500, "Failed to start server with new profile")

                # Wait for server to be ready
                if not self.wait_for_server_ready():
                    raise HTTPException(503, "Server startup timed out")

            # At this point, the correct server should be running
            # Verify server is running before forwarding request
            if (
                not self.cli.llama_server_process
                or self.cli.llama_server_process.poll() is not None
            ):
                raise HTTPException(503, "Llama server is not running")

            # Forward request to the running llama-server
            server_url = f"http://{self.cli.config['host']}:{self.cli.config['port']}/v1/chat/completions"
            console.print(f"[yellow]Forwarding request to {server_url}[/yellow]")
            console.print(
                f"[yellow]Request payload: {json.dumps(request, indent=2)}[/yellow]"
            )

            try:
                # For streaming requests, use stream=True
                is_streaming = request.get("stream", False)
                response = requests.post(
                    server_url,
                    json=request,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream"
                        if is_streaming
                        else "application/json",
                    },
                    timeout=self.timeout,
                    stream=is_streaming,
                )

                # Check for error response
                if response.status_code != 200:
                    try:
                        error_details = (
                            response.json()
                            if response.text
                            else "No error details provided"
                        )
                    except:
                        error_details = response.text or "No error details provided"
                    console.print(
                        f"[red]Llama server error response (Status {response.status_code}): {error_details}[/red]"
                    )
                    # Also log response headers for debugging
                    console.print(
                        f"[red]Response headers: {dict(response.headers)}[/red]"
                    )
                    raise HTTPException(500, f"Llama server error: {error_details}")

                # For streaming responses, return a StreamingResponse
                if is_streaming:

                    async def stream_generator():
                        try:
                            for line in response.iter_lines():
                                if line:
                                    # Send each line as-is
                                    yield line + b"\n"
                        except Exception as e:
                            console.print(
                                f"[red]Error during streaming: {str(e)}[/red]"
                            )
                            yield f'data: {{"error": "{str(e)}"}}\n\n'.encode("utf-8")

                    return StreamingResponse(
                        stream_generator(), media_type="text/event-stream"
                    )

                # For non-streaming, return the JSON response
                return response.json()

            except requests.exceptions.ConnectionError as e:
                console.print(f"[red]Connection error: {str(e)}[/red]")
                raise HTTPException(
                    503, "Could not connect to llama-server. Is it running?"
                )
            except requests.exceptions.RequestException as e:
                console.print(f"[red]Request error: {str(e)}[/red]")
                raise HTTPException(500, f"Error forwarding request: {str(e)}")
            except Exception as e:
                console.print(f"[red]Unexpected error: {str(e)}[/red]")
                raise HTTPException(500, f"Unexpected error: {str(e)}")

    def wait_for_server_ready(self, timeout=30):
        """Wait for llama-server to be ready to accept requests"""
        start = time.time()
        console.print("[yellow]Waiting for server to be ready...[/yellow]")
        while time.time() - start < timeout:
            try:
                # First check if the server process is still running
                if (
                    not self.cli.llama_server_process
                    or self.cli.llama_server_process.poll() is not None
                ):
                    self.cli._handle_error("waiting for server", "Server process has stopped", True)
                    return False

                # Try a basic health check first
                model_url = f"http://{self.cli.config['host']}:{self.cli.config['port']}/v1/models"
                model_res = requests.get(model_url, timeout=2)
                if model_res.status_code == 200:
                    # Try a simple completion to ensure model is fully loaded
                    test_url = f"http://{self.cli.config['host']}:{self.cli.config['port']}/v1/chat/completions"
                    test_payload = {
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 1,
                        "temperature": 0,
                        "stream": False,  # Explicitly disable streaming for test
                    }
                    try:
                        test_res = requests.post(test_url, json=test_payload, timeout=5)
                        if test_res.status_code != 200:
                            error_details = test_res.text or "No error details"
                            console.print(
                                f"[yellow]Server not ready yet. Response: {error_details}[/yellow]"
                            )
                        else:
                            console.print("[green]Server is ready[/green]")
                            return True
                    except requests.RequestException as e:
                        console.print(
                            f"[yellow]Server starting up... ({str(e)})[/yellow]"
                        )
            except requests.RequestException as e:
                console.print(f"[yellow]Waiting for server... ({str(e)})[/yellow]")
            time.sleep(1)
        self.cli._handle_error("waiting for server", "Server startup timed out", True)
        return False

    def start(self):
        if not self.running:
            self.running = True
            self.server = uvicorn.Server(
                config=uvicorn.Config(
                    app=self.app,
                    host=self.host,
                    port=self.port,
                    log_level="info",
                    access_log=True,  # Enable access logs
                )
            )
            self.thread = Thread(target=self._run_server)
            self.thread.start()
            self.cli.console.print(
                f"[green]API server started on {self.host}:{self.port}[/green]"
            )

    def _run_server(self):
        try:
            self.server.run()
        except OSError as e:
            if "address already in use" in str(e):
                self.cli._handle_error(f"starting API server", f"Port {self.port} already in use!", True)
            else:
                self.cli._handle_error("running API server", e, True)
        except Exception as e:
            self.cli._handle_error("running API server", e, True)
        finally:
            self.running = False

    def stop(self):
        if self.running:
            self.running = False
            self.server.should_exit = True
            self.thread.join()


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
            console.print(
                "[yellow]Server is already running and being monitored[/yellow]"
            )
            return

        self.current_profile = profile_name
        self.current_settings = self.cli.get_active_settings(profile_name)

        # Start the server
        success = self.cli.start_server(
            profile_name, background=True, output_queue=self.output_queue
        )
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
                    self.output_queue.put(
                        (
                            "STATUS",
                            "Server stopped with exit code "
                            + str(self.cli.llama_server_process.returncode),
                        )
                    )
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
                        self.server_output = self.server_output[-self.max_log_lines :]

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
                if (
                    self.cli.llama_server_process
                    and self.cli.llama_server_process.poll() is None
                ):
                    status_text = f"[green]Server running with profile: {self.current_profile}[/green]"
                else:
                    status_text = "[red]Server stopped[/red]"

                rprint(
                    Panel.fit(
                        f"Llama Server Monitor - {status_text}", style="bold cyan"
                    )
                )

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
                    "View current settings",
                ]

                choice = numbered_choice("What would you like to do?", options)

                if choice == "Edit settings and restart server":
                    # Stop the server
                    if (
                        self.cli.llama_server_process
                        and self.cli.llama_server_process.poll() is None
                    ):
                        self.cli.stop_server()
                        self.running = False
                        time.sleep(1)  # Give the server time to stop

                    # Edit settings
                    self.cli._interactive_edit_profile(self.current_profile)

                    # Restart the server and monitoring
                    self.start_monitoring(self.current_profile)
                    return

                elif choice == "Stop server and return to main menu":
                    if (
                        self.cli.llama_server_process
                        and self.cli.llama_server_process.poll() is None
                    ):
                        self.cli.stop_server()
                    self.running = False
                    return

                elif choice == "Restart server with current settings":
                    # Stop the server
                    if (
                        self.cli.llama_server_process
                        and self.cli.llama_server_process.poll() is None
                    ):
                        self.cli.stop_server()
                        time.sleep(1)  # Give the server time to stop

                    # Restart it
                    success = self.cli.start_server(
                        self.current_profile,
                        background=True,
                        output_queue=self.output_queue,
                    )
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
            if (
                self.cli.llama_server_process
                and self.cli.llama_server_process.poll() is None
            ):
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
    value: str = typer.Argument(..., help="Setting value"),
):
    """Set a profile setting"""
    cli.set_setting(profile, key, value)


@profile_app.command("clear")
def clear_setting(
    profile: str = typer.Argument(..., help="Profile name"),
    key: str = typer.Argument(..., help="Setting key"),
):
    """Clear a profile setting"""
    cli.clear_setting(profile, key)


@server_app.command("start")
def start_server(
    profile: Optional[str] = typer.Option(None, help="Profile name to use"),
):
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


@api_app.command("start")
def start_api_server():
    """Start the OpenAI-compatible API server"""
    cli.api_server.start()
    console.print("[yellow]Press Ctrl+C to stop the API server[/yellow]")
    try:
        # Keep the main thread alive
        while cli.api_server.running:
            time.sleep(1)
    except KeyboardInterrupt:
        cli.api_server.stop()
        console.print("\n[green]API server stopped[/green]")


@api_app.command("stop")
def stop_api_server():
    """Stop the OpenAI-compatible API server"""
    if cli.api_server.running:
        cli.api_server.stop()
        console.print("[green]API server stopped[/green]")
    else:
        console.print("[yellow]API server is not running[/yellow]")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """LLama Server CLI Tool"""
    try:
        # Set up signal handlers at program level
        def cleanup(signum=None, frame=None):
            if cli.llama_server_process and cli.llama_server_process.poll() is None:
                console.print("[yellow]Stopping server before exit...[/yellow]")
                cli.stop_server()
            if cli.api_server.running:
                console.print("[yellow]Stopping API server...[/yellow]")
                cli.api_server.stop()
            if signum is not None:  # Only exit if called as signal handler
                sys.exit(0)

        signal.signal(signal.SIGINT, cleanup)
        signal.signal(signal.SIGTERM, cleanup)

        # If no subcommand is provided, launch interactive mode
        if ctx.invoked_subcommand is None:
            cli.interactive_config()
    finally:
        # Always ensure servers are stopped when application exits
        cleanup()


# Function to find GGUF models in gguf directory and subdirectories
def find_gguf_models() -> List[str]:
    """Find all GGUF model files in the gguf directory and subdirectories"""
    # Check if gguf directory exists, create it if it doesn't
    if not os.path.exists("gguf"):
        os.makedirs("gguf")
        console.print("[yellow]Created gguf directory for models[/yellow]")
        return []

    # Find all .gguf files in the gguf directory and subdirectories
    models = []
    for root, _, files in os.walk("gguf"):
        for file in files:
            if file.endswith(".gguf"):
                model_path = os.path.join(root, file)
                models.append(model_path)

    # Also look for .gguf files in the current directory
    for file in glob.glob("*.gguf"):
        models.append(file)

    # Sort models by name
    models.sort()

    return models


if __name__ == "__main__":
    app()
