#!/usr/bin/env python3

import asyncio
import glob
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid  # Added for tool call IDs
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

# Rich console for pretty output
console = Console()


def numbered_choice(message, choices):
    """Display a menu with choices using questionary"""
    console.print(f"\n[cyan]{message}[/cyan]")

    try:
        selected = questionary.select(
            "",
            choices=choices,
            qmark="",
            use_shortcuts=True,
            style=questionary.Style(
                [
                    ("selected", "bg:#268bd2 #ffffff"),
                ]
            ),
        ).ask()
        return selected
    except Exception as e:
        console.print(f"[red]Menu error: {e}[/red]")
        return None


class LlamaServerCLI:
    def __init__(self):
        self.console = console
        self.config_path = "config.json"

        # Server state
        self.llama_server_process = None
        self.last_activity_timestamp = time.time()
        self.can_kill = True
        self.inactivity_monitor_thread = None

        # Default configuration
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
            "inactivity_timeout": 300,          # how long the entire server stays alive without any new requests
            "server_ready_timeout": 60,
            "stream_watchdog_timeout": 600,     # how long a single streaming response can maximum take
            "log_llama_server_to_file": False,  # Default to console output for llama-server
            "log_api_server_to_file": False,    # Default to console output for api-server
            "profiles": {"default": {}},
        }
        # Load configuration
        self.config = self.load_config()

        # Create server components
        self.server_monitor = ServerMonitor(self)
        self.api_server = APIServer(self)

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

    def get_active_settings(self, profile_name: Optional[str] = None) -> Dict:
        """Get active settings with proper precedence: profile > global > defaults"""
        # Use provided profile or active profile
        if profile_name is None:
            profile_name = self.config.get("active_profile", "default")

        # Start with defaults
        settings = self.default_config.copy()

        # Apply global settings (excluding profiles and active_profile)
        for key, value in self.config.items():
            if key not in ["profiles", "active_profile"]:
                settings[key] = value

        # Apply profile-specific settings
        profile = self.config.get("profiles", {}).get(profile_name, {})
        for key, value in profile.items():
            settings[key] = value

        return settings

    def build_llama_args(self, profile_name: Optional[str] = None) -> List[str]:
        """Build command line arguments for llama-server based on active settings"""
        settings = self.get_active_settings(profile_name)
        args = ["./llama-server"]

        # Check for required model parameter
        if not settings.get("model"):
            console.print("[red]Error: No model specified in configuration[/red]")
            return []

        # Add string parameters
        for key in ["host", "port"]:
            if key in settings and settings[key] is not None:
                args.extend([f"--{key}", str(settings[key])])

        # Add boolean flags
        for key in [
            "ignore_eos",
            "no_mmap",
            "mlock",
            "embedding",
            "flash_attn",
            "no-perf",
        ]:
            if settings.get(key, False):
                args.append(f"--{key.replace('_', '-')}")

        # Special case for continuous batching
        if settings.get("continuous_batching", True):
            args.append("--cont-batching")
        else:
            args.append("--no-cont-batching")

        excluded_keys = [
            "continuous_batching",
            # Exclude CLI/API server specific settings
            "profiles",
            "active_profile",
            "api_host",
            "api_port",
            "inactivity_timeout",
            "server_ready_timeout",
            "stream_watchdog_timeout",
            "log_llama_server_to_file",
            "log_api_server_to_file",
        ]

        # Add any custom parameters not already handled
        for key, value in settings.items():
            if key not in excluded_keys and value is not None:
                # Format the parameter name with dashes
                if "-" not in key:
                    cmd_key = f"--{key.replace('_', '-')}"
                else:
                    cmd_key = f"--{key}"

                # Add as flag or key-value pair based on type
                if isinstance(value, bool):
                    if value:
                        args.append(cmd_key)
                else:
                    args.extend([cmd_key, str(value)])

        return args

    def start_server(
        self,
        profile_name: Optional[str] = None,
        background: bool = False,
        output_queue: Optional[queue.Queue] = None,
    ) -> bool:
        """Start the llama-server with the specified profile"""
        # Get settings for the selected profile
        settings = self.get_active_settings(profile_name)
        log_to_file = settings.get("log_llama_server_to_file", False)

        # Validate model path - most critical setting
        if not settings.get("model"):
            console.print("[red]ERROR: No model specified in configuration[/red]")
            return False
        if not os.path.exists(settings["model"]):
            console.print(
                f"[red]ERROR: Model file not found: {settings['model']}[/red]"
            )
            return False

        # Build command arguments
        args = self.build_llama_args(profile_name)
        if not args:
            return False

        console.print(
            f"[green]Starting llama-server with command:[/green] {' '.join(args)}"
        )

        try:
            log_file_handle = None
            log_filename = None

            stdout_target = None
            stderr_target = None

            if log_to_file:
                # Create logs directory if it doesn't exist
                os.makedirs("logs", exist_ok=True)
                # Create log file with timestamp
                log_filename = f"logs/llama-server-{time.strftime('%Y%m%d-%H%M%S')}.log"
                log_file_handle = open(log_filename, "w")
                console.print(
                    f"[green]Server output will be logged to: {log_filename}[/green]"
                )
                stdout_target = log_file_handle
                stderr_target = (
                    subprocess.STDOUT
                )  # Redirect stderr to the same log file
            else:
                # If not logging to file
                if background:
                    # Suppress output in background mode if not logging to file
                    stdout_target = subprocess.DEVNULL
                    stderr_target = subprocess.DEVNULL
                    console.print(
                        "[yellow]Server starting in background (output suppressed).[/yellow]"
                    )
                else:
                    # In foreground mode without file logging, pipe output to read later
                    stdout_target = subprocess.PIPE
                    stderr_target = subprocess.PIPE
                    console.print(
                        "[green]Server starting in foreground (output to console).[/green]"
                    )

            # Start the process with configured output redirection
            self.llama_server_process = subprocess.Popen(
                args,
                stdout=stdout_target,
                stderr=stderr_target,
                text=True
                if not log_to_file and not background
                else False,  # Use text mode only for PIPE
                bufsize=1,
                universal_newlines=True
                if not log_to_file and not background
                else False,
            )

            if background:
                # In background mode, start a thread to read output *only if logging to file*
                if log_to_file and output_queue and log_filename:
                    reader_thread = threading.Thread(
                        target=lambda: self._log_file_reader(log_filename, output_queue)
                    )
                    reader_thread.daemon = True
                    reader_thread.start()
                # Start inactivity monitor regardless of logging
                self.start_inactivity_monitor()
                return True
            else:
                # In foreground mode
                if log_to_file and log_filename:
                    # Just indicate logging and wait for the process
                    console.print(
                        f"[bold green]Server running. Output logged to {log_filename}. Press Ctrl+C to stop.[/bold green]"
                    )
                    # Wait for the process to finish directly
                    try:
                        exit_code = self.llama_server_process.wait()
                        if exit_code != 0:
                            console.print(
                                f"[red]Server exited with code {exit_code}[/red]"
                            )
                        else:
                            console.print("[yellow]Server process ended.[/yellow]")
                    except KeyboardInterrupt:
                        console.print(
                            "\n[yellow]Stopping server due to Ctrl+C...[/yellow]"
                        )
                        self.stop_server()  # Attempt graceful shutdown
                        console.print("[yellow]Server stopped.[/yellow]")
                    # Exit code handling is now inside this block.

                elif not log_to_file:
                    # Read directly from process pipes and print to console
                    console.print(
                        "[bold green]Server running (output to console). Press Ctrl+C to stop.[/bold green]"
                    )
                    try:
                        # Read stdout
                        stdout_thread = threading.Thread(
                            target=self._pipe_reader,
                            args=(
                                self.llama_server_process.stdout,
                                "[green][Server][/green] ",
                            ),
                        )
                        stdout_thread.daemon = True
                        stdout_thread.start()
                        # Read stderr
                        stderr_thread = threading.Thread(
                            target=self._pipe_reader,
                            args=(
                                self.llama_server_process.stderr,
                                "[yellow][Server Status/Error][/yellow] ",
                            ),
                        )
                        stderr_thread.daemon = True
                        stderr_thread.start()

                        # Wait for process to finish while threads print output
                        while self.llama_server_process.poll() is None:
                            time.sleep(0.2)

                        # Wait for reader threads to finish
                        stdout_thread.join(timeout=1)
                        stderr_thread.join(timeout=1)

                    except Exception as e:
                        console.print(
                            f"[red]Error reading server output pipes: {e}[/red]"
                        )

                    # Ensure process is finished before checking code
                    exit_code = self.llama_server_process.wait()
                    if exit_code != 0:
                        console.print(f"[red]Server exited with code {exit_code}[/red]")
                    else:
                        console.print("[yellow]Server process ended.[/yellow]")

        except Exception as e:
            console.print(f"[red]Error starting server: {e}[/red]")
            if self.llama_server_process:
                self.stop_server()  # stop_server now handles log file closing
            elif log_file_handle:  # Close handle if Popen failed
                try:
                    log_file_handle.close()
                except:
                    pass
            return False
        finally:
            # Ensure log file handle is closed if it was opened
            if log_file_handle:
                try:
                    log_file_handle.close()
                except:
                    pass

        return True

    def _log_file_reader(
        self, log_filename: str, output_queue: queue.Queue, handle_ending=True
    ):
        """Read output from log file and put it in the queue"""
        try:
            # Open the log file for reading
            with open(log_filename, "r") as f:
                # Start at the beginning of the file
                while True:
                    # Check if the server is still running
                    if self.llama_server_process.poll() is not None:
                        if handle_ending:
                            output_queue.put(("STATUS", "Server process ended"))
                        break

                    # Read new lines
                    line = f.readline()
                    if line:
                        output_queue.put(line.rstrip())
                    else:
                        # No new lines, sleep briefly and try again
                        time.sleep(0.1)
        except Exception as e:
            output_queue.put(f"Error reading log file: {e}")
            if handle_ending:
                output_queue.put(
                    ("STATUS", "Server process monitoring ended due to error")
                )

    def _pipe_reader(self, pipe, prefix=""):
        """Reads lines from a process pipe and prints them."""
        try:
            for line in iter(pipe.readline, ""):
                if line:
                    console.print(f"{prefix}{line.rstrip()}")
            pipe.close()
        except ValueError:  # Handle pipe closed error
            pass
        except Exception as e:
            console.print(f"[red]Error reading pipe: {e}[/red]")

    def stop_server(self) -> None:
        """Stop the running llama-server"""
        if (
            not self.llama_server_process
            or self.llama_server_process.poll() is not None
        ):
            # Try to find and terminate any orphaned processes
            self._terminate_orphaned_processes()
            console.print("[yellow]No running server to stop[/yellow]")
            return

        console.print("[yellow]Stopping llama-server...[/yellow]")
        try:
            # First try graceful termination
            self.llama_server_process.terminate()

            # Wait up to 10 seconds for graceful termination
            try:
                self.llama_server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                console.print("[red]Server not responding, forcing kill...[/red]")
                self.llama_server_process.kill()
                try:
                    self.llama_server_process.wait(timeout=2)
                except:
                    pass

            # Final check for orphaned processes
            self._terminate_orphaned_processes()

            self.llama_server_process = None
            console.print("[green]Server stopped[/green]")

        except Exception as e:
            console.print(f"[red]Error stopping server: {e}[/red]")
            # Still try to terminate orphaned processes
            self._terminate_orphaned_processes()

    def _terminate_orphaned_processes(self) -> None:
        """Find and terminate any orphaned llama-server processes"""
        try:
            if os.name == "posix":  # Linux/Mac
                # Check for any processes matching the pattern, but explicitly exclude Python scripts
                # Use a more specific pattern that won't catch our Python script
                check_result = subprocess.run(
                    ["pgrep", "-f", r"^\./llama-server(\s|$)"],
                    capture_output=True,
                    text=True,
                )

                if check_result.returncode == 0 and check_result.stdout.strip():
                    console.print(
                        "[yellow]Found orphaned llama-server processes. Terminating...[/yellow]"
                    )

                    # Try SIGTERM first with the more specific pattern
                    subprocess.run(
                        ["pkill", "-TERM", "-f", r"^\./llama-server(\s|$)"],
                        capture_output=True,
                    )
                    time.sleep(2)

                    # Force kill any remaining processes with the more specific pattern
                    check_again = subprocess.run(
                        ["pgrep", "-f", r"^\./llama-server(\s|$)"], capture_output=True
                    )
                    if check_again.returncode == 0:
                        subprocess.run(
                            ["pkill", "-KILL", "-f", r"^\./llama-server(\s|$)"],
                            capture_output=True,
                        )

            elif os.name == "nt":  # Windows
                check_result = subprocess.run(
                    ["tasklist", "/fi", "imagename eq llama-server.exe"],
                    capture_output=True,
                    text=True,
                )

                if "llama-server.exe" in check_result.stdout:
                    console.print(
                        "[yellow]Found orphaned llama-server.exe processes. Terminating...[/yellow]"
                    )
                    subprocess.run(
                        ["taskkill", "/IM", "llama-server.exe", "/F"],
                        capture_output=True,
                    )

        except Exception as e:
            console.print(f"[red]Error terminating orphaned processes: {e}[/red]")

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

                # Get active profile
                active_profile = self.config.get("active_profile", "default")
                profiles = list(self.config.get("profiles", {}).keys())

                # If server is running, show its output
                if (
                    self.llama_server_process
                    and self.llama_server_process.poll() is None
                ):
                    server_output = self.server_monitor.get_output()
                    if server_output:
                        console.print("\n[bold]Server Output:[/bold]")
                        for line in server_output[-5:]:  # Show last 5 lines
                            console.print(line)

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
                        "API Server Host:",
                        # Use default_config as fallback for display
                        default=str(
                            self.config.get("api_host", self.default_config["api_host"])
                        ),
                    ).ask()

                    api_port = questionary.text(
                        "API Server Port:",
                        # Use default_config as fallback for display
                        default=str(
                            self.config.get("api_port", self.default_config["api_port"])
                        ),
                    ).ask()

                    # Add server_ready_timeout input
                    server_ready_timeout_str = questionary.text(
                        "Server Ready Timeout (seconds):",
                        default=str(
                            self.config.get(
                                "server_ready_timeout",
                                self.default_config["server_ready_timeout"],
                            )
                        ),
                    ).ask()

                    # Add stream_watchdog_timeout input
                    stream_watchdog_timeout_str = questionary.text(
                        "Stream Watchdog Timeout (seconds):",
                        default=str(
                            self.config.get(
                                "stream_watchdog_timeout",
                                self.default_config["stream_watchdog_timeout"],
                            )
                        ),
                    ).ask()

                    log_api_to_file = questionary.confirm(
                        "Log API Server output to file (in logs/)?",
                        default=bool(self.config.get("log_api_server_to_file", False)),
                    ).ask()

                    log_llama_to_file = questionary.confirm(
                        "Log Llama Server output to file (in logs/)?",
                        default=bool(
                            self.config.get("log_llama_server_to_file", False)
                        ),
                    ).ask()

                    try:
                        api_port = int(api_port)
                        server_ready_timeout = int(server_ready_timeout_str)
                        stream_watchdog_timeout = int(stream_watchdog_timeout_str)

                        is_change = (
                            self.config.get("api_host") != api_host
                            or self.config.get("api_port") != api_port
                            or self.config.get("server_ready_timeout")
                            != server_ready_timeout
                            or self.config.get("streamc_watchdog_timeout")
                            != stream_watchdog_timeout
                            or self.config.get("log_api_server_to_file")
                            != log_api_to_file
                            or self.config.get("log_llama_server_to_file")
                            != log_llama_to_file
                        )

                        self.config["api_host"] = api_host
                        self.config["api_port"] = api_port
                        self.config["server_ready_timeout"] = server_ready_timeout
                        self.config["stream_watchdog_timeout"] = stream_watchdog_timeout
                        self.config["log_api_server_to_file"] = log_api_to_file
                        self.config["log_llama_server_to_file"] = log_llama_to_file

                        self.save_config()
                        console.print(
                            "[green]API server and logging settings updated[/green]"
                        )

                        # Restart API server if its settings changed and it's running
                        if is_change and self.api_server.running:
                            # Check if only llama logging changed - no need to restart API server then
                            api_settings_changed = (
                                self.config.get("api_host") != api_host
                                or self.config.get("api_port") != api_port
                                or self.config.get("server_ready_timeout")
                                != server_ready_timeout
                                or self.config.get("stream_watchdog_timeout")
                                != stream_watchdog_timeout
                                or self.config.get("log_api_server_to_file")
                                != log_api_to_file
                            )
                            if api_settings_changed:
                                restart = questionary.confirm(
                                    "Restart API server with new settings?"
                                ).ask()
                                if restart:
                                    self.api_server.stop()
                                    # Update the server object with new settings
                                    self.api_server.host = api_host
                                    self.api_server.port = api_port
                                    self.api_server.ready_timeout = server_ready_timeout
                                    self.api_server.stream_watchdog_timeout = (
                                        stream_watchdog_timeout
                                    )
                                    # No need to update logging flags here, start() reads from config
                                    self.api_server.start()
                                    console.print(
                                        "[green]API server restarted with new settings[/green]"
                                    )

                        # Restart Llama server if its logging setting changed and it's running
                        llama_log_setting_changed = (
                            self.config.get("log_llama_server_to_file")
                            != log_llama_to_file
                        )
                        if (
                            llama_log_setting_changed
                            and self.llama_server_process
                            and self.llama_server_process.poll() is None
                        ):
                            restart_llama = questionary.confirm(
                                "Restart Llama server with new logging setting?"
                            ).ask()
                            if restart_llama:
                                active_profile = self.config.get(
                                    "active_profile", "default"
                                )
                                self.restart_server_seamless(
                                    active_profile
                                )  # restart_server_seamless handles stop/start

                    except ValueError:
                        self._handle_error(
                            "updating settings",
                            "Port and Timeout values must be numbers",
                            True,
                        )
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
                    if self.server_monitor.start(active_profile):
                        console.print(
                            "[green]Llama Server started in background mode[/green]"
                        )
                    time.sleep(1)

                elif choice == "Stop Llama Server":
                    self.stop_server()
                    self.server_monitor.stop()
                    time.sleep(1)

                elif choice == "Restart Llama Server":
                    self.stop_server()
                    time.sleep(1)
                    if self.server_monitor.start(active_profile):
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

    def _interactive_edit_profile(self, profile_name: str) -> None:
        """Interactive profile editor"""
        try:
            while True:
                console.clear()
                rprint(Panel.fit(f"Editing Profile: {profile_name}", style="bold cyan"))

                # Show current profile settings
                self.show_profile(profile_name)

                # Get settings data
                profile_settings = self.config.get("profiles", {}).get(profile_name, {})
                all_settings = self.get_active_settings(profile_name)

                common_settings = [
                    "model",
                    "ctx_size",
                    "n_gpu_layers",
                    "threads",
                    "temp",
                    "top_k",
                    "top_p",
                    "batch_size",
                    "host",
                    "port",
                    "flash_attn",
                    "mlock",
                    "no_mmap",
                    "continuous_batching",
                    "inactivity_timeout",
                ]

                # Add any existing custom settings
                for key in profile_settings:
                    if key not in common_settings and key not in [
                        "profiles",
                        "active_profile",
                    ]:
                        common_settings.append(key)

                # Build menu options
                choices = [
                    f"Set {key} (current: {all_settings.get(key, 'Not set')})"
                    for key in common_settings
                ] + ["Add custom setting", "Clear a setting", "Back to main menu"]

                # Show menu
                choice = numbered_choice("Choose setting to edit:", choices)
                if not choice:
                    break

                # Handle menu selection
                if choice == "Back to main menu":
                    break

                elif choice == "Add custom setting":
                    key = questionary.text("Enter setting name:").ask()
                    if key and key.strip():
                        value = questionary.text(f"Enter value for {key}:").ask()
                        if value is not None:
                            self.set_setting(profile_name, key.strip(), value)

                elif choice == "Clear a setting":
                    if not profile_settings:
                        console.print(
                            "[yellow]No settings to clear in this profile.[/yellow]"
                        )
                        time.sleep(1)
                        continue

                    clear_choices = list(profile_settings.keys()) + [
                        "Back to settings menu"
                    ]
                    setting_to_clear = numbered_choice(
                        "Select setting to clear:", clear_choices
                    )
                    if setting_to_clear and setting_to_clear != "Back to settings menu":
                        self.clear_setting(profile_name, setting_to_clear)

                else:
                    # Extract setting name from choice
                    setting = choice.split(" ")[1]

                    # Handle model selection specially
                    if setting == "model":
                        models = find_gguf_models()
                        if models:
                            models.append("Enter custom path manually")
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
                        else:
                            console.print(
                                "[yellow]No GGUF models found. Enter path manually:[/yellow]"
                            )
                            value = questionary.text(
                                "Enter path to model file:",
                                default=str(all_settings.get(setting, "")),
                            ).ask()

                    # Handle different setting types appropriately
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
                        "inactivity_timeout",
                    ]:
                        value = questionary.text(
                            f"Enter value for {setting} (number):",
                            default=str(all_settings.get(setting, "")),
                        ).ask()
                    elif setting in [
                        "flash_attn",
                        "no_mmap",
                        "mlock",
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

                    # Apply the setting if we got a value
                    if value is not None:
                        self.set_setting(profile_name, setting, value)

                # Brief pause to show feedback
                time.sleep(0.5)

        except KeyboardInterrupt:
            pass  # Return to previous menu on Ctrl+C

    def restart_server_seamless(self, profile_name: str) -> bool:
        """Restart the server seamlessly in the background"""
        if (
            not self.llama_server_process
            or self.llama_server_process.poll() is not None
        ):
            return False

        # Stop the current server
        self.stop_server()

        # Start new server in background
        bg_queue = queue.Queue()
        success = self.start_server(
            profile_name, background=True, output_queue=bg_queue
        )

        # Reset activity state
        self.last_activity_timestamp = time.time()
        self.can_kill = False

        return success

    def start_inactivity_monitor(self):
        """Start the inactivity monitor thread if not already running"""
        # Reset activity timestamp
        self.last_activity_timestamp = time.time()

        # Only start if not already running
        if (
            self.inactivity_monitor_thread is None
            or not self.inactivity_monitor_thread.is_alive()
        ):
            self.inactivity_monitor_thread = threading.Thread(
                target=self._inactivity_monitor_loop
            )
            self.inactivity_monitor_thread.daemon = True
            self.inactivity_monitor_thread.start()

    def update_activity_timestamp(self):
        """Update the last activity timestamp to prevent server shutdown"""
        previous_state = self.can_kill
        self.last_activity_timestamp = time.time()
        self.can_kill = False

        # Log the state change for debugging
        # if previous_state != self.can_kill:
        #     pass # console.print(f"[dim]Inactivity monitor: Setting can_kill from {previous_state} to {self.can_kill}[/dim]")

    def _inactivity_monitor_loop(self):
        """Monitor inactivity and shut down server when idle for too long"""

        # Add a counter for periodic logging
        # log_counter = 0

        while True:
            try:
                # Exit if server is not running
                if (
                    self.llama_server_process is None
                    or self.llama_server_process.poll() is not None
                ):
                    break

                # Get configured timeout
                timeout_seconds = self.config.get(
                    "inactivity_timeout", self.default_config["inactivity_timeout"]
                )

                # Calculate time since last activity
                current_time = time.time()
                idle_time = current_time - self.last_activity_timestamp

                # Check if API server has active requests
                api_is_active = False
                api_request_count = 0
                api_stream_count = 0

                if hasattr(self, "api_server") and self.api_server.running:
                    with self.api_server.active_requests_lock:
                        api_request_count = self.api_server.active_requests
                    with self.api_server.active_streams_lock:
                        api_stream_count = self.api_server.active_streams

                    if api_request_count > 0 or api_stream_count > 0:
                        api_is_active = True
                        self.last_activity_timestamp = current_time

                # If we've been idle for longer than the timeout and we're allowed to kill
                if idle_time >= timeout_seconds and self.can_kill and not api_is_active:
                    console.print(
                        f"[yellow]Server idle for {int(idle_time)} seconds, shutting down...[/yellow]"
                    )
                    # console.print(f"[yellow]Debug info: can_kill={self.can_kill}, api_active={api_is_active}[/yellow]")
                    self.stop_server()
                    break

                # Force shutdown after 3x timeout period regardless of state
                elif idle_time >= (timeout_seconds * 3):
                    console.print(
                        f"[red]Server idle for {int(idle_time)} seconds (3x timeout)! Forcing shutdown...[/red]"
                    )
                    # console.print(f"[red]Debug info: can_kill={self.can_kill}, api_active={api_is_active}[/red]")
                    if hasattr(self, "api_server") and self.api_server.running:
                        self.api_server.reset_counters()
                    self.stop_server()
                    break

            except Exception as e:
                console.print(f"[red]Error in inactivity monitor: {e}[/red]")

            # Check every 10 seconds
            time.sleep(10)


class APIServer:
    def __init__(self, cli):
        self.cli = cli
        self.app = FastAPI()
        self.router = APIRouter()
        self.setup_routes()

        # Server state
        self.server = None
        self.running = False
        self.thread = None

        # Configuration
        self.host = self.cli.config.get("api_host", "0.0.0.0")
        self.port = self.cli.config.get("api_port", self.cli.default_config["api_port"])
        self.ready_timeout = self.cli.config.get(
            "server_ready_timeout", self.cli.default_config["server_ready_timeout"]
        )
        self.stream_watchdog_timeout = self.cli.config.get(
            "stream_watchdog_timeout",
            self.cli.default_config["stream_watchdog_timeout"],
        )
        self.timeout = 30  # Timeout for server requests
        self.log_file = None

        # Request tracking
        self.active_requests = 0
        self.active_streams = 0
        self.active_requests_lock = threading.Lock()
        self.active_streams_lock = threading.Lock()
        self.lock = threading.Lock()  # Lock for model switching

    def setup_routes(self):
        self.router.add_api_route("/v1/models", self.list_models, methods=["GET"])
        self.router.add_api_route(
            "/v1/models/{model}", self.model_info, methods=["GET"]
        )
        self.router.add_api_route(
            "/v1/chat/completions", self.create_chat_completion, methods=["POST"]
        )
        self.app.include_router(self.router)

        # Add client disconnect middleware
        @self.app.middleware("http")
        async def handle_client_disconnect(request, call_next):
            try:
                return await call_next(request)
            except (ConnectionResetError, BrokenPipeError, socket.error) as e:
                console.print(
                    f"[yellow]Client disconnected during request: {str(e)}[/yellow]"
                )
                # Return a special response for client disconnects
                # Note: This might not reach the client as they've already disconnected
                from fastapi.responses import Response

                return Response(status_code=499, content="Client Disconnected")

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

        # Check for unsupported combination early
        is_streaming = request.get("stream", False)
        has_tools = bool(request.get("tools"))
        
        # Llama.cpp does not support tool calls in streaming mode
        # You will just get: Cannot use tools with stream

        # We need to detect if we need to simulate streaming for tools
        simulate_streaming = is_streaming and has_tools
        
        if simulate_streaming:
            console.print("[yellow]Client requested streaming with tools - llama.cpp doesn't support this combination.[/yellow]")
            console.print("[yellow]Simulating streaming response by chunking a non-streaming response...[/yellow]")

        with self.lock:
            # --- Start: Add check and auto-start logic ---
            server_needs_start = False
            if (
                self.cli.llama_server_process is None
                or self.cli.llama_server_process.poll() is not None
            ):
                # console.print(
                #    f"[yellow]Llama server is stopped. Request for profile '{model}' requires starting it.[/yellow]"
                # )
                server_needs_start = True
            # --- End: Add check and auto-start logic ---

            current_profile = self.cli.config.get("active_profile", "default")
            profile_needs_switch = model != current_profile

            # Check if we can skip restart because requested profile is already running
            if not server_needs_start and not profile_needs_switch:
                console.print(
                    f"[green]Profile '{model}' is already running. Reusing existing server.[/green]"
                )
            # Only start or switch if necessary
            elif server_needs_start or profile_needs_switch:
                # Stop server only if it's running and needs switching
                if profile_needs_switch and not server_needs_start:
                    console.print(
                        f"[yellow]Switching from profile {current_profile} to {model}[/yellow]"
                    )
                # Removed extra parenthesis here

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

                # Update the active profile in config (do this whether starting or switching)
                self.cli.config["active_profile"] = model
                self.cli.save_config()  # Save immediately

                # Start server with the target profile
                console.print(
                    f"[yellow]Starting server with profile {model}...[/yellow]"
                )
                # Use a temporary queue, its output isn't critical here
                bg_queue = queue.Queue()
                if not self.cli.start_server(
                    model, background=True, output_queue=bg_queue
                ):
                    # Restore previous profile if start fails? Maybe too complex.
                    # For now, just report failure.
                    raise HTTPException(
                        500, f"Failed to start server with profile '{model}'"
                    )

                # Wait for server to be ready (important after starting)
                # Remove the argument here
                if not self.wait_for_server_ready():
                    raise HTTPException(
                        503, f"Server startup timed out for profile '{model}'"
                    )

                # Explicitly ensure the inactivity monitor is running
                # console.print("[yellow]Starting inactivity monitor for new model...[/yellow]")
                self.cli.start_inactivity_monitor()

            # --- End: Modify condition ---

            # At this point, the correct server should be running
            # Verify server is running before forwarding request (redundant check, but safe)
            if (
                not self.cli.llama_server_process
                or self.cli.llama_server_process.poll() is not None
            ):
                # This should ideally not happen if wait_for_server_ready passed
                console.print(
                    "[red]Error: Server process check failed after startup/switch attempt.[/red]"
                )
                raise HTTPException(
                    503, "Llama server is not running despite startup attempt"
                )

            # Update activity timestamp and disable can_kill flag
            self.cli.update_activity_timestamp()

            # Track active requests
            with self.active_requests_lock:
                self.active_requests += 1
                # console.print(f"[dim]API request start: active_requests incremented to {self.active_requests}[/dim]")

            is_streaming = request.get("stream", False)
            if is_streaming:
                with self.active_streams_lock:
                    self.active_streams += 1
                    # console.print(f"[dim]API stream start: active_streams incremented to {self.active_streams}[/dim]")

            # Create a copy of the request for llama.cpp
            llama_payload = request.copy()
            
            # If we need to simulate streaming, remove stream flag for llama.cpp request
            if simulate_streaming:
                llama_payload["stream"] = False

            # Forward request to the running llama-server
            server_url = f"http://{self.cli.config['host']}:{self.cli.config['port']}/v1/chat/completions"
            # console.print(f"[yellow]Forwarding request to {server_url}[/yellow]")
            # console.print(
            #     f"[yellow]Request payload: {json.dumps(request, indent=2)}[/yellow]"
            # )

            try:
                # Start timing for usage metrics
                start_time = time.time()
                load_start_time = start_time

                # For streaming requests, use stream=True
                is_streaming = request.get("stream", False)
                
                # Set appropriate socket timeout
                # Use a much longer timeout for simulated streaming because we need to wait for complete model response
                if simulate_streaming:
                    # For simulated streaming, we need a much longer timeout (5 minutes) to get the full response
                    request_timeout = 600  # 5 minutes for simulated streaming with tools
                    console.print(f"[yellow]Using extended timeout ({request_timeout}s) for simulated streaming with tools[/yellow]")
                elif is_streaming:
                    # For regular streaming, shorter timeout is fine as we process chunks incrementally
                    request_timeout = 10  # Slightly longer than before for streaming
                else:
                    # For normal non-streaming requests
                    request_timeout = self.timeout

                # Special handling for simulate_streaming case to catch timeouts
                if simulate_streaming:
                    try:
                        console.print(f"[yellow]Making non-streaming request to llama.cpp for simulated streaming...[/yellow]")
                        response = requests.post(
                            server_url,
                            json=llama_payload,
                            headers={
                                "Content-Type": "application/json",
                                "Accept": "application/json",
                            },
                            timeout=request_timeout,
                            stream=False,
                        )
                        # Record load time (time to first response)
                        load_duration = time.time() - load_start_time
                        console.print(f"[green]Received complete response from llama.cpp in {load_duration:.2f}s[/green]")
                        try:
                            raw_response_data_for_simulate_streaming = response.json()
                            console.print(f"[cyan][DIAGNOSTIC] Raw llama.cpp response (for simulate_streaming):\n{json.dumps(raw_response_data_for_simulate_streaming, indent=2)}[/cyan]")
                        except json.JSONDecodeError:
                            console.print(f"[red][DIAGNOSTIC] Failed to decode JSON from llama.cpp response. Status: {response.status_code}, Text: {response.text[:500]}...[/red]")
                            raw_response_data_for_simulate_streaming = None # Ensure it's defined

                    except requests.exceptions.Timeout:
                        console.print(f"[red]Request to llama.cpp timed out after {request_timeout}s when preparing for simulated streaming[/red]")
                        console.print(f"[yellow]The model is taking too long to generate a complete response with tool calls.[/yellow]")
                        console.print(f"[yellow]Recommend using non-streaming mode with tools for this model.[/yellow]")
                        raise HTTPException(
                            504,  # Gateway Timeout
                            "Request timed out. The model is taking too long to generate a complete response with tool calls. Try without streaming."
                        )
                    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
                        console.print(f"[red]Connection error when preparing for simulated streaming: {str(e)}[/red]")
                        self._check_and_kill_server_on_error()
                        raise HTTPException(503, f"Could not connect to llama-server: {str(e)}. Is it running?")
                else:
                    # Normal request handling for non-simulated cases
                    response = requests.post(
                        server_url,
                        json=llama_payload,
                        headers={
                            "Content-Type": "application/json",
                            "Accept": "text/event-stream"
                            if (is_streaming and not simulate_streaming)
                            else "application/json",
                        },
                        timeout=request_timeout, 
                        stream=(is_streaming and not simulate_streaming),
                    )
                    # Record load time (time to first response)
                    load_duration = time.time() - load_start_time

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

                # For simulated streaming responses (tools + stream), manually break up the response
                if simulate_streaming:
                    # Use the logged raw_response_data_for_simulate_streaming if available
                    response_json = raw_response_data_for_simulate_streaming if raw_response_data_for_simulate_streaming is not None else response.json()
                    total_duration = time.time() - start_time
                    
                    console.print(f"[green]Successfully generated complete tool call response in {round(total_duration, 2)} seconds[/green]")
                    console.print(f"[yellow]Now simulating streaming of tool calls to client...[/yellow]")
                    
                    async def simulated_stream_generator():
                        try:
                            # 1. Get the complete response from llama.cpp
                            if not response_json.get("choices"):
                                console.print("[red]Failed to get valid response for simulated streaming[/red]")
                                console.print(f"[yellow]Response from llama.cpp: {json.dumps(response_json, indent=2)}[/yellow]")
                                
                                # Try to return a meaningful error to the client
                                error_msg = "Invalid response from language model server"
                                if "error" in response_json:
                                    error_msg = response_json.get("error", {}).get("message", error_msg)
                                
                                # Send error as an event
                                yield f"data: {json.dumps({'error': error_msg})}\n\n".encode("utf-8")
                                yield b"data: [DONE]\n\n"
                                return
                            
                            # 2. Extract the message content
                            llama_choice = response_json.get("choices", [{}])[0]
                            llama_message = llama_choice.get("message", {})
                            llama_finish_reason = llama_choice.get("finish_reason")
                            
                            # Construct base response template - ensure consistent fields with OpenAI spec
                            base_chunk = {
                                "id": response_json.get("id", f"chatcmpl-{uuid.uuid4()}"),
                                "object": "chat.completion.chunk",
                                "created": response_json.get("created", int(time.time())),
                                "model": response_json.get("model", model),
                                "system_fingerprint": response_json.get("system_fingerprint", "fp_" + model.replace("-", "_").lower()),
                            }
                            
                            # 3. First yield the role
                            role_chunk = base_chunk.copy()
                            role_chunk["choices"] = [{
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None
                            }]
                            yield f"data: {json.dumps(role_chunk)}\n\n".encode("utf-8")
                            await asyncio.sleep(0.01)  # Small delay to simulate streaming
                            
                            # 4. If this is a tool call
                            if llama_message.get("tool_calls") or llama_finish_reason == "tool_calls":
                                llama_tool_calls = llama_message.get("tool_calls", [])
                                
                                # For each tool call, yield in chunks
                                for tc_index, tc in enumerate(llama_tool_calls):
                                    # Ensure we have function details and arguments
                                    function_details = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
                                    arguments = function_details.get("arguments", "")
                                    # Ensure arguments is a string
                                    if not isinstance(arguments, str):
                                        try:
                                            arguments = json.dumps(arguments)
                                        except TypeError:
                                            arguments = "{}"
                                    
                                    # Generate tool ID if missing
                                    tool_id = tc.get("id", f"call_{uuid.uuid4()}")
                                    
                                    # Initial tool call chunk (with name and ID)
                                    tool_init_chunk = base_chunk.copy()
                                    tool_init_chunk["choices"] = [{
                                        "index": 0,
                                        "delta": {
                                            "tool_calls": [{
                                                "index": tc_index,
                                                "id": tool_id,
                                                "type": "function",
                                                "function": {
                                                    "name": function_details.get("name", ""),
                                                    "arguments": ""
                                                }
                                            }]
                                        },
                                        "finish_reason": None
                                    }]
                                    yield f"data: {json.dumps(tool_init_chunk)}\n\n".encode("utf-8")
                                    await asyncio.sleep(0.03)  # Slightly longer delay for tool init
                                    
                                    # Chunk the arguments (roughly character by character or small groups)
                                    # This simulates how the arguments would be streamed
                                    chunk_size = 4  # Small chunks to simulate realistic streaming
                                    for i in range(0, len(arguments), chunk_size):
                                        arg_chunk = arguments[i:i+chunk_size]
                                        arg_delta_chunk = base_chunk.copy()
                                        arg_delta_chunk["choices"] = [{
                                            "index": 0,
                                            "delta": {
                                                "tool_calls": [{
                                                    "index": tc_index,
                                                    "function": {
                                                        "arguments": arg_chunk
                                                    }
                                                }]
                                            },
                                            "finish_reason": None
                                        }]
                                        yield f"data: {json.dumps(arg_delta_chunk)}\n\n".encode("utf-8")
                                        await asyncio.sleep(0.02)  # Small delay between argument chunks
                                    
                                    # Add a slightly longer delay after arguments complete
                                    # This helps clients properly finish processing the tool call arguments
                                    await asyncio.sleep(0.1)
                            
                            # 5. If this is regular content
                            elif llama_message.get("content"):
                                content = llama_message.get("content", "")
                                # Chunk the content (roughly character by character or small groups)
                                chunk_size = 4  # Small chunks to simulate streaming
                                for i in range(0, len(content), chunk_size):
                                    content_chunk = content[i:i+chunk_size]
                                    content_delta_chunk = base_chunk.copy()
                                    content_delta_chunk["choices"] = [{
                                        "index": 0,
                                        "delta": {"content": content_chunk},
                                        "finish_reason": None
                                    }]
                                    yield f"data: {json.dumps(content_delta_chunk)}\n\n".encode("utf-8")
                                    await asyncio.sleep(0.02)  # Small delay between content chunks
                            
                            # 6. Prepare usage info (moved to be populated before creating the final chunk)
                            info = {}
                            # Extract usage info from response_json["usage"]
                            if "usage" in response_json:
                                info.update({
                                    "prompt_tokens": response_json["usage"].get("prompt_tokens", 0),
                                    "completion_tokens": response_json["usage"].get("completion_tokens", 0),
                                    "total_tokens": response_json["usage"].get("total_tokens", 0),
                                })
                            
                            # Extract timing info if available from response_json["timings"]
                            if "timings" in response_json:
                                timings = response_json["timings"]
                                # Convert milliseconds to seconds where needed
                                info.update({
                                    "prompt_eval_count": timings.get("prompt_n", 0),
                                    "prompt_eval_duration": round(timings.get("prompt_ms", 0) / 1000, 2),
                                    "eval_count": timings.get("predicted_n", 0),
                                    "eval_duration": round(timings.get("predicted_ms", 0) / 1000, 2),
                                    "tokens_per_second": round(timings.get("predicted_per_second", 0), 2),
                                })
                                
                            # Add timing information we tracked
                            info.update({
                                "total_duration": round(total_duration, 2),
                                "load_duration": round(load_duration, 2),
                            })

                            # 7. Final chunk with finish_reason and combined usage info
                            final_chunk_payload = base_chunk.copy()
                            final_reason = "tool_calls" if llama_message.get("tool_calls") else llama_finish_reason
                            final_chunk_payload["choices"] = [{
                                "index": 0,
                                "delta": {}, # Delta is empty for the final chunk
                                "finish_reason": final_reason
                            }]
                            if info: # Add usage to the final chunk if info is populated
                                final_chunk_payload["usage"] = info
                            
                            yield f"data: {json.dumps(final_chunk_payload)}\n\n".encode("utf-8")
                            
                            # The separate yield for usage info and associated sleeps have been removed.
                            
                            # Finally, send [DONE]
                            yield b"data: [DONE]\n\n"
                            
                        except Exception as e:
                            console.print(f"[red]Error in simulated streaming: {type(e).__name__}: {e}[/red]")
                            import traceback
                            traceback.print_exc()
                            # Try to gracefully end the stream
                            yield b"data: [DONE]\n\n"
                        finally:
                            # Clean up counters
                            with self.active_streams_lock:
                                self.active_streams -= 1
                                if self.active_streams < 0:
                                    self.active_streams = 0
                    
                    # Create and return the streaming response
                    response_obj = StreamingResponse(
                        simulated_stream_generator(),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Accel-Buffering": "no"
                        }
                    )
                    
                    # Set a timer to ensure the server can be killed if streams get abandoned
                    def stream_watchdog():
                        # Wait a reasonable time for normal streaming completion
                        stream_timeout = (
                            self.stream_watchdog_timeout
                        )  # Use the configured timeout
                        stream_start = time.time()

                        try:
                            # Wait for either stream completion or timeout
                            while time.time() - stream_start < stream_timeout:
                                # Check if we've completed normally
                                with self.active_streams_lock:
                                    if self.active_streams == 0:
                                        # console.print("[dim]Stream completed normally.[/dim]")
                                        return
                                time.sleep(1)

                            # If we're here, watchdog timed out - handle abandoned stream
                            # console.print("[yellow]Stream watchdog: Stream may be abandoned.[/yellow]")

                            # Forcibly reset counters to ensure clean exit
                            with self.active_streams_lock:
                                if self.active_streams > 0:
                                    console.print(
                                        f"[yellow]Stream watchdog: Forcing reset of {self.active_streams} abandoned streams[/yellow]"
                                    )
                                    self.active_streams = 0

                            with self.active_requests_lock:
                                if self.active_requests > 0:
                                    # console.print(f"[yellow]Stream watchdog: Forcing reset of {self.active_requests} abandoned requests[/yellow]")
                                    self.active_requests = 0

                            # When it sets can_kill to true:
                            previous_state = self.cli.can_kill
                            self.cli.can_kill = True
                            # if previous_state != self.cli.can_kill:
                            #     console.print(f"[yellow]Stream watchdog: Setting can_kill from {previous_state} to {self.cli.can_kill}[/yellow]")

                        except Exception as e:
                            console.print(f"[red]Stream watchdog error: {e}[/red]")

                    # Start watchdog in background thread
                    watchdog = threading.Thread(target=stream_watchdog)
                    watchdog.daemon = True
                    watchdog.start()

                    # Return the streaming response
                    return response_obj

                # For regular streaming responses, use server-provided streaming
                elif is_streaming:
                    # Initialize tracking variables
                    has_usage = False
                    has_timings = False
                    final_chunk = None

                    async def stream_generator():
                        nonlocal has_usage, has_timings, final_chunk

                        try:
                            # Set a separate timeout for each iteration
                            iteration_timeout = (
                                time.time() + 10
                            )  # 10-second max per iteration

                            for line in response.iter_lines():
                                # Update activity timestamp on each chunk
                                self.cli.update_activity_timestamp()

                                # Check for timeout on each iteration
                                if time.time() > iteration_timeout:
                                    # console.print("[yellow]Stream iteration timeout - client may be slow or disconnected[/yellow]")

                                    # Increment the timeout count and check if we've had too many consecutive timeouts
                                    consecutive_timeouts = (
                                        getattr(self, "_timeout_count", 0) + 1
                                    )
                                    setattr(
                                        self, "_timeout_count", consecutive_timeouts
                                    )

                                    # If we've had 3 consecutive timeouts, assume the client is gone
                                    if consecutive_timeouts >= 3:
                                        console.print(
                                            "[red]Three consecutive timeouts - forcing connection cleanup[/red]"
                                        )
                                        # Force-reset counters and raise an exception to break the loop
                                        self._clean_exit_on_disconnect()
                                        # Raise an exception to break out of the stream
                                        raise ConnectionResetError(
                                            "Forced disconnect after multiple timeouts"
                                        )

                                    # Reset timeout for next iteration
                                    iteration_timeout = time.time() + 10

                                if line:
                                    # Store the data for analysis
                                    if line.startswith(b"data: "):
                                        json_str = line[6:].decode("utf-8")
                                        if json_str.strip() and json_str != "[DONE]":
                                            try:
                                                chunk_data = json.loads(json_str)
                                                # Keep track of the final non-[DONE] chunk
                                                final_chunk = chunk_data
                                                # Check if it contains usage or timings
                                                if "usage" in chunk_data:
                                                    has_usage = True
                                                if "timings" in chunk_data:
                                                    has_timings = True
                                            except Exception:
                                                pass  # Ignore parsing errors, just pass through

                                    # Send each line as-is
                                    yield line + b"\n"

                                    # Reset consecutive timeout counter when data is flowing
                                    setattr(self, "_timeout_count", 0)

                                    # Send a final message with usage statistics
                            total_duration = time.time() - start_time

                            # Create info using available data
                            info = {}

                            # From final chunk if available
                            if final_chunk:
                                if has_usage and "usage" in final_chunk:
                                    info.update(
                                        {
                                            "prompt_tokens": final_chunk["usage"].get(
                                                "prompt_tokens", 0
                                            ),
                                            "completion_tokens": final_chunk[
                                                "usage"
                                            ].get("completion_tokens", 0),
                                            "total_tokens": final_chunk["usage"].get(
                                                "total_tokens", 0
                                            ),
                                        }
                                    )

                                if has_timings and "timings" in final_chunk:
                                    timings = final_chunk["timings"]
                                    info.update(
                                        {
                                            "prompt_eval_count": timings.get(
                                                "prompt_n", 0
                                            ),
                                            "prompt_eval_duration": round(
                                                timings.get("prompt_ms", 0) / 1000, 2
                                            ),
                                            "eval_count": timings.get("predicted_n", 0),
                                            "eval_duration": round(
                                                timings.get("predicted_ms", 0) / 1000, 2
                                            ),
                                            "tokens_per_second": round(
                                                timings.get("predicted_per_second", 0),
                                                2,
                                            ),
                                        }
                                    )

                            # Add timing information we tracked
                            info.update(
                                {
                                    "total_duration": round(total_duration, 2),
                                    "load_duration": round(load_duration, 2),
                                }
                            )

                            # Send a final data message with usage info
                            # final_message = json.dumps({"usage": info})
                            # yield f"data: {final_message}\n\n".encode("utf-8")

                            # Send the standard [DONE] signal as the very last message
                            yield "data: [DONE]\n\n"

                            # console.print(
                            #     f"[green]Stream completed. Usage info: {json.dumps(info, indent=2)}[/green]"
                            # )

                        except (
                            BrokenPipeError,
                            ConnectionResetError,
                            socket.error,
                            requests.exceptions.ChunkedEncodingError,
                            requests.exceptions.ConnectionError,
                            TimeoutError,
                        ) as e:
                            # These are expected when client disconnects
                            console.print(
                                f"[yellow]Client disconnected during streaming: {str(e)}[/yellow]"
                            )
                            console.print("Terminating stream gracefully.")
                            # Clean up state to ensure timer restarts
                            self._clean_exit_on_disconnect()
                            # No need to yield anything here as the client is gone
                            return
                        finally:
                            # Decrement active streams counter
                            with self.active_streams_lock:
                                self.active_streams -= 1
                                # console.print(f"[dim]API stream end: active_streams decremented to {self.active_streams}[/dim]")

                            # Clean up state after request completes or client disconnects
                            with self.active_requests_lock:
                                self.active_requests -= 1
                                # console.print(f"[dim]API request end: active_requests decremented to {self.active_requests}[/dim]")

                            # Log completion or disconnection info
                            if not hasattr(response, "closed") or response.closed:
                                console.print(
                                    "[yellow]Client likely disconnected. Terminating stream gracefully.[/yellow]"
                                )

                    response_obj = StreamingResponse(
                        stream_generator(),
                        media_type="text/event-stream",
                        status_code=200,
                        # Set headers to disable caching for streaming response
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Accel-Buffering": "no",  # Disable buffering in Nginx
                        },
                    )

                    # Set a timer to ensure the server can be killed if streams get abandoned
                    def stream_watchdog():
                        # Wait a reasonable time for normal streaming completion
                        stream_timeout = (
                            self.stream_watchdog_timeout
                        )  # Use the configured timeout
                        stream_start = time.time()

                        try:
                            # Wait for either stream completion or timeout
                            while time.time() - stream_start < stream_timeout:
                                # Check if we've completed normally
                                with self.active_streams_lock:
                                    if self.active_streams == 0:
                                        # console.print("[dim]Stream completed normally.[/dim]")
                                        return
                                time.sleep(1)

                            # If we're here, watchdog timed out - handle abandoned stream
                            # console.print("[yellow]Stream watchdog: Stream may be abandoned.[/yellow]")

                            # Forcibly reset counters to ensure clean exit
                            with self.active_streams_lock:
                                if self.active_streams > 0:
                                    console.print(
                                        f"[yellow]Stream watchdog: Forcing reset of {self.active_streams} abandoned streams[/yellow]"
                                    )
                                    self.active_streams = 0

                            with self.active_requests_lock:
                                if self.active_requests > 0:
                                    # console.print(f"[yellow]Stream watchdog: Forcing reset of {self.active_requests} abandoned requests[/yellow]")
                                    self.active_requests = 0

                            # When it sets can_kill to true:
                            previous_state = self.cli.can_kill
                            self.cli.can_kill = True
                            # if previous_state != self.cli.can_kill:
                            #     console.print(f"[yellow]Stream watchdog: Setting can_kill from {previous_state} to {self.cli.can_kill}[/yellow]")

                        except Exception as e:
                            console.print(f"[red]Stream watchdog error: {e}[/red]")

                    # Start watchdog in background thread
                    watchdog = threading.Thread(target=stream_watchdog)
                    watchdog.daemon = True
                    watchdog.start()

                    # Return the streaming response
                    return response_obj

                # For non-streaming, use server-provided metrics
                response_json = response.json()
                total_duration = time.time() - start_time

                if response_json.get("choices") and len(response_json["choices"]) > 0:
                    llama_choice = response_json["choices"][0]
                    llama_message = llama_choice.get("message", {})
                    llama_finish_reason = llama_choice.get("finish_reason")

                    # Check for tool calls
                    if llama_message.get("tool_calls") or llama_finish_reason == "tool_calls":
                        openai_tool_calls = []
                        # Process each tool call
                        llama_tc_list = llama_message.get("tool_calls", [])
                        for i, tc in enumerate(llama_tc_list):
                            # Get function details
                            function_details = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
                            # Ensure arguments are a JSON string
                            arguments = function_details.get("arguments", "")
                            if not isinstance(arguments, str):
                                try:
                                    arguments = json.dumps(arguments)
                                except (TypeError, ValueError):
                                    arguments = "{}"

                            # Create OpenAI-compatible tool call
                            openai_tool_calls.append({
                                "id": tc.get("id", f"call_{uuid.uuid4()}"),
                                "type": "function",
                                "function": {
                                    "name": function_details.get("name", ""),
                                    "arguments": arguments
                                }
                            })

                        # Update the response format for OpenAI compatibility
                        response_json["choices"][0]["message"] = {
                            "role": llama_message.get("role", "assistant"),
                            "content": None,  # Must be null when tool_calls are present
                            "tool_calls": openai_tool_calls
                        }
                        response_json["choices"][0]["finish_reason"] = "tool_calls"

                # Create info field for client expected format
                info = {}

                # Copy existing usage data if available
                if "usage" in response_json:
                    info.update(
                        {
                            "prompt_tokens": response_json["usage"].get(
                                "prompt_tokens", 0
                            ),
                            "completion_tokens": response_json["usage"].get(
                                "completion_tokens", 0
                            ),
                            "total_tokens": response_json["usage"].get(
                                "total_tokens", 0
                            ),
                        }
                    )

                # Copy timings data if available
                if "timings" in response_json:
                    timings = response_json["timings"]
                    # Convert milliseconds to seconds where needed
                    info.update(
                        {
                            "prompt_eval_count": timings.get("prompt_n", 0),
                            "prompt_eval_duration": round(
                                timings.get("prompt_ms", 0) / 1000, 2
                            ),
                            "eval_count": timings.get("predicted_n", 0),
                            "eval_duration": round(
                                timings.get("predicted_ms", 0) / 1000, 2
                            ),
                            "tokens_per_second": round(
                                timings.get("predicted_per_second", 0), 2
                            ),
                        }
                    )

                # Add total timing information
                info.update(
                    {
                        "total_duration": round(total_duration, 2),
                        "load_duration": round(load_duration, 2),
                    }
                )

                # Update usage field with collected info
                if "usage" not in response_json:
                    response_json["usage"] = {}
                response_json["usage"].update(info)

                # console.print(
                #     f"[green]Request completed. Usage info: {json.dumps(info, indent=2)}[/green]"
                # )
                return response_json

            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
            ) as e:
                console.print(f"[red]Connection error: {str(e)}[/red]")
                raise HTTPException(
                    503, "Could not connect to llama-server. Is it running?"
                )
            except (
                requests.exceptions.RequestException,
                requests.exceptions.ChunkedEncodingError,
            ) as e:
                console.print(f"[red]Request error: {str(e)}[/red]")
                raise HTTPException(500, f"Error forwarding request: {str(e)}")
            except (
                BrokenPipeError,
                ConnectionResetError,
                socket.error,
                TimeoutError,
            ) as e:
                console.print(f"[yellow]Client connection error: {str(e)}[/yellow]")
                console.print("Client likely disconnected during the request.")
                # Use our utility to properly reset state
                self._clean_exit_on_disconnect(self.active_requests)
                # For API consistency, still raise an exception
                raise HTTPException(499, "Client Closed Request")
            except Exception as e:
                console.print(f"[red]Unexpected error: {str(e)}[/red]")
                raise HTTPException(500, f"Unexpected error: {str(e)}")
            finally:
                # Decrement active requests counter
                with self.active_requests_lock:
                    self.active_requests -= 1

                # Only allow server to be killed if all requests and streams are complete
                with self.active_streams_lock:
                    all_done = self.active_requests == 0 and self.active_streams == 0

                # If there are no active requests or streams, allow the server to be killed
                if all_done:
                    previous_state = self.cli.can_kill
                    self.cli.can_kill = True
                    # if previous_state != self.cli.can_kill:
                    #     console.print(f"[dim]API completed: Setting can_kill from {previous_state} to {self.cli.can_kill}[/dim]")

    def wait_for_server_ready(self):
        """Wait for llama-server to be ready to accept requests"""
        start = time.time()
        # console.print("[yellow]Waiting for server to be ready...[/yellow]")

        while time.time() - start < self.ready_timeout:
            try:
                # Check if the server process is still running
                if (
                    not self.cli.llama_server_process
                    or self.cli.llama_server_process.poll() is not None
                ):
                    console.print("[red]Server process has stopped[/red]")
                    return False

                # Try a simple completion to ensure model is fully loaded
                test_url = f"http://{self.cli.config['host']}:{self.cli.config['port']}/v1/chat/completions"
                test_payload = {
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 1,
                    "temperature": 0,
                    "stream": False,
                }

                test_res = requests.post(test_url, json=test_payload, timeout=5)
                if test_res.status_code == 200:
                    # console.print("[green]Server is ready[/green]")
                    return True
                else:
                    # console.print(f"[yellow]Server not ready yet. Response: {test_res.status_code}[/yellow]")
                    pass
            except requests.RequestException:
                # console.print("[yellow]Waiting for server...[/yellow]")
                pass

            time.sleep(1)

        console.print(
            f"[red]Server startup timed out after {self.ready_timeout} seconds[/red]"
        )
        return False

    def _clean_exit_on_disconnect(self, active_requests_before=None):
        """Reset request counters when client disconnects"""
        console.print(
            "[yellow]Client disconnected - cleaning up request state[/yellow]"
        )

        # Reset request counters
        with self.active_requests_lock:
            self.active_requests = 0

        with self.active_streams_lock:
            self.active_streams = 0

        # Enable server shutdown
        previous_state = self.cli.can_kill
        self.cli.can_kill = True
        # if previous_state != self.cli.can_kill:
        #     console.print(f"[dim]Clean exit: Setting can_kill from {previous_state} to {self.cli.can_kill}[/dim]")
        
    def _check_and_kill_server_on_error(self):
        """Checks server status after connection error and attempts kill if unresponsive."""
        if self.cli.llama_server_process and self.cli.llama_server_process.poll() is not None:
            console.print("[yellow]Server process already exited after connection error.[/yellow]")
            self.cli.llama_server_process = None  # Ensure state is clean
            return

        if self.cli.llama_server_process:
            console.print("[yellow]Attempting to terminate unresponsive server after connection error...[/yellow]")
            try:
                # Send SIGTERM first
                self.cli.llama_server_process.terminate()
                try:
                    self.cli.llama_server_process.wait(timeout=3)  # Short wait for terminate
                    console.print("[green]Server terminated gracefully after error.[/green]")
                except subprocess.TimeoutExpired:
                    console.print("[yellow]Server did not terminate gracefully, sending SIGKILL...[/yellow]")
                    self.cli.llama_server_process.kill()
                    self.cli.llama_server_process.wait(timeout=3)  # Wait for kill
                    console.print("[green]Server killed after error.[/green]")
            except Exception as e:
                console.print(f"[red]Error trying to terminate/kill server process: {e}[/red]")
            finally:
                self.cli.llama_server_process = None  # Mark as stopped

    def reset_counters(self):
        """Force reset all request and stream counters and enable server shutdown"""
        console.print("[yellow]Resetting request counters[/yellow]")

        with self.active_requests_lock:
            self.active_requests = 0

        with self.active_streams_lock:
            self.active_streams = 0

        # Enable server shutdown
        self.cli.can_kill = True

        console.print("[green]Counters reset, server shutdown enabled[/green]")

    def start(self):
        if not self.running:
            try:
                log_to_file = self.cli.config.get("log_api_server_to_file", False)
                log_filename = None
                log_config = None  # Default to Uvicorn's console logging

                # Create logs directory if it doesn't exist
                os.makedirs("logs", exist_ok=True)

                if log_to_file:
                    # Create log file with timestamp for the API server
                    log_filename = (
                        f"logs/api-server-{time.strftime('%Y%m%d-%H%M%S')}.log"
                    )
                    # Open the handle for potential writes in _run_server
                    try:
                        self.log_file = open(log_filename, "w")
                        console.print(
                            f"[green]API server output will be logged to: {log_filename}[/green]"
                        )
                    except Exception as e:
                        console.print(
                            f"[red]Error opening API log file {log_filename}: {e}[/red]"
                        )
                        self.log_file = None  # Ensure handle is None if open failed
                        log_to_file = False  # Fallback to console logging

                    # Configure Uvicorn logging only if file was opened successfully
                    if self.log_file:
                        import copy  # Import copy module

                        from uvicorn.config import LOGGING_CONFIG

                        log_config = copy.deepcopy(LOGGING_CONFIG)
                        log_config["handlers"]["api_file"] = {
                            "class": "logging.FileHandler",
                            "formatter": "default",
                            "filename": log_filename,
                        }
                        log_config["loggers"]["uvicorn"] = {
                            "handlers": ["api_file"],
                            "level": "INFO",
                            "propagate": False,
                        }
                        log_config["loggers"]["uvicorn.error"] = {
                            "handlers": ["api_file"],
                            "level": "INFO",
                            "propagate": False,
                        }
                        log_config["loggers"]["uvicorn.access"] = {
                            "handlers": ["api_file"],
                            "level": "INFO",
                            "propagate": False,
                        }
                else:
                    console.print("[green]API server logging to console.[/green]")
                    self.log_file = (
                        None  # Ensure log_file is None when not logging to file
                    )

                self.running = True
                self.server = uvicorn.Server(
                    config=uvicorn.Config(
                        app=self.app,
                        host=self.host,
                        port=self.port,
                        log_level="info",
                        access_log=True,
                        log_config=log_config,  # Pass None to use Uvicorn defaults (console)
                        timeout_keep_alive=5,
                        timeout_graceful_shutdown=10,
                    )
                )

                self.thread = Thread(target=self._run_server)
                self.thread.start()
                console.print(
                    f"[green]API server started on {self.host}:{self.port}[/green]"
                )
            except Exception as e:
                self.cli._handle_error("starting API server", e, True)
                self.running = False
                # Ensure log file handle is closed on error
                if hasattr(self, "log_file") and self.log_file:
                    try:
                        self.log_file.close()
                    except:
                        pass
                    self.log_file = None

    def _run_server(self):
        log_to_file = self.cli.config.get("log_api_server_to_file", False)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        target_stream = None
        devnull_handle = None  # Keep track if we open devnull

        if log_to_file:
            # Try to use the configured log file
            if hasattr(self, "log_file") and self.log_file and not self.log_file.closed:
                target_stream = self.log_file
                # console.print("[dim]API Server: Redirecting output to log file.[/dim]", file=original_stderr) # Debug
            else:
                # Fallback if log file wasn't opened correctly
                print(
                    "[red]Error: API log file handle not available. Suppressing API output.[/red]",
                    file=original_stderr,
                )
                log_to_file = False  # Force fallback to devnull

        # If file logging is disabled (either by config or fallback)
        if not log_to_file:
            try:
                devnull_handle = open(os.devnull, "w")
                target_stream = devnull_handle
                # console.print("[dim]API Server: Redirecting output to os.devnull.[/dim]", file=original_stderr) # Debug
            except OSError as e:
                print(
                    f"[red]Error opening os.devnull: {e}. API output might appear on console.[/red]",
                    file=original_stderr,
                )
                target_stream = None  # Avoid redirection if devnull fails

        try:
            redirected = False
            if target_stream:
                sys.stdout = target_stream
                sys.stderr = target_stream
                redirected = True

            # Uvicorn server runs here
            self.server.run()

        except OSError as e:
            # Log errors to original stderr before restoring, as current stderr goes to file
            print(f"[Error running API server OS Error]: {e}", file=original_stderr)
            if "address already in use" in str(e):
                self.cli._handle_error(
                    "starting API server", f"Port {self.port} already in use!", True
                )
            else:
                self.cli._handle_error("running API server", e, True)
        except (BrokenPipeError, ConnectionResetError, socket.error) as e:
            # Log connection errors to original stderr
            print(
                f"[Client connection error in server]: {str(e)}", file=original_stderr
            )
            print(
                "This is likely due to a client disconnection and is not a serious issue.",
                file=original_stderr,
            )
            # Attempt to continue if possible
            if self.running:
                print(
                    "[Attempting to continue server operation...]", file=original_stderr
                )
        except Exception as e:
            # Log other exceptions to original stderr
            print(f"[Error running API server Exception]: {e}", file=original_stderr)
            self.cli._handle_error("running API server", e, True)
        finally:
            if redirected:  # Only restore if we actually redirected
                sys.stdout = original_stdout
                sys.stderr = original_stderr

            self.running = False  # This should be outside the conditional block

            # Close the log file handle *only* if it was the target and it's the instance's handle
            if (
                target_stream is self.log_file
                and hasattr(self, "log_file")
                and self.log_file
            ):
                try:
                    self.log_file.close()
                    self.log_file = None  # Clear handle after closing
                except Exception as e:
                    print(
                        f"[red]Error closing API log file: {e}[/red]",
                        file=original_stderr,
                    )

            # Close the devnull handle if it was opened
            if devnull_handle:
                try:
                    devnull_handle.close()
                except Exception:
                    pass  # Ignore errors closing devnull

    def stop(self):
        if self.running:
            self.running = False
            if self.server:  # Check if server object exists
                self.server.should_exit = True
            if self.thread:  # Check if thread exists
                self.thread.join(timeout=10)  # Add timeout


class ServerMonitor:
    """Simple monitor for capturing server output to a queue"""

    def __init__(self, cli):
        self.cli = cli
        self.output_queue = queue.Queue()
        self.running = False
        self.monitor_thread = None
        self.server_output = []
        self.max_log_lines = 20  # Maximum number of log lines to display
        self.current_log_file = None

    def start(self, profile_name: str) -> bool:
        """Start server with the specified profile and capture output"""
        settings = self.cli.get_active_settings(profile_name)
        log_to_file = settings.get("log_llama_server_to_file", False)

        # Start the server
        success = self.cli.start_server(
            profile_name, background=True, output_queue=self.output_queue
        )
        if not success:
            return False

        self.running = True
        self.server_output = []  # Clear previous output

        if log_to_file:
            # Start the monitoring thread only if logging to file
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            self.server_output.append("[dim]Monitoring server log file...[/dim]")
        else:
            # Indicate that monitoring is not active
            self.server_output.append(
                "[yellow]Logging to file disabled - output not monitored.[/yellow]"
            )

        return True

    def stop(self) -> None:
        """Stop monitoring (but not the server)"""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)

    def _monitor_loop(self) -> None:
        """Background thread to monitor server output"""
        while self.running and self.cli.llama_server_process:
            try:
                # Check if server is still running
                if self.cli.llama_server_process.poll() is not None:
                    self.running = False
                    self.server_output.append(
                        f"[yellow]Server stopped with code {self.cli.llama_server_process.returncode}[/yellow]"
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

    def get_output(self) -> List[str]:
        """Get the current server output logs"""
        return self.server_output.copy()


# Create singleton instance
cli = LlamaServerCLI()


@app.command("list-profiles")
def list_profiles():
    """List all available profiles"""
    cli.list_profiles()


@app.command("show-profile")
def show_profile(name: str = typer.Argument(..., help="Profile name")):
    """Show details of a specific profile"""
    cli.show_profile(name)


@app.command("create-profile")
def create_profile(name: str = typer.Argument(..., help="Profile name")):
    """Create a new profile"""
    cli.create_profile(name)


@app.command("delete-profile")
def delete_profile(name: str = typer.Argument(..., help="Profile name")):
    """Delete an existing profile"""
    cli.delete_profile(name)


@app.command("use-profile")
def set_active_profile(name: str = typer.Argument(..., help="Profile name")):
    """Set active profile"""
    cli.set_active_profile(name)


@app.command("set-setting")
def set_setting(
    profile: str = typer.Argument(..., help="Profile name"),
    key: str = typer.Argument(..., help="Setting key"),
    value: str = typer.Argument(..., help="Setting value"),
):
    """Set a profile setting"""
    cli.set_setting(profile, key, value)


@app.command("clear-setting")
def clear_setting(
    profile: str = typer.Argument(..., help="Profile name"),
    key: str = typer.Argument(..., help="Setting key"),
):
    """Clear a profile setting"""
    cli.clear_setting(profile, key)


@app.command("start-server")
def start_server(
    profile: Optional[str] = typer.Option(None, help="Profile name to use"),
):
    """Start the server with specified profile (or active profile if none specified)"""
    cli.start_server(profile)


@app.command("stop-server")
def stop_server():
    """Stop the running server"""
    cli.stop_server()


@app.command("config")
def interactive_config():
    """Start interactive configuration mode"""
    cli.interactive_config()


@app.command("start-api")
def start_api_server():
    """Start the OpenAI-compatible API server"""
    cli.api_server.start()
    # Start the inactivity monitor if server starts successfully and it's not already running
    if cli.api_server.running:
        # Only start the inactivity monitor if llama-server is running
        if cli.llama_server_process and cli.llama_server_process.poll() is None:
            cli.start_inactivity_monitor()

    console.print("[yellow]Press Ctrl+C to stop the API server[/yellow]")
    try:
        # Keep the main thread alive
        while cli.api_server.running:
            time.sleep(1)
    except KeyboardInterrupt:
        cli.api_server.stop()
        console.print("\n[green]API server stopped[/green]")


@app.command("stop-api")
def stop_api_server():
    """Stop the OpenAI-compatible API server"""
    if cli.api_server.running:
        cli.api_server.stop()
        console.print("[green]API server stopped[/green]")
    else:
        console.print("[yellow]API server is not running[/yellow]")


@app.command("reset-api")
def reset_api_counters():
    """Reset all API request counters and enable server shutdown"""
    if cli.api_server.running:
        cli.api_server.reset_counters()
        console.print("[green]API counters reset, server shutdown enabled[/green]")
    else:
        console.print("[yellow]API server is not running[/yellow]")


@app.command("restart-monitor")
def restart_inactivity_monitor():
    """Restart the inactivity monitor for the currently running server"""
    # Check if server is running but monitor is not
    server_running = False
    monitor_running = False

    # Check if server is running via system processes
    try:
        if os.name == "posix":  # Linux/Mac
            check_result = subprocess.run(
                # Use the same more specific pattern as in _terminate_orphaned_processes
                ["pgrep", "-f", r"^\./llama-server(\s|$)"],
                capture_output=True,
                text=True,
            )
            if check_result.returncode == 0 and check_result.stdout.strip():
                server_running = True
        elif os.name == "nt":  # Windows
            check_result = subprocess.run(
                # Be more specific with Windows process name
                ["tasklist", "/fi", "imagename eq llama-server.exe"],
                capture_output=True,
                text=True,
            )
            if "llama-server.exe" in check_result.stdout:
                server_running = True
    except:
        pass

    # Check if monitor thread is running
    monitor_running = (
        cli.inactivity_monitor_thread is not None
        and cli.inactivity_monitor_thread.is_alive()
    )

    if not server_running:
        console.print(
            "[yellow]No llama-server process found running. Start the server first.[/yellow]"
        )
        return

    if monitor_running:
        console.print("[yellow]Inactivity monitor is already running.[/yellow]")
        return

    # Server is running but monitor is not - restart it
    console.print("[green]Restarting inactivity monitor for running server...[/green]")

    # Reset the process reference if needed by finding the actual PID
    if cli.llama_server_process is None or cli.llama_server_process.poll() is not None:
        try:
            # Find PID of running llama-server with the safer pattern
            if os.name == "posix":  # Linux/Mac
                check_result = subprocess.run(
                    ["pgrep", "-f", r"^\./llama-server(\s|$)"],
                    capture_output=True,
                    text=True,
                )
                if check_result.returncode == 0 and check_result.stdout.strip():
                    # Just take the first PID if multiple are found
                    pid = int(check_result.stdout.strip().split("\n")[0])
                    # Create a new Process object from the PID
                    cli.llama_server_process = subprocess.Popen(
                        ["true"],  # Dummy command that doesn't do anything
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                    # Replace the pid with the one we found
                    cli.llama_server_process.pid = pid
                    console.print(
                        f"[green]Reconnected to llama-server process (PID: {pid})[/green]"
                    )

            # Windows version is more complex, skip for now
        except Exception as e:
            console.print(f"[yellow]Could not reconnect to process: {e}[/yellow]")
            console.print(
                "[yellow]Monitor will still restart but might not detect the server correctly.[/yellow]"
            )

    # Reset counters in API server if it's running
    if cli.api_server.running:
        cli.api_server.reset_counters()

    # Start monitor
    cli.start_inactivity_monitor()
    console.print("[green]Inactivity monitor restarted successfully[/green]")


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

        signal.signal(signal.SIGINT, cleanup)
        signal.signal(signal.SIGTERM, cleanup)

        # If no subcommand is provided, launch interactive mode
        if ctx.invoked_subcommand is None:
            cli.interactive_config()
    finally:
        # Always ensure servers and monitor are stopped when application exits
        cleanup()  # This already calls the modified cleanup


# Function to find GGUF models in gguf directory and subdirectories
def find_gguf_models() -> List[str]:
    """Find all GGUF model files in the current directory and gguf subdirectory"""
    models = []

    # Create gguf directory if it doesn't exist
    if not os.path.exists("gguf"):
        os.makedirs("gguf")
        console.print("[yellow]Created gguf directory for models[/yellow]")

    # Search in current directory
    models.extend(glob.glob("*.gguf"))

    # Search in gguf directory and its subdirectories
    models.extend(glob.glob("gguf/**/*.gguf", recursive=True))

    # Sort models alphabetically
    models.sort()

    return models


if __name__ == "__main__":
    app()
