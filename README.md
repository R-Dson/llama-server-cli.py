# Llama Server CLI

A simple, user-friendly CLI tool for managing, and running, llama-server with multiple configuration profiles and providing OpenAI-compatible API access.

## Installation

1. Place the `llama-server-cli.py` file in the same directory as your `llama-server` executable.

2. Install the required Python packages:

```bash
pip install typer rich questionary prompt_toolkit fastapi uvicorn
```

3. Create a `gguf` folder for your models if you don't already have one:

```bash
mkdir -p gguf
```


## Recommended Directory Structure

```
.
├── llama-server     # The llama-server executable
├── llama-server-cli.py     # The CLI tool
├── config.json      # Auto-generated config file (don't edit directly)
└── gguf/            # Folder containing your GGUF model files
    ├── model1.gguf
    └── model2.gguf
```

## Usage

### Interactive Mode (Recommended)

Simply run the tool without arguments to enter interactive mode:

```bash
python llama-server-cli.py
```

In interactive mode, you can:
- Switch between profiles
- Edit settings
- Create and delete profiles
- Start the server in the background
- Change settings with instant auto-restart
- Select models from a list of available GGUF files
- Configure and manage the OpenAI-compatible API server

### Command-Line Arguments (Advanced)

You can also use command-line arguments for specific operations:

```bash
# Create a profile
python llama-server-cli.py profile create my_profile

# Set model path
python llama-server-cli.py profile set my_profile model ./gguf/my_model.gguf

# See all available profiles
python llama-server-cli.py profile list

# Start server with a specific profile
python llama-server-cli.py server start --profile my_profile

# Start OpenAI-compatible API server
python llama-server-cli.py api start

# Stop API server (if needed)
python llama-server-cli.py api stop
```

## Quick Start

1. Create a profile for your model:

```bash
python llama-server-cli.py
```

2. From the interactive menu:
   - Choose "Create new profile"
   - Enter a name for your profile
   - Select "Edit current profile settings"
   - Choose "Set model" and select your model from the list
   - Adjust other settings as needed
   - Start the server with your profile

## OpenAI API Compatibility

The tool includes an OpenAI-compatible API server that allows you to use your local llama-server with applications that support the OpenAI API format:

1. **API Configuration**: Configure the API server host and port through the interactive menu or config.json
2. **Automatic Profile Switching**: The API server automatically switches between profiles based on the requested model
3. **Chat Completions**: Supports the `/v1/chat/completions` endpoint with streaming capabilities
4. **Models Endpoint**: Exposes profiles as models via the `/v1/models` endpoint

## Background Server Operation

The CLI tool allows you to:

1. **Start the server in the background** and immediately return to the main menu
2. **Continue using the CLI** while the server is running
3. **See server status** in the main menu
4. **Restart or stop the server** directly from the main menu
5. **Change settings on-the-fly** - any settings change automatically restarts the server seamlessly

## Real-time Configuration Changes

1. **Automatic server restart** - when you change any setting, the server automatically restarts in the background
2. **Instant profile switching** - change profiles and the server immediately uses the new configuration
3. **Seamless experience** - all restarts happen invisibly without disrupting your workflow
4. **Cleanup on exit** - the server is always properly terminated when you exit the CLI

## Navigation

The interface is designed to be intuitive and easy to navigate:

1. **Arrow keys** - Navigate through menu options
2. **Number keys (1-9)** - Quickly select menu options by number
3. **Enter key** - Select the highlighted option

## Model Selection

When setting the model path, the tool will:
1. Automatically scan the `gguf` directory for model files
2. Present a list of all available models
3. Allow you to select a model from the list
4. Also offer an option to enter a custom path manually

This makes it easy to switch between models without having to remember exact file paths.

## Configuration Options

Default settings:
- `ctx_size`: 2048 (context size)
- `threads`: -1 (use all available threads)
- `temp`: 0.8 (temperature)
- `flash_attn`: True (enables Flash Attention)
- `continuous_batching`: True (enables continuous batching)
- `api_host`: 0.0.0.0 (API server listen address)
- `api_port`: 8000 (API server port)

You can customize these and many other options through the interactive menu.

## Key Features

- Multiple configuration profiles
- Interactive menus with both arrow and number key navigation
- Background server operation
- Automatic model discovery and selection
- Instant, seamless configuration changes
- Config file managed automatically
- Automatic server process management
- OpenAI-compatible API interface
- Simple single-file deployment

## Troubleshooting

- If the server fails to start, check the error messages for required settings
- Ensure your model file path is correct
- Verify that you have sufficient RAM/VRAM for your model size
- For API issues, check that both the llama-server and API server are running
- Port conflicts can be resolved by changing the port in the API settings
