# Llama Server CLI

A simple, user-friendly CLI tool for managing, and running, [llama.cpp](https://github.com/ggml-org/llama.cpp)'s [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/examples/server) with multiple configuration profiles and providing OpenAI-compatible API access.

## Key Features
*   **Simple Single-File Deployment** - easy to deploy and get started.
*   **Interactive Menus** - navigate options intuitively using arrow or number keys.
*   **Multiple Configuration Profiles** - easily manage and switch between different setups.
*   **Background Server Operation** - the server runs unobtrusively in the background.
*   **Automatic Model Discovery** - automatically finds and selects compatible models.
*   **Instant Config Changes** - apply settings immediately via profile switching or direct edits.
*   **Automated Config File** - the configuration file is managed automatically for you.
*   **Automatic Server Management** - handles seamless restarts on changes and cleanup on exit.
*   **OpenAI-Compatible API** - provides an interface compatible with OpenAI standards. (Still some TODO)

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

<details>

<summary>Command-Line Arguments (Advanced)</summary>

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
</details>

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

1. **Start the server in the background**
2. **Continue using the CLI** while the server is running
3. **See server status** in the main menu
4. **Restart or stop the server** directly from the main menu
5. **Change settings on-the-fly** - any settings change automatically restarts the server seamlessly

## Model Selection

When setting the model path, the tool will:
1. Automatically scan the `gguf` directory for model files
2. Present a list of all available models
3. Allow you to select a model from the list
4. Also offer an option to enter a custom path manually

## TODO
- Add better logging for llama-server
- Add better logging for llama-server-api

## Troubleshooting

- Use the keyboard arrows to re-draw the interface if the interface is not displaying correctly.
- If the server fails to start, check the error messages for required settings
- Ensure your model file path is correct
- Verify that you have sufficient RAM/VRAM for your model size
- For API issues, check that both the llama-server and API server are running
- Port conflicts can be resolved by changing the port in the API settings
