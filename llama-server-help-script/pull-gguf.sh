#!/bin/bash

set -e

# Constants
USER_AGENT="Huggingface-GGUF-Downloader/1.0"
DATE=$(date +%Y%m%d)
DEFAULT_OUTPUT="gguf/"
ERROR_LOG="$DEFAULT_OUTPUT/error.log"
HEAD_URL=""
OUT_URL=""
FILENAME=""
HF_TOKEN=""

# Parse collected input arguments to determine the download URL
parse_input() {
    local num_args=$#
    local arg1="${1:-}"
    local arg2="${2:-}"

    # Clear previous results
    OUT_URL=""
    FILENAME=""
    HEAD_URL=""

    if [ "$num_args" -eq 0 ]; then
        echo "ERROR: No input arguments provided." >&2
        return 1
    fi

    # Case 1: Direct full URL to .gguf file
    if [[ "$arg1" =~ ^https://.*\.gguf$ ]] && [ "$num_args" -eq 1 ]; then
        OUT_URL="$arg1"
        echo "Parsed Input (Direct URL): $OUT_URL"
    # Case 2: Repo URL (ending with /) + filename
    elif [[ "$arg1" =~ ^https://.*/$ ]] && [ "$num_args" -eq 2 ]; then
        local base_url="${arg1%/}" # Remove trailing slash
        OUT_URL="$base_url/resolve/main/$arg2"
        echo "Parsed Input (Repo URL/ + Filename): $OUT_URL"
    # Case 3: Repo URL (not ending with /) + filename
    elif [[ "$arg1" =~ ^https:// ]] && [ "$num_args" -eq 2 ]; then
        # Check if it looks like a repo URL (e.g. https://huggingface.co/user/repo)
        if [[ "$arg1" =~ ^https://huggingface\.co/[^/]+/[^/]+/?$ ]]; then
             local base_url="${arg1%/}" # Remove trailing slash if present
             OUT_URL="$base_url/resolve/main/$arg2"
             echo "Parsed Input (Repo URL + Filename): $OUT_URL"
        else
             echo "ERROR: First argument looks like URL, but not a recognized repo URL format, with second argument: $arg1" >&2
             return 1
        fi
    # Case 4: Single argument like repo/name/filename.gguf
    elif [[ "$arg1" =~ / ]] && [[ "$arg1" =~ \.gguf$ ]] && [ "$num_args" -eq 1 ]; then
         local filename
         filename=$(basename "$arg1")
         local repo
         repo=$(dirname "$arg1")
         if [[ "$repo" == "." ]]; then
            # Handle case where input is just "filename.gguf" (no slashes)
            echo "ERROR: Single argument has .gguf suffix, but no repository path specified: $arg1" >&2
            return 1
         fi
         # Remove leading ./ if dirname added it
         repo=${repo#./}
         OUT_URL="https://huggingface.co/$repo/resolve/main/$filename"
         echo "Parsed Input (Single Arg Repo/File): $OUT_URL"
    # Case 5: repo/name + filename (space separated)
    elif [[ "$arg1" =~ / ]] && [[ ! "$arg1" =~ ^https?:// ]] && [ "$num_args" -eq 2 ]; then
         # Assume arg1 is repo, arg2 is filename
         local repo="${arg1%/}" # Remove trailing slash if present
         local filename="$arg2"
         OUT_URL="https://huggingface.co/$repo/resolve/main/$filename"
         echo "Parsed Input (Repo Name + Filename): $OUT_URL"
    else
        # Unrecognized format
        echo "ERROR: Cannot parse input format:" >&2
        printf "  Args(%d): %s\n" $# "$*" >&2
        return 1 # Indicate failure
    fi

    # If OUT_URL was successfully set, derive FILENAME and HEAD_URL
    if [ -n "$OUT_URL" ]; then
        FILENAME=$(basename "$OUT_URL")
        # Handle potential query parameters in URL before basename
        FILENAME=${FILENAME%%\?*} # Remove query string if any
        HEAD_URL="http${OUT_URL:4}" # Swap https for http for HEAD request
        # Basic validation that derived filename looks like GGUF
        if [[ ! "$FILENAME" =~ \.gguf$ ]]; then
             echo "WARNING: Derived filename '$FILENAME' does not end with .gguf from URL '$OUT_URL'" >&2
        fi
        return 0 # Indicate success
    else
         # This path should ideally not be reached if logic above is complete
         echo "ERROR: Internal parsing error, OUT_URL could not be determined." >&2
         return 1 # Indicate failure
    fi
}

# Check if file exists on Huggingface
check_file_exists() {
    curl -s --head --location "$HEAD_URL" | grep -i "HTTP/2 200" > /dev/null 2>&1
    return $?
}

# Handle authentication headers if token provided
get_auth_headers() {
    if [ -n "$HF_TOKEN" ]; then
        echo "--header 'Authorization: Bearer $HF_TOKEN'"
    fi
}

# Main download function with resume support
download_file() {
    local output_file="$DEFAULT_OUTPUT/$FILENAME"

    # Resume flag if file exists
    local resume_flag=""
    if [ -f "$output_file" ]; then
        resume_flag="-C -"
    fi

    # Perform download with progress bar
    curl $resume_flag \
        --location-trusted \
        --header "User-Agent: $USER_AGENT" \
        $(get_auth_headers) \
        --progress-bar \
        --output "$output_file" \
        "$OUT_URL"

    # Verify download
    if [ $? -eq 0 ]; then
        echo "SUCCESS: File saved to $output_file"
    else
        echo "$(date +'%Y-%m-%d %T') ERROR: Download failed" >> "$ERROR_LOG"
        return 1
    fi
}

# Parse command line arguments
input_args=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -o|--output) OUTPUT_DIR="$2"; shift 2 ;;
        --token) HF_TOKEN="$2"; shift 2 ;;
        --help) show_help; exit 0 ;;
        *) input_args+=("$1"); shift ;; # Collect non-option args
    esac
done

# Show help message
show_help() {
    echo "Usage: $0 [OPTIONS] <input>"
    echo "Input can be:"
    echo "  Direct URL to .gguf file"
    echo "  Repo URL and filename separated by space/slash"
    echo "  Repo name and filename separated by space/slash"
    echo ""
    echo "Options:"
    echo "  -o,--output   Specify output directory (Note: Files always save to gguf/)"
    echo "  --token       HuggingFace API token"
    echo "  --help        Show this help"
}

# Main execution
main() {
    current_dir=$(basename "$(pwd)")
    if [ "$current_dir" = "llama-server-help-script" ]; then
        cd ..
    fi

    # Call parse_input with collected args
    if ! parse_input "${input_args[@]}"; then
        # parse_input already printed an error message
        show_help
        exit 1
    fi

    # Create the default output directory if it doesn't exist
    mkdir -p "$DEFAULT_OUTPUT" || {
        echo "ERROR: Could not create output directory '$DEFAULT_OUTPUT'" >&2
        exit 1 # Exit if directory cannot be created
    }

    # Check file exists on server
    if check_file_exists; then
        echo "Found file: $FILENAME"
        download_file
    else
        echo "ERROR: File not found at $OUT_URL" >&2
    fi
}


# Run the script
main "$@"
