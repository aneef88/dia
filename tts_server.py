# tts_server.py (Server-Side Prompt Handling - Final Review)

import sys
import io
import os
import random
import traceback
from pathlib import Path
import argparse
import logging # Use standard logging module

# Third-party imports
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel # Using base model for condensed style
from typing import Optional

# --- Logging Setup ---
# Configure logging for better output control than print statements
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Configuration Defaults ---
# These can be overridden by environment variables or CLI arguments
# Environment variables take precedence over defaults, CLI args take precedence over environment variables
DEFAULT_PROMPTS_DIR_STR = os.getenv("NARI_PROMPTS_DIR", "prompts")
DEFAULT_PROMPT_ID = os.getenv("NARI_DEFAULT_PROMPT", "default_voice")
DEFAULT_MODEL_NAME = os.getenv("NARI_MODEL_NAME", "nari-labs/Dia-1.6B")
DEFAULT_HOST = os.getenv("NARI_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("NARI_PORT", "8210"))
DEFAULT_USE_TORCH_COMPILE_STR = os.getenv("NARI_USE_COMPILE", "true").lower()

# --- Effective Configuration (Set in main block) ---
# These will hold the final configuration values used by the application after parsing args/env
EFFECTIVE_PROMPTS_DIR: Path = Path(DEFAULT_PROMPTS_DIR_STR)
EFFECTIVE_DEFAULT_PROMPT_ID: str = DEFAULT_PROMPT_ID
EFFECTIVE_MODEL_NAME: str = DEFAULT_MODEL_NAME
EFFECTIVE_HOST: str = DEFAULT_HOST
EFFECTIVE_PORT: int = DEFAULT_PORT
EFFECTIVE_USE_TORCH_COMPILE: bool = DEFAULT_USE_TORCH_COMPILE_STR == "true"

# Global variable to hold the loaded model and its properties
tts_model = None
model_sample_rate = 44100 # Default, might be updated if model provides it
compute_device = torch.device("cpu") # Default, updated during model load

# --- Pydantic Model for API Request ---
class TTSRequest(BaseModel):
    """Defines the structure for TTS API requests."""
    text: str                       # Text to be synthesized into speech.
    prompt_id: Optional[str] = None # Identifier (base filename) for the server-side prompt (e.g., 'voice1'). If None, the server's default prompt is used.
    max_new_tokens: int = 3072      # Maximum number of new audio tokens to generate. Should be > 0.
    cfg_scale: float = 3.0          # Guidance scale for classifier-free guidance. Should be >= 1.0.
    temperature: float = 1.3        # Sampling temperature. Higher values increase randomness. Should be > 0.0.
    top_p: float = 0.95             # Nucleus sampling probability threshold. Should be > 0.0 and <= 1.0.
    cfg_filter_top_k: int = 30      # Filters guidance logits to top K during CFG. Should be >= 1.
    speed_factor: float = 1.0       # Factor to speed up (>1.0) or slow down (<1.0) the audio. Should be > 0.1.
    seed: Optional[int] = None      # Random seed for reproducible generation.

    # Add simple validation within Pydantic if desired (requires Field)
    # or perform validation within the endpoint function if sticking to basic BaseModel.
    # @validator('speed_factor')
    # def speed_factor_must_be_positive(cls, v):
    #     if v <= 0.1:
    #         raise ValueError('speed_factor must be greater than 0.1')
    #     return v
    # (Similar validators for other fields)

    class Config:
        # Example for generating OpenAPI schema documentation
        schema_extra = {
            "example": {
                "text": "[S1] This is a test sentence.",
                "prompt_id": "default_voice",
                "temperature": 0.8,
                "seed": 1234
            }
        }

# --- Helper Functions ---
def load_model():
    """Loads the Nari Dia TTS model based on effective configuration."""
    global tts_model, model_sample_rate, compute_device
    if tts_model is not None:
        logger.info("Model already loaded.")
        return

    logger.info("--- Loading TTS Model ---")
    logger.info(f"Model Name: {EFFECTIVE_MODEL_NAME}")

    logger.info("Determining compute device...")
    if torch.cuda.is_available():
        compute_device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Check if MPS is available and works
        try:
            torch.zeros(1, device=torch.device("mps"))
            compute_device = torch.device("mps")
        except Exception:
            logger.warning("MPS available but verification failed, falling back to CPU.")
            compute_device = torch.device("cpu")
    else:
        compute_device = torch.device("cpu")
    logger.info(f"Using device: {compute_device}")

    logger.info("Loading model weights...")
    try:
        # Dynamically import here to allow installation without dia initially?
        from dia.model import Dia
        if not hasattr(Dia, 'from_pretrained'):
             raise ImportError("Imported 'Dia' class does not have 'from_pretrained' method.")

        # Determine compute type based on device
        compute_dtype = torch.float16 if compute_device != torch.device("cpu") else torch.float32
        logger.info(f"Using compute dtype: {compute_dtype}")

        tts_model = Dia.from_pretrained(
            EFFECTIVE_MODEL_NAME,
            compute_dtype=compute_dtype,
            device=compute_device
        )
        # Future: Attempt to get sample rate from model config if available
        # model_sample_rate = getattr(tts_model, 'config.sample_rate', 44100)
        logger.info("Nari Dia model loaded successfully.")
        logger.info(f"Model sample rate set to: {model_sample_rate}")

    except ImportError as ie:
        logger.error(f"Could not import Dia model. Ensure 'dia' package is installed correctly.", exc_info=True)
        tts_model = None # Ensure model is None if loading fails
    except Exception as e:
        logger.error(f"Error loading Nari Dia model weights: {e}", exc_info=True)
        tts_model = None # Ensure model is None if loading fails

def set_seed(seed: int):
    """Sets random seeds for Python, NumPy, and PyTorch for reproducibility."""
    logger.info(f"Setting random seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # Important for multi-GPU setups
    # Optional: Force deterministic algorithms (can impact performance and might not guarantee full determinism)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # If using CUDA >= 10.2

# --- Application Factory ---
def create_app() -> FastAPI:
    """
    Creates, configures, and returns the FastAPI application instance.
    Sets up startup/shutdown events.
    """
    logger.info("--- Initializing FastAPI Application ---")
    app = FastAPI(
        title="Nari TTS API Server",
        description=f"FastAPI server for {EFFECTIVE_MODEL_NAME} TTS, using server-side prompts.",
        version="0.3.1" # Version bump for final review cleanup
    )

    @app.on_event("startup")
    async def startup_event():
        """Load model and validate prompts directory on server startup."""
        logger.info("--- Server Startup Sequence ---")
        load_model() # Load the model
        if tts_model is None:
            logger.error("Model loading failed during startup. TTS endpoint will be unavailable.")
            # Consider raising RuntimeError here if server should NOT start without a model:
            # raise RuntimeError("Model failed to load during startup.")

        # Validate Prompts Directory (uses effective config set before server start)
        logger.info(f"Validating prompts directory: {EFFECTIVE_PROMPTS_DIR.resolve()}")
        if not EFFECTIVE_PROMPTS_DIR.is_dir():
            logger.warning(f"Prompts directory '{EFFECTIVE_PROMPTS_DIR}' not found or is not a directory.")
            logger.warning("Prompt selection will fail unless the directory exists and is populated.")
        else:
            logger.info(f"Using prompts directory: {EFFECTIVE_PROMPTS_DIR.resolve()}")
            # Check default prompt files if a default ID is configured
            if EFFECTIVE_DEFAULT_PROMPT_ID:
                default_wav = EFFECTIVE_PROMPTS_DIR / f"{EFFECTIVE_DEFAULT_PROMPT_ID}.wav"
                default_txt = EFFECTIVE_PROMPTS_DIR / f"{EFFECTIVE_DEFAULT_PROMPT_ID}.txt"
                if not default_wav.is_file() or not default_txt.is_file():
                    logger.warning(f"Default prompt files ('{EFFECTIVE_DEFAULT_PROMPT_ID}.wav/.txt')")
                    logger.warning(f"not found in '{EFFECTIVE_PROMPTS_DIR}'. Requests without a prompt_id will fail.")
                else:
                    logger.info(f"Default prompt ID '{EFFECTIVE_DEFAULT_PROMPT_ID}' files found.")
            else:
                 logger.info("No default prompt ID configured.")
        logger.info("Server startup complete.")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Perform cleanup on server shutdown."""
        logger.info("--- Server Shutdown ---")
        global tts_model
        # Add any cleanup logic here if needed (e.g., releasing GPU memory explicitly)
        tts_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Server shutdown complete.")


    # --- API Endpoint (/tts) ---
    @app.post(
        "/tts",
        response_class=StreamingResponse,
        tags=["TTS Generation"],
        summary="Generate Speech Audio",
        response_description="Streaming WAV audio data.",
        responses={
            200: {"description": "Successful TTS Generation", "content": {"audio/wav": {}}},
            400: {"description": "Invalid input parameters (e.g., empty text, invalid values)"},
            404: {"description": "Requested prompt_id not found"},
            500: {"description": "Internal server error during generation or processing"},
            503: {"description": "Model not loaded or unavailable"},
        }
    )
    async def synthesize(req: TTSRequest):
        """
        Synthesizes speech from text using a specified or default voice prompt.
        Returns streaming WAV audio. Validates input parameters.
        """
        # Use global model variable
        if tts_model is None:
             logger.error("TTS request failed: Model not loaded.")
             raise HTTPException(status_code=503, detail="Model is not loaded or unavailable.")

        # --- Input Validation ---
        if not req.text or req.text.isspace():
             logger.error("TTS request failed: Text input is empty.")
             raise HTTPException(status_code=400, detail="Text input cannot be empty.")
        # Add validation for other parameters if not using Pydantic Field validation
        if req.max_new_tokens <= 0:
            raise HTTPException(status_code=400, detail="max_new_tokens must be positive.")
        if req.cfg_scale < 1.0:
             raise HTTPException(status_code=400, detail="cfg_scale must be >= 1.0.")
        if req.temperature <= 0.0:
             raise HTTPException(status_code=400, detail="temperature must be positive.")
        if not (0.0 < req.top_p <= 1.0):
             raise HTTPException(status_code=400, detail="top_p must be between 0.0 (exclusive) and 1.0 (inclusive).")
        if req.cfg_filter_top_k < 1:
             raise HTTPException(status_code=400, detail="cfg_filter_top_k must be >= 1.")
        if req.speed_factor <= 0.1:
             raise HTTPException(status_code=400, detail="speed_factor must be > 0.1.")
        # --- End Input Validation ---


        logger.info("--- Received TTS Request ---")
        # Log key parameters concisely
        logger.info(f"Text: '{req.text[:80]}...', Prompt ID: {req.prompt_id}, Seed: {req.seed}")

        prompt_audio_path: Optional[Path] = None
        prompt_transcript: Optional[str] = None
        text_for_generate: str = req.text

        try:
            # --- Determine and Validate Prompt ---
            # Use provided prompt_id or fall back to the server's default
            selected_prompt_id = req.prompt_id if req.prompt_id else EFFECTIVE_DEFAULT_PROMPT_ID
            logger.info(f"Processing with prompt_id: {selected_prompt_id or 'None (using default/no prompt)'}")

            if selected_prompt_id:
                # Construct absolute paths for clarity and robustness
                wav_path = (EFFECTIVE_PROMPTS_DIR / f"{selected_prompt_id}.wav").resolve()
                txt_path = (EFFECTIVE_PROMPTS_DIR / f"{selected_prompt_id}.txt").resolve()

                # Validate prompt files exist
                if not wav_path.is_file():
                     logger.error(f"Prompt audio file not found for ID '{selected_prompt_id}' at: {wav_path}")
                     raise HTTPException(status_code=404, detail=f"Prompt audio file not found for ID: '{selected_prompt_id}'")
                if not txt_path.is_file():
                     logger.error(f"Prompt transcript file not found for ID '{selected_prompt_id}' at: {txt_path}")
                     raise HTTPException(status_code=404, detail=f"Prompt transcript file not found for ID: '{selected_prompt_id}'")

                # Read transcript file
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        prompt_transcript = f.read().strip()
                    if not prompt_transcript:
                        logger.error(f"Prompt transcript file is empty for ID '{selected_prompt_id}': {txt_path}")
                        raise HTTPException(status_code=500, detail=f"Prompt transcript file for ID '{selected_prompt_id}' is empty.")
                    logger.info(f"Successfully read transcript for '{selected_prompt_id}'.")
                except Exception as e:
                    logger.error(f"Failed to read transcript file {txt_path}: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Error reading transcript file for ID '{selected_prompt_id}'.")

                # Assign path and prepend transcript for generation
                prompt_audio_path = wav_path
                text_for_generate = f"{prompt_transcript} {req.text}"
                logger.info(f"Using prompt audio: {prompt_audio_path}")

            else:
                # This case occurs if req.prompt_id is None AND EFFECTIVE_DEFAULT_PROMPT_ID is empty/None
                logger.info("No prompt_id requested and no default configured. Generating without voice prompt.")

            # --- Set Seed ---
            if req.seed is not None:
                set_seed(req.seed)
            else:
                 logger.info("Using random seed.") # Default PyTorch/NumPy seed behavior

            # --- TTS Generation ---
            logger.info(f"Calling model.generate...")
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()

            with torch.inference_mode():
                # Pass parameters directly from the validated request object
                wav_output = tts_model.generate(
                    text=text_for_generate,
                    audio_prompt=str(prompt_audio_path) if prompt_audio_path else None,
                    max_tokens=req.max_new_tokens,
                    cfg_scale=req.cfg_scale,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    cfg_filter_top_k=req.cfg_filter_top_k,
                    use_torch_compile=EFFECTIVE_USE_TORCH_COMPILE
                 )

            end_time.record()
            torch.cuda.synchronize() # Wait for GPU operations to complete
            generation_time = start_time.elapsed_time(end_time) / 1000.0 # Time in seconds
            logger.info(f"Model generation complete in {generation_time:.2f} seconds.")
            # --- End TTS Generation ---

            # --- Post-processing ---
            if wav_output is None or len(wav_output) == 0:
                logger.error("Model generation resulted in empty audio data.")
                raise HTTPException(status_code=500, detail="Model failed to generate audio data (returned None or empty array).")

            raw_samples = len(wav_output)
            logger.info(f"Generated {raw_samples} raw audio samples.")
            # Ensure float32 for processing, defensive copy
            processed_wav = np.copy(wav_output).astype(np.float32)

            # Apply speed factor
            if req.speed_factor != 1.0:
                logger.info(f"Applying speed factor: {req.speed_factor}")
                orig_len = len(processed_wav)
                # Ensure target_len calculation avoids division by zero if speed_factor validation failed
                target_len = int(orig_len / max(0.1, req.speed_factor))
                if target_len > 0:
                    xs = np.linspace(0, orig_len - 1, target_len)
                    processed_wav = np.interp(xs, np.arange(orig_len), processed_wav).astype(np.float32)
                    logger.info(f"Audio resampled from {orig_len} to {len(processed_wav)} samples.")
                else:
                    logger.warning("Speed factor resulted in zero target length. Skipping speed adjustment.")

            # Normalize audio (prevents clipping)
            max_abs_val = np.max(np.abs(processed_wav))
            if max_abs_val > 1e-6: # Avoid division by zero for silent/near-silent audio
                logger.info(f"Normalizing audio from max abs value: {max_abs_val:.4f}")
                processed_wav = processed_wav / max_abs_val
            else:
                logger.info("Skipping normalization, audio is near silent.")

            # Check for NaN values after processing
            if np.isnan(processed_wav).any():
                logger.error("Generated audio contains NaN values after processing.")
                # Consider saving the problematic array for debugging?
                raise HTTPException(status_code=500, detail="Audio processing resulted in NaN values.")
            # --- End Post-processing ---


            # --- Prepare and Stream WAV Response ---
            final_samples = len(processed_wav)
            logger.info(f"Preparing WAV buffer for {final_samples} processed samples...")
            buf = io.BytesIO()
            try:
                 # Write the processed audio data to the buffer
                 sf.write(buf, processed_wav, model_sample_rate, format="WAV", subtype='FLOAT')
                 buf.seek(0)
                 content_length = buf.getbuffer().nbytes
                 logger.info(f"Prepared WAV buffer ({content_length} bytes). Sending response.")
                 # Set headers for client download and streaming
                 headers = {
                     'Content-Disposition': 'attachment; filename="naritts_output.wav"',
                     'Content-Length': str(content_length),
                     'Accept-Ranges': 'bytes', # Optional: indicates support for range requests
                 }
                 return StreamingResponse(buf, media_type="audio/wav", headers=headers)
            except Exception as sf_err:
                 logger.error(f"Failed to write WAV data to buffer: {sf_err}", exc_info=True)
                 raise HTTPException(status_code=500, detail="Failed to encode audio to WAV format.")

        except HTTPException:
             # Re-raise known HTTP exceptions (like 400, 404, 503) directly
             raise
        except Exception as e:
            # Catch any other unexpected errors during the process
            logger.error(f"Unexpected error during TTS request processing: {e}", exc_info=True)
            # Return a generic 500 error to the client
            raise HTTPException(status_code=500, detail="Internal server error during TTS generation.")

    logger.info("FastAPI application creation complete.")
    return app

# --- Server Runner ---
def ensure_prompts_dir(prompts_dir_path: Path):
    """Creates the prompts directory and a placeholder file if it doesn't exist."""
    if not prompts_dir_path.exists():
        logger.warning(f"Default prompts directory '{prompts_dir_path}' not found. Creating...")
        try:
            prompts_dir_path.mkdir(parents=True, exist_ok=True)
            placeholder_path = prompts_dir_path / "PLACEHOLDER_PUT_PROMPTS_HERE.txt"
            # Only create placeholder if it doesn't exist (avoids overwriting)
            if not placeholder_path.exists():
                with open(placeholder_path, "w") as f:
                    f.write("Place your mono .wav prompt files and corresponding .txt transcript files in this directory.\n")
                    f.write("The filename base (e.g., 'my_voice' for 'my_voice.wav' and 'my_voice.txt') is used as the 'prompt_id'.\n")
                logger.info(f"Created prompts directory and placeholder file at '{prompts_dir_path}'.")
            else:
                logger.info(f"Prompts directory created at '{prompts_dir_path}'. Placeholder file already exists.")
        except OSError as e:
            logger.error(f"Could not create prompts directory '{prompts_dir_path}': {e}", exc_info=True)
    elif not prompts_dir_path.is_dir():
         # Path exists but isn't a directory - this is a problem
         logger.error(f"Path specified for prompts ('{prompts_dir_path}') exists but is not a directory.")
         # Potentially exit or raise an error here if this is critical


def run_server(host: str, port: int, reload: bool, prompts_dir: Path, default_prompt: str, model_name: str, use_compile: bool):
    """Sets effective configuration globally and runs the Uvicorn server."""
    # Set global effective config variables used by create_app and load_model
    global EFFECTIVE_PROMPTS_DIR, EFFECTIVE_DEFAULT_PROMPT_ID, EFFECTIVE_MODEL_NAME
    global EFFECTIVE_HOST, EFFECTIVE_PORT, EFFECTIVE_USE_TORCH_COMPILE
    EFFECTIVE_PROMPTS_DIR = prompts_dir
    EFFECTIVE_DEFAULT_PROMPT_ID = default_prompt
    EFFECTIVE_MODEL_NAME = model_name
    EFFECTIVE_HOST = host
    EFFECTIVE_PORT = port
    EFFECTIVE_USE_TORCH_COMPILE = use_compile

    logger.info("--- Starting Server ---")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Reload: {reload}")
    logger.info(f"Effective Prompts Directory: {EFFECTIVE_PROMPTS_DIR.resolve()}")
    logger.info(f"Effective Default Prompt ID: {EFFECTIVE_DEFAULT_PROMPT_ID or 'None'}")
    logger.info(f"Effective Model Name: {EFFECTIVE_MODEL_NAME}")
    logger.info(f"Use Torch Compile: {EFFECTIVE_USE_TORCH_COMPILE}")


    # Create prompts dir only if using the default path and it doesn't exist
    # Check against the default *string* before it was cast to Path
    if str(prompts_dir) == DEFAULT_PROMPTS_DIR_STR:
        ensure_prompts_dir(prompts_dir)
    elif not prompts_dir.is_dir():
         logger.warning(f"Custom prompts directory specified ('{prompts_dir}') does not exist or is not a directory.")

    if reload:
        # When reloading, Uvicorn restarts the process. Pass config via environment variables
        # so the new process picks them up correctly.
        logger.info("Running Uvicorn with reload enabled. Configuration passed via environment variables.")
        env_config = {
            "NARI_PROMPTS_DIR": str(EFFECTIVE_PROMPTS_DIR),
            "NARI_DEFAULT_PROMPT": EFFECTIVE_DEFAULT_PROMPT_ID or "", # Env var needs a string
            "NARI_MODEL_NAME": EFFECTIVE_MODEL_NAME,
            "NARI_HOST": EFFECTIVE_HOST,
            "NARI_PORT": str(EFFECTIVE_PORT),
            "NARI_USE_COMPILE": "true" if EFFECTIVE_USE_TORCH_COMPILE else "false",
        }
        uvicorn.run(
            "tts_server:create_app", # Path to the app factory function
            host=host,
            port=port,
            reload=True,
            factory=True, # Indicates the app path is a factory function
            env=env_config
        )
    else:
        # When not reloading, create the app instance directly and pass it to Uvicorn
        logger.info("Running Uvicorn without reload.")
        app_instance = create_app()
        uvicorn.run(app_instance, host=host, port=port, reload=False)


# --- Main Execution Guard ---
if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(
        description=f"Run Nari TTS FastAPI Server ({DEFAULT_MODEL_NAME})",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help message
        )
    parser.add_argument("--host", type=str, default=DEFAULT_HOST,
                        help="Host address to bind the server to.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="Port number to bind the server to.")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload mode for development. Requires Uvicorn[standard].")
    parser.add_argument("--prompts-dir", type=str, default=DEFAULT_PROMPTS_DIR_STR,
                        help="Directory containing prompt .wav and corresponding .txt files.")
    parser.add_argument("--default-prompt", type=str, default=DEFAULT_PROMPT_ID,
                        help="Base filename (without extension) of the default prompt to use if none is specified in the request.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                        help="Name or path of the Hugging Face model to load (e.g., 'nari-labs/Dia-1.6B').")
    parser.add_argument("--use-torch-compile", action=argparse.BooleanOptionalAction, default=DEFAULT_USE_TORCH_COMPILE_STR == "true",
                        help="Enable Torch Compile for potentially faster inference (requires compatible PyTorch/GPU). Use --no-use-torch-compile to disable.")

    cli_args = parser.parse_args()

    # Determine final configuration (CLI args override environment vars/defaults)
    final_prompts_dir = Path(cli_args.prompts_dir)
    # Handle potential empty string from env/default for default_prompt
    final_default_prompt = cli_args.default_prompt if cli_args.default_prompt else ""

    # Run the server with the final configuration values
    run_server(
        host=cli_args.host,
        port=cli_args.port,
        reload=cli_args.reload,
        prompts_dir=final_prompts_dir,
        default_prompt=final_default_prompt,
        model_name=cli_args.model_name,
        use_compile=cli_args.use_torch_compile
    )