# Troubleshooting Guide for Voxy

This guide addresses common issues that may arise when using the Voxy module with the CSM-1B model.

## Model Downloads Repeatedly

**Issue**: The CSM model downloads every time you run your script.

**Solution**: 

The updated module now caches the model in `~/.cache/voxy/models` and reuses it across sessions. If you're still experiencing repeated downloads, check:

1. That you have write permissions for `~/.cache/voxy/models`
2. That the environment running your script has a consistent home directory
3. That the Hugging Face cache is not being cleared between runs

You can also explicitly provide the model path to avoid downloading:

```python
model = create_speech_model(model_path="/path/to/downloaded/ckpt.pt")
```

## MPS Device Compatibility Issues

**Issue**: Error message: `NotImplementedError: Output channels > 65536 not supported at the MPS device`

**Solution**:

The CSM-1B model is not compatible with Apple's Metal Performance Shaders (MPS) due to architectural limitations. The model contains operations with more than 65,536 channels, which exceeds the MPS backend's capabilities.

The updated module automatically falls back to CPU when running on macOS with Apple Silicon. To explicitly control this behavior:

```python
# Force CPU usage
model = create_speech_model(device="cpu")

# Use CUDA if available
model = create_speech_model(device="cuda")
```

## Import Errors for CSM Dependencies

**Issue**: `ImportError` when trying to use the CSM model

**Solution**:

Ensure you have all the required dependencies installed:

1. Clone the CSM repository:
   ```bash
   git clone https://github.com/SesameAILabs/csm.git
   cd csm
   ```

2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Set the Python path to include the CSM directory:
   ```python
   import sys
   sys.path.append("/path/to/csm")
   ```

4. Disable torch compilation which can cause issues:
   ```bash
   export NO_TORCH_COMPILE=1
   ```
   
## Out of Memory Errors

**Issue**: CUDA out of memory error when running the model

**Solution**:

1. Try using a smaller audio sample for voice cloning
2. Reduce the `max_length_ms` parameter when generating speech:
   ```python
   audio = model.generate_speech(
       text="Hello world",
       voice_profile=profile,
       max_length_ms=5000  # Reduced from default
   )
   ```

3. If using a multi-GPU system, ensure PyTorch is only using one GPU:
   ```python
   import os
   os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the first GPU
   ```

## Whisper Installation Issues

**Issue**: Cannot use the `audio_to_text` function

**Solution**:

Install Whisper and its dependencies:

```bash
pip install openai-whisper
```

If you prefer not to use Whisper, you can provide your own transcripts when cloning voices:

```python
voice_profile = model.clone_voice(
    audio_input="sample_voice.wav",
    transcript="This is the transcript of my sample voice.",
)
```

## Audio Format Issues

**Issue**: Errors when loading audio files

**Solution**:

Ensure you have FFmpeg installed for broad audio format support:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg
```

Also check that your audio files are not corrupted and are in a supported format (WAV is most reliable).

## Hugging Face Authentication Issues

**Issue**: Cannot download the model due to authentication errors

**Solution**:

1. Ensure you're logged in to Hugging Face:
   ```bash
   huggingface-cli login
   ```

2. Accept the model terms on the Hugging Face website:
   - Visit https://huggingface.co/sesame/csm-1b
   - Click "Access repository" and accept the terms
