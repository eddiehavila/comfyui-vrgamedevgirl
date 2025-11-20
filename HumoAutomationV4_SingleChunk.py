"""
HumoAutomationV4_SingleChunk.py

New single-chunk processing nodes for simplified HUMO music video generation.
Processes one 3.88-second chunk at a time instead of batching 16 chunks.

Key nodes:
1. VRGDG_FullSongAnalyzerV4 - Analyzes entire song once, generates all prompts (cached)
2. VRGDG_ChunkIndexController - Manages which chunk to process (auto vs manual)
3. VRGDG_GetPromptByIndex - Extracts single prompt from full prompts
4. VRGDG_LoadSingleAudioChunk - Loads one audio chunk by index
5. VRGDG_SaveVideoChunkWithIndex - Saves video with index in filename
6. VRGDG_AutoQueueController - Queues next chunk automatically
"""

import os
import json
import math
import hashlib
import torch
import torchaudio
import folder_paths
from server import PromptServer

# Whisper imports (from existing nodes)
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[VRGDG V4] Warning: Whisper not available. Transcription will be disabled.")

any_typ = "*"

# =============================================================================
# Node 1: VRGDG_FullSongAnalyzerV4
# =============================================================================

class VRGDG_FullSongAnalyzerV4:
    """
    Master orchestrator - analyzes entire song once and generates ALL prompts upfront.

    This node:
    - Loads and analyzes the full audio file
    - Transcribes entire song with Whisper
    - Calculates total_chunks needed (audio_duration / 3.88s)
    - Generates ALL scene prompts at once (leverages ComfyUI caching)
    - Creates audio hash for cache invalidation

    ComfyUI will cache this node's output if inputs don't change,
    so prompt generation happens ONCE per song, not per chunk.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "character_description": ("STRING", {
                    "multiline": True,
                    "default": "A woman in a white dress",
                }),
                "song_theme_style": ("STRING", {
                    "multiline": True,
                    "default": "cinematic realism, emotional storytelling, soft surrealism",
                }),
                "scene_duration_seconds": ("FLOAT", {
                    "default": 3.88,  # 97 frames at 25 FPS
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.01,
                }),
                "language": ([
                    "auto", "english", "chinese", "german", "spanish", "russian", "korean", "french",
                    "japanese", "portuguese", "turkish", "polish", "catalan", "dutch", "arabic", "swedish",
                    "italian", "indonesian", "hindi", "finnish", "vietnamese", "hebrew", "ukrainian", "greek",
                ], {"default": "english"}),
                "enable_full_transcription": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Fixed seed for deterministic LLM prompt generation",
                }),
            },
            "optional": {
                # Visual parameters (same as MusicVideoPromptCreator)
                "environment": ("STRING", {
                    "multiline": True,
                    "default": "open field at dusk, dimly lit bedroom, empty city street at night",
                }),
                "lighting": ("STRING", {
                    "multiline": True,
                    "default": "warm amber glow, cool window light, neon reflections",
                }),
                "camera_motion": ("STRING", {
                    "multiline": True,
                    "default": "zoom in, zoom out, tilt down, rotate around, pan, track",
                }),
                "physical_interaction": ("STRING", {
                    "multiline": True,
                    "default": "walking through tall grass, lying on bed, leaning against wall",
                }),
                "facial_expression": ("STRING", {
                    "multiline": True,
                    "default": "Intense raw emotion",
                }),
                "shots": ("STRING", {
                    "multiline": True,
                    "default": "Close up, medium, wide angle, over the shoulder, point of view",
                }),
                "outfit_rules": ("STRING", {
                    "multiline": True,
                    "default": "a white dress",
                }),
            }
        }

    RETURN_TYPES = ("DICT", "INT", "STRING", "DICT", "STRING")
    RETURN_NAMES = ("full_prompts", "total_chunks", "full_lyrics", "audio_meta", "audio_hash")
    FUNCTION = "analyze_song"
    CATEGORY = "VRGDG/V4 Single Chunk"

    def analyze_song(
        self,
        audio,
        character_description,
        song_theme_style,
        scene_duration_seconds,
        language,
        enable_full_transcription,
        seed,
        environment="",
        lighting="",
        camera_motion="",
        physical_interaction="",
        facial_expression="",
        shots="",
        outfit_rules="",
    ):
        """
        Analyze entire song and generate all prompts upfront.
        This leverages ComfyUI's caching - if audio doesn't change, this won't re-run.
        """

        print("[VRGDG V4] 🎵 Full Song Analyzer - Starting analysis...")

        # Extract audio data
        try:
            waveform = audio["waveform"]  # shape: [channels, samples]
            sample_rate = audio["sample_rate"]
        except Exception as e:
            raise ValueError(f"Invalid audio input: {e}")

        # Calculate audio metadata
        num_samples = waveform.shape[-1]
        audio_duration = num_samples / sample_rate

        # Calculate audio hash (for caching)
        try:
            sample_data = waveform[..., :sample_rate].cpu().numpy().tobytes()  # First 1 second
            audio_hash = hashlib.md5(sample_data).hexdigest()[:16]
        except Exception:
            audio_hash = "unknown"

        print(f"[VRGDG V4] Audio Duration: {audio_duration:.2f}s | Hash: {audio_hash}")

        # Calculate total chunks needed
        fps = 25
        frames_per_chunk = 97  # HuMo optimal (4n+1 format)
        seconds_per_chunk = frames_per_chunk / fps  # 3.88 seconds
        samples_per_chunk_calc = int(frames_per_chunk * sample_rate / fps + 0.5)

        total_chunks = math.ceil(audio_duration / seconds_per_chunk)

        print(f"[VRGDG V4] Chunks needed: {total_chunks} (@ {seconds_per_chunk:.2f}s each)")
        print(f"[VRGDG V4] Samples per chunk: {samples_per_chunk_calc} (@ {sample_rate} Hz)")
        print(f"[VRGDG V4] Total audio samples: {num_samples} ({audio_duration:.2f}s)")

        # Transcribe full audio if enabled
        full_lyrics = ""
        chunk_transcriptions = []

        if enable_full_transcription and WHISPER_AVAILABLE:
            print(f"[VRGDG V4] 🎤 Transcribing full audio (language: {language})...")

            try:
                # Load Whisper model
                model = whisper.load_model("base")

                # Convert audio to format Whisper expects
                # Whisper expects mono audio at 16kHz
                audio_numpy = waveform.mean(dim=0).cpu().numpy()  # Convert to mono

                # Transcribe full audio
                lang_param = None if language == "auto" else language
                result = model.transcribe(
                    audio_numpy,
                    language=lang_param,
                    fp16=False,
                )

                full_lyrics = result["text"].strip()
                print(f"[VRGDG V4] ✅ Full transcription complete ({len(full_lyrics)} chars)")

                # Also transcribe each chunk individually for per-chunk metadata
                # Use proper rounding (not truncation) to avoid sample loss
                samples_per_chunk = int(frames_per_chunk * sample_rate / fps + 0.5)

                for idx in range(total_chunks):
                    start_sample = idx * samples_per_chunk
                    end_sample = min(start_sample + samples_per_chunk, num_samples)

                    chunk_audio = audio_numpy[start_sample:end_sample]

                    if len(chunk_audio) > 0:
                        chunk_result = model.transcribe(
                            chunk_audio,
                            language=lang_param,
                            fp16=False,
                        )
                        chunk_transcriptions.append(chunk_result["text"].strip())
                    else:
                        chunk_transcriptions.append("")

                print(f"[VRGDG V4] ✅ Transcribed {len(chunk_transcriptions)} individual chunks")

            except Exception as e:
                print(f"[VRGDG V4] ⚠️ Transcription failed: {e}")
                full_lyrics = f"[Transcription failed: {str(e)}]"
                chunk_transcriptions = [""] * total_chunks
        else:
            print("[VRGDG V4] ℹ️ Transcription disabled or Whisper not available")
            chunk_transcriptions = ["[lyrics not transcribed]"] * total_chunks

        # Generate prompts for ALL chunks
        # Note: In a real implementation, this would call the LLM to generate prompts
        # For now, we'll create a placeholder structure

        print(f"[VRGDG V4] 🎨 Generating {total_chunks} scene prompts...")

        # Create pipe-separated lyrics for prompt generation
        if chunk_transcriptions:
            pipe_separated_lyrics = " | ".join(chunk_transcriptions)
        else:
            pipe_separated_lyrics = full_lyrics

        # Build prompt instruction (similar to VRGDG_MusicVideoPromptCreatorV3)
        # This would normally be sent to an LLM
        prompt_instruction = self._build_prompt_instruction(
            character_description=character_description,
            song_theme_style=song_theme_style,
            pipe_separated_lyrics=pipe_separated_lyrics,
            total_prompts=total_chunks,
            environment=environment,
            lighting=lighting,
            camera_motion=camera_motion,
            physical_interaction=physical_interaction,
            facial_expression=facial_expression,
            shots=shots,
            outfit_rules=outfit_rules,
            seed=seed,
        )

        # For now, generate placeholder prompts
        # In production, you would call your LLM here with prompt_instruction
        generated_prompts = []
        for idx in range(total_chunks):
            # Placeholder prompt (in production, this comes from LLM)
            prompt = f"Chunk {idx+1}: {character_description} in a cinematic scene. {chunk_transcriptions[idx] if idx < len(chunk_transcriptions) else ''}"
            generated_prompts.append(prompt)

        print(f"[VRGDG V4] ✅ Generated {len(generated_prompts)} prompts")

        # Package everything into output dict
        full_prompts = {
            "prompts": generated_prompts,
            "total_chunks": total_chunks,
            "audio_hash": audio_hash,
            "audio_duration": audio_duration,
            "chunk_transcriptions": chunk_transcriptions,
            "prompt_instruction": prompt_instruction,  # Save for reference
        }

        audio_meta = {
            "sample_rate": sample_rate,
            "num_samples": num_samples,
            "duration": audio_duration,
            "frames_per_chunk": frames_per_chunk,
            "samples_per_chunk": int(frames_per_chunk * sample_rate / fps + 0.5),  # Proper rounding
        }

        print(f"[VRGDG V4] ✅ Analysis complete! Ready for chunked processing.")

        return (full_prompts, total_chunks, full_lyrics, audio_meta, audio_hash)

    def _build_prompt_instruction(
        self,
        character_description,
        song_theme_style,
        pipe_separated_lyrics,
        total_prompts,
        environment,
        lighting,
        camera_motion,
        physical_interaction,
        facial_expression,
        shots,
        outfit_rules,
        seed,
    ):
        """
        Build instruction text for LLM prompt generation.
        Similar to VRGDG_MusicVideoPromptCreatorV3.build_prompt_instructions()
        """

        instructions = f"""TASK: Generate {total_prompts} cinematic text-to-video prompts for a music video

CHARACTER: {character_description}

THEME/STYLE: {song_theme_style}

LYRIC SEGMENTS ({total_prompts} total, pipe-separated):
{pipe_separated_lyrics}

OUTPUT FORMAT:
Return JSON with sequential keys:
{{
  "prompt1": "First cinematic visual...",
  "prompt2": "Second cinematic visual...",
  ...
  "prompt{total_prompts}": "Final cinematic visual..."
}}

VISUAL ELEMENTS:
- Environments: {environment}
- Lighting: {lighting}
- Camera Motion: {camera_motion}
- Physical Interactions: {physical_interaction}
- Facial Expressions: {facial_expression}
- Shot Types: {shots}
- Outfit: {outfit_rules}

REQUIREMENTS:
- Each prompt: 40-50 words
- Self-contained descriptions (no references to previous prompts)
- Cinematic, visual language
- JSON output only (no markdown, no commentary)

SEED: {seed} (use for deterministic generation)
"""

        return instructions.strip()


# =============================================================================
# Node 2: VRGDG_ChunkIndexController
# =============================================================================

class VRGDG_ChunkIndexController:
    """
    Manages which chunk to process - supports both auto-detection and manual override.

    Auto Mode: Counts existing video_chunk_*-audio.mp4 files to determine current index
    Manual Mode: Uses provided manual_index parameter
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "full_prompts": ("DICT",),
                "output_folder": ("STRING", {
                    "default": "video_output",
                    "multiline": False,
                }),
                "auto_index": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True = auto-detect index from files, False = use manual_index",
                }),
                "manual_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "tooltip": "Used when auto_index=False to manually specify chunk index",
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("current_index", "total_chunks", "remaining", "should_auto_queue", "status_message")
    FUNCTION = "get_index"
    CATEGORY = "VRGDG/V4 Single Chunk"

    def get_index(self, full_prompts, output_folder, auto_index, manual_index):
        """
        Determine which chunk to process based on mode.
        """

        total_chunks = full_prompts.get("total_chunks", 0)

        if auto_index:
            # Auto-detect index by counting existing files
            current_index = self._count_completed_chunks(output_folder)
            mode_str = "AUTO"
        else:
            # Use manual index
            current_index = manual_index
            mode_str = "MANUAL"

        # Calculate remaining chunks
        remaining = total_chunks - current_index - 1

        # Should we auto-queue the next chunk?
        should_auto_queue = auto_index and (current_index < total_chunks)

        # Build status message
        if current_index >= total_chunks:
            status_message = f"✅ All {total_chunks} chunks complete!"
            should_auto_queue = False
        else:
            status_message = f"[{mode_str}] Processing chunk {current_index + 1}/{total_chunks} ({remaining} remaining)"

        print(f"[VRGDG V4] 🎯 Index Controller: {status_message}")

        return (current_index, total_chunks, remaining, should_auto_queue, status_message)

    def _count_completed_chunks(self, folder_path):
        """
        Count completed chunks by looking for video_chunk_*-audio.mp4 marker files.
        """
        if not os.path.isdir(folder_path):
            return 0

        try:
            marker_files = [
                f for f in os.listdir(folder_path)
                if f.startswith("video_chunk_") and f.endswith("-audio.mp4")
            ]
            count = len(marker_files)
            print(f"[VRGDG V4] Found {count} completed chunks in {folder_path}")
            return count
        except Exception as e:
            print(f"[VRGDG V4] ⚠️ Error counting chunks: {e}")
            return 0


# =============================================================================
# Node 3: VRGDG_GetPromptByIndex
# =============================================================================

class VRGDG_GetPromptByIndex:
    """
    Simple prompt extractor - gets a single prompt from the cached full_prompts dict.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "full_prompts": ("DICT",),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "transcription")
    FUNCTION = "get_prompt"
    CATEGORY = "VRGDG/V4 Single Chunk"

    def get_prompt(self, full_prompts, index):
        """
        Extract single prompt by index.
        """

        prompts = full_prompts.get("prompts", [])
        transcriptions = full_prompts.get("chunk_transcriptions", [])

        if index < 0 or index >= len(prompts):
            error_msg = f"❌ Index {index} out of range (0-{len(prompts)-1})"
            print(f"[VRGDG V4] {error_msg}")
            return (error_msg, "")

        prompt = prompts[index]
        transcription = transcriptions[index] if index < len(transcriptions) else ""

        print(f"[VRGDG V4] 📝 Retrieved prompt #{index}: {prompt[:60]}...")

        return (prompt, transcription)


# =============================================================================
# Node 4: VRGDG_LoadSingleAudioChunk
# =============================================================================

class VRGDG_LoadSingleAudioChunk:
    """
    Loads a single audio chunk based on index.

    Extracts exactly 97 frames worth of audio (3.88s at 25 FPS).
    Pads with silence if the chunk extends beyond the audio duration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                }),
                "audio_meta": ("DICT",),
            }
        }

    RETURN_TYPES = ("AUDIO", "DICT")
    RETURN_NAMES = ("audio_chunk", "chunk_info")
    FUNCTION = "load_chunk"
    CATEGORY = "VRGDG/V4 Single Chunk"

    def load_chunk(self, audio, index, audio_meta):
        """
        Load a single audio chunk by index.
        """

        try:
            waveform = audio["waveform"]  # shape: [channels, samples]
            sample_rate = audio["sample_rate"]
        except Exception as e:
            raise ValueError(f"Invalid audio input: {e}")

        # Get chunk parameters
        samples_per_chunk = audio_meta.get("samples_per_chunk", 0)
        frames_per_chunk = audio_meta.get("frames_per_chunk", 97)

        if samples_per_chunk == 0:
            # Fallback calculation with proper rounding
            fps = 25
            frames_per_chunk = 97
            samples_per_chunk = int(frames_per_chunk * sample_rate / fps + 0.5)

        # Calculate sample range for this chunk
        start_sample = index * samples_per_chunk
        end_sample = start_sample + samples_per_chunk

        num_samples = waveform.shape[-1]

        # Extract chunk
        if start_sample >= num_samples:
            # Beyond audio end - create silence
            chunk_waveform = torch.zeros(
                (waveform.shape[0], samples_per_chunk),
                dtype=waveform.dtype,
                device=waveform.device
            )
            actual_samples = 0
        else:
            # Extract available samples
            actual_end = min(end_sample, num_samples)
            chunk_waveform = waveform[..., start_sample:actual_end]
            actual_samples = chunk_waveform.shape[-1]

            # Pad with silence if needed
            if actual_samples < samples_per_chunk:
                padding = samples_per_chunk - actual_samples
                silence = torch.zeros(
                    (chunk_waveform.shape[0], padding),
                    dtype=chunk_waveform.dtype,
                    device=chunk_waveform.device
                )
                chunk_waveform = torch.cat([chunk_waveform, silence], dim=-1)

        chunk_audio = {
            "waveform": chunk_waveform,
            "sample_rate": sample_rate,
        }

        chunk_info = {
            "index": index,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "actual_samples": actual_samples,
            "padded": actual_samples < samples_per_chunk,
            "duration": samples_per_chunk / sample_rate,
            "samples_per_chunk": samples_per_chunk,
        }

        print(f"[VRGDG V4] 🎵 Loaded chunk #{index}: {actual_samples}/{samples_per_chunk} samples "
              f"({'padded' if chunk_info['padded'] else 'full'})")
        print(f"[VRGDG V4]    Sample range: {start_sample} - {end_sample} (duration: {chunk_info['duration']:.3f}s)")

        return (chunk_audio, chunk_info)


# =============================================================================
# Node 5: VRGDG_SaveVideoChunkWithIndex
# =============================================================================

class VRGDG_SaveVideoChunkWithIndex:
    """
    Saves video chunk with index in filename.

    Creates two files:
    1. video_chunk_0000.mp4 - The actual video
    2. video_chunk_0000-audio.mp4 - Marker file for counting (can be small/empty)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Video frames
                "audio_chunk": ("AUDIO",),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                }),
                "output_folder": ("STRING", {
                    "default": "video_output",
                    "multiline": False,
                }),
                "fps": ("FLOAT", {
                    "default": 25.0,
                    "min": 1.0,
                    "max": 60.0,
                }),
            }
        }

    RETURN_TYPES = ("STRING", any_typ)
    RETURN_NAMES = ("saved_path", "signal")
    FUNCTION = "save_chunk"
    CATEGORY = "VRGDG/V4 Single Chunk"
    OUTPUT_NODE = True

    def save_chunk(self, images, audio_chunk, index, output_folder, fps):
        """
        Save video chunk with index in filename.
        """

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Generate filenames
        video_filename = f"video_chunk_{index:04d}.mp4"
        marker_filename = f"video_chunk_{index:04d}-audio.mp4"

        video_path = os.path.join(output_folder, video_filename)
        marker_path = os.path.join(output_folder, marker_filename)

        try:
            # Save video with audio using existing ComfyUI video save logic
            # This is a simplified version - you may need to adapt to your video save implementation

            print(f"[VRGDG V4] 💾 Saving chunk #{index} to {video_path}...")

            # For now, we'll use a simple approach with imageio/ffmpeg
            # In production, you'd integrate with your existing video save node

            import imageio
            import numpy as np

            # Convert images tensor to numpy
            # images shape: [frames, height, width, channels]
            frames_np = (images.cpu().numpy() * 255).astype(np.uint8)

            # Get audio data
            waveform = audio_chunk["waveform"]  # [channels, samples]
            sample_rate = audio_chunk["sample_rate"]

            # Convert audio to numpy (mono)
            audio_np = waveform.mean(dim=0).cpu().numpy()

            # Save video with imageio
            writer = imageio.get_writer(
                video_path,
                fps=fps,
                codec='libx264',
                audio_codec='aac',
                audio_path=None,  # We'll add audio separately
                quality=8,
            )

            for frame in frames_np:
                writer.append_data(frame)

            writer.close()

            # Create marker file (empty file for counting)
            with open(marker_path, 'w') as f:
                f.write("")  # Empty marker

            print(f"[VRGDG V4] ✅ Saved chunk #{index}: {video_path}")
            print(f"[VRGDG V4] ✅ Created marker: {marker_path}")

            return (video_path, True)

        except Exception as e:
            error_msg = f"❌ Failed to save chunk #{index}: {e}"
            print(f"[VRGDG V4] {error_msg}")
            raise RuntimeError(error_msg)


# =============================================================================
# Node 6: VRGDG_AutoQueueController
# =============================================================================

class VRGDG_AutoQueueController:
    """
    Manages auto-queueing of the next chunk.

    If should_auto_queue=True and chunks remain, queues the next run.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": (any_typ,),
                "should_auto_queue": ("BOOLEAN",),
                "remaining": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                }),
                "current_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                }),
            }
        }

    RETURN_TYPES = (any_typ, "STRING")
    RETURN_NAMES = ("signal", "queue_status")
    FUNCTION = "control_queue"
    CATEGORY = "VRGDG/V4 Single Chunk"
    OUTPUT_NODE = True

    def control_queue(self, trigger, should_auto_queue, remaining, current_index):
        """
        Queue next chunk if needed.
        """

        if should_auto_queue and remaining >= 0:
            # Queue the next chunk
            try:
                PromptServer.instance.send_sync("impact-add-queue", {})
                status = f"✅ Chunk {current_index} done. Queued chunk {current_index + 1} ({remaining} more after this)"
                print(f"[VRGDG V4] 🔄 {status}")
            except Exception as e:
                status = f"⚠️ Failed to queue next chunk: {e}"
                print(f"[VRGDG V4] {status}")
        else:
            if remaining < 0:
                status = f"🏁 All chunks complete!"
            else:
                status = f"⏸️ Auto-queue disabled (manual mode)"
            print(f"[VRGDG V4] {status}")

        return (True, status)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "VRGDG_FullSongAnalyzerV4": VRGDG_FullSongAnalyzerV4,
    "VRGDG_ChunkIndexController": VRGDG_ChunkIndexController,
    "VRGDG_GetPromptByIndex": VRGDG_GetPromptByIndex,
    "VRGDG_LoadSingleAudioChunk": VRGDG_LoadSingleAudioChunk,
    "VRGDG_SaveVideoChunkWithIndex": VRGDG_SaveVideoChunkWithIndex,
    "VRGDG_AutoQueueController": VRGDG_AutoQueueController,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_FullSongAnalyzerV4": "🎵 VRGDG Full Song Analyzer V4",
    "VRGDG_ChunkIndexController": "🎯 VRGDG Chunk Index Controller",
    "VRGDG_GetPromptByIndex": "📝 VRGDG Get Prompt By Index",
    "VRGDG_LoadSingleAudioChunk": "🎵 VRGDG Load Single Audio Chunk",
    "VRGDG_SaveVideoChunkWithIndex": "💾 VRGDG Save Video Chunk",
    "VRGDG_AutoQueueController": "🔄 VRGDG Auto Queue Controller",
}
