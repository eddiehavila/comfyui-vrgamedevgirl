"""
HumoAutomationV4_SingleChunk.py

New single-chunk processing nodes for simplified HUMO music video generation.
Processes one 3.88-second chunk at a time instead of batching 16 chunks.

Key nodes:
1. VRGDG_FullSongAnalyzerV4 - Analyzes entire song once, generates all prompts (cached)
   - Includes frames_per_chunk as separate INT output for video combine nodes

2. VRGDG_ChunkIndexController - Manages which chunk to process (auto vs manual)
   - Auto mode: Counts existing video_chunk_*.done marker files
   - Manual mode: Uses provided index

3. VRGDG_GetPromptByIndex - Extracts single prompt from full prompts by index

4. VRGDG_LoadSingleAudioChunk - Loads one audio chunk by index with precise timing

5. VRGDG_SaveVideoChunkWithIndex - Saves video chunk with proper naming
   - VHS mode: Moves/renames VHS_VideoCombine output to proper location
   - Direct mode: Saves video+audio with ffmpeg (fallback)
   - Creates marker files for auto-indexing

6. VRGDG_AutoQueueController - Manages auto-queueing and completion
   - Queues next chunk automatically when chunks remain
   - Signals final combine when all chunks are complete

7. VRGDG_CombineAllChunks - Combines all video chunks into final video
   - Uses ffmpeg concat to merge all video_chunk_*.mp4 files
   - Triggered after last chunk is processed

Workflow:
1. FullSongAnalyzer runs once (cached), generates all prompts upfront
2. ChunkIndexController determines current chunk to process
3. GetPromptByIndex extracts prompt for current chunk
4. LoadSingleAudioChunk loads audio for current chunk
5. Video generation happens (HuMo, AnimateDiff, etc.)
6. SaveVideoChunkWithIndex saves video as video_chunk_0000.mp4 in output folder
7. AutoQueueController either:
   - Queues next chunk if more remain, OR
   - Triggers CombineAllChunks to create final video
8. CombineAllChunks merges all chunks into final_video.mp4
"""

import os
import json
import math
import hashlib
import torch
import torchaudio
import folder_paths
from server import PromptServer

# Whisper imports (using Hugging Face Transformers like existing nodes)
try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[VRGDG V4] Warning: Whisper (Transformers) not available. Transcription will be disabled.")

any_typ = "*"

# =============================================================================
# Helper Classes and Constants
# =============================================================================

class SafeDict(dict):
    """
    Dictionary that returns placeholder as-is for missing keys
    instead of raising KeyError. Used for safe template formatting.
    """
    def __missing__(self, key):
        return f"{{{key}}}"

# Default LLM instruction template with placeholders
DEFAULT_LLM_INSTRUCTIONS = """Generate a cinematic text-to-video prompt for a music video

You are a prompt engineer, aiming to write user inputs into high-quality prompts for music video generation.

Task requirements:
1. Reasonably infer and add details to make the video more complete and appealing without altering the original intent
2. Enhance the main features in user descriptions (appearance, expression, posture, etc.), visual style, spatial relationships, and shot scales
3. Output the entire prompt in English
4. Prompts should match the user's intent and accurately reflect the specified style
5. Emphasize motion information and different camera movements
6. Add natural actions of the target using simple and direct verbs
7. The prompt should be around 80-100 words long
8. PROMPT FORMULA: Subject + Scene + Motion + Camera Language + Atmosphere + Styling
9. Do not split the prompt in sections
10. DO NOT include component titles (Subject Description, Scene Description, etc.)
11. Do not use "+" to join parts
12. The generated prompt MUST be one single paragraph in natural English
13. Directly write the prompt without extra responses (no markdown, no commentary):

Prompt examples:
- The camera starts with a full screen of antique wooden screens, and slowly pans to the left, revealing an ancient-style girl sitting behind the screen. The girl is wearing Shu embroidered Hanfu, her hair is tied up high, and she is conducting an online video conference.
- A knight in shining armor stands by a medieval castle gate at dusk. He mounts a dragon and takes off into the sky as the camera pulls back. Cinematic lighting, glowing sunset clouds.
- A lone astronaut wanders through an alien forest at twilight. The camera tracks from behind through misty trees. Soft bioluminescent glow from plants lights the scene, creating a mysterious, awe-inspiring atmosphere.

VISUAL ELEMENTS (select one RANDOM entry from each category):
- Character: {required_CHARACTER}
- Environments: {optional_ENVIRONMENT}
- Lighting: {optional_LIGHTING}
- Camera Motion: {optional_CAMERA_MOTION}
- Physical Interactions: {optional_PHYSICAL_INTERACTION}
- Facial Expressions: {optional_FACIAL_EXPRESSION}
- Shot Types: {optional_SHOTS}
- Outfit: {optional_OUTFIT_RULES}



INFER THE OVERALL THEME, STORY, AND MOOD FROM THE FULL LYRICS: 
{internal_FULL_LYRICS}

THE PROMPT MUST BE RELEVANT TO THIS SECTION OF THE LYRICS:

"""

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
    - Handles chunks with no/minimal lyrics using placeholder words
    - Generates ALL scene prompts at once (leverages ComfyUI caching)
    - Creates audio hash for cache invalidation
    - Outputs pipe-separated lyrics and LLM instruction for external prompt generation
    - Outputs frames_per_chunk as a separate INT parameter (can be connected to video combine node)

    ComfyUI will cache this node's output if inputs don't change,
    so prompt generation happens ONCE per song, not per chunk.

    Placeholder words feature:
    - Automatically detects chunks with insufficient lyrics (below min_lyric_words threshold)
    - Replaces empty chunks with descriptive placeholders (e.g., "instrumental break")
    - Ensures LLM always has context to generate appropriate scene prompts
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
                "overlap_lyric_seconds": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Seconds of lyrics to include before/after each chunk for LLM context",
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Fixed seed for deterministic LLM prompt generation",
                }),
                "placeholder_words": ("STRING", {
                    "multiline": True,
                    "default": "instrumental break, music interlude, atmospheric moment",
                    "tooltip": "Comma-separated words/phrases to use for chunks with no/minimal lyrics. Gives LLM context for instrumental sections.",
                }),
                "min_lyric_words": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 20,
                    "tooltip": "Minimum word count to consider lyrics valid. Below this, placeholder words are used instead.",
                }),
            },
            "optional": {
                # LLM instruction template with placeholders
                "llm_instructions_template": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_LLM_INSTRUCTIONS,
                    "tooltip": "Template for LLM instructions. Use placeholders: {required_CHARACTER}, {optional_ENVIRONMENT}, {optional_LIGHTING}, {optional_CAMERA_MOTION}, {optional_PHYSICAL_INTERACTION}, {optional_FACIAL_EXPRESSION}, {optional_SHOTS}, {optional_OUTFIT_RULES}",
                }),
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
                    "default": "a white dress, a black dress",
                }),
            }
        }

    RETURN_TYPES = ("DICT", "INT", "STRING", "DICT", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("full_prompts", "total_chunks", "full_lyrics", "audio_meta", "audio_hash", "pipe_separated_lyrics", "llm_instructions", "frames_per_chunk")
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
        overlap_lyric_seconds,
        seed,
        placeholder_words,
        min_lyric_words,
        llm_instructions_template="",
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

        print("\n" + "="*80)
        print("[VRGDG V4] 🎵 Full Song Analyzer - Starting analysis...")
        print("[VRGDG V4] Parameters:")
        print(f"  - Character: {character_description[:50]}...")
        print(f"  - Theme: {song_theme_style[:50]}...")
        print(f"  - Language: {language}")
        print(f"  - Transcription: {'enabled' if enable_full_transcription else 'disabled'}")
        print(f"  - Overlap context: {overlap_lyric_seconds}s")
        print(f"  - Seed: {seed}")
        print("="*80)

        # Extract audio data
        print("[VRGDG V4] Step 1/5: Extracting audio data...")
        try:
            waveform = audio["waveform"]  # shape: [channels, samples]
            sample_rate = audio["sample_rate"]
            print(f"[VRGDG V4]   ✓ Audio shape: {waveform.shape}, Sample rate: {sample_rate}Hz")
        except Exception as e:
            raise ValueError(f"Invalid audio input: {e}")

        # Calculate audio metadata
        print("[VRGDG V4] Step 2/5: Calculating audio metadata...")
        num_samples = waveform.shape[-1]
        audio_duration = num_samples / sample_rate

        # Calculate audio hash (for caching)
        try:
            sample_data = waveform[..., :sample_rate].cpu().numpy().tobytes()  # First 1 second
            audio_hash = hashlib.md5(sample_data).hexdigest()[:16]
        except Exception:
            audio_hash = "unknown"

        print(f"[VRGDG V4]   ✓ Duration: {audio_duration:.2f}s | Hash: {audio_hash}")

        # Calculate total chunks needed
        print("[VRGDG V4] Step 3/5: Calculating chunk breakdown...")
        fps = 25
        frames_per_chunk = 97  # HuMo optimal (4n+1 format)

        # Use scene_duration_seconds to calculate frames if it differs from default
        expected_duration = frames_per_chunk / fps  # 3.88 seconds
        if abs(scene_duration_seconds - expected_duration) > 0.01:
            print(f"[VRGDG V4]   ⚠️ scene_duration_seconds ({scene_duration_seconds}s) differs from default ({expected_duration}s)")
            print(f"[VRGDG V4]   Recalculating frames_per_chunk to match requested duration...")
            frames_per_chunk = int(scene_duration_seconds * fps)
            print(f"[VRGDG V4]   New frames_per_chunk: {frames_per_chunk}")

        seconds_per_chunk = frames_per_chunk / fps

        total_chunks = math.ceil(audio_duration / seconds_per_chunk)

        print(f"[VRGDG V4]   ✓ FPS: {fps}")
        print(f"[VRGDG V4]   ✓ Frames per chunk: {frames_per_chunk}")
        print(f"[VRGDG V4]   ✓ Seconds per chunk: {seconds_per_chunk:.3f}s")
        print(f"[VRGDG V4]   ✓ Chunks needed: {total_chunks} (@ {seconds_per_chunk:.2f}s each)")
        print(f"[VRGDG V4]   ✓ Total video duration: ~{total_chunks * seconds_per_chunk:.2f}s")

        # Transcribe full audio if enabled
        full_lyrics = ""
        chunk_transcriptions = []

        print("[VRGDG V4] Step 4/5: Transcription phase...")
        if enable_full_transcription and WHISPER_AVAILABLE:
            print(f"[VRGDG V4] 🎤 Transcribing full audio (language: {language})...")

            try:
                # Load Whisper model (using Transformers)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[VRGDG V4]   Loading Whisper model on {device}...")
                model_name = "openai/whisper-large-v3"
                processor = WhisperProcessor.from_pretrained(model_name)
                model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device).eval()
                print("[VRGDG V4]   ✓ Model loaded")

                # Convert audio to format Whisper expects
                # Whisper expects mono audio at 16kHz
                print("[VRGDG V4]   Converting audio to mono @ 16kHz...")
                print(f"[VRGDG V4]   Original waveform shape: {waveform.shape}, sample_rate: {sample_rate}Hz")
                target_sr = 16000

                # Convert to mono
                if waveform.dim() == 2:
                    # Shape is [channels, samples]
                    waveform_mono = waveform.mean(dim=0)  # Convert to mono
                elif waveform.dim() == 3:
                    # Shape is [batch, channels, samples]
                    waveform_mono = waveform.squeeze(0).mean(dim=0)
                else:
                    # Already mono
                    waveform_mono = waveform.squeeze()

                print(f"[VRGDG V4]   After mono conversion: {waveform_mono.shape}")

                # Resample if needed
                if sample_rate != target_sr:
                    print(f"[VRGDG V4]   Resampling from {sample_rate}Hz to {target_sr}Hz...")
                    waveform_mono = torchaudio.functional.resample(waveform_mono, sample_rate, target_sr)
                    print(f"[VRGDG V4]   After resampling: {waveform_mono.shape}")

                print(f"[VRGDG V4]   ✓ Audio prepared: {waveform_mono.shape} ({waveform_mono.size(0)} samples = {waveform_mono.size(0)/target_sr:.2f}s)")

                # Transcribe full audio in chunks (30 second max)
                print("[VRGDG V4]   Starting full audio transcription...")
                max_length = target_sr * 30  # 30 seconds
                total_len = waveform_mono.size(0)
                audio_chunks = [waveform_mono[i:i + max_length] for i in range(0, total_len, max_length)]

                full_transcriptions = []
                for i, chunk in enumerate(audio_chunks):
                    print(f"[VRGDG V4]     Processing audio segment {i+1}/{len(audio_chunks)}...", end="\r")

                    # Pad short chunks if using auto language detection
                    if language == "auto" and chunk.size(0) < max_length:
                        pad_len = max_length - chunk.size(0)
                        chunk = torch.nn.functional.pad(chunk, (0, pad_len))

                    inputs = processor(
                        chunk.cpu().numpy(),
                        sampling_rate=target_sr,
                        return_tensors="pt",
                        padding="longest",
                        truncation=False
                    )
                    input_features = inputs["input_features"].to(device)

                    # Generate transcription
                    if language == "auto":
                        generated_ids = model.generate(input_features)
                    else:
                        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
                        generated_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

                    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    full_transcriptions.append(transcription)

                full_lyrics = " ".join(full_transcriptions).strip()
                print(f"\n[VRGDG V4]   ✅ Full transcription complete ({len(full_lyrics)} chars)")

                # Also transcribe each chunk individually for per-chunk metadata
                # Use overlap_lyric_seconds to include context before/after each chunk
                print(f"[VRGDG V4]   Starting per-chunk transcription ({total_chunks} chunks)...")
                print(f"[VRGDG V4]   Overlap context: ±{overlap_lyric_seconds}s")

                samples_per_chunk_resampled = int(frames_per_chunk * target_sr / fps)
                overlap_samples = int(overlap_lyric_seconds * target_sr)

                print(f"[VRGDG V4]   Samples per chunk: {samples_per_chunk_resampled}")
                print(f"[VRGDG V4]   Overlap samples: {overlap_samples}")

                for idx in range(total_chunks):
                    try:
                        # Calculate chunk boundaries
                        chunk_start = idx * samples_per_chunk_resampled
                        chunk_end = chunk_start + samples_per_chunk_resampled

                        # Extend boundaries by overlap for transcription (to get context)
                        trans_start = max(0, chunk_start - overlap_samples)
                        trans_end = min(waveform_mono.size(0), chunk_end + overlap_samples)

                        chunk_audio = waveform_mono[trans_start:trans_end]

                        print(f"[VRGDG V4]     Chunk {idx+1}/{total_chunks}: samples {trans_start}-{trans_end} ({chunk_audio.size(0)} samples)")

                        if chunk_audio.size(0) > 0:
                            # Pad if too short
                            if language == "auto" and chunk_audio.size(0) < max_length:
                                pad_len = max_length - chunk_audio.size(0)
                                chunk_audio = torch.nn.functional.pad(chunk_audio, (0, pad_len))

                            inputs = processor(
                                chunk_audio.cpu().numpy(),
                                sampling_rate=target_sr,
                                return_tensors="pt",
                                padding="longest",
                                truncation=False
                            )
                            input_features = inputs["input_features"].to(device)

                            if language == "auto":
                                generated_ids = model.generate(input_features)
                            else:
                                forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
                                generated_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

                            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                            transcription = transcription.strip()
                            chunk_transcriptions.append(transcription)

                            # Show result for debugging
                            preview = transcription[:40] if transcription else "[empty]"
                            print(f"[VRGDG V4]       → Result: \"{preview}...\" ({len(transcription.split())} words)")
                        else:
                            print(f"[VRGDG V4]       → Skipped (no audio)")
                            chunk_transcriptions.append("")

                    except Exception as e:
                        print(f"[VRGDG V4]       ⚠️ Error transcribing chunk {idx+1}: {e}")
                        chunk_transcriptions.append("")

                print(f"\n[VRGDG V4]   ✅ Transcribed {len(chunk_transcriptions)} individual chunks (with context)")

                # Show word count stats to verify overlap is working
                if chunk_transcriptions:
                    word_counts = [len(t.split()) for t in chunk_transcriptions if t]
                    if word_counts:
                        avg_words = sum(word_counts) / len(word_counts)
                        print(f"[VRGDG V4]   📊 Average words per chunk: {avg_words:.1f} (includes overlap context)")
                        print(f"[VRGDG V4]   📊 Word count range: {min(word_counts)}-{max(word_counts)} words")

                        # Show preview of first 3 chunks to verify split
                        print(f"[VRGDG V4]   📝 Chunk lyrics preview:")
                        for i in range(min(3, len(chunk_transcriptions))):
                            preview = chunk_transcriptions[i][:60] if chunk_transcriptions[i] else "[empty]"
                            word_count = len(chunk_transcriptions[i].split()) if chunk_transcriptions[i] else 0
                            print(f"[VRGDG V4]      Chunk {i}: \"{preview}...\" ({word_count} words)")

            except Exception as e:
                print(f"[VRGDG V4] ⚠️ Transcription failed: {e}")
                import traceback
                traceback.print_exc()
                full_lyrics = f"[Transcription failed: {str(e)}]"
                chunk_transcriptions = [""] * total_chunks
        else:
            print("[VRGDG V4] ℹ️ Transcription disabled or Whisper not available")
            chunk_transcriptions = ["[lyrics not transcribed]"] * total_chunks

        # Generate prompts for ALL chunks
        # Note: In a real implementation, this would call the LLM to generate prompts
        # For now, we'll create a placeholder structure

        print("[VRGDG V4] Step 5/5: Prompt generation phase...")
        print(f"[VRGDG V4] 🎨 Generating {total_chunks} scene prompts...")

        # Process placeholder words for empty/minimal lyrics
        print("[VRGDG V4]   Processing placeholder words for empty chunks...")
        print(f"[VRGDG V4]   Min lyric words threshold: {min_lyric_words}")
        print(f"[VRGDG V4]   Placeholder options: {placeholder_words}")

        # Parse placeholder words into a list
        placeholder_list = [p.strip() for p in placeholder_words.split(",") if p.strip()]
        if not placeholder_list:
            placeholder_list = ["instrumental break"]  # Fallback

        # Process chunk transcriptions - replace empty/minimal with placeholders
        processed_chunk_lyrics = []
        empty_chunk_count = 0

        import random
        random.seed(seed)  # Use seed for deterministic placeholder selection

        for idx, transcription in enumerate(chunk_transcriptions):
            # Count words in transcription
            word_count = len(transcription.split()) if transcription else 0

            if word_count < min_lyric_words:
                # Use placeholder - cycle through or pick randomly based on index
                placeholder = placeholder_list[idx % len(placeholder_list)]
                processed_chunk_lyrics.append(placeholder)
                empty_chunk_count += 1

                if idx < 5:  # Show first few replacements
                    print(f"[VRGDG V4]     Chunk {idx}: Empty ({word_count} words) → \"{placeholder}\"")
            else:
                # Keep original transcription
                processed_chunk_lyrics.append(transcription)

        if empty_chunk_count > 0:
            print(f"[VRGDG V4]   ✓ Replaced {empty_chunk_count}/{total_chunks} empty chunks with placeholders")
        else:
            print(f"[VRGDG V4]   ✓ All chunks have sufficient lyrics (no placeholders needed)")

        # Create pipe-separated lyrics for prompt generation
        print("[VRGDG V4]   Preparing lyrics for prompt generation...")
        if processed_chunk_lyrics:
            pipe_separated_lyrics = " | ".join(processed_chunk_lyrics)
        else:
            pipe_separated_lyrics = full_lyrics
        print(f"[VRGDG V4]   ✓ Lyrics prepared ({len(pipe_separated_lyrics)} chars)")

        # Build LLM instructions using template with placeholder replacement
        print("[VRGDG V4]   Building LLM instructions from template...")

        # Build parameter mapping for placeholder replacement
        llm_params = {
            'required_CHARACTER': character_description,
            'optional_ENVIRONMENT': environment,
            'optional_LIGHTING': lighting,
            'optional_CAMERA_MOTION': camera_motion,
            'optional_PHYSICAL_INTERACTION': physical_interaction,
            'optional_FACIAL_EXPRESSION': facial_expression,
            'optional_SHOTS': shots,
            'optional_OUTFIT_RULES': outfit_rules,
            'internal_FULL_LYRICS': full_lyrics,
        }

        # Use default template if none provided or if empty
        template = llm_instructions_template if llm_instructions_template and llm_instructions_template.strip() else DEFAULT_LLM_INSTRUCTIONS

        # Replace placeholders in template
        llm_instructions = self._replace_llm_placeholders(template, llm_params)

        print(f"[VRGDG V4]   ✓ Instructions built from template ({len(llm_instructions)} chars)")
        print(f"[VRGDG V4]   Template placeholders replaced: {list(llm_params.keys())}")

        # Generate WAN2.1 compatible prompts
        print("[VRGDG V4]   Generating WAN2.1 compatible prompts...")
        print(f"[VRGDG V4]   Using seed {seed} for deterministic randomness")

        # Parse all visual options once before loop
        import random
        random_gen = random.Random(seed)  # Deterministic randomness

        env_opts = self._parse_visual_options(environment)
        light_opts = self._parse_visual_options(lighting)
        camera_opts = self._parse_visual_options(camera_motion)
        phys_opts = self._parse_visual_options(physical_interaction)
        face_opts = self._parse_visual_options(facial_expression)
        shot_opts = self._parse_visual_options(shots)
        outfit_opts = self._parse_visual_options(outfit_rules)

        print(f"[VRGDG V4]   Visual options parsed:")
        print(f"[VRGDG V4]     - Environments: {len(env_opts)} options")
        print(f"[VRGDG V4]     - Lighting: {len(light_opts)} options")
        print(f"[VRGDG V4]     - Camera motions: {len(camera_opts)} options")
        print(f"[VRGDG V4]     - Physical interactions: {len(phys_opts)} options")
        print(f"[VRGDG V4]     - Facial expressions: {len(face_opts)} options")
        print(f"[VRGDG V4]     - Shot types: {len(shot_opts)} options")
        print(f"[VRGDG V4]     - Outfits: {len(outfit_opts)} options")

        generated_prompts = []
        for idx in range(total_chunks):
            # Get chunk lyrics (with placeholders for empty chunks)
            chunk_lyrics = processed_chunk_lyrics[idx] if idx < len(processed_chunk_lyrics) else ''

            # Build WAN2.1 compatible prompt
            prompt = self._build_wan21_prompt(
                character_description=character_description,
                song_theme_style=song_theme_style,
                environment_options=env_opts,
                lighting_options=light_opts,
                camera_motion_options=camera_opts,
                physical_interaction_options=phys_opts,
                facial_expression_options=face_opts,
                shots_options=shot_opts,
                outfit_options=outfit_opts,
                chunk_lyrics=chunk_lyrics,
                random_gen=random_gen,
            )

            generated_prompts.append(prompt)
            if (idx + 1) % 10 == 0 or idx == total_chunks - 1:
                print(f"[VRGDG V4]     Generated {idx+1}/{total_chunks} prompts...")

        print(f"[VRGDG V4]   ✅ Generated {len(generated_prompts)} WAN2.1 prompts")

        # Show preview of first 3 prompts to verify variety
        print(f"[VRGDG V4]   📝 Prompt preview (first 3):")
        for i in range(min(3, len(generated_prompts))):
            preview = generated_prompts[i][:120] if generated_prompts[i] else "[empty]"
            print(f"[VRGDG V4]      Prompt {i}: \"{preview}...\"")
            # Show word count
            word_count = len(generated_prompts[i].split())
            print(f"[VRGDG V4]               ({word_count} words)")

        # Package everything into output dict
        full_prompts = {
            "prompts": generated_prompts,
            "total_chunks": total_chunks,
            "audio_hash": audio_hash,
            "audio_duration": audio_duration,
            "chunk_transcriptions": chunk_transcriptions,  # Original transcriptions
            "processed_chunk_lyrics": processed_chunk_lyrics,  # Transcriptions with placeholders applied
            "llm_instructions": llm_instructions,  # Save for reference
            "empty_chunk_count": empty_chunk_count,  # How many chunks had placeholders
        }

        # Calculate samples per chunk
        samples_per_chunk_calculated = int(frames_per_chunk * sample_rate / fps)

        audio_meta = {
            "sample_rate": sample_rate,
            "num_samples": num_samples,
            "duration": audio_duration,
            "frames_per_chunk": frames_per_chunk,
            "samples_per_chunk": samples_per_chunk_calculated,
            "fps": fps,  # Add fps for debugging
        }

        # Debug logging for audio_meta
        print(f"[VRGDG V4] Audio metadata created:")
        print(f"  - sample_rate: {sample_rate} Hz")
        print(f"  - fps: {fps}")
        print(f"  - frames_per_chunk: {frames_per_chunk}")
        print(f"  - samples_per_chunk: {samples_per_chunk_calculated}")
        print(f"  - Expected chunk duration: {samples_per_chunk_calculated / sample_rate:.3f}s")

        print("="*80)
        print(f"[VRGDG V4] ✅ Analysis complete! Ready for chunked processing.")
        print(f"[VRGDG V4] Summary:")
        print(f"  - Total chunks: {total_chunks}")
        print(f"  - Audio duration: {audio_duration:.2f}s")
        print(f"  - Prompts generated: {len(generated_prompts)}")
        print(f"  - Transcriptions per chunk: {len(chunk_transcriptions)}")
        print(f"  - Empty chunks (using placeholders): {empty_chunk_count}")
        print(f"  - Audio hash: {audio_hash}")
        print(f"  - Frames per chunk: {frames_per_chunk} (output as separate parameter)")
        print(f"[VRGDG V4] Data structure:")
        print(f"  - full_prompts['prompts'][i] = prompt for chunk i")
        print(f"  - full_prompts['chunk_transcriptions'][i] = original lyrics for chunk i")
        print(f"  - full_prompts['processed_chunk_lyrics'][i] = lyrics with placeholders applied")
        print(f"[VRGDG V4] Outputs:")
        print(f"  - pipe_separated_lyrics: {len(pipe_separated_lyrics)} chars (includes placeholders)")
        print(f"  - llm_instructions: {len(llm_instructions)} chars")
        print(f"  - frames_per_chunk: {frames_per_chunk} frames (can be connected to video combine node)")
        print("="*80 + "\n")

        return (full_prompts, total_chunks, full_lyrics, audio_meta, audio_hash, pipe_separated_lyrics, llm_instructions, frames_per_chunk)

    def _replace_llm_placeholders(self, template, params):
        """
        Replace visual parameter placeholders in LLM instruction template.

        Args:
            template: String with placeholders like {required_CHARACTER}, {optional_ENVIRONMENT}
            params: Dict mapping placeholder names to values

        Returns:
            Template with placeholders replaced

        Example:
            template = "Character: {required_CHARACTER}, Scene: {optional_ENVIRONMENT}"
            params = {'required_CHARACTER': 'A man', 'optional_ENVIRONMENT': 'open field'}
            result = "Character: A man, Scene: open field"
        """
        safe_params = SafeDict(params)
        return template.format_map(safe_params)

    def _parse_visual_options(self, option_string):
        """
        Parse comma-separated options into a list.
        Returns empty list if string is empty.

        Example: "zoom in, zoom out, tilt down" -> ["zoom in", "zoom out", "tilt down"]
        """
        if not option_string or not option_string.strip():
            return []
        return [opt.strip() for opt in option_string.split(',') if opt.strip()]

    def _build_wan21_prompt(
        self,
        character_description,
        song_theme_style,
        environment_options,
        lighting_options,
        camera_motion_options,
        physical_interaction_options,
        facial_expression_options,
        shots_options,
        outfit_options,
        chunk_lyrics,
        random_gen,
    ):
        """
        Build WAN2.1 compatible prompt using the formula:
        Subject + Scene + Motion + Camera + Atmosphere + Styling

        Each chunk gets random selections from visual parameter options.
        """

        parts = []

        # 1. SUBJECT (Subject Description)
        # Start with character base
        subject = character_description

        # Add outfit if available
        if outfit_options:
            outfit = random_gen.choice(outfit_options)
            subject = f"{subject} in {outfit}"

        parts.append(subject)

        # 2. SCENE (Scene Description)
        scene_parts = []

        # Add environment
        if environment_options:
            env = random_gen.choice(environment_options)
            scene_parts.append(f"in {env}" if not env.startswith(("in ", "at ", "on ")) else env)

        if scene_parts:
            parts.append(" ".join(scene_parts))

        # 3. MOTION (Motion Description)
        if physical_interaction_options:
            motion = random_gen.choice(physical_interaction_options)
            parts.append(motion)

        # 4. CAMERA LANGUAGE
        camera_parts = []

        # Add shot type
        if shots_options:
            shot = random_gen.choice(shots_options)
            camera_parts.append(shot)

        # Add camera motion
        if camera_motion_options:
            cam_motion = random_gen.choice(camera_motion_options)
            camera_parts.append(cam_motion)

        if camera_parts:
            camera_text = ", ".join(camera_parts)
            parts.append(f"The camera {camera_text}")

        # 5. ATMOSPHERE (Emotion/Expression)
        if facial_expression_options:
            expression = random_gen.choice(facial_expression_options)
            parts.append(f"capturing {expression}")
        elif lighting_options:
            # Use lighting as atmosphere if no expression
            light = random_gen.choice(lighting_options)
            parts.append(f"with {light}")

        # 6. STYLING
        # Parse song_theme_style in case it has multiple options
        style_options = self._parse_visual_options(song_theme_style)
        if style_options:
            style = random_gen.choice(style_options)
            parts.append(f"The atmosphere is {style}")
        elif song_theme_style:
            # Use as-is if not comma-separated
            parts.append(f"The atmosphere is {song_theme_style}")

        # Join all parts into a natural sentence
        # Use proper punctuation
        if len(parts) == 0:
            return f"{character_description} in a cinematic scene"

        # Build the prompt
        prompt = parts[0]  # Subject

        if len(parts) > 1:
            # Add scene and motion with commas
            middle_parts = []
            for i, part in enumerate(parts[1:], 1):
                if part.startswith("The camera") or part.startswith("capturing") or part.startswith("with ") or part.startswith("The atmosphere"):
                    # These start new sentences
                    break
                else:
                    middle_parts.append(part)

            if middle_parts:
                prompt += " " + ", ".join(middle_parts)

            # Add camera and atmosphere parts
            remaining_start = 1 + len(middle_parts)
            for part in parts[remaining_start:]:
                if part.startswith("The camera") or part.startswith("The atmosphere"):
                    prompt += ". " + part
                else:
                    prompt += ", " + part

        # Ensure it ends with a period
        if not prompt.endswith('.'):
            prompt += '.'

        return prompt


# =============================================================================
# Node 2: VRGDG_ChunkIndexController
# =============================================================================

class VRGDG_ChunkIndexController:
    """
    Manages which chunk to process - supports both auto-detection and manual override.

    Auto Mode: Counts existing video_chunk_*.done files to determine current index
    Manual Mode: Uses provided manual_index parameter

    Folder change detection:
    - Tracks the last used output folder
    - When folder changes, resets to index 0 (once per folder change)
    - Helps when switching between different songs/projects
    """

    # Class variable to track last folder for change detection
    _last_folder = None

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

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Force re-execution every time by returning a unique value.
        This prevents ComfyUI from caching the node's output.
        Without this, the index stays at 0 and auto-queueing fails.
        """
        import time
        return float(time.time())

    def get_index(self, full_prompts, output_folder, auto_index, manual_index):
        """
        Determine which chunk to process based on mode.
        Detects folder changes to reset index.
        """

        print("\n" + "="*80)
        print("[VRGDG V4] 🎯 Chunk Index Controller - Determining chunk to process...")
        print(f"[VRGDG V4] Mode: {'AUTO' if auto_index else 'MANUAL'}")
        print(f"[VRGDG V4] Output folder (raw): {output_folder}")

        # Resolve output folder to absolute path for consistent comparison
        if not os.path.isabs(output_folder):
            output_base = folder_paths.get_output_directory()
            output_folder_abs = os.path.join(output_base, output_folder)
        else:
            output_folder_abs = output_folder

        print(f"[VRGDG V4] Output folder (absolute): {output_folder_abs}")

        # Detect folder change
        folder_changed = False
        if VRGDG_ChunkIndexController._last_folder is None:
            print(f"[VRGDG V4] First run - initializing folder tracking")
            VRGDG_ChunkIndexController._last_folder = output_folder_abs
        elif VRGDG_ChunkIndexController._last_folder != output_folder_abs:
            folder_changed = True
            print(f"[VRGDG V4] 🔄 FOLDER CHANGE DETECTED!")
            print(f"[VRGDG V4]   Previous: {VRGDG_ChunkIndexController._last_folder}")
            print(f"[VRGDG V4]   Current:  {output_folder_abs}")
            print(f"[VRGDG V4]   → Resetting index to 0")
            VRGDG_ChunkIndexController._last_folder = output_folder_abs

        total_chunks = full_prompts.get("total_chunks", 0)
        print(f"[VRGDG V4] Total chunks in song: {total_chunks}")

        if auto_index:
            # Auto-detect index by counting existing files
            if folder_changed:
                # Folder changed, start from 0 (will count files to verify)
                print("[VRGDG V4] Folder changed - starting fresh from index 0")
                current_index = 0
                # Still check if there are any completed chunks in new folder
                completed_count = self._count_completed_chunks(output_folder_abs)
                if completed_count > 0:
                    current_index = completed_count
                    print(f"[VRGDG V4]   Found {completed_count} existing chunks in new folder")
            else:
                # Normal auto-detection
                print("[VRGDG V4] Scanning output folder for completed chunks...")
                current_index = self._count_completed_chunks(output_folder_abs)
                print(f"[VRGDG V4]   ✓ Found {current_index} completed chunks")
            mode_str = "AUTO"
        else:
            # Use manual index
            current_index = manual_index
            mode_str = "MANUAL"
            print(f"[VRGDG V4]   Using manual index: {current_index}")

        # Calculate remaining chunks
        remaining = total_chunks - current_index - 1
        print(f"[VRGDG V4] Current index: {current_index}")
        print(f"[VRGDG V4] Remaining after this: {remaining}")

        # Should we auto-queue the next chunk?
        should_auto_queue = auto_index and (current_index < total_chunks)

        # Build status message
        if current_index >= total_chunks:
            status_message = f"✅ All {total_chunks} chunks complete!"
            should_auto_queue = False
            print(f"[VRGDG V4] Status: ALL CHUNKS COMPLETE!")
        else:
            status_message = f"[{mode_str}] Processing chunk {current_index + 1}/{total_chunks} ({remaining} remaining)"
            print(f"[VRGDG V4] Status: Processing chunk {current_index + 1}/{total_chunks}")
            print(f"[VRGDG V4] Auto-queue next: {should_auto_queue}")

        print("="*80 + "\n")

        return (current_index, total_chunks, remaining, should_auto_queue, status_message)

    def _count_completed_chunks(self, folder_path):
        """
        Count completed chunks by looking for video_chunk_*.done marker files.
        Falls back to counting .mp4 files if no .done markers found (for compatibility).
        """
        if not os.path.isdir(folder_path):
            print(f"[VRGDG V4]   ℹ️ Folder doesn't exist yet: {folder_path}")
            return 0

        try:
            print(f"[VRGDG V4]   Scanning folder: {folder_path}")

            # List all files for debugging
            all_files = os.listdir(folder_path)
            print(f"[VRGDG V4]   Total files in folder: {len(all_files)}")
            if len(all_files) > 0 and len(all_files) <= 20:
                print(f"[VRGDG V4]   All files: {', '.join(sorted(all_files))}")

            # Look for .done marker files (new format)
            marker_files = [
                f for f in all_files
                if f.startswith("video_chunk_") and f.endswith(".done")
            ]

            print(f"[VRGDG V4]   .done marker files found: {len(marker_files)}")

            # Fallback: count .mp4 files if no .done markers found (backwards compatibility)
            if len(marker_files) == 0:
                marker_files = [
                    f for f in all_files
                    if f.startswith("video_chunk_") and f.endswith(".mp4")
                ]
                if len(marker_files) > 0:
                    print(f"[VRGDG V4]   Using .mp4 files for counting (legacy mode)")
                    print(f"[VRGDG V4]   .mp4 chunk files found: {len(marker_files)}")

            count = len(marker_files)
            if count > 0:
                print(f"[VRGDG V4]   Marker files: {', '.join(sorted(marker_files)[:5])}" + ("..." if len(marker_files) > 5 else ""))
            else:
                print(f"[VRGDG V4]   No marker files found - will start from index 0")

            print(f"[VRGDG V4]   Total completed chunks: {count}")
            print(f"[VRGDG V4]   Next chunk index will be: {count}")
            return count
        except Exception as e:
            print(f"[VRGDG V4] ⚠️ Error counting chunks: {e}")
            import traceback
            traceback.print_exc()
            return 0


# =============================================================================
# Node 3: VRGDG_GetPromptByIndex
# =============================================================================

class VRGDG_GetPromptByIndex:
    """
    Simple prompt extractor - gets a single prompt from the cached full_prompts dict.

    Outputs:
    - prompt: The generated prompt for this chunk
    - transcription: Original lyrics transcribed from audio (may be empty)
    - processed_lyrics: Lyrics with placeholder words applied for empty chunks
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

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "transcription", "processed_lyrics", "llm_instructions")
    FUNCTION = "get_prompt"
    CATEGORY = "VRGDG/V4 Single Chunk"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Force re-execution when index changes.
        """
        index = kwargs.get('index', 0)
        return float(index)

    def get_prompt(self, full_prompts, index):
        """
        Extract single prompt by index.
        Returns both original transcription and processed lyrics (with placeholders).
        Also returns the full LLM instructions for reference.
        """

        print("\n" + "="*80)
        print("[VRGDG V4] 📝 Get Prompt By Index - Extracting prompt...")
        print(f"[VRGDG V4] Requested index: {index}")

        prompts = full_prompts.get("prompts", [])
        transcriptions = full_prompts.get("chunk_transcriptions", [])
        processed_lyrics = full_prompts.get("processed_chunk_lyrics", [])
        llm_instructions = full_prompts.get("llm_instructions", "")
        print(f"[VRGDG V4] Available prompts: {len(prompts)}")

        if index < 0 or index >= len(prompts):
            error_msg = f"❌ Index {index} out of range (0-{len(prompts)-1})"
            print(f"[VRGDG V4] {error_msg}")
            print("="*80 + "\n")
            return (error_msg, "", "", "")

        prompt = prompts[index]
        transcription = transcriptions[index] if index < len(transcriptions) else ""
        processed = processed_lyrics[index] if index < len(processed_lyrics) else transcription

        print(f"[VRGDG V4]   ✓ Extracted prompt for chunk #{index}")
        print(f"[VRGDG V4]   ✓ Prompt length: {len(prompt)} chars")

        if transcription:
            word_count = len(transcription.split())
            print(f"[VRGDG V4]   ✓ Original lyrics: {word_count} words")
            print(f"[VRGDG V4]   ✓ Original preview: \"{transcription[:60]}...\"")
        else:
            print(f"[VRGDG V4]   ✓ No original transcription")

        if processed != transcription:
            print(f"[VRGDG V4]   ✓ Processed lyrics (placeholder): \"{processed}\"")
        else:
            print(f"[VRGDG V4]   ✓ Processed lyrics: Same as original")

        if llm_instructions:
            print(f"[VRGDG V4]   ✓ LLM instructions: {len(llm_instructions)} chars")

        print(f"[VRGDG V4] Prompt preview: \"{prompt[:100]}...\"")
        print("="*80 + "\n")

        return (prompt, transcription, processed, llm_instructions)


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

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Force re-execution when index changes.
        """
        index = kwargs.get('index', 0)
        return float(index)

    def load_chunk(self, audio, index, audio_meta):
        """
        Load a single audio chunk by index.
        """

        print("\n" + "="*80)
        print("[VRGDG V4] 🎵 Load Single Audio Chunk - Extracting audio...")
        print(f"[VRGDG V4] Requested chunk index: {index}")

        try:
            waveform = audio["waveform"]  # shape: [channels, samples]
            sample_rate = audio["sample_rate"]
            print(f"[VRGDG V4]   ✓ Audio loaded: {waveform.shape} @ {sample_rate}Hz")
        except Exception as e:
            raise ValueError(f"Invalid audio input: {e}")

        # Get chunk parameters from audio_meta
        print(f"[VRGDG V4] Audio metadata received:")
        print(f"  - Keys in audio_meta: {list(audio_meta.keys())}")
        for key, value in audio_meta.items():
            print(f"  - {key}: {value}")

        meta_sample_rate = audio_meta.get("sample_rate", 0)
        frames_per_chunk = audio_meta.get("frames_per_chunk", 97)
        fps = audio_meta.get("fps", 25)

        # CRITICAL: Check if sample rates match
        if meta_sample_rate != sample_rate:
            print(f"[VRGDG V4] ⚠️ SAMPLE RATE MISMATCH DETECTED!")
            print(f"[VRGDG V4]   audio_meta sample_rate: {meta_sample_rate} Hz")
            print(f"[VRGDG V4]   Actual audio sample_rate: {sample_rate} Hz")
            print(f"[VRGDG V4]   This will cause incorrect chunk durations!")
            print(f"[VRGDG V4]   Recalculating samples_per_chunk using ACTUAL sample rate...")

        # ALWAYS recalculate samples_per_chunk using the ACTUAL audio sample rate
        # This ensures correct duration even if audio was resampled between nodes
        samples_per_chunk = int(frames_per_chunk * sample_rate / fps)

        print(f"[VRGDG V4] Chunk calculation:")
        print(f"  - Formula: frames_per_chunk * sample_rate / fps")
        print(f"  - Calculation: {frames_per_chunk} * {sample_rate} / {fps} = {samples_per_chunk}")

        print(f"[VRGDG V4] Final chunk parameters:")
        print(f"  - FPS: {fps}")
        print(f"  - Frames per chunk: {frames_per_chunk}")
        print(f"  - Samples per chunk: {samples_per_chunk}")
        print(f"  - Sample rate (ACTUAL): {sample_rate} Hz")
        print(f"  - Expected duration: {samples_per_chunk / sample_rate:.3f}s")

        # Calculate sample range for this chunk
        start_sample = index * samples_per_chunk
        end_sample = start_sample + samples_per_chunk

        num_samples = waveform.shape[-1]
        print(f"[VRGDG V4] Sample range: {start_sample} to {end_sample} (total: {num_samples})")

        # Extract chunk
        if start_sample >= num_samples:
            # Beyond audio end - create silence
            print("[VRGDG V4] ⚠️ Beyond audio end - creating silence")
            print(f"[VRGDG V4]   waveform shape: {waveform.shape}")

            # Create silence with same shape as waveform (all dims except last)
            silence_shape = list(waveform.shape)
            silence_shape[-1] = samples_per_chunk  # Replace last dim with chunk size

            chunk_waveform = torch.zeros(
                silence_shape,
                dtype=waveform.dtype,
                device=waveform.device
            )
            print(f"[VRGDG V4]   silence shape: {chunk_waveform.shape}")
            actual_samples = 0
        else:
            # Extract available samples
            actual_end = min(end_sample, num_samples)
            chunk_waveform = waveform[..., start_sample:actual_end]
            actual_samples = chunk_waveform.shape[-1]
            print(f"[VRGDG V4] Extracted {actual_samples} samples")

            # Pad with silence if needed
            if actual_samples < samples_per_chunk:
                padding = samples_per_chunk - actual_samples
                print(f"[VRGDG V4] Padding with {padding} samples of silence")
                print(f"[VRGDG V4]   chunk_waveform shape: {chunk_waveform.shape}")

                # Create silence with same shape as chunk_waveform (all dims except last)
                silence_shape = list(chunk_waveform.shape)
                silence_shape[-1] = padding  # Replace last dim with padding size

                silence = torch.zeros(
                    silence_shape,
                    dtype=chunk_waveform.dtype,
                    device=chunk_waveform.device
                )
                print(f"[VRGDG V4]   silence shape: {silence.shape}")
                chunk_waveform = torch.cat([chunk_waveform, silence], dim=-1)
                print(f"[VRGDG V4]   padded shape: {chunk_waveform.shape}")

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
            "actual_duration": actual_samples / sample_rate,
            "samples_per_chunk": samples_per_chunk,
            "frames_per_chunk": frames_per_chunk,
            "fps": fps,
            "sample_rate": sample_rate,
        }

        print(f"[VRGDG V4] ✅ Chunk #{index} loaded:")
        print(f"  - Samples: {actual_samples}/{samples_per_chunk} ({'padded' if chunk_info['padded'] else 'full'})")
        print(f"  - Expected duration: {chunk_info['duration']:.3f}s")
        print(f"  - Actual duration: {chunk_info['actual_duration']:.3f}s")
        print(f"  - Calculation: {samples_per_chunk} samples / {sample_rate} Hz = {chunk_info['duration']:.3f}s")
        print("="*80 + "\n")

        return (chunk_audio, chunk_info)


# =============================================================================
# Node 5: VRGDG_SaveVideoChunkWithIndex
# =============================================================================

class VRGDG_SaveVideoChunkWithIndex:
    """
    Saves video chunk with index in filename.

    Can work in two modes:
    1. VHS mode: Auto-detects VHS_VideoCombine output and moves/renames files
    2. Direct mode: Takes images and audio, saves with ffmpeg (fallback)

    Creates files:
    1. video_chunk_0000.mp4 - The actual video (with audio)
    2. video_chunk_0000.png - Workflow metadata (if available)
    3. video_chunk_0000.done - Empty marker file for auto-indexing
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                }),
                "output_folder": ("STRING", {
                    "default": "video_output",
                    "multiline": False,
                }),
            },
            "optional": {
                "images": ("IMAGE",),  # Video frames (for direct save)
                "audio_chunk": ("AUDIO",),  # Audio (for direct save)
                "vhs_filenames": ("VHS_FILENAMES",),  # VHS output (for VHS mode)
                "fps": ("FLOAT", {
                    "default": 25.0,
                    "min": 1.0,
                    "max": 60.0,
                }),
            }
        }

    RETURN_TYPES = ("STRING", any_typ, "INT")
    RETURN_NAMES = ("saved_path", "signal", "index")
    FUNCTION = "save_chunk"
    CATEGORY = "VRGDG/V4 Single Chunk"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Force re-execution when index changes.
        """
        index = kwargs.get('index', 0)
        return float(index)

    def save_chunk(self, index, output_folder, images=None, audio_chunk=None, vhs_filenames=None, fps=25.0):
        """
        Save video chunk with index in filename.
        Supports two modes:
        1. VHS mode: Move/rename VHS-saved file
        2. Direct mode: Save video+audio with ffmpeg (fallback)
        """

        print("\n" + "="*80)
        print("[VRGDG V4] 💾 Save Video Chunk - Starting save process...")
        print(f"[VRGDG V4] Chunk index: {index}")
        print(f"[VRGDG V4] Output folder (raw): {output_folder}")
        print(f"[VRGDG V4] Mode: {'VHS' if vhs_filenames is not None else 'Direct'}")

        # Resolve output folder path
        # If relative path, make it relative to ComfyUI output directory
        if not os.path.isabs(output_folder):
            output_base = folder_paths.get_output_directory()
            output_folder = os.path.join(output_base, output_folder)
            print(f"[VRGDG V4] Resolved to absolute path: {output_folder}")

        # Create output folder if it doesn't exist
        print(f"[VRGDG V4] Ensuring output folder exists...")
        os.makedirs(output_folder, exist_ok=True)
        print(f"[VRGDG V4]   ✓ Folder ready: {output_folder}")

        # Generate filenames
        video_filename = f"video_chunk_{index:04d}.mp4"
        marker_filename = f"video_chunk_{index:04d}.done"  # Marker for completion tracking

        video_path = os.path.join(output_folder, video_filename)
        marker_path = os.path.join(output_folder, marker_filename)

        print(f"[VRGDG V4] Target files:")
        print(f"  - Video: {video_filename}")
        print(f"  - Marker: {marker_filename} (for auto-indexing)")

        try:
            # Choose mode based on available inputs
            # Priority: Direct mode (if we have images+audio) > VHS mode (fallback)

            import glob
            import shutil
            output_base = folder_paths.get_output_directory()

            # MODE 1: Direct mode - create video with ffmpeg (PREFERRED)
            if images is not None and audio_chunk is not None:
                print(f"[VRGDG V4] Using Direct mode (ffmpeg)...")

                import subprocess
                import numpy as np
                import tempfile

                # Save frames to temporary video file
                temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                temp_video_path = temp_video.name
                temp_video.close()

                # Save audio to temporary file
                temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_audio_path = temp_audio.name
                temp_audio.close()

                print(f"[VRGDG V4]   Converting {images.shape[0]} frames to video...")
                frames_np = (images.cpu().numpy() * 255).astype(np.uint8)

                # Save frames with imageio (no audio first)
                import imageio
                with imageio.get_writer(temp_video_path, fps=fps, codec='libx264', quality=8) as writer:
                    for i, frame in enumerate(frames_np):
                        writer.append_data(frame)
                        if (i + 1) % 25 == 0:
                            print(f"[VRGDG V4]     Progress: {i+1}/{len(frames_np)} frames", end="\r")
                print(f"\n[VRGDG V4]   ✓ Frames written to temp video")

                # Save audio
                print(f"[VRGDG V4]   Saving audio...")
                waveform = audio_chunk["waveform"]
                sample_rate = audio_chunk["sample_rate"]

                print(f"[VRGDG V4]     Original waveform shape: {waveform.shape}")

                # Use soundfile to save WAV (more reliable than torchaudio)
                # torchaudio has a bug that causes std::length_error
                try:
                    print(f"[VRGDG V4]     Attempting to save with soundfile library...")
                    import soundfile as sf

                    # Handle different waveform shapes
                    # Common formats: [channels, samples] or [batch, channels, samples]
                    save_waveform = waveform.cpu()

                    # Remove batch dimension if present
                    while save_waveform.dim() > 2:
                        save_waveform = save_waveform.squeeze(0)
                    print(f"[VRGDG V4]     After squeeze: {save_waveform.shape}")

                    # Ensure we have 2D: [channels, samples]
                    if save_waveform.dim() == 1:
                        save_waveform = save_waveform.unsqueeze(0)  # Add channel dimension

                    # soundfile expects [samples, channels], so transpose
                    audio_np = save_waveform.t().numpy()  # [samples, channels]
                    print(f"[VRGDG V4]     Audio shape for soundfile: {audio_np.shape} (samples, channels)")

                    sf.write(temp_audio_path, audio_np, sample_rate, subtype='PCM_16')
                    print(f"[VRGDG V4]   ✓ Audio saved to temp file (soundfile)")

                except ImportError:
                    print(f"[VRGDG V4]     soundfile not available, falling back to torchaudio")
                    try:
                        # Prepare waveform for torchaudio (expects [channels, samples])
                        save_waveform = waveform.cpu()
                        while save_waveform.dim() > 2:
                            save_waveform = save_waveform.squeeze(0)
                        if save_waveform.dim() == 1:
                            save_waveform = save_waveform.unsqueeze(0)

                        torchaudio.save(temp_audio_path, save_waveform, sample_rate)
                        print(f"[VRGDG V4]   ✓ Audio saved to temp file (torchaudio)")
                    except Exception as e:
                        print(f"[VRGDG V4]     ❌ torchaudio.save failed: {e}")
                        import traceback
                        traceback.print_exc()
                        raise
                except Exception as e:
                    print(f"[VRGDG V4]     ❌ soundfile.write failed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise

                # Combine video + audio with ffmpeg
                print(f"[VRGDG V4]   Combining video and audio with ffmpeg...")
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-y',  # Overwrite output
                    '-i', temp_video_path,  # Video input
                    '-i', temp_audio_path,  # Audio input
                    '-c:v', 'copy',  # Copy video codec
                    '-c:a', 'aac',  # Audio codec
                    '-b:a', '192k',  # Audio bitrate
                    '-shortest',  # Match shortest stream
                    video_path
                ]

                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"[VRGDG V4] FFmpeg stderr: {result.stderr}")
                    raise RuntimeError(f"FFmpeg failed: {result.stderr}")

                print(f"[VRGDG V4]   ✓ Video+audio combined successfully")

                # Clean up temp files
                os.remove(temp_video_path)
                os.remove(temp_audio_path)
                print(f"[VRGDG V4]   ✓ Temp files cleaned up")

            # MODE 2: VHS mode - move/rename VHS-saved file (FALLBACK)
            else:
                print(f"[VRGDG V4] Direct mode inputs not available, trying VHS mode...")

                # Look for recent AnimateDiff files with audio in output folder
                pattern = os.path.join(output_base, "AnimateDiff_*-audio.mp4")
                vhs_files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

                if vhs_files:
                    print(f"[VRGDG V4] Using VHS mode (fallback)...")

                    source_path = vhs_files[0]  # Most recent
                    print(f"[VRGDG V4]   Found VHS file: {os.path.basename(source_path)}")
                    print(f"[VRGDG V4]   Moving to: {video_path}")

                    # Move and rename the main video file (with audio)
                    shutil.move(source_path, video_path)
                    print(f"[VRGDG V4]   ✓ Moved: {os.path.basename(source_path)} → {os.path.basename(video_path)}")

                    # Remove the non-audio version (duplicate)
                    source_noaudio = source_path.replace("-audio.mp4", ".mp4")
                    if os.path.exists(source_noaudio):
                        os.remove(source_noaudio)
                        print(f"[VRGDG V4]   ✓ Removed: {os.path.basename(source_noaudio)} (non-audio duplicate)")

                    print(f"[VRGDG V4]   ✓ Video chunk saved successfully (VHS mode)")

                else:
                    # Neither mode available
                    error_details = []
                    error_details.append("No VHS files found in output folder")
                    if images is None:
                        error_details.append("images input is None")
                    if audio_chunk is None:
                        error_details.append("audio_chunk input is None")

                    error_msg = "Cannot save video: " + ", ".join(error_details)
                    print(f"[VRGDG V4] ❌ {error_msg}")
                    print(f"[VRGDG V4] Suggestions:")
                    print(f"[VRGDG V4]   1. Connect images and audio_chunk inputs for Direct mode, OR")
                    print(f"[VRGDG V4]   2. Ensure VHS_VideoCombine runs before this node and saves files")
                    raise ValueError(error_msg)

            # Create marker file for counting (in both modes)
            print(f"[VRGDG V4] Creating marker file...")
            with open(marker_path, 'w') as f:
                f.write("")  # Empty marker
            print(f"[VRGDG V4]   ✓ Marker created: {marker_filename}")

            # Get file size
            if os.path.exists(video_path):
                video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                print(f"[VRGDG V4] ✅ Chunk #{index} saved successfully!")
                print(f"[VRGDG V4] File size: {video_size_mb:.2f} MB")
                print(f"[VRGDG V4] Location: {video_path}")
            else:
                raise FileNotFoundError(f"Video file was not created: {video_path}")

            print("="*80 + "\n")

            return (video_path, True, index)

        except Exception as e:
            error_msg = f"❌ Failed to save chunk #{index}: {e}"
            print(f"[VRGDG V4] {error_msg}")
            import traceback
            traceback.print_exc()
            print("="*80 + "\n")
            raise RuntimeError(error_msg)


# =============================================================================
# Node 6: VRGDG_AutoQueueController
# =============================================================================

class VRGDG_AutoQueueController:
    """
    Manages auto-queueing of the next chunk.

    If should_auto_queue=True and chunks remain, queues the next run.
    When all chunks are complete (remaining < 0), outputs trigger_combine signal.

    Folder change detection:
    - Tracks the last used output folder
    - When folder changes, resets trigger_combine to False
    - Prevents premature combine triggering when switching projects
    """

    # Class variable to track last folder for change detection
    _last_folder = None

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
                "output_folder": ("STRING", {
                    "default": "video_output",
                    "multiline": False,
                }),
            }
        }

    RETURN_TYPES = (any_typ, "STRING", "BOOLEAN")
    RETURN_NAMES = ("signal", "queue_status", "trigger_combine")
    FUNCTION = "control_queue"
    CATEGORY = "VRGDG/V4 Single Chunk"
    OUTPUT_NODE = True

    def control_queue(self, trigger, should_auto_queue, remaining, current_index, output_folder):
        """
        Queue next chunk if needed.
        When all chunks are done, signal to trigger final combine.
        Detects folder changes to reset trigger_combine.
        """

        print("\n" + "="*80)
        print("[VRGDG V4] 🔄 Auto Queue Controller - Managing queue...")
        print(f"[VRGDG V4] Current chunk: {current_index}")
        print(f"[VRGDG V4] Remaining chunks: {remaining}")
        print(f"[VRGDG V4] Should auto-queue: {should_auto_queue}")
        print(f"[VRGDG V4] Output folder (raw): {output_folder}")

        # Resolve output folder to absolute path for consistent comparison
        if not os.path.isabs(output_folder):
            output_base = folder_paths.get_output_directory()
            output_folder_abs = os.path.join(output_base, output_folder)
        else:
            output_folder_abs = output_folder

        print(f"[VRGDG V4] Output folder (absolute): {output_folder_abs}")

        # Detect folder change
        folder_changed = False
        if VRGDG_AutoQueueController._last_folder is None:
            print(f"[VRGDG V4] First run - initializing folder tracking")
            VRGDG_AutoQueueController._last_folder = output_folder_abs
        elif VRGDG_AutoQueueController._last_folder != output_folder_abs:
            folder_changed = True
            print(f"[VRGDG V4] 📁 Folder changed detected!")
            print(f"[VRGDG V4]   Previous: {VRGDG_AutoQueueController._last_folder}")
            print(f"[VRGDG V4]   Current:  {output_folder_abs}")
            print(f"[VRGDG V4]   🔄 Resetting trigger_combine to False")
            VRGDG_AutoQueueController._last_folder = output_folder_abs

        # Reset trigger_combine to False on folder change
        trigger_combine = False

        # Fix: Only queue if remaining > 0 (not >= 0)
        # remaining = 0 means we just processed the last chunk, don't queue another!
        if should_auto_queue and remaining > 0:
            # Queue the next chunk
            print(f"[VRGDG V4] Queueing next chunk ({current_index + 1})...")
            try:
                PromptServer.instance.send_sync("impact-add-queue", {})
                status = f"✅ Chunk {current_index} done. Queued chunk {current_index + 1} ({remaining} more after this)"
                print(f"[VRGDG V4]   ✓ Successfully queued next chunk")
                print(f"[VRGDG V4] Status: {status}")
            except Exception as e:
                status = f"⚠️ Failed to queue next chunk: {e}"
                print(f"[VRGDG V4] ❌ Queue failed: {e}")
        else:
            if remaining <= 0 and should_auto_queue and not folder_changed:
                # All chunks complete (or just finished the last one)
                # Only trigger combine if folder hasn't changed
                status = f"🏁 All chunks complete! Ready to combine."
                trigger_combine = True
                print(f"[VRGDG V4] 🏁 All chunks have been processed!")
                print(f"[VRGDG V4] 🎬 Triggering final video combine...")
            elif folder_changed:
                status = f"📁 Folder changed - trigger_combine reset to False"
                print(f"[VRGDG V4] 📁 Folder changed - not triggering combine")
            else:
                status = f"⏸️ Auto-queue disabled (manual mode)"
                print(f"[VRGDG V4] ⏸️ Auto-queue disabled (manual mode)")
            print(f"[VRGDG V4] Status: {status}")

        print("="*80 + "\n")

        return (True, status, trigger_combine)


# =============================================================================
# Node 7: VRGDG_CombineAllChunks
# =============================================================================

class VRGDG_CombineAllChunks:
    """
    Combines all video chunks into a final video.

    Searches for all video_chunk_*.mp4 files in the output folder,
    concatenates them in order using ffmpeg, and saves as final_video.mp4.

    Should be triggered after all chunks are processed (when remaining = 0).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("BOOLEAN",),  # Signal from auto-queue controller
                "output_folder": ("STRING", {
                    "default": "video_output",
                    "multiline": False,
                }),
                "final_video_name": ("STRING", {
                    "default": "final_video.mp4",
                    "multiline": False,
                }),
                "total_chunks": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                }),
            }
        }

    RETURN_TYPES = ("STRING", any_typ)
    RETURN_NAMES = ("final_video_path", "signal")
    FUNCTION = "combine_chunks"
    CATEGORY = "VRGDG/V4 Single Chunk"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Force execution when trigger changes to True.
        """
        trigger = kwargs.get('trigger', False)
        # If trigger is True (boolean), return unique value to force execution
        if trigger is True:
            import time
            return float(time.time())
        return False

    def combine_chunks(self, trigger, output_folder, final_video_name, total_chunks):
        """
        Combine all video chunks into final video using ffmpeg concat.
        """

        print("\n" + "="*80)
        print("[VRGDG V4] 🎬 Combine All Chunks - Checking trigger...")
        print(f"[VRGDG V4] Trigger value: {trigger} (type: {type(trigger)})")

        # Check if we should actually combine
        # Trigger should be True (boolean) from AutoQueueController's trigger_combine output
        if trigger is not True:
            print(f"[VRGDG V4] ℹ️ Trigger is not True - skipping combine")
            print(f"[VRGDG V4] (Connect AutoQueueController's trigger_combine output to this node's trigger input)")
            print("="*80 + "\n")
            return ("", False)

        print("[VRGDG V4] ✓ Trigger is True - proceeding with combine")
        print(f"[VRGDG V4] Output folder (raw): {output_folder}")
        print(f"[VRGDG V4] Expected chunks: {total_chunks}")
        print(f"[VRGDG V4] Final video name: {final_video_name}")

        # Ensure filename includes "final" for clarity
        if "final" not in final_video_name.lower():
            final_video_name = f"final_{final_video_name}"
            print(f"[VRGDG V4] Added 'final' prefix: {final_video_name}")

        # Ensure filename has .mp4 extension
        if not final_video_name.endswith('.mp4'):
            final_video_name = f"{final_video_name}.mp4"
            print(f"[VRGDG V4] Added .mp4 extension: {final_video_name}")

        # Resolve output folder to absolute path (same as SaveVideoChunk)
        if not os.path.isabs(output_folder):
            output_base = folder_paths.get_output_directory()
            output_folder = os.path.join(output_base, output_folder)
            print(f"[VRGDG V4] Resolved to absolute path: {output_folder}")

        try:
            # Find all chunk files
            import glob
            pattern = os.path.join(output_folder, "video_chunk_*.mp4")
            # Only include .mp4 files, exclude any marker files (.done, -audio.mp4, etc.)
            chunk_files = [
                f for f in sorted(glob.glob(pattern))
                if f.endswith(".mp4") and not f.endswith("-audio.mp4")
            ]

            print(f"[VRGDG V4] Found {len(chunk_files)} chunk files:")
            for i, f in enumerate(chunk_files[:10]):  # Show first 10
                print(f"[VRGDG V4]   {i}: {os.path.basename(f)}")
            if len(chunk_files) > 10:
                print(f"[VRGDG V4]   ... and {len(chunk_files) - 10} more")

            if len(chunk_files) == 0:
                raise FileNotFoundError(f"No video chunks found in {output_folder}")

            # Verify we have all expected chunks
            if total_chunks > 0 and len(chunk_files) < total_chunks:
                print(f"[VRGDG V4] ⚠️ Warning: Expected {total_chunks} chunks but found {len(chunk_files)}")
                print(f"[VRGDG V4]   Proceeding with available chunks...")

            # Create concat file for ffmpeg
            import tempfile
            concat_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            concat_file_path = concat_file.name

            print(f"[VRGDG V4] Creating concat file...")
            for chunk_file in chunk_files:
                # ffmpeg concat format requires: file '/path/to/file.mp4'
                # Escape single quotes in path
                escaped_path = chunk_file.replace("'", "'\\''")
                concat_file.write(f"file '{escaped_path}'\n")
            concat_file.close()
            print(f"[VRGDG V4]   ✓ Concat file created: {concat_file_path}")

            # Prepare output path
            final_video_path = os.path.join(output_folder, final_video_name)

            # Run ffmpeg concat
            print(f"[VRGDG V4] Running ffmpeg concat...")
            print(f"[VRGDG V4]   Combining {len(chunk_files)} chunks...")

            import subprocess
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-f', 'concat',  # Concat demuxer
                '-safe', '0',  # Allow absolute paths
                '-i', concat_file_path,  # Input concat file
                '-c', 'copy',  # Copy streams (no re-encoding)
                final_video_path
            ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[VRGDG V4] FFmpeg stderr: {result.stderr}")
                raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")

            # Clean up concat file
            os.remove(concat_file_path)
            print(f"[VRGDG V4]   ✓ Concat file cleaned up")

            # Verify output
            if os.path.exists(final_video_path):
                file_size_mb = os.path.getsize(final_video_path) / (1024 * 1024)
                print(f"[VRGDG V4] ✅ Final video created successfully!")
                print(f"[VRGDG V4] File: {final_video_path}")
                print(f"[VRGDG V4] Size: {file_size_mb:.2f} MB")
                print(f"[VRGDG V4] Combined {len(chunk_files)} chunks")
            else:
                raise FileNotFoundError(f"Final video was not created: {final_video_path}")

            print("="*80 + "\n")

            return (final_video_path, True)

        except Exception as e:
            error_msg = f"❌ Failed to combine chunks: {e}"
            print(f"[VRGDG V4] {error_msg}")
            import traceback
            traceback.print_exc()
            print("="*80 + "\n")
            raise RuntimeError(error_msg)


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
    "VRGDG_CombineAllChunks": VRGDG_CombineAllChunks,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_FullSongAnalyzerV4": "🎵 VRGDG Full Song Analyzer V4",
    "VRGDG_ChunkIndexController": "🎯 VRGDG Chunk Index Controller",
    "VRGDG_GetPromptByIndex": "📝 VRGDG Get Prompt By Index",
    "VRGDG_LoadSingleAudioChunk": "🎵 VRGDG Load Single Audio Chunk",
    "VRGDG_SaveVideoChunkWithIndex": "💾 VRGDG Save Video Chunk",
    "VRGDG_AutoQueueController": "🔄 VRGDG Auto Queue Controller",
    "VRGDG_CombineAllChunks": "🎬 VRGDG Combine All Chunks",
}
