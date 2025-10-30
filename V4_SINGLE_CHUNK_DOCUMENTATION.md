# V4 Single-Chunk Processing Documentation

## Overview

The V4 Single-Chunk system simplifies HUMO music video generation by processing **one 3.88-second chunk at a time** instead of batching 16 chunks together. This eliminates the complexity of handling partial sets and manual intervention requirements.

---

## Key Improvements

### Before (V3 - 16-Chunk Batching)
```
❌ Processes 16 chunks at once (62.08 seconds)
❌ Requires manual intervention for songs that don't divide evenly
❌ Complex logic for partial last sets
❌ Manual node muting required
❌ Can't easily regenerate specific chunks
```

### After (V4 - Single-Chunk)
```
✅ Processes 1 chunk at a time (3.88 seconds)
✅ Fully automatic - no manual intervention
✅ Simple index-based processing
✅ No manual node muting needed
✅ Easy to regenerate any specific chunk
✅ Leverages ComfyUI caching for prompt generation
```

---

## Architecture

### Workflow Flow

```
1. LoadAudio → VRGDG_FullSongAnalyzerV4 (CACHED - runs once per song!)
   ↓
   Outputs:
   - full_prompts (DICT): All scene prompts for entire song
   - total_chunks (INT): Total chunks needed
   - full_lyrics (STRING): Complete transcription
   - audio_meta (DICT): Audio metadata

2. full_prompts + output_folder → VRGDG_ChunkIndexController
   ↓
   Outputs:
   - current_index (INT): Which chunk to process now
   - should_auto_queue (BOOLEAN): Queue next chunk?
   - remaining (INT): Chunks left

3a. full_prompts + current_index → VRGDG_GetPromptByIndex
    ↓
    Output: Single prompt string for this chunk

3b. audio + current_index + audio_meta → VRGDG_LoadSingleAudioChunk
    ↓
    Output: Single 3.88s audio chunk (padded if needed)

4. prompt → WanVideoTextEncode → text_embeds

5. WanVideoSampler (model + image_embeds + text_embeds + audio)
   ↓
   Output: Single video chunk (97 frames)

6. video + audio + index → VRGDG_SaveVideoChunkWithIndex
   ↓
   Saves as: video_chunk_0000.mp4, video_chunk_0001.mp4, etc.

7. trigger + should_auto_queue → VRGDG_AutoQueueController
   ↓
   Queues next chunk if needed
```

---

## New Nodes Reference

### 1. VRGDG_FullSongAnalyzerV4
**Purpose:** Master orchestrator - analyzes entire song once and generates all prompts upfront

**Inputs:**
- `audio` (AUDIO): Full audio file
- `character_description` (STRING): Character details
- `song_theme_style` (STRING): Visual theme/style
- `scene_duration_seconds` (FLOAT): 3.88s (97 frames @ 25 FPS)
- `language` (ENUM): Whisper language for transcription
- `enable_full_transcription` (BOOLEAN): Enable/disable transcription
- `seed` (INT): Fixed seed for deterministic LLM output
- Visual parameters: `environment`, `lighting`, `camera_motion`, etc.

**Outputs:**
- `full_prompts` (DICT): All prompts + metadata
- `total_chunks` (INT): Total chunks needed
- `full_lyrics` (STRING): Complete transcription
- `audio_meta` (DICT): Audio metadata
- `audio_hash` (STRING): Hash for cache validation

**Key Feature:** This node leverages ComfyUI's built-in caching. If the audio file hasn't changed, ComfyUI will use the cached output instead of re-running prompt generation!

---

### 2. VRGDG_ChunkIndexController
**Purpose:** Manages which chunk to process (auto vs manual mode)

**Inputs:**
- `full_prompts` (DICT): From FullSongAnalyzerV4
- `output_folder` (STRING): Where videos are saved
- `auto_index` (BOOLEAN): True = auto-detect, False = manual
- `manual_index` (INT): Used when auto_index=False

**Outputs:**
- `current_index` (INT): Which chunk to process
- `total_chunks` (INT): Total chunks
- `remaining` (INT): Chunks left to process
- `should_auto_queue` (BOOLEAN): Whether to queue next
- `status_message` (STRING): Human-readable status

**Modes:**

**Auto Mode** (`auto_index=True`):
- Counts existing `video_chunk_*-audio.mp4` files
- Automatically determines next chunk to process
- Default mode for full song processing

**Manual Mode** (`auto_index=False`):
- Uses `manual_index` parameter
- Perfect for regenerating specific chunks
- Example: Set `manual_index=7` to regenerate chunk 7

---

### 3. VRGDG_GetPromptByIndex
**Purpose:** Extracts single prompt from cached full_prompts

**Inputs:**
- `full_prompts` (DICT): All prompts
- `index` (INT): Which prompt to get

**Outputs:**
- `prompt` (STRING): Scene prompt for this chunk
- `transcription` (STRING): Lyrics for this chunk

**Simple Logic:**
```python
prompt = full_prompts["prompts"][index]
transcription = full_prompts["chunk_transcriptions"][index]
```

---

### 4. VRGDG_LoadSingleAudioChunk
**Purpose:** Loads one audio chunk based on index

**Inputs:**
- `audio` (AUDIO): Full audio file
- `index` (INT): Which chunk to load
- `audio_meta` (DICT): Metadata from analyzer

**Outputs:**
- `audio_chunk` (AUDIO): Single 3.88s audio segment
- `chunk_info` (DICT): Metadata about this chunk

**Features:**
- Extracts exactly 97 frames worth of audio
- Automatically pads with silence if last chunk is short
- No manual handling needed for partial chunks!

---

### 5. VRGDG_SaveVideoChunkWithIndex
**Purpose:** Saves video chunk with index in filename

**Inputs:**
- `images` (IMAGE): Video frames from HUMO
- `audio_chunk` (AUDIO): Corresponding audio
- `index` (INT): Current chunk number
- `output_folder` (STRING): Save location
- `fps` (FLOAT): Frame rate (default 25.0)

**Outputs:**
- `saved_path` (STRING): Path to saved file
- `signal` (ANY): Completion signal

**Creates:**
- `video_chunk_0000.mp4` - The actual video
- `video_chunk_0000-audio.mp4` - Marker file for auto-indexing

**File Naming Examples:**
```
video_chunk_0000.mp4  (chunk 0)
video_chunk_0001.mp4  (chunk 1)
video_chunk_0002.mp4  (chunk 2)
...
video_chunk_0042.mp4  (chunk 42)
```

---

### 6. VRGDG_AutoQueueController
**Purpose:** Manages auto-queueing of next chunk

**Inputs:**
- `trigger` (ANY): Execution trigger
- `should_auto_queue` (BOOLEAN): From ChunkIndexController
- `remaining` (INT): Chunks remaining
- `current_index` (INT): Current chunk number

**Outputs:**
- `signal` (ANY): Completion signal
- `queue_status` (STRING): Human-readable status

**Logic:**
```python
if should_auto_queue and remaining >= 0:
    queue_next_chunk()  # Automatically queue next run
else:
    stop()  # All done or manual mode
```

---

## Usage Scenarios

### Scenario 1: Process Entire Song Automatically

**Setup:**
1. Connect your audio file to `VRGDG_FullSongAnalyzerV4`
2. Set `auto_index=True` in `VRGDG_ChunkIndexController`
3. Configure your character description and visual parameters
4. Click "Queue Prompt" ONCE

**What Happens:**
1. First chunk processes
2. Automatically queues chunk 2
3. Chunk 2 processes, queues chunk 3
4. ... continues automatically
5. Last chunk processes, stops

**No manual intervention required!**

---

### Scenario 2: Regenerate Specific Chunk

**Setup:**
1. Set `auto_index=False` in `VRGDG_ChunkIndexController`
2. Set `manual_index=7` (to regenerate chunk 7)
3. Optionally adjust prompt parameters
4. Click "Queue Prompt"

**What Happens:**
1. Only chunk 7 processes
2. Overwrites existing `video_chunk_0007.mp4`
3. Stops (no auto-queue in manual mode)

**Perfect for tweaking specific scenes!**

---

### Scenario 3: Resume Interrupted Processing

**Setup:**
1. Set `auto_index=True` in `VRGDG_ChunkIndexController`
2. Click "Queue Prompt"

**What Happens:**
1. Node counts existing chunks (e.g., finds chunks 0-14)
2. Automatically starts at chunk 15
3. Continues from where you left off
4. Processes remaining chunks

**No manual index tracking needed!**

---

## Technical Details

### Chunk Specifications
- **Duration:** 3.88 seconds
- **Frames:** 97 frames @ 25 FPS
- **Format:** 4n+1 (HuMo optimal: 97 = 4×24+1)
- **Audio:** Stereo, padded with silence if needed

### File Naming Convention
```
video_chunk_{index:04d}.mp4        # Actual video
video_chunk_{index:04d}-audio.mp4  # Marker for counting
```

### Caching Behavior
`VRGDG_FullSongAnalyzerV4` uses ComfyUI's built-in caching:
- **Cache Key:** Audio file path + hash + parameters
- **Cache Hit:** Prompts reused instantly (no LLM call)
- **Cache Miss:** New prompts generated (if audio changed)

This means:
- ✅ Prompt generation happens ONCE per song
- ✅ Subsequent chunks use cached prompts
- ✅ Changing visual parameters invalidates cache
- ✅ Same audio = instant prompt reuse

---

## Comparison: V3 vs V4

| Feature | V3 (16-Chunk Batching) | V4 (Single-Chunk) |
|---------|------------------------|-------------------|
| **Chunks per run** | 16 (62.08s) | 1 (3.88s) |
| **Manual intervention** | Required for partial sets | Never |
| **Node muting** | Manual (complex) | Automatic |
| **Regenerate specific chunk** | Difficult | Easy |
| **Workflow complexity** | High (16x nodes) | Low (1x nodes) |
| **Auto-queue** | Partial (leaves last manual) | Full (all chunks) |
| **ComfyUI caching** | Limited | Optimized |
| **Partial song handling** | Manual | Automatic |

---

## Migration from V9 to V10 Workflow

### Old V9 Workflow Nodes (Remove/Replace):
1. ❌ `VRGDG_LoadAudioSplit_HUMO_TranscribeV3` → Replace with `VRGDG_FullSongAnalyzerV4`
2. ❌ `VRGDG_PromptSplitterJson` → Replace with `VRGDG_GetPromptByIndex`
3. ❌ 16x `WanVideoTextEncode` → Keep only 1
4. ❌ 16x `WanVideoSampler` → Keep only 1
5. ❌ `VRGDG_CombinevideosV3` → Remove (no longer needed)

### New V10 Workflow Nodes (Add):
1. ✅ `VRGDG_FullSongAnalyzerV4` (replaces audio splitter)
2. ✅ `VRGDG_ChunkIndexController` (new - index management)
3. ✅ `VRGDG_GetPromptByIndex` (new - prompt extraction)
4. ✅ `VRGDG_LoadSingleAudioChunk` (new - single chunk loader)
5. ✅ `VRGDG_SaveVideoChunkWithIndex` (new - indexed save)
6. ✅ `VRGDG_AutoQueueController` (new - queue management)

### Connection Changes:
```
OLD:
LoadAudio → Split16 → PromptCreator → Splitter → 16xEncode → 16xHUMO → Combine

NEW:
LoadAudio → Analyzer (cached!) → IndexController → GetPrompt → 1xEncode → 1xHUMO → SaveChunk → AutoQueue
             ↓                      ↓
             └─────────────────────→ LoadChunk
```

---

## Tips & Best Practices

### 1. Set a Fixed Seed
Always use the same `seed` value in `VRGDG_FullSongAnalyzerV4` for consistent prompt generation across runs.

### 2. Use Auto Mode for Full Songs
For processing entire songs, always use `auto_index=True` and let the system handle everything.

### 3. Manual Mode for Tweaking
When you want to regenerate specific chunks (e.g., adjust a scene), use manual mode with the specific index.

### 4. Monitor Output Folder
The output folder will contain:
- `video_chunk_0000.mp4`, `video_chunk_0001.mp4`, etc. (actual videos)
- `video_chunk_0000-audio.mp4`, etc. (marker files for counting)
- Marker files can be empty - they're just for auto-index tracking

### 5. Final Video Assembly
After all chunks are processed, you can concatenate them using:
- `VRGDG_CreateFinalVideo` (existing node)
- Or any video concatenation tool (ffmpeg, etc.)

### 6. Prompt Customization
The `VRGDG_FullSongAnalyzerV4` uses placeholder prompts in the current implementation. To integrate with your LLM:

**Option A:** Modify `_build_prompt_instruction()` to call your LLM API
**Option B:** Use the existing `VRGDG_MusicVideoPromptCreatorV3` approach (connect externally)

---

## Troubleshooting

### Problem: Chunks not auto-queueing
**Solution:** Check `auto_index=True` in `VRGDG_ChunkIndexController`

### Problem: Wrong chunk processing
**Solution:** Verify marker files exist: `ls video_chunk_*-audio.mp4` in output folder

### Problem: Prompt generation runs every time
**Solution:** Ensure audio file path and parameters haven't changed (ComfyUI caching)

### Problem: Want to start fresh
**Solution:** Delete all `video_chunk_*.mp4` files from output folder

### Problem: Last chunk is silent
**Solution:** This is expected! Last chunk may have 1-3 seconds of silence padding (not a problem)

---

## Performance Notes

### Memory Usage
- **V3:** Processes 16 chunks → High memory (16x HUMO models)
- **V4:** Processes 1 chunk → Lower memory (1x HUMO model)

### Processing Time
- **V3:** 16 chunks in parallel (if GPU supports)
- **V4:** 1 chunk at a time (sequential)
- **Trade-off:** Lower memory vs longer total time

### Caching Benefits
- Prompt generation: Once per song (not per chunk)
- Audio analysis: Once per song (not per chunk)
- LLM calls: Drastically reduced

---

## Example: 3-Minute Song

### Old V3 Workflow:
- Duration: 180 seconds
- Chunks needed: 46.4 (requires 47 chunks)
- Sets needed: 3 sets (16 + 16 + 15)
- **Manual intervention:** Required for last set (mute group 16)
- Runs: 3 manual runs

### New V4 Workflow:
- Duration: 180 seconds
- Chunks needed: 47 chunks (180 / 3.88 = 46.4 → 47)
- **Manual intervention:** None
- Runs: 1 click → auto-queues 47 times

---

## Conclusion

The V4 Single-Chunk system transforms HUMO music video generation from a complex, manual process into a fully automated workflow. By processing one chunk at a time and leveraging ComfyUI's caching, you get:

- ✅ Simplicity
- ✅ Automation
- ✅ Flexibility
- ✅ Performance

No more manual node muting, no more partial set calculations, no more intervention. Just load your audio, configure your style, and click Queue!

---

**For questions or issues, please check the GitHub repository issues page.**

*Happy music video making! 🎵🎬*
