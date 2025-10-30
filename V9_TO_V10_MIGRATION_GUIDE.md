# Migration Guide: V9 (16-Chunk) → V10 (Single-Chunk)

## Quick Summary

**V9 Workflow:** Processes 16 chunks (62 seconds) at once using 16 parallel HUMO nodes
**V10 Workflow:** Processes 1 chunk (3.88 seconds) at a time with automatic queueing

---

## Why Migrate?

### Problems V9 Has:
- ❌ Songs that don't divide evenly into 16-chunk groups require manual intervention
- ❌ Must manually mute unused nodes for partial last sets
- ❌ Complex logic for calculating sets and groups
- ❌ Can't easily regenerate specific chunks without editing workflow
- ❌ Large memory footprint (16x HUMO models in memory)

### What V10 Fixes:
- ✅ Fully automatic - handles any song length
- ✅ No manual node muting ever needed
- ✅ Simple index-based processing
- ✅ Easy chunk regeneration (just set manual index)
- ✅ Lower memory usage (1x HUMO model)
- ✅ Leverages ComfyUI caching for prompt generation

---

## Step-by-Step Migration

### Step 1: Identify V9 Nodes to Remove

Open your `WanHumoMVC_V9.json` workflow and locate these nodes:

#### Remove These:
1. **VRGDG_LoadAudioSplit_HUMO_TranscribeV3**
   - This splits audio into 16 chunks
   - ❌ Replace with: `VRGDG_FullSongAnalyzerV4`

2. **VRGDG_PromptSplitterJson**
   - This splits prompts into 16 outputs
   - ❌ Replace with: `VRGDG_GetPromptByIndex`

3. **15 extra WanVideoTextEncode nodes**
   - V9 has 16 of these
   - ❌ Keep only 1, delete the other 15

4. **15 extra WanVideoSampler nodes**
   - V9 has 16 of these (the HUMO nodes)
   - ❌ Keep only 1, delete the other 15

5. **VRGDG_CombinevideosV3**
   - Combines 16 videos into one
   - ❌ Delete entirely (not needed in V10)

### Step 2: Add V10 Nodes

Add these new nodes to your workflow:

1. **VRGDG_FullSongAnalyzerV4** (🎵)
   - Analyzes entire song and generates all prompts
   - Replaces: VRGDG_LoadAudioSplit_HUMO_TranscribeV3

2. **VRGDG_ChunkIndexController** (🎯)
   - Manages which chunk to process
   - New functionality

3. **VRGDG_GetPromptByIndex** (📝)
   - Extracts single prompt by index
   - Replaces: VRGDG_PromptSplitterJson

4. **VRGDG_LoadSingleAudioChunk** (🎵)
   - Loads one audio chunk
   - New functionality

5. **VRGDG_SaveVideoChunkWithIndex** (💾)
   - Saves video with index in filename
   - New functionality

6. **VRGDG_AutoQueueController** (🔄)
   - Auto-queues next chunk
   - New functionality

### Step 3: Make Connections

Here's how to wire up the V10 workflow:

```
1. LoadAudio (your existing audio loader)
   ↓
2. VRGDG_FullSongAnalyzerV4
   Inputs:
   - audio: from LoadAudio
   - character_description: "A woman in a white dress"
   - song_theme_style: "cinematic realism, emotional..."
   - scene_duration_seconds: 3.88
   - language: "english"
   - enable_full_transcription: True
   - seed: 42 (fixed for caching)
   - [all visual parameters]

   Outputs: full_prompts, total_chunks, full_lyrics, audio_meta, audio_hash

3. VRGDG_ChunkIndexController
   Inputs:
   - full_prompts: from FullSongAnalyzerV4.full_prompts
   - output_folder: "video_output"
   - auto_index: True (for automatic processing)
   - manual_index: 0 (ignored when auto_index=True)

   Outputs: current_index, total_chunks, remaining, should_auto_queue, status_message

4a. VRGDG_GetPromptByIndex
    Inputs:
    - full_prompts: from FullSongAnalyzerV4.full_prompts
    - index: from ChunkIndexController.current_index

    Output: prompt, transcription

4b. VRGDG_LoadSingleAudioChunk
    Inputs:
    - audio: from LoadAudio (same audio as step 1)
    - index: from ChunkIndexController.current_index
    - audio_meta: from FullSongAnalyzerV4.audio_meta

    Output: audio_chunk, chunk_info

5. WanVideoTextEncode (keep your existing one)
   Inputs:
   - text: from GetPromptByIndex.prompt
   - [other existing inputs: model, etc.]

   Output: text_embeds

6. WanVideoSampler (keep your existing one)
   Inputs:
   - model: [your existing model connection]
   - image_embeds: [your existing reference image embeddings]
   - text_embeds: from WanVideoTextEncode
   - audio: from LoadSingleAudioChunk.audio_chunk
   - [all other existing parameters]

   Output: video frames (IMAGE)

7. VRGDG_SaveVideoChunkWithIndex
   Inputs:
   - images: from WanVideoSampler output
   - audio_chunk: from LoadSingleAudioChunk.audio_chunk
   - index: from ChunkIndexController.current_index
   - output_folder: "video_output"
   - fps: 25.0

   Outputs: saved_path, signal

8. VRGDG_AutoQueueController
   Inputs:
   - trigger: from SaveVideoChunkWithIndex.signal
   - should_auto_queue: from ChunkIndexController.should_auto_queue
   - remaining: from ChunkIndexController.remaining
   - current_index: from ChunkIndexController.current_index

   Outputs: signal, queue_status
```

---

## Visual Comparison

### V9 Architecture (Complex)
```
LoadAudio
   ↓
VRGDG_LoadAudioSplit_HUMO_TranscribeV3 (splits to 16)
   ↓ (16 audio outputs)
   ↓
VRGDG_MusicVideoPromptCreatorV3
   ↓
VRGDG_PromptSplitterJson
   ↓ (16 prompt outputs)
   ↓
[WanVideoTextEncode #1]  [WanVideoTextEncode #2]  ...  [WanVideoTextEncode #16]
   ↓                        ↓                              ↓
[WanVideoSampler #1]     [WanVideoSampler #2]     ...  [WanVideoSampler #16]
   ↓                        ↓                              ↓
   └────────────────────────┴──────────────────────────────┘
                            ↓
                  VRGDG_CombinevideosV3
                            ↓
                      Final Video
```

### V10 Architecture (Simple)
```
LoadAudio
   ↓
VRGDG_FullSongAnalyzerV4 (cached!)
   ↓
VRGDG_ChunkIndexController
   ↓
   ├─→ VRGDG_GetPromptByIndex ──→ WanVideoTextEncode ──┐
   │                                                    ↓
   └─→ VRGDG_LoadSingleAudioChunk ──────────→ WanVideoSampler
                                                        ↓
                                         VRGDG_SaveVideoChunkWithIndex
                                                        ↓
                                           VRGDG_AutoQueueController
                                                        ↓
                                              (queues next chunk)
```

---

## Configuration Changes

### V9 Settings (Delete These):
- ❌ `groups_in_last_set` - No longer needed
- ❌ `total_sets` - No longer needed
- ❌ `context_1` through `context_16` - No longer needed
- ❌ Manual muter node settings - No longer needed

### V10 Settings (Use These):
- ✅ `auto_index` - True for automatic, False for manual
- ✅ `manual_index` - Set specific chunk to regenerate
- ✅ `seed` - Fixed value for consistent prompts
- ✅ `output_folder` - Where chunks are saved

---

## Testing Your Migration

### Test 1: Process a Short Song (< 16 chunks)

**Setup:**
1. Load a 30-second song
2. Set `auto_index=True`
3. Click "Queue Prompt" once

**Expected Result:**
- Should process 8 chunks automatically (30s / 3.88s ≈ 8)
- Should create files: `video_chunk_0000.mp4` through `video_chunk_0007.mp4`
- Should stop automatically after chunk 7

### Test 2: Process a Long Song (> 16 chunks)

**Setup:**
1. Load a 2-minute song
2. Set `auto_index=True`
3. Click "Queue Prompt" once

**Expected Result:**
- Should process 31 chunks automatically (120s / 3.88s ≈ 31)
- Should create files: `video_chunk_0000.mp4` through `video_chunk_0030.mp4`
- Should stop automatically after chunk 30
- **No manual intervention needed!**

### Test 3: Regenerate Specific Chunk

**Setup:**
1. After processing a song, identify a chunk you want to redo (e.g., chunk 5)
2. Set `auto_index=False`
3. Set `manual_index=5`
4. Optionally adjust visual parameters
5. Click "Queue Prompt"

**Expected Result:**
- Should process only chunk 5
- Should overwrite `video_chunk_0005.mp4`
- Should NOT queue other chunks
- Should stop after chunk 5

---

## Common Migration Issues

### Issue 1: "Prompts regenerating every chunk"
**Cause:** ComfyUI cache is being invalidated
**Solution:**
- Use a fixed `seed` value
- Don't change audio file path
- Don't change visual parameters between runs

### Issue 2: "Wrong chunk index being processed"
**Cause:** Marker files missing or incorrect
**Solution:**
- Check output folder for `video_chunk_*-audio.mp4` files
- Delete all chunks to start fresh
- Ensure `auto_index=True` for automatic detection

### Issue 3: "Auto-queue not working"
**Cause:** `auto_index` or `should_auto_queue` is False
**Solution:**
- Set `auto_index=True` in ChunkIndexController
- Verify connections to AutoQueueController

### Issue 4: "Last chunk has silence"
**Cause:** Audio duration doesn't divide evenly into 3.88s chunks
**Solution:**
- This is expected behavior!
- Last chunk is padded with silence (1-3 seconds max)
- Not a problem for final video

### Issue 5: "Memory usage same as V9"
**Cause:** Multiple WanVideoSampler nodes still in workflow
**Solution:**
- Delete 15 of the 16 WanVideoSampler nodes
- Keep only 1 WanVideoSampler
- Same for WanVideoTextEncode

---

## Workflow File Updates

If you're editing the JSON directly:

### Find and Remove:
```json
{
  "type": "VRGDG_LoadAudioSplit_HUMO_TranscribeV3",
  ...
}
```

### Find and Remove:
```json
{
  "type": "VRGDG_CombinevideosV3",
  ...
}
```

### Find and Remove 15 of these (keep 1):
```json
{
  "type": "WanVideoTextEncode",
  ...
}
```

### Find and Remove 15 of these (keep 1):
```json
{
  "type": "WanVideoSampler",
  ...
}
```

### Add New Nodes:
See the V4_SINGLE_CHUNK_DOCUMENTATION.md for complete node specifications.

---

## Rollback Plan

If you need to go back to V9:

1. **Keep your V9 workflow file as backup** (`WanHumoMVC_V9.json`)
2. Save V10 workflow as new file (`WanHumoMVC_V10_SingleChunk.json`)
3. If issues arise, load V9 workflow from backup

**Recommendation:** Don't delete V9 workflow until V10 is fully tested!

---

## Benefits After Migration

### Immediate Benefits:
- ✅ No more manual node muting
- ✅ No more partial set calculations
- ✅ No more manual intervention
- ✅ Automatic processing for any song length

### Long-term Benefits:
- ✅ Easy chunk regeneration
- ✅ Lower memory usage
- ✅ Faster prompt generation (cached)
- ✅ Simpler workflow maintenance
- ✅ Better debugging (one chunk at a time)

---

## Next Steps

1. **Backup your V9 workflow**
2. **Create new V10 workflow file**
3. **Follow the connection steps above**
4. **Test with a short song first**
5. **Once confident, process full songs**
6. **Enjoy fully automated music video generation!**

---

## Support

If you encounter issues during migration:

1. Check the V4_SINGLE_CHUNK_DOCUMENTATION.md for detailed node reference
2. Verify all connections match the diagram above
3. Test with auto_index=True for automatic processing
4. Check output folder for marker files
5. Report issues on GitHub with workflow JSON and error logs

---

**Happy migrating! The V10 single-chunk system will make your workflow much simpler and more reliable.** 🎵🎬
