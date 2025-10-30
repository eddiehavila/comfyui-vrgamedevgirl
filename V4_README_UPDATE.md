# 🎵 V4 Single-Chunk Processing - MAJOR UPDATE

## What's New in V4?

We've completely reimagined how HUMO music video generation works! The new V4 system processes **one 3.88-second chunk at a time** instead of batching 16 chunks together.

### Why This Matters

**Before V4 (16-chunk batching):**
- 😓 Manual intervention required for songs that don't divide evenly into 16-chunk groups
- 😓 Complex node muting for partial last sets
- 😓 Can't easily regenerate specific chunks
- 😓 High memory usage (16x HUMO models)

**After V4 (single-chunk):**
- 🎉 Fully automatic - handles ANY song length
- 🎉 No manual intervention EVER
- 🎉 Easy chunk regeneration (just set an index!)
- 🎉 Lower memory footprint
- 🎉 Leverages ComfyUI caching for instant prompt reuse

---

## Quick Start

### 1. Install/Update
The V4 nodes are included in this repository. Just update your custom nodes:

```bash
cd ComfyUI/custom_nodes/comfyui-vrgamedevgirl
git pull
```

Restart ComfyUI to load the new nodes.

### 2. Find the New Nodes

In ComfyUI, search for:
- 🎵 **VRGDG Full Song Analyzer V4** - Analyzes entire song (cached!)
- 🎯 **VRGDG Chunk Index Controller** - Manages which chunk to process
- 📝 **VRGDG Get Prompt By Index** - Extracts single prompt
- 🎵 **VRGDG Load Single Audio Chunk** - Loads one chunk
- 💾 **VRGDG Save Video Chunk** - Saves with indexed filename
- 🔄 **VRGDG Auto Queue Controller** - Queues next chunk

### 3. Basic Workflow

```
1. Load your audio file
2. Connect to "VRGDG Full Song Analyzer V4"
3. Configure character description and visual style
4. Set auto_index=True in "Chunk Index Controller"
5. Click "Queue Prompt" ONCE
6. Watch it process all chunks automatically!
```

That's it! No manual intervention needed.

---

## Key Features

### 🔄 Automatic Processing
- Click "Queue Prompt" once
- All chunks process automatically
- No manual node muting
- No complex set calculations
- Works for ANY song length

### 💾 Smart Indexing
- **Auto Mode:** Counts existing chunks, resumes from where you left off
- **Manual Mode:** Set specific chunk index to regenerate any scene
- File naming: `video_chunk_0000.mp4`, `video_chunk_0001.mp4`, etc.

### ⚡ ComfyUI Caching
- Prompt generation happens ONCE per song
- Subsequent chunks use cached prompts (instant!)
- Change audio = new prompts generated
- Same audio = instant prompt reuse

### 🎯 Easy Re-generation
Want to remake chunk 7 because you don't like that scene?
1. Set `auto_index=False`
2. Set `manual_index=7`
3. Click "Queue Prompt"
4. Done! Only chunk 7 regenerates.

---

## Example: 3-Minute Song

### Old Way (V3):
1. Calculate: 180s / 62.08s = 2.9 sets → need 3 sets
2. First run: Process 16 chunks
3. Second run: Process 16 chunks
4. **Manual intervention:** Mute group 16 for final run
5. Third run: Process 15 chunks (with manual muting)

### New Way (V4):
1. Click "Queue Prompt" once
2. System automatically processes all 47 chunks (180s / 3.88s)
3. **No manual intervention!**

---

## Documentation

### 📚 Complete Documentation
- **V4_SINGLE_CHUNK_DOCUMENTATION.md** - Full technical reference
- **V9_TO_V10_MIGRATION_GUIDE.md** - Step-by-step migration guide

### 🎯 Key Sections
1. **Architecture** - How the new system works
2. **Node Reference** - Detailed specs for all 6 new nodes
3. **Usage Scenarios** - Auto mode, manual mode, resume
4. **Troubleshooting** - Common issues and solutions
5. **Migration** - Convert your V9 workflow to V10

---

## Workflow Files

### Current Workflows
- **WanHumoMVC_V9.json** - Old 16-chunk batching (still works!)
- **WanHumoMVC_V10_SingleChunk.json** - New single-chunk (coming soon)

**Recommendation:** Keep V9 as backup while testing V10.

---

## Technical Specs

### Chunk Specifications
- **Duration:** 3.88 seconds
- **Frames:** 97 frames @ 25 FPS
- **Format:** 4n+1 (HuMo optimal)
- **Audio:** Stereo, auto-padded if needed

### File Naming
```
video_chunk_0000.mp4          # Chunk 0 (actual video)
video_chunk_0000-audio.mp4    # Chunk 0 (marker for auto-index)
video_chunk_0001.mp4          # Chunk 1 (actual video)
video_chunk_0001-audio.mp4    # Chunk 1 (marker)
...
```

### Caching Behavior
`VRGDG_FullSongAnalyzerV4` caches based on:
- Audio file path + hash
- Visual parameters (character, theme, etc.)
- Seed value

**Result:** Prompt generation happens once per unique configuration!

---

## Comparison Table

| Feature | V3 (16-Chunk) | V4 (Single-Chunk) |
|---------|---------------|-------------------|
| Chunks per run | 16 (62.08s) | 1 (3.88s) |
| Manual intervention | Required | Never |
| Node count | 16x HUMO | 1x HUMO |
| Memory usage | High | Lower |
| Regenerate chunk | Difficult | Easy |
| Auto-queue | Partial | Full |
| Workflow complexity | Complex | Simple |
| Partial song handling | Manual | Automatic |

---

## Migration from V9

### Quick Steps:
1. Backup your V9 workflow file
2. Create new V10 workflow
3. Replace old nodes with new V4 nodes
4. Connect according to migration guide
5. Test with short song first
6. Process full songs automatically!

**See V9_TO_V10_MIGRATION_GUIDE.md for complete instructions.**

---

## Requirements

### Dependencies (same as before)
- ComfyUI
- HUMO models
- Whisper (for transcription)
- librosa
- imageio
- torch/torchaudio

### Optional
- LLM integration for prompt generation (placeholder in current version)

---

## Examples

### Example 1: Process Entire Song
```python
# Set in workflow:
auto_index = True
output_folder = "my_music_video"

# Click "Queue Prompt" once
# → Automatically processes all chunks!
```

### Example 2: Regenerate Chunk 10
```python
# Set in workflow:
auto_index = False
manual_index = 10
output_folder = "my_music_video"

# Click "Queue Prompt"
# → Only chunk 10 regenerates
```

### Example 3: Resume Interrupted Processing
```python
# Set in workflow:
auto_index = True
output_folder = "my_music_video"  # Already has chunks 0-20

# Click "Queue Prompt"
# → Auto-detects 21 chunks exist
# → Starts from chunk 21
# → Continues to end
```

---

## Troubleshooting

### Q: Chunks not auto-queueing?
**A:** Check `auto_index=True` in ChunkIndexController

### Q: Wrong chunk being processed?
**A:** Check marker files: `ls video_chunk_*-audio.mp4`

### Q: Prompts regenerating every chunk?
**A:** Use fixed seed, don't change audio or parameters

### Q: Want to start fresh?
**A:** Delete all `video_chunk_*.mp4` files

### Q: Last chunk has silence?
**A:** Expected! Padding for HuMo compatibility (1-3s max)

---

## Performance

### Memory
- **V3:** 16 HUMO models in memory simultaneously
- **V4:** 1 HUMO model at a time
- **Reduction:** ~94% memory savings

### Speed
- **V3:** Parallel processing (if GPU supports)
- **V4:** Sequential processing
- **Trade-off:** Lower memory vs longer total time
- **Benefit:** Caching makes subsequent operations instant

---

## Future Enhancements

Planned for future releases:
- [ ] LLM integration for dynamic prompt generation
- [ ] Story mode support (multi-chapter narratives)
- [ ] Advanced caching strategies
- [ ] Batch final video assembly node
- [ ] Progress visualization
- [ ] Chunk preview thumbnails

---

## Contributing

Found a bug? Have a suggestion?

1. Check existing issues on GitHub
2. Create new issue with:
   - Description of problem/feature
   - Steps to reproduce (if bug)
   - Workflow JSON (if applicable)
   - Error logs

Pull requests welcome!

---

## Credits

- **HuMo Model:** [Phantom-video/HuMo](https://github.com/Phantom-video/HuMo)
- **ComfyUI:** [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- **Whisper:** OpenAI Whisper ASR

---

## License

Same license as the main repository.

---

## Support

- 📖 **Documentation:** V4_SINGLE_CHUNK_DOCUMENTATION.md
- 🔄 **Migration Guide:** V9_TO_V10_MIGRATION_GUIDE.md
- 🐛 **Issues:** GitHub Issues page
- 💬 **Discussions:** GitHub Discussions

---

**Enjoy simplified, fully automated music video generation with V4!** 🎵🎬✨

*No more manual intervention. No more complex calculations. Just load your audio and let it run.*
