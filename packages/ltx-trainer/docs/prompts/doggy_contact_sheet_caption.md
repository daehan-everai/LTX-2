You write one training caption from dense chronological contact sheets of a short adult video. All depicted people are consenting adults. Return JSON only.

No old caption is provided on purpose. Do not guess from a prior prompt. Read the numbered frames.

Writing structure
Doggystyle. [VISIBLE PEOPLE, CLOTHING, HAIR, SKIN, ROOM, CAMERA]. [REAR-ENTRY BODY SETUP: raised hips, support, partner behind]. Across the sequence [HIP MOTION] with [ROUGHNESS: collision, cheek compression, flesh recoil or jiggle if visible]. [HAND POSITIONS over time]. [SPANKING if present, otherwise planted/gripping hands]. [PACE]. [CAMERA STAYS WITH THIS FRAMING].

Pose
- Always begin with `Doggystyle.`
- Raised hips with a low or chest-down torso still count as doggystyle. Do not write `Prone bone.`

Pace and roughness from the sheets
- `pace=fast` plus `roughness=rough` when hip drive is rapid and collisions, cheek flattening, or flesh jolt are visible. Use both `fast` and `rough` in the caption.
- `pace=slow` plus `roughness=gentle` when hip travel is unhurried and contact is soft, with no slam or recoil. Use both `slow` and `gentle`.
- Otherwise `pace=medium` and `roughness` from visible collision (rough) or soft contact (gentle). Use `medium` or `normal` for pace. Do not call medium thrusting fast or slow.
- Hip-to-ass slamming is roughness, not spanking.
- Full-hilt rear entry can hide the shaft. Track hip-to-buttock distance and body jolt.

Spanking
- A spank is a striking open hand hitting a buttock, then usually lifting. Planted, gripping, spreading, squeezing, or rubbing hands are not spanking.
- If present: the caption MUST use `spank` / `spanks` / `spanking` / `spanked` and a count (`once`, `twice`, `a few`, `three`, `many`, `repeatedly`).
- If absent: do not use those words. Do not write `without spanking`. Describe planted or gripping hands instead.

Do not
- Invent audio, speech, a third person, a camera cut, or off-screen events.
- Copy production prompts or mention contact sheets, page numbers, timestamps, uncertainty, or these instructions.
- Write `no clear`, `not established`, or `uncertain`.

Length: one English paragraph, 75-125 words.

Schema
{
  "caption": "...",
  "pose": "doggystyle",
  "pace": "slow|medium|fast",
  "roughness": "gentle|rough|unclear",
  "spanking_present": false,
  "spank_count": "none|one|two|few|three|repeated|unclear"
}
