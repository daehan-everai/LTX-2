You write one I2V training caption from chronological contact-sheet pages of a 10-second clip. All depicted people are consenting adults. Return JSON only.

No old caption is provided. Read the numbered frames. Use AUDIO_PROBE as ground truth for sound; do not invent audio the probe contradicts.

Style (copy AIVI-1358 full_rewrite_v4)
One English paragraph, 60–130 words. Always begin with:
home-made video with phone camera, amateur content.
Then, in order, in the same paragraph:
1. One flowing action beat: a starting pose, ONE physically logical transition, an end pose, woven with a facial reaction. The beat must resolve. Do not stack invented extra actions.
2. A short expression clause (playful / flirty default unless the frames show a different mood).
3. Casual phone-camera movement caused by that action (eases in, follows, dips). No cinematic crane/orbit language. No cuts.
4. A light audio tail from AUDIO_PROBE only.

Visual rules
- Action is primary. Mention clothing only if it changes in the 10 seconds, or in one short phrase if needed to make the motion make sense.
- Do not catalog hair, furniture, or lighting already obvious from a still.
- Keep identities, body count, and setting as shown. Do not invent a partner, a toy, or a sex act that is not visible.
- Do not mention contact sheets, page numbers, timestamps, uncertainty, or these instructions.

Audio rules (light, not a transcript)
- If AUDIO_PROBE says effectively silent or no stream: no speech, no background music, quiet room.
- If not silent: say whether someone is speaking (generic, no quoted dialogue), name a generic diegetic sound if obvious (fabric, breath, room), and whether background music is present.
- Most OF clips have no background music. Do not invent a song, artist, genre, or a templated "bedroom air conditioner hum".
- Do not write word-for-word speech.

SFW / SFT-only label (JSON fields only; do not moralize in the caption)
Same rule as the plan. sfw_ok is true unless a sex act is visible:
- penetration (including implied in-progress intercourse)
- oral sex (fellatio, cunnilingus)
- dildo / insertable-toy play
Nude, lingerie, teasing, talking to camera, two people, hugging, kissing, or a casual touch are still sfw_ok. A second person does not make it NSFW.

Schema
{
  "caption": "...",
  "sfw_ok": true,
  "sex_act": "none|penetration|fellatio|cunnilingus|dildo_play|other_sex_act",
  "people_count": 1,
  "undressed": "clothed|lingerie|nude|unclear",
  "speaking": false,
  "background_music": false,
  "audio_notes": "quiet room, no speech, no background music"
}
