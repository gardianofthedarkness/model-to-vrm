# Flat colours and parts for an anime character's 3D model

You are colouring and labelling a 3D model of an anime character (Rikka) so it matches her original drawing with clean,
flat anime colours, and so every part says what it is. The model came from an AI generator: its texture is roughly right
but blurry, and its 51 pieces ("units") do not always follow real part boundaries (one piece can hold face skin AND hair).

## Files
- `setting_reference.jpg` - the original drawing. It is the truth for colours and parts.
- `renders/compare.jpg` - top row: the model with its blurry AI texture; bottom row: the model in the flat colours you have
  set so far (front, back, head). MAGENTA = a unit with no colour yet.
- `cards/card_<unit>.jpg` - a unit outlined in magenta (left zoom, right whole model). Make cards with the card tool.
- `sheet.json` - what is recorded per unit. `NOTES.md` - your memory: read it first, keep it up to date.

## Tools (run with bash from this folder; Python is E:/splat/.venv/Scripts/python.exe)
- `PY tools/show.py [unit...]` - list units and what is recorded (no args: all units + how many still lack a colour)
- `PY tools/card.py <unit> [...]` - make cards and print each unit's MEASURED candidate colours
- `PY tools/split.py <unit>` - carve a unit into its colour regions (<unit>.1, <unit>.2, ...), each with a card and candidates.
  Use it when a unit mixes things (e.g. skin and hair, a garment and its stripes, leg skin and a sock).
- `PY tools/merge.py <unit>` - undo a split (or `merge.py A B C` to join units into A)
- `PY tools/set.py <unit> name=... category=... material=... stiffness=... colour=#hex movable=yes|no notes="..."`
  category: skin | hair | cloth | leather | rigid accessory | eye | mouth | other; stiffness: soft | medium | stiff.
  colour must be one of the unit's measured candidates (or its parent's); the tool refuses anything else.
- `PY tools/render.py` - re-render `renders/compare.jpg` from the sheet (takes about a minute)

## How to work (a loop)
1. Read NOTES.md. Look at `setting_reference.jpg`.
2. Make cards for the units you have not seen (`card.py`), look at them, and set each unit: name, category, material,
   stiffness, colour (the flat anime base colour, not a shading tone), movable (does it move on its own: iris, mouth, eyelid,
   hair lock = yes; button, stripe, badge = no).
3. When a card shows a unit holding two different things, split it, look at the regions' cards, and set each region.
   Very small regions that are only texture noise can be merged back into their neighbour.
4. Things that are one garment or one material get the SAME colour (all skin one skin colour, the whole cardigan one white...).
5. Render, read `renders/compare.jpg`, compare the bottom row with the drawing. Note what is wrong in NOTES.md, fix it, render again.
6. Stop when `show.py` reports no unit without a colour and the flat render matches the drawing as closely as the units allow.
   Write a final summary at the end of NOTES.md: what you did, what you split, what still looks wrong and why.

Rules: do not edit the tools, data.npz, owner.npy or sheet.json by hand - only through the tools. Do not invent colours.
Keep NOTES.md short and current: a list of decisions (unit -> what and why) and open problems.
