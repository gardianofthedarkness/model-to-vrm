# Build the parts tree of an anime character's 3D model (entity matching)

The model is Rikka (see `setting_reference.jpg`, the original drawing: the truth). It has already been cut into PARTS:
- **colour regions**: big parts found by colour differencing on the model's texture (hair, skin, cardigan, skirt...).
  Each also names the Tripo piece it lies on (a hint, sometimes wrong: Tripo pieces mix things).
- **sam3 details**: small parts the SAM 3 segmenter found on close-ups (eye, iris, pupil, eye white, eyelash, eyebrow,
  mouth, nose, ear, button, badge, cuff stripe, pocket stripe). They are trusted for WHAT they are; check WHERE they are.

Your job: compose ONE tree of the character from these parts.

## Rules
1. Start from the whole (`body`, already there). Under it make the big entities first: `skin`, `hair`, and one node per
   garment (cardigan, shirt, collar/bow, skirt, socks, shoes, scrunchies...). Each big entity gets one overall colour.
2. **Same thing = same node.** All skin (face, neck, hands, legs) is ONE `skin` node with one colour. All pieces of one
   garment go in that garment's node (left and right sleeves, front and back of the cardigan). Do NOT make a node per piece.
   Colour regions that are only shading of something (a darker fold of the cardigan, a shadow under the hair) belong to
   that thing's node.
3. Then hang the smaller things under what they sit on: eyes and mouth under the face area of `skin` (make `face` a
   subpart of skin if useful), iris and eye white under eye, pupil and highlight under iris, buttons/badge/pockets under the
   cardigan, cuff stripes under the sleeve... Use the sam3 detail's `inside` hint and its card.
4. A node's colour is computed from its parts (you never type a colour). A `subpart` (face, sleeve, pocket) shows its
   entity's colour (it is the same stuff); a `detail` (eye, button, stripe) shows its own. If a node's colour is wrong, a wrong part is in it:
   look at the cards and move the part.
5. Mark `movable=yes` for things that move on their own: iris/pupil, eyelids, mouth, eyebrows, hair locks, skirt, bow ends.
6. A colour region can hold two things (the colour carving merges small neighbours). Look at the render: a patch of
   the wrong colour (pink on hair, dark on skin) means a mixed part; find it with cards and split it.
6b. Every part must end up in exactly one node. Tiny parts that are only texture noise go into the node they sit on.
7. SAM 3 can be wrong. Check every sam3 detail on its card against the drawing: a "nose" bigger than an eye, an "iris"
   on hair, a "pupil" covering the whole coloured disc. A wrong detail goes into the node it really is (usually
   what it sits on, e.g. skin), and you note it in NOTES.md. A detail with the wrong NAME but the right shape (the
   blue disc called "pupil") goes into the right node (iris).
8. The colour region hint "None" means the Tripo piece had no name; judge from colour, centre and card.

## Tools (bash from this folder; PY = E:/splat/.venv/Scripts/python.exe)
- `PY tools/parts.py [unassigned]` - list the parts (id, kind, points, colour, centre, hint, node)
- `PY tools/card.py <id> [...]` - cards/card_<id>.jpg: the part in magenta on the front, back and best close-up render
- `PY tools/node.py <node> parent=<node> kind=entity|garment|subpart|detail movable=yes|no notes="..."` - make/edit a node
- `PY tools/assign.py <node> <id> [...]` - put parts in a node (moves them if already assigned)
- `PY tools/split.py <id>` - split a part that holds two things (e.g. skin with a hair lock) by colour into new parts;
  the pieces keep the old node, then reassign them. It refuses parts that are only one colour.
- `PY tools/tree.py` - print the tree with each node's colour and how many parts are unassigned
- `PY tools/render.py` - renders/compare.jpg: top = original, bottom = each part in its node's colour (magenta = unassigned)

## Loop
1. Read NOTES.md (your memory) and the drawing.
2. List parts. Look at cards of parts you are unsure about (you can pass many ids to card.py at once). Big parts first.
3. Build nodes and assign. Render, compare the bottom row with the drawing, fix, render again.
4. Stop when tree.py says 0 unassigned and the render matches the drawing's parts. Write in NOTES.md: the final tree in short,
   decisions that were not obvious, and what still looks wrong and why.

Do not edit tools, part_list.json, parts.npz or tree.json by hand. Keep NOTES.md short and current.
