# Part 1 summary

## Top-3 neighbours before/after

| word   | pre top1         | post top1         | pre top2          | post top2          | pre top3           | post top3        |
|:-------|:-----------------|:------------------|:------------------|:-------------------|:-------------------|:-----------------|
| cast   | casts (0.722)    | actors (0.653)    | casting (0.719)   | supporting (0.645) | Cast (0.664)       | ensemble (0.593) |
| score  | scoring (0.720)  | morricone (0.675) | scores (0.660)    | music (0.671)      | scored (0.638)     | ennio (0.665)    |
| plot   | plots (0.762)    | storyline (0.750) | Plot (0.652)      | story (0.711)      | plotting (0.633)   | plotline (0.678) |
| screen | screens (0.773)  | screens (0.597)   | onscreen (0.612)  | onscreen (0.523)   | LCD_screen (0.560) | stage (0.471)    |
| review | reviewed (0.663) | comment (0.660)   | reviewing (0.661) | reviews (0.617)    | reviews (0.638)    | nixflix (0.581)  |

## Vector shift

| word   |   cos(original, fine-tuned) |   shift (1 - cos) |   L2 distance |
|:-------|----------------------------:|------------------:|--------------:|
| score  |                      0.5299 |            0.4701 |        3.6714 |
| review |                      0.5384 |            0.4616 |        3.4615 |
| plot   |                      0.5929 |            0.4071 |        2.8017 |
| cast   |                      0.6138 |            0.3862 |        2.7176 |
| screen |                      0.6497 |            0.3503 |        2.6774 |

- Most shifted: **score** (cos = 0.5299)
- Least shifted: **screen** (cos = 0.6497)
