# Tokenizer vocabulary — NOT 5 symbols

Finding **G7** (cycle 2, measured): **8.90%** of polymer residues in the sampled
structures fall outside `{A,C,G,U}`:

| Class | Count | Representable in a 5-symbol vocab? |
|---|---|---|
| DNA (DA/DC/DG/DT/DU) — hybrid duplexes | 17,767 | **no** |
| UNK — identity unmodelled | 7,036 | only as `N` |
| Inosine and other true RNA modifications | 2,634 | **no** |

elDORS is pre-normalised to 5 symbols, which is why vocab-5 looked sufficient
for *pretraining*. **Structures are not normalised**, so the structural side
needs more. Nature has >170 RNA modifications and tRNA is among the most
heavily modified.

## Decision

Two vocabularies, one model:

- **Pretraining (sequence)**: 5 symbols + specials. elDORS is already normalised;
  widening here would add dead tokens, which is the NucleicBERT vocab-25 mistake.
- **Structural (mmCIF-derived)**: 5 + `DNA{A,C,G,T,U}` + `I` (inosine) +
  `UNK` + a learned `MOD` embedding keyed by the mmCIF `comp_id`, falling back
  to the parent base. Modified residues then have geometry predicted rather than
  being silently mapped to `N`.

**Until this is implemented, the honest statement is**: modified residues are
mapped to `N` and their geometry is not predicted. That must appear in any
claim about handling "any RNA".
