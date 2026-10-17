# Dataset Loading — Analisi e Ottimizzazione

## Situazione attuale

```
DISCO (.pt files)
     │
     ▼
__init__()
  ├─ legge ogni file una volta  →  estrae solo bar_positions
  ├─ costruisce self.chunks = [(file_idx, start_bar), ...]
  └─ BUTTA VIA i dati letti (garbage collected)

     │  training loop
     ▼

__getitem__(idx)
  ├─ per ogni sample: rilegge il file da disco
  ├─ estrae gli 8 bar del chunk
  ├─ applica masking
  └─ restituisce il sample

     │
     ▼

DataLoader (4 workers paralleli)
  └─ ogni worker chiama __getitem__ indipendentemente
     → 4 letture disco simultanee, nessuna cache condivisa
```

**Risultato**: con ~10 chunk per file, ogni `.pt` viene letto dal disco ~10 volte per epoch.
Con 50k file sono ~500k letture disco/epoch.

---

## Cosa modifichiamo

```
DISCO (.pt files)
     │
     ▼
__init__()
  ├─ legge ogni file una volta
  ├─ costruisce self.chunks = [(file_idx, start_bar), ...]
  └─ MANTIENE i dati in self._data_cache = {file_idx: data}  ← NUOVO

     │  training loop
     ▼

__getitem__(idx)
  ├─ legge self._data_cache[file_idx]  →  O(1), zero I/O disco
  ├─ estrae gli 8 bar del chunk
  ├─ applica masking
  └─ restituisce il sample
```

---

## Perché

| | Prima | Dopo |
|---|---|---|
| Letture disco/epoch | ~500k (50k file × 10 chunk) | 0 durante il training |
| Letture disco totali (training) | ~500k × N epoche | 50k (solo al caricamento iniziale) |
| Overhead `torch.load` | ad ogni sample | una tantum in `__init__` |
| Costo | bottleneck I/O → CPU worker in attesa | RAM occupata (~1.5 GB per 50k file) |

Il trade-off è semplice: si paga una volta in RAM invece di pagare ad ogni step in I/O disco.
Dato che il cluster ha abbondante RAM e i worker del DataLoader sono altrimenti in attesa del disco,
il guadagno è netto.

---

## Stima RAM

| Dataset | File | Stima media/file | RAM totale |
|---|---|---|---|
| preprocessed_v2_5k | ~3k | ~30 KB | ~90 MB |
| preprocessed_v2_50k | ~50k | ~30 KB | ~1.5 GB |

---

## File coinvolti

- `midit/data/lakh_dataset.py` — modifica in `__init__` (cache) e `__getitem__` (lettura da cache)
