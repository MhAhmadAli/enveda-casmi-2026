"""Masses, adduct arithmetic, molecular formulae, and InChIKey14 canonicalisation."""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

ELECTRON = 0.00054857990946

# Monoisotopic masses of the most abundant isotope (CODATA/AME2020 derived).
MONO = {
    "H": 1.0078250319, "D": 2.0141017780, "C": 12.0, "N": 14.0030740052,
    "O": 15.9949146221, "F": 18.99840320, "Na": 22.98976928, "Mg": 23.98504170,
    "Si": 27.9769265327, "P": 30.97376151, "S": 31.97207069, "Cl": 34.96885271,
    "K": 38.96370649, "Ca": 39.96259098, "Fe": 55.93493633, "Co": 58.93319429,
    "Ni": 57.93534241, "Cu": 62.92959772, "Zn": 63.92914201, "As": 74.92159457,
    "Se": 79.91652132, "Br": 78.91833710, "I": 126.904473, "B": 11.0093054,
    "Li": 7.01600344, "Al": 26.98153853, "Mn": 54.93804391, "Hg": 201.9706434,
    "Pt": 194.9647917, "Sn": 119.90220163, "Ag": 106.9050916, "Au": 196.96656879,
    "Ba": 137.90524700, "Cd": 113.90336509, "Cr": 51.94050623, "Mo": 97.90540482,
    "Pb": 207.9766521, "Ti": 47.94794198, "V": 50.94395704, "W": 183.95093092,
}

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> Counter:
    """'C6H12O6' -> Counter({'C': 6, 'H': 12, 'O': 6}). Ignores charge suffixes."""
    core = formula.strip().rstrip("+-")
    out: Counter = Counter()
    pos = 0
    for m in _FORMULA_TOKEN.finditer(core):
        if m.start() != pos:
            raise ValueError(f"unparsable formula {formula!r} at {pos}")
        pos = m.end()
        out[m.group(1)] += int(m.group(2)) if m.group(2) else 1
    if pos != len(core):
        raise ValueError(f"unparsable formula {formula!r} at {pos}")
    return out


def formula_mass(formula: str | Counter) -> float:
    counts = formula if isinstance(formula, Counter) else parse_formula(formula)
    return sum(MONO[el] * n for el, n in counts.items())


def formula_str(counts: Counter) -> str:
    """Hill notation."""
    parts = []
    for el in ("C", "H"):
        n = counts.get(el, 0)
        if n:
            parts.append(el + (str(n) if n > 1 else ""))
    for el in sorted(k for k in counts if k not in ("C", "H")):
        n = counts[el]
        if n:
            parts.append(el + (str(n) if n > 1 else ""))
    return "".join(parts)


# --- adducts ---------------------------------------------------------------

_ADDUCT_RE = re.compile(r"^\[(\d*)M(?P<mods>(?:[+-][A-Za-z0-9]+)*)\](?P<chg>\d*[+-])$")


@lru_cache(maxsize=4096)
def parse_adduct(adduct: str) -> tuple[int, float, int]:
    """Return (n_mers, mass_delta, charge) for an adduct string like '[M+CH2O2-H]-'.

    mass_delta is the mass added to n_mers * M, before the electron correction,
    so: mz = (n_mers * M + mass_delta - charge * ELECTRON) / abs(charge)
    """
    m = _ADDUCT_RE.match(adduct.strip().replace(" ", ""))
    if not m:
        raise ValueError(f"unparsable adduct {adduct!r}")
    n_mers = int(m.group(1)) if m.group(1) else 1
    chg_s = m.group("chg")
    mag = int(chg_s[:-1]) if len(chg_s) > 1 else 1
    charge = mag if chg_s[-1] == "+" else -mag
    delta = 0.0
    for sign, token in re.findall(r"([+-])([A-Za-z0-9]+)", m.group("mods")):
        mass = formula_mass(token)
        delta += mass if sign == "+" else -mass
    return n_mers, delta, charge


def precursor_to_neutral_mass(precursor_mz: float, adduct: str) -> float:
    """Neutral monoisotopic mass M implied by an observed precursor m/z."""
    n_mers, delta, charge = parse_adduct(adduct)
    return (precursor_mz * abs(charge) - delta + charge * ELECTRON) / n_mers


def neutral_mass_to_precursor(mass: float, adduct: str) -> float:
    n_mers, delta, charge = parse_adduct(adduct)
    return (n_mers * mass + delta - charge * ELECTRON) / abs(charge)


# --- structure canonicalisation -------------------------------------------

_taut = None


def _tautomer_enumerator():
    global _taut
    if _taut is None:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        _taut = rdMolStandardize.TautomerEnumerator()
    return _taut


def inchikey14(smiles: str, tautomer_canonical: bool = True) -> str | None:
    """Connectivity block (first 14 chars) of the InChIKey of the tautomer-canonical
    form of `smiles`. This is the unit the competition metric compares."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if tautomer_canonical:
        try:
            mol = _tautomer_enumerator().Canonicalize(mol)
        except Exception:
            pass
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    return key[:14] if key else None

def neutral_masses(precursor_mz, adducts) -> "np.ndarray":
    """Vectorised neutral monoisotopic mass for many spectra at once.

    Adducts repeat heavily (a few dozen distinct strings over millions of spectra),
    so parse each distinct string once and apply it to all its rows. Rows whose
    adduct cannot be parsed come back as NaN rather than raising, since library
    metadata contains occasional malformed values.
    """
    import numpy as np

    pmz = np.asarray(precursor_mz, dtype=np.float64)
    add = np.asarray(adducts, dtype=object)
    out = np.full(pmz.shape, np.nan, dtype=np.float64)
    for a in set(add.tolist()):
        try:
            n_mers, delta, charge = parse_adduct(str(a))
        except Exception:
            continue
        sel = add == a
        out[sel] = (pmz[sel] * abs(charge) - delta + charge * ELECTRON) / n_mers
    return out
