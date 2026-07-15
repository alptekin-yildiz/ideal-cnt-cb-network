"""
Filler material definitions for the reference model.

Physical and electrical properties for the carbon black (CB) and carbon
nanotube (CNT) fillers used by the percolation threshold generators.

Active material constructors
----------------------------
carbon_black()   : FillerMaterial - spherical CB particle model
carbon_nanotube(): FillerMaterial - wavy MWCNT rod model
"""

from dataclasses import dataclass


@dataclass
class FillerMaterial:
    """Filler material properties."""
    name: str
    filler_type: str           # "CB" or "CNT"
    density: float             # g/cm^3
    conductivity: float        # S/m

    # Geometric parameters
    diameter_nm: float         # Particle diameter [nm]
    length_nm: float = 0.0     # CNT only: contour length [nm]
    aspect_ratio: float = 1.0  # L/d

    # CNT waviness parameters
    waviness: float = 1.0      # cos(bend angle) between successive segments (1.0 = straight)
    n_segments: int = 10       # Number of worm-like chain segments


def carbon_black(
    diameter_nm: float = 148.0,
    conductivity: float = 1e4,
) -> FillerMaterial:
    """
    Create a carbon black (CB) filler definition.

    Parameters
    ----------
    diameter_nm : float
        Effective CB particle diameter [nm] used by the ideal-network
        model, where one sphere is one conducting unit. For fused carbon
        black this is the effective aggregate diameter (the CLI defaults
        and the companion paper centre on 148 nm); bare primary particles
        are typically 20-80 nm.
    conductivity : float
        Intrinsic conductivity [S/m]. Typical range: 1e3-1e5 S/m.
    """
    return FillerMaterial(
        name="Carbon Black",
        filler_type="CB",
        density=1.80,            # g/cm^3 - CB skeletal density (helium pycnometry ~1.8-2.1)
                                 # Catalog values near 0.4 g/cm^3 are bulk/apparent density.
        conductivity=conductivity,
        diameter_nm=diameter_nm,
        length_nm=diameter_nm,   # Spherical particle: L = d
        aspect_ratio=1.0,
        waviness=1.0,
        n_segments=1,
    )


def carbon_nanotube(
    diameter_nm: float = 10.0,
    length_um: float = 5.0,
    waviness: float = 0.7,
    n_segments: int = 10,
    conductivity: float = 1e5,
) -> FillerMaterial:
    """
    Create a carbon nanotube (CNT) filler definition.

    Parameters
    ----------
    diameter_nm : float
        CNT diameter [nm]. Typical range: MWCNT 5-20 nm, SWCNT 1-2 nm.
    length_um : float
        CNT contour length [um]. Typical range: 1-25 um. The default (5 um)
        is a generic MWCNT; the generate/ CLIs and the companion paper
        centre on shorter 0.5 um (500 nm) populations and always pass the
        length explicitly.
    waviness : float
        Cosine of the bend angle between successive chain segments
        (a local orientation correlation, not an end-to-end-to-contour
        ratio). 1.0 = straight; lower values = wavier. The realized
        RMS end-to-end-to-contour ratio is lower than w itself
        (about 0.645 for w = 0.7 with 10 equal segments).
    n_segments : int
        Number of worm-like chain segments.
    conductivity : float
        Intrinsic conductivity [S/m]. Typical range: 1e5-1e6 S/m.
    """
    length_nm = length_um * 1e3  # um -> nm
    return FillerMaterial(
        name="Carbon Nanotube",
        filler_type="CNT",
        density=1.75,            # g/cm^3 - MWCNT skeletal density (Zhang 2024/Al-Saleh 2009: 1.3-1.75)
                                 # Bulk powder values near 0.05-0.15 g/cm^3 are not used here.
        conductivity=conductivity,
        diameter_nm=diameter_nm,
        length_nm=length_nm,
        aspect_ratio=length_nm / diameter_nm,
        waviness=waviness,
        n_segments=n_segments,
    )
