"""
Simulation mode registry.
Author contribution: Paul Völker implemented this project part.
"""

from modes import wave_field
from modes import particle_hits

MODES = {
    wave_field.TITLE: wave_field,
    particle_hits.TITLE: particle_hits,
}