"""
Simulation mode registry.
Author contribution: Paul Völker implemented this project part.
"""


from . import wave_field
from . import particle_hits
from . import bohmian_paths

MODES = {
    wave_field.TITLE: wave_field,
    particle_hits.TITLE: particle_hits,
    bohmian_paths.TITLE: bohmian_paths,
}