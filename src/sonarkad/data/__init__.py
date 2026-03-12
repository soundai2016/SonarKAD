"""Datasets and data readers."""

from .sio import SIOReader, SIOHeader, read_sio_header
from .swellex96 import (
    TonalExtractionConfig,
    extract_tonal_rl_db,
    get_tone_frequencies,
    load_range_table,
    find_range_table_file,
    load_element_depths_m,
    load_vla_depths_m,
    save_processed_npz,
)
from .ctd import (
    CTDCast,
    load_ctd_casts,
    representative_sound_speed_profile,
    estimate_depth_averaged_c0,
)

__all__ = [
    "SIOReader",
    "SIOHeader",
    "read_sio_header",
    "TonalExtractionConfig",
    "extract_tonal_rl_db",
    "get_tone_frequencies",
    "load_range_table",
    "find_range_table_file",
    "load_element_depths_m",
    "load_vla_depths_m",
    "save_processed_npz",
    "CTDCast",
    "load_ctd_casts",
    "representative_sound_speed_profile",
    "estimate_depth_averaged_c0",
]
