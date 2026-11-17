"""Load config.toml and streamers.toml, filling in defaults for anything missing.

Why defaults in code AND a config file?
  config.toml is where you tune things. The defaults below mean an old or partial
  config.toml still works, and the tests can run without any config file at all.
"""

from __future__ import annotations

import copy
try:
    import tomllib  # built into Python 3.11+
except ModuleNotFoundError:  # Python 3.10: same reader, installed from pip as "tomli"
    import tomli as tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict = {   'general': {'top_n': 15, 'top_n_detect': 40, 'explore_share': 0.15},
    'chat': {   'lag_seconds': 6,
                'smooth_seconds': 5,
                'baseline_seconds': 300,
                'per_user_cap': 2,
                'hype_ratio': 0.5,
                'hype_words': [   'kekw',
                                  'omegalul',
                                  'lul',
                                  'lmao',
                                  'lmfao',
                                  'pog',
                                  'pogchamp',
                                  'poggers',
                                  'pogu',
                                  'w',
                                  'ww',
                                  'www',
                                  'monkas',
                                  'pepelaugh',
                                  'icant',
                                  'sadge',
                                  'holy',
                                  'wtf',
                                  'bro',
                                  '💀',
                                  '😂',
                                  '🤣',
                                  '😭',
                                  '🔥'],
                'hype_phrases': ['clip it', 'clip that', 'no way', 'oh my god', 'omg'],
                'bots': [   'fossabot',
                            'nightbot',
                            'streamelements',
                            'streamlabs',
                            'moobot',
                            'wizebot',
                            'sery_bot',
                            'soundalerts',
                            'pokemoncommunitygame',
                            'kofistreambot',
                            'botrixoficial',
                            'kickbot'],
                'moods': {   'funny': [   'kekw',
                                          'omegalul',
                                          'lul',
                                          'lmao',
                                          'lmfao',
                                          'icant',
                                          'pepelaugh',
                                          'lol',
                                          'haha',
                                          '😂',
                                          '🤣',
                                          '💀',
                                          '😭'],
                             'hype': [   'pog',
                                         'pogchamp',
                                         'poggers',
                                         'pogu',
                                         'w',
                                         'lets go',
                                         'clip it',
                                         'clip that',
                                         'goat',
                                         '🔥',
                                         'insane'],
                             'tense': ['monkas', 'monkaw', '😬', 'holy', 'nervous', 'dont', 'noooo'],
                             'confusion': ['huh', 'what', 'wtf', 'wait what', 'how', '🤔'],
                             'fail': ['l', 'f', 'rip', 'unlucky', 'noob', 'fail', 'ratio', '🤡']}},
    'audio': {'sample_rate': 8000, 'baseline_seconds': 300, 'music_removal': 'off'},
    'signals': {   'floor': {   'chat': 3.0,
                                'unique': 2.0,
                                'hype': 1.0,
                                'audio': 2.0,
                                'viewer_clips': 0.5,
                                'spam': 1.0},
                   'viewer_clip_sigma': 8,
                   'viewer_clip_center': 0.6},
    'weights': {   'chat': 1.0,
                   'unique': 0.8,
                   'hype': 1.2,
                   'audio': 0.7,
                   'viewer_clips': 1.5,
                   'spam': 0.6},
    'scoring': {'z_cap': 8, 'both_bonus': 0.5, 'smooth_seconds': 3},
    'peaks': {'threshold': 2.0, 'min_gap_seconds': 45},
    'clip': {   'min_seconds': 15,
                'max_seconds': 50,
                'target_seconds': 30,
                'pre_roll': 8,
                'build_up_max': 20,
                'post_roll': 3,
                'tail_max': 45,
                'decay_ratio': 0.35,
                'keep_before_ratio': 0.45,
                'snap_back_max': 8,
                'snap_forward_max': 6,
                'peak_guard': 3,
                'viewer_clip_bounds': True},
    'whisper': {   'model': 'large-v3',
                   'device': 'cuda',
                   'compute_type': 'int8_float16',
                   'window_padding': 10,
                   'fallback_model': 'small',
                   'sentence_gap': 0.8},
    'launch': {   'gta6_us_launch_utc': '2026-11-19T00:00:00-05:00',
                  'title_patterns': ['gta ?6', 'gta ?vi\\b', 'grand theft auto ?(vi|6)\\b'],
                  'launch_mode': True},
    'framecheck': {   'enabled': True,
                      'analysis_format': 'bv*[height<=360]+ba/b[height<=360]/worst',
                      'letterbox_share': 0.1,
                      'letterbox_luma': 20,
                      'letterbox_seconds': 4,
                      'hud_missing_ratio': 0.45,
                      'hud_missing_seconds': 5,
                      'blank_std': 8,
                      'blank_seconds': 3},
    'judge': {   'enabled': True,
                 'model': 'claude-haiku-5-5',
                 'effort': 'low',
                 'max_candidates': 40,
                 'max_tokens': 4000,
                 'chat_sample': 30,
                 'frames': 15,
                 'frame_width': 512,
                 'window_before': 8,
                 'window_after': 6,
                 'fallbacks': True,
                 'warn_on': ['high'],
                 'cutscene_quiet_seconds': 12,
                 'chat_quiet_z': 0.0,
                 'neutral_score': 5,
                 'rank_weight': 0.3,
                 'signal_power': 0.5,
                 'ai_power': 2.0},
    'download': {'padding': 2, 'preview_format': 'bv*[height<=720]+ba/b[height<=720]/b'},
    'edit': {   'top_n': 10,
                'source_format': 'bv*[height<=1080]+ba/b[height<=1080]/b',
                'fps': 30,
                'preset': 'veryfast',
                'crf': 20,
                'facecam_share': 0.35,
                'blur_strength': 10,
                'font': 'Arial Black',
                'caption_size': 84,
                'caption_max_words': 3,
                'caption_max_chars': 16,
                'caption_gap': 0.6,
                'caption_uppercase': True,
                'caption_highlight': '&H00FFFF&',
                'caption_lead': 0.1,
                'caption_y_no_facecam': 0.62,
                'caption_y_crop': 0.62,
                'hook': True,
                'hook_size': 64,
                'hook_seconds': 2.2,
                'hook_y': 0.17,
                'banner': True,
                'banner_size': 44,
                'credit': True,
                'credit_size': 34,
                'dead_air': True,
                'dead_air_gap': 0.7,
                'dead_air_keep': 0.15,
                'dead_air_quiet_db': 4,
                'zoom': True,
                'zoom_factor': 1.12,
                'zoom_seconds': 1.4,
                'teaser_seconds': 0,
                'loudness_lufs': -14,
                'duck_music_db': 0,
                'safe_top': 288,
                'safe_bottom': 1248,
                'safe_left': 120,
                'safe_right': 888,
                'cover': True},
    'queue': {   'host': '0.0.0.0',
                 'port': 8000,
                 'trim_step': 2,
                 'min_seconds': 5,
                 'max_seconds': 90,
                 'submit_minutes': 30},
    'live': {   'poll_seconds': 60,
                'check_seconds': 5,
                'window_seconds': 900,
                'warmup_seconds': 240,
                'edge_seconds': 60,
                'confirm_seconds': 45,
                'max_peaks_per_check': 10,
                'buffer_minutes': 6,
                'segment_seconds': 10,
                'buffer_format': 'best[height<=1080]/best',
                'x264_threads': 4,
                'whisper_model': ''}}


def _merge(base: dict, override: dict) -> dict:
    """Recursively copy `override` on top of `base` (nested tables are merged, not replaced)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path | None = None) -> dict:
    """Return the full config as a nested dict, e.g. cfg["peaks"]["threshold"]."""
    path = Path(path) if path else PROJECT_ROOT / "config.toml"
    if not path.exists():
        return copy.deepcopy(DEFAULTS)
    with path.open("rb") as f:  # tomllib wants binary mode
        user_cfg = tomllib.load(f)
    return _merge(DEFAULTS, user_cfg)


def load_campaigns(path: Path | None = None) -> dict[str, dict]:
    """Campaign profiles from campaigns.toml: {name: settings}. Empty if there's no file.

    A profile holds what a clipping campaign requires (disclosure text, credit handle, tags,
    platforms, length limit, banned things). The editor and the review queue apply it.
    """
    path = Path(path) if path else PROJECT_ROOT / "campaigns.toml"
    if not path.exists():
        return {}
    with path.open("rb") as f:
        data = tomllib.load(f)
    return {name: (info or {}) for name, info in data.get("campaigns", {}).items()}


def load_streamers(path: Path | None = None) -> dict[str, dict]:
    """Return the allow-list as {login_name_lowercase: settings}."""
    path = Path(path) if path else PROJECT_ROOT / "streamers.toml"
    if not path.exists():
        return {}
    with path.open("rb") as f:
        data = tomllib.load(f)
    return {name.lower(): (info or {}) for name, info in data.get("streamers", {}).items()}
