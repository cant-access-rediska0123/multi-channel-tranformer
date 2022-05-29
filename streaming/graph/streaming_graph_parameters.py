from dataclasses import dataclass


@dataclass
class StreamingGraphParameters:
    window_size_ms: int
    window_shift_ms: int
    min_word_score: int
    noise_duration_ms: float
    timings_unstability_coefficient: float
    combine_words_pause_duration_threshold_ms: float
    long_pause_fix_duration_ms: float
    optimize_graph_min_sz: int
    optimize_graph_min_window_ms: int
    speakers_num: int
    use_word_duration: bool = False
