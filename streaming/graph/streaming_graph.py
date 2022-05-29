import abc
import copy
import logging
from collections import defaultdict
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import numpy as np

from aligner.aligned_word import FrameAlignedWords, TimeAlignedWord
from data.text.text import Text
from streaming.graph.graph import Graph
from streaming.graph.streaming_graph_parameters import StreamingGraphParameters
from streaming.graph.streaming_graph_word import StreamingGraphWord
from streaming.recognition_model.recognition_model import RecognitionModelOutput, TimeAlignedWords

LOG = logging.getLogger()


def _weighted_average(a, b, wa, wb):
    return (wa * a + wb * b) / (wa + wb)


def _merge_words(w1: StreamingGraphWord, w2: StreamingGraphWord) -> StreamingGraphWord:
    assert w1.text == w2.text
    return StreamingGraphWord(
        w1.text,
        _weighted_average(w1.start_ms, w2.start_ms, w1.score, w2.score),
        _weighted_average(w1.end_ms, w2.end_ms, w1.score, w2.score),
        w1.score + w2.score)


def _intersection_len(word1: TimeAlignedWord, word2: TimeAlignedWord) -> float:
    return max(0.0, min(word1.end_ms, word2.end_ms) - max(word1.start_ms, word2.start_ms))


def _preprocess_word(word: TimeAlignedWord) -> TimeAlignedWord:
    word.text = word.text.lower()
    return word


def _is_probable_same(w1: TimeAlignedWord, w2: TimeAlignedWord, params: StreamingGraphParameters) -> bool:
    if w1.text != w2.text:
        return False
    intersection = params.noise_duration_ms + params.timings_unstability_coefficient * _intersection_len(w1, w2)
    return intersection > min(w1.duration_ms, w2.duration_ms)


class StreamingGraph:
    def __init__(self, params: StreamingGraphParameters):
        self._params = params

        self._graph: Graph[StreamingGraphWord] = Graph[StreamingGraphWord]()
        self._fake_start_word_id = self._graph.add_node()
        self._fake_start_word = StreamingGraphWord(Text('START'), -10, -1, 1000000)
        self._graph.add_node_info(self._fake_start_word_id, self._fake_start_word)

        self._word_to_id: Dict[StreamingGraphWord, int] = {self._fake_start_word: self._fake_start_word_id}

        self._best_path_dynamic: Dict[int, Optional[Tuple[float, Optional[int]]]] = {}

        self._last_ms = 0.0
        self._last_accumulated_ms = 0.0
        self._accumulated_best_path: List[StreamingGraphWord] = [self._fake_start_word]
        self._last_accumulated_word_id: int = self._fake_start_word_id

    @property
    def word_to_id(self) -> Dict[StreamingGraphWord, int]:
        return self._word_to_id

    @property
    def graph(self) -> Graph[StreamingGraphWord]:
        return self._graph

    def add_hypothesis_window(self, words: List[TimeAlignedWord], window_start_ms: float, window_end_ms: float,
                              is_first_hypothesis: bool, is_last_hypothesis: bool):
        words = [_preprocess_word(word) for word in words]
        words = self._filter_irrelevant_words(words)
        words = self._combine_repeated_words(words)

        prev_word_id: int = self._fake_start_word_id
        for aligned_word in words:
            word_start_ms, word_end_ms = \
                aligned_word.start_ms + window_start_ms, aligned_word.end_ms + window_start_ms
            score_to_add: int = 1
            if is_last_hypothesis:
                score_to_add += int(aligned_word.start_ms / self._params.window_shift_ms)
            if is_first_hypothesis:
                score_to_add += int((window_end_ms - word_start_ms) / self._params.window_shift_ms)

            word = StreamingGraphWord(
                text=aligned_word.text,
                start_ms=word_start_ms,
                end_ms=word_end_ms,
                score=score_to_add,
            )

            added_word_id = self._add_word_to_graph(word, prev_word_id)
            self._graph.add_edge(self._fake_start_word_id, added_word_id)
            self._graph.add_edge(prev_word_id, added_word_id)
            prev_word_id = added_word_id

        self._last_ms = max(self._last_ms, window_end_ms)

    def calculate_hypothesis(self) -> List[TimeAlignedWord]:
        self._remove_cycles()

        path: List[StreamingGraphWord] = self._calculate_best_path()

        res: List[TimeAlignedWord] = []
        for i in path[1:]:
            res.append(copy.deepcopy(i))

        if self._should_cache(path):
            self._cache_path(path)

        return res

    def lock_word(self, word_id: int):
        if word_id in [self._fake_start_word_id, self._last_accumulated_word_id]:
            return
        word = self._graph.get_node_info(word_id)
        assert word.score > 0
        word.score *= -1
        self._graph.add_node_info(word_id, word)

    def unlock_word(self, word_id: int):
        if word_id in [self._fake_start_word_id, self._last_accumulated_word_id]:
            return
        word = self._graph.get_node_info(word_id)
        if word is None:  # word was cached, whatever
            return
        assert word.score < 0
        word.score *= -1
        self._graph.add_node_info(word_id, word)

    def _cache_path(self, best_path: List[StreamingGraphWord]):
        final_ms = self._last_ms - self._params.optimize_graph_min_window_ms
        self._accumulated_best_path = [
            copy.deepcopy(w) for w in best_path
            if w.end_ms + self._params.noise_duration_ms / 2 < final_ms]
        self._last_accumulated_word_id = self._word_to_id[self._accumulated_best_path[-1]]
        self._last_accumulated_ms = self._last_ms

        ids_to_delete = []
        for word, i in self._word_to_id.items():
            if i in [self._fake_start_word_id, self._last_accumulated_word_id]:
                continue
            if word.end_ms + self._params.noise_duration_ms / 2 > final_ms:
                continue
            ids_to_delete.append(i)
        for i in ids_to_delete:
            del self._word_to_id[self._graph.get_node_info(i)]
            self._graph.remove_node(i)

    def _should_cache(self, path: List[StreamingGraphWord]) -> bool:
        return len(path) > len(self._accumulated_best_path) and \
               self._last_ms - self._last_accumulated_ms > self._params.optimize_graph_min_window_ms and \
               len(self._word_to_id) >= self._params.optimize_graph_min_sz

    def _combine_repeated_words(self, words: List[TimeAlignedWord]) -> List[TimeAlignedWord]:
        res: List[TimeAlignedWord] = []
        i = 0
        while i < len(words):
            word: TimeAlignedWord = words[i]
            text = word.text
            i += 1
            while i < len(words):
                nxt_word = words[i]
                if nxt_word.text != text:
                    break
                if nxt_word.start_ms - word.end_ms > self._params.combine_words_pause_duration_threshold_ms:
                    break
                word = TimeAlignedWord(
                    text=Text(word.text + ' ' + nxt_word.text),
                    start_ms=min(word.start_ms, nxt_word.start_ms),
                    end_ms=max(word.end_ms, nxt_word.end_ms),
                )
                i += 1
            res.append(word)
        return res

    def _add_edges_for_long_pauses(self, id1):
        word1 = self._graph.get_node_info(id1)
        for id2, word2 in self._graph:
            if word1.end_ms + self._params.long_pause_fix_duration_ms < word2.start_ms:
                self._graph.add_edge(id1, id2)
            if word2.end_ms + self._params.long_pause_fix_duration_ms < word1.start_ms:
                self._graph.add_edge(id2, id1)

    def _remove_cycles(self):
        for id1, word1 in self._graph:
            for id2 in self._graph.get_forward_edges(id1):
                word2 = self._graph.get_node_info(id2)
                if word1.start_ms >= word2.start_ms:
                    self._graph.remove_edge(id1, id2)

    def _filter_irrelevant_words(self, words: List[TimeAlignedWord]) -> List[TimeAlignedWord]:
        res: List[TimeAlignedWord] = []
        for word in words:
            if word.duration_ms < self._params.noise_duration_ms:
                continue
            res.append(word)
        return res

    def _find_best_word_match(self, word: StreamingGraphWord, prev_word_id: int) -> Optional[int]:
        prev_word = self._graph.get_node_info(prev_word_id)
        best_match_id, best_start_frame = None, None
        for id_to_match, word_to_match in self._graph:
            if id_to_match == prev_word_id:
                continue
            if not _is_probable_same(word_to_match, word, self._params):
                continue
            if word_to_match.start_ms > prev_word.start_ms and (
                    best_match_id is None or word_to_match.start_ms < best_start_frame):
                best_match_id = id_to_match
                best_start_frame = word_to_match.start_ms
        if best_match_id is None and word in self._word_to_id:
            best_match_id = self._word_to_id[word]
        return best_match_id

    def _add_word_to_graph(self, word: StreamingGraphWord, prev_word_id: int) -> int:
        best_match_id = self._find_best_word_match(word, prev_word_id)

        if best_match_id is None:
            new_word_id = self._graph.add_node()
        else:
            word = _merge_words(word, self._graph.get_node_info(best_match_id))
            ids_to_merge = {best_match_id}
            del self._word_to_id[self._graph.get_node_info(best_match_id)]
            while word in self._word_to_id:
                old_id = self._word_to_id[word]
                del self._word_to_id[word]
                ids_to_merge.add(old_id)
                word = _merge_words(word, self._graph.get_node_info(old_id))
            new_word_id = self._graph.merge_nodes(ids_to_merge)
            if self._last_accumulated_word_id in ids_to_merge:
                self._last_accumulated_word_id = new_word_id

        self._graph.add_node_info(new_word_id, word)
        self._word_to_id[word] = new_word_id

        self._add_edges_for_long_pauses(new_word_id)

        return new_word_id

    def _calculate_best_path(self) -> List[StreamingGraphWord]:
        self._best_path_dynamic: Dict[int, Optional[Tuple[float, Optional[int]]]] = defaultdict(lambda: None)
        last_word_id = self._last_accumulated_word_id
        self._calculate_dynamic(last_word_id)

        res: List[StreamingGraphWord] = self._accumulated_best_path.copy()

        while self._best_path_dynamic[last_word_id][1] is not None:
            last_word_id = self._best_path_dynamic[last_word_id][1]
            res.append(self._graph.get_node_info(last_word_id))

        return res.copy()

    def _word_score(self, word: StreamingGraphWord):
        if self._params.use_word_duration:
            return word.score * (word.duration_ms + self._params.noise_duration_ms)
        return word.score

    def _calculate_dynamic(self, word_id: int) -> Tuple[float, Optional[int]]:
        assert word_id in [a for a, _ in self._graph]
        if self._best_path_dynamic[word_id] is not None:
            return self._best_path_dynamic[word_id]
        word = self._graph.get_node_info(word_id)
        if word.score < self._params.min_word_score:
            self._best_path_dynamic[word_id] = (0, None)
            return 0, None
        max_nxt_score, nxt_word_id = 0, None
        for nxt_id in self._graph.get_forward_edges(word_id):
            nxt_score, _ = self._calculate_dynamic(nxt_id)
            if max_nxt_score is None or nxt_score > max_nxt_score:
                max_nxt_score = nxt_score
                nxt_word_id = nxt_id
        self._best_path_dynamic[word_id] = (max_nxt_score + self._word_score(word), nxt_word_id)
        return self._best_path_dynamic[word_id]


def _graphs_similarity_score(texts: TimeAlignedWords, graphs: List[Graph[StreamingGraphWord]]):
    similarity = 0.0
    for text, graph in zip(texts, graphs):
        for new_word in text:
            for _, graph_word in graph:
                if graph_word.text == new_word.text:
                    start_diff = (graph_word.start_ms - new_word.start_ms) / 3000
                    end_diff = (graph_word.end_ms - new_word.end_ms) / 3000
                    score = graph_word.score * len(graph_word.text)
                    similarity += score * np.exp(-start_diff ** 2)
                    similarity += score * np.exp(-end_diff ** 2)
    return similarity


def _texts_similarity_score(cur_texts: TimeAlignedWords,
                            prev_texts: FrameAlignedWords,
                            window_start_ms: float,
                            match_score=3,
                            mismatch_score=-0.1):
    prev_texts = TimeAlignedWords(
        [[w for w in sp_hyp if w.end_ms > window_start_ms - 1000]
         for sp_hyp in prev_texts])
    score = 0
    for text1, text2 in zip(prev_texts, cur_texts):
        n, m = len(text1), len(text2)
        dp = np.full((n + 1, m + 1), fill_value=-np.inf)
        dp[:, 0] = 0
        for i in range(n + 1):
            for j in range(m + 1):
                if i >= 1 and j >= 1 and text1[i - 1].text == text2[j - 1].text:
                    dp[i, j] = max(dp[i, j], dp[i - 1, j - 1] + match_score)
                if i >= 1:
                    dp[i, j] = max(dp[i, j], dp[i - 1, j] + mismatch_score)
                if j >= 1:
                    dp[i, j] = max(dp[i, j], dp[i, j - 1] + mismatch_score)
        score += dp[n, :].max()
    return score


class StreamingGraphInterface:
    @abc.abstractmethod
    def __init__(self, _: StreamingGraphParameters):
        pass

    @abc.abstractmethod
    def add_hypothesis_window(self, outp: RecognitionModelOutput, window_start_ms: float, window_end_ms: float,
                              is_first_hypothesis: bool, is_last_hypothesis: bool):
        pass

    @abc.abstractmethod
    def calculate_hypothesis(self) -> FrameAlignedWords:
        pass

    def calculate_text_hypothesis(self) -> List[Text]:
        al: FrameAlignedWords = self.calculate_hypothesis()
        texts = [Text(' '.join([word.text for word in sp_hyp])) for sp_hyp in al]
        return texts


class MultiSpeakerStreamingGraph(StreamingGraphInterface):
    def __init__(self, params: StreamingGraphParameters):
        self._speakers_num: int = params.speakers_num
        self._streaming_graphs: List[StreamingGraph] = [StreamingGraph(params) for _ in range(params.speakers_num)]
        self._prev_texts: FrameAlignedWords = FrameAlignedWords([])
        self._params = params

    def _find_best_speakers_permutation(self, words_in_window: TimeAlignedWords,
                                        window_start_ms: float) -> TimeAlignedWords:
        best_score, best_permutation = -np.inf, None
        for pi, words_permutation in zip(permutations(range(len(words_in_window))), permutations(words_in_window)):
            words_in_audio: TimeAlignedWords = TimeAlignedWords([
                [TimeAlignedWord(word.text,
                                 word.start_ms + window_start_ms,
                                 word.end_ms + window_start_ms)
                 for word in speaker_words]
                for speaker_words in words_permutation])

            # print(list(pi), ':')
            # score = _texts_similarity_score(words, self._prev_texts, window_start_ms)
            score = _graphs_similarity_score(words_in_audio, [s.graph for s in self._streaming_graphs])
            # print('\t', score)

            if best_permutation is None or (score, len(words_permutation[0])) > (best_score, len(best_permutation[0])):
                best_score = score
                best_permutation = copy.deepcopy(words_permutation)
        return best_permutation

    def _lock_repeated_words(self):
        words_to_unlock: List[Tuple[int, int]] = []
        for sp1 in range(len(self._streaming_graphs)):
            for sp2 in range(len(self._streaming_graphs)):
                if sp1 == sp2:
                    continue
                for word1, id1 in self._streaming_graphs[sp1].word_to_id.items():
                    if word1.score <= 0:
                        continue
                    for word2, id2 in self._streaming_graphs[sp2].word_to_id.items():
                        if word2.score <= 0 or word1.score <= 0 or word1.score > word2.score:
                            continue
                        if not _is_probable_same(word1, word2, self._params):
                            continue
                        self._streaming_graphs[sp1].lock_word(id1)
                        words_to_unlock.append((sp1, id1))
        return words_to_unlock

    def _unlock_repeated_words(self, words_to_unlock: List[Tuple[int, int]]):
        for sp, word_id in words_to_unlock:
            self._streaming_graphs[sp].unlock_word(word_id)

    def add_hypothesis_window(self, output: RecognitionModelOutput, window_start_ms: float, window_end_ms: float,
                              is_first_hypothesis: bool, is_last_hypothesis: bool):
        words = self._find_best_speakers_permutation(output.alignment, window_start_ms)
        # print('NEW WINDOW')
        # for w in words:
        #     print('\t', w)
        # print()

        for i, speaker_words in enumerate(words):
            self._streaming_graphs[i].add_hypothesis_window(
                speaker_words, window_start_ms, window_end_ms, is_first_hypothesis, is_last_hypothesis)

        self._prev_texts = self.calculate_hypothesis()
        # print('MAX PATH')
        # for w in self._prev_texts:
        #     print('\t', w)
        # print()
        # print('=' * 100)

    def calculate_hypothesis(self) -> FrameAlignedWords:
        locked_words = self._lock_repeated_words()

        words = FrameAlignedWords([])
        for i in range(self._speakers_num):
            words.append(self._streaming_graphs[i].calculate_hypothesis().copy())

        self._unlock_repeated_words(locked_words)

        return words
