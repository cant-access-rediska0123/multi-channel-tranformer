import re
from typing import Dict, List, Optional

from num2words import num2words

from data.text.text import Text
from data.trans import Trans
from data.utterance.utterance import RawUtterance


class TextProcessor(Trans[RawUtterance, RawUtterance]):
    def __init__(self,
                 letters: List[str],
                 replace_numbers_lang: Optional[str],
                 replacement_rules: Dict[str, str],
                 lower: bool,
                 add_begin_end_spaces: bool = False,
                 **kwargs):
        self._lower = lower
        self.letters = ''.join(letters)
        self.token_regex = re.compile(f'([{self.letters}]+|[0-9-]+|.?)')
        self.non_letters_regex = re.compile(f'[^{self.letters}]+')
        self.replace_numbers_lang = replace_numbers_lang

        self.add_begin_end_spaces = add_begin_end_spaces

        self.replacement_rules = {}
        for src, dst in replacement_rules.items():
            self.replacement_rules[self._normalize_text(Text(src))] = self._normalize_text(Text(dst))

    def _normalize_text(self, text: Text) -> Text:
        if self._lower:
            text = text.lower()
        else:
            text = text.upper()
        text = Text(' '.join(filter(len, text.split())))
        if self.add_begin_end_spaces:
            text = Text(' ' + text + ' ')
        return text

    def _num_to_word(self, token: Text):
        if token.isdigit():
            try:
                return num2words(token, lang=self.replace_numbers_lang)
            except:
                pass
        return token

    def process_text(self, text: Text) -> Text:
        text = self._normalize_text(text)

        words = []
        for word in self.token_regex.findall(text):
            words.append(word)
        text = ' '.join(words)

        for src, dst in self.replacement_rules.items():
            text = text.replace(src, dst)

        if self.replace_numbers_lang:
            text = ' '.join(map(lambda token: self._num_to_word(token), text.split()))

        text = Text(self.non_letters_regex.sub(' ', text))
        text = self._normalize_text(text)
        return text

    def apply(self, utt: RawUtterance, **kwargs) -> RawUtterance:
        return RawUtterance(self.process_text(utt.text), utt.audio, utt.uuid, utt.speaker_id)
