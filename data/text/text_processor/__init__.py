from data.text.text_processor.multispeaker_text_processor import MultiSpeakerTextProcessor
from data.text.text_processor.text_processor import TextProcessor
from data.trans import Trans
from factory.factory import Factory

Factory.register(Trans, {
    "text_processor": TextProcessor,
    "multispeaker_text_processor": MultiSpeakerTextProcessor,
})
