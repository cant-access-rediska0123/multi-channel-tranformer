import nemo.collections.asr as nemo_asr
import torch


class NemoJasper:
    def __init__(self, device):
        self._jasper = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="stt_en_jasper10x5dr")
        self._device = device
        self._jasper.to(self._device)
        self._jasper.eval()

    def __call__(self, audios: torch.Tensor, audio_lengths: torch.Tensor) -> torch.Tensor:
        assert audios.shape[1] == 1
        audios = audios[:, 0, :].to(self._device)
        audio_lengths = audio_lengths.to(self._device)
        with torch.no_grad():
            ctc_logprobs, encoded_len, greedy_predictions = self._jasper(
                input_signal=audios,
                input_signal_length=audio_lengths)
        return ctc_logprobs
