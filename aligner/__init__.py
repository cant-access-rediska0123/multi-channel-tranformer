from aligner.time_aligner import TimeAligner
from aligner.time_ctc_aligner import TimeCtcAligner

from factory.factory import Factory

Factory.register(TimeAligner, {
    'time_ctc_aligner': TimeCtcAligner
})
