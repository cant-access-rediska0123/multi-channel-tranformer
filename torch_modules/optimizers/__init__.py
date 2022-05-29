from factory.factory import Factory
from torch_modules.optimizers.adam import AdamW
from torch_modules.optimizers.novograd import NovoGrad
from torch_modules.optimizers.optimizer import Optimizer

Factory.register(Optimizer, {
    "adamw": AdamW,
    "novograd": NovoGrad
})
