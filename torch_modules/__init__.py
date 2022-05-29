import torch

from factory.factory import Factory
from torch_modules.models.jasper.jasper import JasperEncoder
from torch_modules.optimizers.adam import AdamW
from torch_modules.optimizers.novograd import NovoGrad

Factory.register(torch.optim.Optimizer, {
    "adamw": AdamW,
    "novograd": NovoGrad
})

Factory.register(
    torch.nn.Module, {
        "jasper_encoder": JasperEncoder,
    })
