from .coupling import AffineCouplingLayer
from .flow_model import FlowModel
from .flow_model import FlowTransformLayer
from .flow_utils import FlowBatchNorm
from .flow_utils import FlowLayerStacker
from torch.nn import Module

class RealNVP(FlowModel):
    def _create_network(self, config_dic: dict[str, int]) -> Module:
        if "num_layers" not in config_dic.keys():
            raise ValueError(
                "RealNVP requires 'num_layers' in config_dic."
            )
        num_layers: int = config_dic["num_layers"]
        layers: list[FlowTransformLayer] = []
        for i in range(num_layers):
            layers.append(
                AffineCouplingLayer(
                    input_shape=self.input_shape,
                    split_pattern="checkerboard",
                    is_odd=i%2!=0
                )
            )
        net: Module = FlowLayerStacker(layers)
        return net