from flows import AffineCouplingLayer
from flows import FlowBatchNorm
from flows import FlowModel
from flows import FlowLayerStacker
from flows import FlowTransformLayer
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
            layers.append(FlowBatchNorm(input_shape=self.input_shape))
            layers.append(
                AffineCouplingLayer(
                    input_shape=self.input_shape,
                    split_pattern="checkerboard",
                    is_odd=i%2!=0
                )
            )
        net: Module = FlowLayerStacker(layers)
        return net