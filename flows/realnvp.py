from .coupling import AffineCouplingLayer
from .coupling import ReshapeLayer
from .coupling import Squeeze2dLayer
from .flow_model import FlowModel
from .flow_model import FlowTransformLayer
from .flow_utils import FlowLayerStacker
from numpy import ndarray
from torch.nn import Module

class RealNVP(FlowModel):
    def _create_network(self, config_dic: dict[str, int]) -> Module:
        if "num_layers" not in config_dic.keys():
            raise ValueError(
                "RealNVP requires 'num_layers' in config_dic."
            )
        num_layers: int = config_dic["num_layers"]
        layers: list[FlowTransformLayer] = []
        if len(self.input_shape) == 1:
            for i in range(num_layers):
                layers.append(
                    AffineCouplingLayer(
                        input_shape=self.input_shape,
                        split_pattern="checkerboard",
                        is_odd=i%2!=0
                    )
                )
        elif len(self.input_shape) == 3:
            if "num_squeeze" not in config_dic.keys():
                raise ValueError(
                    "RealNVP for image requires 'num_squeeze' in config_dic."
                )
            num_squeeze: int = config_dic["num_squeeze"]
            c, h, w = self.input_shape
            if (
                h % (2**num_squeeze) != 0 or w % (2**num_squeeze) != 0
            ):
                raise ValueError(
                    f"the size of input image: {(h, w)} is not suitable for squeezing " +
                    f"{num_squeeze} times."
                )
            num_mid_layers_list: list[int] = [
                num_layers // (num_squeeze + 1)
            ] * num_squeeze
            num_mid_layers_list.append(
                num_layers - sum(num_mid_layers_list)
            )
            split_patterns: list[str] = ["checkerboard", "channelwise"]
            c_, h_, w_ = c*(4**num_squeeze), h//(2**num_squeeze), w//(2**num_squeeze)
            layers.append(
                ReshapeLayer(
                    input_shape=self.input_shape,
                    output_shape=[c_,h_,w_]
                )
            )
            for i, num_mid_layers in enumerate(num_mid_layers_list):
                for j in range(num_mid_layers):
                    layers.append(
                        AffineCouplingLayer(
                            input_shape=[c,h,w],
                            split_pattern=split_patterns[i%2],
                            is_odd=j%2!=0
                        )
                    )
                if not i == len(num_mid_layers_list)-1:
                    layers.append(
                        Squeeze2dLayer(
                            input_shape=[c_,h_,w_]
                        )
                    )
                    c_, h_, w_ = c_//4, h_*2, w_*2
                    if c_ <= 2:
                        split_patterns = ["checkerboard", "checkerboard"]

        net: Module = FlowLayerStacker(layers)
        return net