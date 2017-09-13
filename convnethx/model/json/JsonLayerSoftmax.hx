package convnethx.model.json;

import convnethx.type.LayerType;

typedef JsonLayerSoftmax = {
    var out_depth:Int;
    var out_sx:Int;
    var out_sy:Int;
    var num_inputs:Int;
    var layer_type:LayerType;
}
