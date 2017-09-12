package convnethx.model.json;

import convnethx.type.LayerType;

typedef JsonLayerInput = {
    var layer_type:LayerType;
    var out_depth:Int;
    var out_sx:Int;
    var out_sy:Int;
}
