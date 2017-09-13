package convnethx.model.json;

import convnethx.type.LayerType;

typedef JsonLayerTanh = {
    var out_depth:Int;
    var out_sx:Int;
    var out_sy:Int;

    var layer_type:LayerType;
}
