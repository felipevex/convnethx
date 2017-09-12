package convnethx.model.json;

import convnethx.type.LayerType;

typedef JsonLayerFC = {
    var out_depth:Int;
    var out_sx:Int;
    var out_sy:Int;

    var layer_type:LayerType;
    var num_inputs:Int;
    var l1_decay_mul:Float;
    var l2_decay_mul:Float;

    var filters:Array<JsonVol>;

    var biases:JsonVol;
}
