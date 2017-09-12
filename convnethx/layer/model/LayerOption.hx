package convnethx.layer.model;

import convnethx.type.LayerType;

typedef LayerOption = {
    @:optional var layer_type:LayerType;
    @:optional var activation:LayerType;

    @:optional var in_depth:Int;
    @:optional var in_sx:Int;
    @:optional var in_sy:Int;

    @:optional var out_sx:Int;
    @:optional var out_sy:Int;
    @:optional var out_depth:Int;

    @:optional var num_neurons:Int;

    @:optional var num_classes:Int;

    @:optional var bias_pref:Float;

    @:optional var group_size:Int;

    @:optional var drop_prob:Float;

    @:optional var l1_decay_mul:Float;
    @:optional var l2_decay_mul:Float;
}
