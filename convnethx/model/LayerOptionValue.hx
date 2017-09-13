package convnethx.model;

import convnethx.type.LayerType;

typedef LayerOptionValue = {
    @:optional var layer_type:LayerType;
    @:optional var activation:LayerType;

    @:optional var in_depth:Null<Int>;
    @:optional var in_sx:Null<Int>;
    @:optional var in_sy:Null<Int>;

    @:optional var out_sx:Null<Int>;
    @:optional var out_sy:Null<Int>;
    @:optional var out_depth:Null<Int>;

    @:optional var num_neurons:Null<Int>;

    @:optional var num_classes:Null<Int>;

    @:optional var bias_pref:Null<Float>;

    @:optional var group_size:Null<Int>;

    @:optional var drop_prob:Null<Float>;

    @:optional var l1_decay_mul:Null<Float>;
    @:optional var l2_decay_mul:Null<Float>;
}
