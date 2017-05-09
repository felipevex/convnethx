package convnethx;

typedef Opt = {
    @:optional var width:Int;
    @:optional var height:Int;

    @:optional var depth:Int;
    @:optional var sx:Int;
    @:optional var sy:Int;

    @:optional var out_depth:Int;
    @:optional var out_sx:Int;
    @:optional var out_sy:Int;

    @:optional var in_depth:Int;
    @:optional var in_sx:Int;
    @:optional var in_sy:Int;

    @:optional var filters:Int;
    @:optional var stride:Int;
    @:optional var pad:Int;

    @:optional var l1_decay_mul:Float;
    @:optional var l2_decay_mul:Float;

    @:optional var bias_pref:Float;

    @:optional var num_neurons:Int;
    @:optional var drop_prob:Float;
}
