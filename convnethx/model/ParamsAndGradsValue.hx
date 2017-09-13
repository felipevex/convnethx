package convnethx.model;

import haxe.io.Float64Array;

typedef ParamsAndGradsValue = {
    var params:Float64Array;
    var grads:Float64Array;
    var l1_decay_mul:Null<Float>;
    var l2_decay_mul:Null<Float>;
}
