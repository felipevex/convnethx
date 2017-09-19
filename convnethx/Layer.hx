package convnethx;

import convnethx.model.ParamsAndGradsValue;
import convnethx.type.LayerType;

class Layer {

    public var num_inputs:Int;

    public var in_act:Vol;
    public var out_act:Vol;

    public var in_depth:Int;
    public var in_sx:Int;
    public var in_sy:Int;

    public var out_depth:Int;
    public var out_sx:Int;
    public var out_sy:Int;

    public var sx:Int;
    public var sy:Int;

    public var l1_decay_mul:Null<Float>;
    public var l2_decay_mul:Null<Float>;

    public var layer_type:LayerType;

    public var filters:Array<Vol>;
    public var biases:Vol;

    public var stride:Int;
    public var pad:Int;

    public function new() {

    }

    public function forward(V:Vol, is_training:Bool = false):Vol {
        return V; // simply identity function for now
    }

    public function backward(y:Null<Int> = null):Null<Float> {
        return null;
    }

    public function getParamsAndGrads():Array<ParamsAndGradsValue> {
        return [];
    }

}
