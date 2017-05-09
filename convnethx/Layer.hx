package convnethx;

class Layer {

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

    public var l1_decay_mul:Float;
    public var l2_decay_mul:Float;

    public var layer_type:LayerType;

    public var filters:Array<Vol>;
    public var biases:Vol;

    public function new(opt:Opt) {

    }

    public function forward(V:Vol, is_training:Bool):Vol {
        return V; // simply identity function for now
    }

    public function backward():Void {

    }

    public function getParamsAndGrads():Array<Dynamic> {
        return [];
    }

    public function toJSON():Dynamic {
        var json = {};
        return json;
    }


    public function fromJSON(json:Dynamic) {

    }
}
