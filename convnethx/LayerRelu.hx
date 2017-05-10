package convnethx;

import haxe.io.Float64Array;

/**
* Implements ReLU nonlinearity elementwise
* x -> max(0, x)
* the output is in [0, inf)
**/
class LayerRelu extends Layer {

    public function new(opt:Opt) {
        super(opt);

        // computed
        this.out_sx = opt.in_sx;
        this.out_sy = opt.in_sy;
        this.out_depth = opt.in_depth;
        this.layer_type = LayerType.RELU;
    }

    override public function forward(V:Vol, is_training:Bool):Vol {
        this.in_act = V;

        var V2:Vol = V.clone();
        var N:Int = V.w.length;
        var V2w:Float64Array = V2.w;

        for (i in 0 ... N) {
            if (V2w[i] < 0) V2w[i] = 0; // threshold at 0
        }

        this.out_act = V2;

        return this.out_act;
    }

    override public function backward(y:Array<Float> = null):Null<Float> {
        var V:Vol = this.in_act; // we need to set dw of this
        var V2:Vol = this.out_act;
        var N:Int = V.w.length;

        V.dw = Utils.zeros(N); // zero out gradient wrt data

        for(i in 0 ... N) {
            if (V2.w[i] <= 0) V.dw[i] = 0; // threshold
            else V.dw[i] = V2.dw[i];
        }

        return null;
    }

    override public function toJSON():Dynamic {
        var json:Dynamic = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        return json;
    }

    override public function fromJSON(json:Dynamic):Void {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type;
    }
}
