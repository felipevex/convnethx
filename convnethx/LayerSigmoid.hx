package convnethx;


/**
* Implements Sigmoid nnonlinearity elementwise
* x -> 1/(1+e^(-x))
* so the output is between 0 and 1.
**/
import haxe.io.Float64Array;
class LayerSigmoid extends Layer {

    public function new(opt:Opt) {
        super(opt);

        // computed
        this.out_sx = opt.in_sx;
        this.out_sy = opt.in_sy;
        this.out_depth = opt.in_depth;
        this.layer_type = LayerType.SIGMOID;
    }

    override public function forward(V:Vol, is_training:Bool = false):Vol {
        this.in_act = V;

        var V2:Vol = V.cloneAndZero();
        var N:Int = V.w.length;

        var V2w:Float64Array = V2.w;
        var Vw:Float64Array = V.w;

        for(i in 0 ... N) {
            V2w[i] = 1.0 / (1.0 + Math.exp(-Vw[i]));
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
            var v2wi:Float = V2.w[i];
            V.dw[i] =  v2wi * (1.0 - v2wi) * V2.dw[i];
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
