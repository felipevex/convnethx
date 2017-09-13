package convnethx.layer.nonlinearities;

/**
* Implements Tanh nnonlinearity
* elementwise x -> tanh(x)
* so the output is between -1 and 1.
**/
import convnethx.model.json.JsonLayerTanh;
import convnethx.type.LayerType;
import convnethx.layer.model.LayerOption;

class LayerTanh extends Layer {

    public function new(option:LayerOption) {
        super();

        if (option.in_sx == null || option.in_sy == null || option.in_depth == null) throw "Missing arguments for Tanh Layer";

        // computed
        this.out_sx = option.in_sx;
        this.out_sy = option.in_sy;
        this.out_depth = option.in_depth;
        this.layer_type = LayerType.TANH;
    }

    override public function forward(V:Vol, is_training:Bool = false):Vol {
        this.in_act = V;

        var V2:Vol = V.cloneAndZero();
        var N:Int = V.w.length;

        for(i in 0 ... N) {
            V2.w[i] = Utils.tanh(V.w[i]);
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
            V.dw[i] = (1.0 - v2wi * v2wi) * V2.dw[i];
        }

        return null;
    }

    public function toJSON():JsonLayerTanh {
        var json:JsonLayerTanh = {
            layer_type : this.layer_type,
            out_depth : this.out_depth,
            out_sx : this.out_sx,
            out_sy : this.out_sy
        };

        return json;
    }

    public function fromJSON(json:JsonLayerTanh):Void {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type;
    }
}
