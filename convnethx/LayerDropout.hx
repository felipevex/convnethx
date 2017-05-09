package convnethx;

import haxe.io.Float64Array;

/**
* An inefficient dropout layer
* Note this is not most efficient implementation since the layer before
* computed all these activations and now we're just going to drop them :(
* same goes for backward pass. Also, if we wanted to be efficient at test time
* we could equivalently be clever and upscale during train and copy pointers during test
*
* TODO: make more efficient.
**/
class LayerDropout extends Layer {

    public var drop_prob:Float;
    public var dropped:Float64Array;

    public function new(opt:Opt) {
        super(opt);

        this.out_sx = opt.in_sx;
        this.out_sy = opt.in_sy;
        this.out_depth = opt.in_depth;

        this.drop_prob = opt.drop_prob != null ? opt.drop_prob : 0.5;
        this.dropped = Utils.zeros(this.out_sx * this.out_sy * this.out_depth);

        this.layer_type = LayerType.DROPOUT;
    }

    override public function forward(V:Vol, is_training:Bool = false):Vol {
        this.in_act = V;

        // default is prediction mode

        var V2:Vol = V.clone();
        var N:Int = V.w.length;

        if(is_training) {

            // do dropout

            for (i in 0 ... N) {
                if(Math.random() < this.drop_prob) {
                    // drop!
                    V2.w[i] = 0;
                    this.dropped[i] = true;
                } else {
                    this.dropped[i] = false;
                }
            }

        } else {
            // scale the activations during prediction
            for(i in 0 ... N) {
                V2.w[i] *= this.drop_prob;
            }
        }

        this.out_act = V2;

        return this.out_act; // dummy identity function for now
    }

    override public function backward():Void {
        var V:Vol = this.in_act; // we need to set dw of this
        var chain_grad:Vol = this.out_act;

        var N:Int = V.w.length;
        V.dw = Utils.zeros(N); // zero out gradient wrt data

        for(i in 0 ... N) {
            if (!(this.dropped[i])) {
                V.dw[i] = chain_grad.dw[i]; // copy over the gradient
            }
        }
    }

    override public function toJSON():Dynamic {
        var json:Dynamic = {};

        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        json.drop_prob = this.drop_prob;

        return json;
    }


    override public function fromJSON(json:Dynamic) {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type;
        this.drop_prob = json.drop_prob;
    }
}
