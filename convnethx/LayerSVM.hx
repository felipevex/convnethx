package convnethx;

class LayerSVM extends Layer {

    public function new(opt:Opt) {
        super(opt);

        // computed
        this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
        this.out_depth = this.num_inputs;
        this.out_sx = 1;
        this.out_sy = 1;
        this.layer_type = LayerType.SVM;
    }

    override public function forward(V:Vol, is_training:Bool = false):Vol {
        this.in_act = V;
        this.out_act = V; // nothing to do, output raw scores
        return V;
    }

    override public function backward(y:Array<Float>):Float {
        // compute and accumulate gradient wrt weights and bias of this layer
        var x:Vol = this.in_act;
        x.dw = Utils.zeros(x.w.length); // zero out the gradient of input Vol

        // we're using structured loss here, which means that the score
        // of the ground truth should be higher than the score of any other
        // class, by a margin
        var yscore:Float = x.w[Std.int(y[0])]; // score of ground truth
        var margin:Float = 1.0;
        var loss:Float = 0.0;

        for(i in 0 ... this.out_depth) {
            if(Std.int(y[0] == i)) { continue; }

            var ydiff:Float = -yscore + x.w[i] + margin;

            if(ydiff > 0) {
                // violating dimension, apply loss
                x.dw[i] += 1;
                x.dw[y] -= 1;

                loss += ydiff;
            }
        }

        return loss;
    }

    override public function toJSON():Dynamic {
        var json:Dynamic = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        json.num_inputs = this.num_inputs;
        return json;
    }
    
    override public function fromJSON(json:Dynamic):Void {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type;
        this.num_inputs = json.num_inputs;
    }
}
