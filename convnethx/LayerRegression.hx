package convnethx;

class LayerRegression extends Layer {

    public function new(opt:Opt) {
        super(opt);

        // computed
        this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
        this.out_depth = this.num_inputs;
        this.out_sx = 1;
        this.out_sy = 1;
        this.layer_type = LayerType.REGRESSION;
    }

    override public function forward(V:Vol, is_training:Bool = false):Vol {
        this.in_act = V;
        this.out_act = V;

        return V; // identity function
    }

    /**
    * y is a list here of size num_inputs (array length > 2)
    * or it can be a number if only one value is regressed (array length == 1)
    * or it can be a struct [0] = (i)dim [1] = (x)val where we only want to
    * regress on dimension i and asking it to have value x (array length == 2)
    **/
    override public function backward(y:Array<Float> = null):Null<Float> {

        // compute and accumulate gradient wrt weights and bias of this layer
        var x:Vol = this.in_act;
        x.dw = Utils.zeros(x.w.length); // zero out the gradient of input Vol

        var loss:Float = 0.0;

        if(y.length > 2) {
            for(i in 0 ... this.out_depth) {
                var dy:Float = x.w[i] - y[i];
                x.dw[i] = dy;

                loss += 0.5 * dy * dy;
            }
        } else if(y.length == 1) {
            // lets hope that only one number is being regressed
            var dy = x.w[0] - y[0];
            x.dw[0] = dy;
            loss += 0.5 * dy * dy;
        } else if (y.length == 2) {

            // assume it is a struct with entries .dim and .val
            // and we pass gradient only along dimension dim to be equal to val
            // TODO validate this haxe adaptation

            var i:Int = Std.int(y[0]); // DIM
            var yi:Float = y[1]; // VAL;
            var dy:Float = x.w[i] - yi;

            x.dw[i] = dy;
            loss += 0.5 * dy * dy;
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
