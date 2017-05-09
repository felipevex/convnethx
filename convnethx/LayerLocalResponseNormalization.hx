package convnethx;

class LayerLocalResponseNormalization extends Layer {

    public var k:Int;
    public var n:Int;
    public var alpha:Int;
    public var beta:Int;

    public var S_cache_:Vol;

    public function new(opt:Opt) {
        super(opt);

        // required
        this.k = opt.k;
        this.n = opt.n;
        this.alpha = opt.alpha;
        this.beta = opt.beta;

        // computed
        this.out_sx = opt.in_sx;
        this.out_sy = opt.in_sy;
        this.out_depth = opt.in_depth;
        this.layer_type = LayerType.LNR;

        // checks
        if (this.n % 2 == 0) {
            trace('WARNING n should be odd for LRN layer');
        }
    }

    override public function forward(V:Vol, is_training:Bool):Vol {
        this.in_act = V;

        var A:Vol = V.cloneAndZero();
        this.S_cache_ = V.cloneAndZero();

        var n2:Int = Math.floor(this.n/2);

        for (x in 0 ... V.sx) {
            for(y in 0 ... V.sy) {
                for(i in 0 ... V.depth) {

                    var ai:Float = V.get(x,y,i);

                    // normalize in a window of size n

                    var den:Float = 0.0;

                    for (j in Math.max(0, i - n2) ... Math.min(i + n2 + 1, V.depth)) {
                        var aa:Float = V.get(x, y, j);

                        den += aa * aa;
                    }

                    den *= this.alpha / this.n;
                    den += this.k;

                    this.S_cache_.set(x, y, i, den); // will be useful for backprop

                    den = Math.pow(den, this.beta);
                    A.set(x, y, i, ai / den);
                }
            }
        }

        this.out_act = A;
        return this.out_act; // dummy identity function for now
    }

    override public function backward(y:Array<Float> = null):Float {
        // evaluate gradient wrt data
        var V:Vol = this.in_act; // we need to set dw of this
        V.dw = Utils.zeros(V.w.length); // zero out gradient wrt data

        var A:Vol = this.out_act; // computed in forward pass
        var n2:Int = Math.floor(this.n/2);

        for(x in 0 ... V.sx) {
            for(y in 0 ... V.sy) {
                for(i in 0 ... V.depth) {

                    var chain_grad:Float = this.out_act.get_grad(x, y, i);

                    var S:Float = this.S_cache_.get(x, y, i);
                    var SB:Float = Math.pow(S, this.beta);
                    var SB2:Float = SB*SB;

                    // normalize in a window of size n
                    for(j in Math.max(0, i-n2) ... Math.min(i + n2 + 1, V.depth)) {
                        var aj:Float = V.get(x, y, j);
                        var g:Float = -aj * this.beta * Math.pow(S, this.beta - 1) * this.alpha / this.n * 2 * aj;

                        if(j == i) g += SB;

                        g /= SB2;
                        g *= chain_grad;

                        V.add_grad(x, y, j, g);
                    }
                }
            }
        }

        return 0;
    }

    override public function toJSON():Dynamic {
        var json:Dynamic = {};
        json.k = this.k;
        json.n = this.n;
        json.alpha = this.alpha; // normalize by size
        json.beta = this.beta;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.out_depth = this.out_depth;
        json.layer_type = this.layer_type;
        return json;
    }

    override public function fromJSON(json:Dynamic):Void {
        this.k = json.k;
        this.n = json.n;
        this.alpha = json.alpha; // normalize by size
        this.beta = json.beta;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.out_depth = json.out_depth;
        this.layer_type = json.layer_type;
    }
}
