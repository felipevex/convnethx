package convnethx;

class LayerInput extends Layer {

    public function new(opt:Opt) {
        super(opt);

        this.out_depth = Utils.getopt(opt, ['out_depth', 'depth'], 0);
        this.out_sx = Utils.getopt(opt, ['out_sx', 'sx', 'width'], 1);
        this.out_sy = Utils.getopt(opt, ['out_sy', 'sy', 'height'], 1);

        this.layer_type = LayerType.INPUT;
    }

    override public function forward(V:Vol, is_training:Bool = false):Vol {
        this.in_act = V;
        this.out_act = V;

        return this.out_act; // simply identity function for now
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
