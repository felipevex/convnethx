package convnethx.layer.input;

import convnethx.model.json.JsonLayerInput;
import convnethx.layer.model.LayerOption;
import convnethx.type.LayerType;

class LayerInput extends Layer {

    public function new(option:LayerOption) {
        super();

        this.out_depth = option.out_depth == null ? 0 : option.out_depth;
        this.out_sx = option.out_sx == null ? 1 : option.out_sx;
        this.out_sy = option.out_sy == null ? 1 : option.out_sy;

        this.layer_type = LayerType.INPUT;
    }

    override public function forward(V:Vol, is_training:Bool = false):Vol {
        this.in_act = V;
        this.out_act = V;

        return this.out_act; // simply identity function for now
    }

    public function toJSON():JsonLayerInput {
        var json:JsonLayerInput = {
            layer_type : this.layer_type,
            out_depth : this.out_depth,
            out_sx : this.out_sx,
            out_sy : this.out_sy
        };

        return json;
    }

    public function fromJSON(json:JsonLayerInput):Void {
        this.layer_type = json.layer_type;
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
    }
}
