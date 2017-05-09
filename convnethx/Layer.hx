package convnethx;

class Layer {

    public function new(opt:Opt) {

    }

    public function forward(V:Vol, is_training:Bool):Vol {
        return V; // simply identity function for now
    }

    public function backward() {

    }

    public function getParamsAndGrads():Array<Dynamic> {
        return [];
    }

    public function toJSON() {
        var json = {};
        return json;
    }


    public function fromJSON(json:Dynamic) {

    }
}
