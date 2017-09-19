package convnethx.maestro;

import haxe.ds.StringMap;
import convnethx.trainer.Trainer;
import convnethx.type.LayerType;
import convnethx.helper.LayerOptionHelper;

class Maestro {

    private var inputLen:Int = 0;

    private var categories:Array<String>;
    private var trainData:StringMap<Array<Array<Float>>>;

    private var net:Net;

    private var neurons:Int;

    public function new(neurons:Int = 3) {
        this.neurons = neurons;
        this.trainData = new StringMap<Array<Array<Float>>>();
    }

    public function addRule(combination:Array<Float>, category:String):Void {

        if (this.inputLen > 0) if (combination.length != this.inputLen) throw "All train data must have same length";

        this.inputLen = combination.length;

        if (!this.trainData.exists(category)) this.trainData.set(category, []);

        var data:Array<Array<Float>> = this.trainData.get(category);
        data.push(combination);
    }

    public function train(trainerIteractions:Int = 2000):Void {
        this.categories = [for (item in this.trainData.keys()) item];
        var categoryLen:Int = this.categories.length;

        this.net = new Net();
        this.net.makeLayers(
            [
                LayerOptionHelper.createInput(1, 1, this.inputLen),
                LayerOptionHelper.createFC(this.neurons, LayerType.TANH),
                LayerOptionHelper.createSoftmax(categoryLen)
            ]
        );

        var trainer:Trainer = new Trainer(this.net);

        for (iter in 0 ... trainerIteractions) {
            for (i in 0 ... this.categories.length) {
                var d:Array<Array<Float>> = this.trainData.get(this.categories[i]);
                for (data in d) trainer.train(new Vol(data), i);

            }
        }

    }

    public function test(data:Array<Float>):String {
        var prediction:Vol = this.net.forward(new Vol(data));

        var values:Array<Float> = Utils.convertToFloatArray(prediction.w);
        var maxIndex:Int = 0;
        var maxValue:Float = values[0];

        for (i in 0 ... values.length) {
            if (values[i] > maxValue) {
                maxValue = values[i];
                maxIndex = i;
            }
        }

        //trace("CONFIDENCE: " + this.categories[maxIndex] + " " + maxValue);

        return this.categories[maxIndex];
    }
}
