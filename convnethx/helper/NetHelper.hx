package convnethx.helper;

import convnethx.layer.dotproduct.LayerFullyConn;
import convnethx.type.LayerType;
import convnethx.layer.model.LayerOption;

class NetHelper {

    public static function createLayers(options:Array<LayerOption>):Array<Layer> {

        var result:Array<Layer> = [];

        options = desugar(options);

        var prevLayer:Layer = null;

        for (option in options) {

            if (prevLayer != null) {
                option.in_sx = prevLayer.out_sx;
                option.in_sy = prevLayer.out_sy;
                option.in_depth = prevLayer.out_depth;
            }

            var currentLayer:Layer = switch(option.layer_type) {
                case LayerType.INPUT : new LayerInput(option);
                case LayerType.FC : new LayerFullyConn(option);
                case _ : null;
            }

            if (currentLayer != null) {
                result.push(currentLayer);

                prevLayer = currentLayer;
            }
        }

        return result;
    }

    private static function desugar(options:Array<LayerOption>):Array<LayerOption> {

        var result:Array<LayerOption> = [];

        for (option in options) {

            switch (option.layer_type) {

                case LayerType.SOFTMAX | LayerType.SVM : {
                    // add an fc layer here, there is no reason the user should
                    // have to worry about this and we almost always want to
                    result.push(
                        LayerOptionHelper.createFC(option.num_classes)
                    );
                }

                case LayerType.REGRESSION : {
                    // add an fc layer here, there is no reason the user should
                    // have to worry about this and we almost always want to

                    result.push(
                        LayerOptionHelper.createFC(option.num_neurons)
                    );
                }

                case LayerType.FC | LayerType.CONV : {
                    if (option.bias_pref == null) {
                        option.bias_pref = 0;

                        if (option.activation != null && option.activation == LayerType.RELU) {
                            // relus like a bit of positive bias to get gradients early
                            // otherwise it's technically possible that a relu unit will never turn on (by chance)
                            // and will never get any gradient and never contribute any computation. Dead relu.

                            option.bias_pref = 0.1;
                        }
                    }
                }

                case _ : {
                    //
                }
            }

            result.push(option);

            if (option.activation != null) {

                switch (option.activation) {

                    case LayerType.RELU : {
                        result.push(LayerOptionHelper.createRelu());
                    }

                    case LayerType.SIGMOID : {
                        result.push(LayerOptionHelper.createSigmoid());
                    }

                    case LayerType.TANH : {
                        result.push(LayerOptionHelper.createTANH());
                    }

                    case LayerType.MAXOUT : {
                        result.push(LayerOptionHelper.createMaxOut(option.group_size == null ? 2 : option.group_size));
                    }

                    case _ : {
                        throw 'ERROR unsupported activation ${option.activation}';
                    }
                 }
            }

            if (option.drop_prob != null && option.layer_type != LayerType.DROPOUT) {
                result.push(LayerOptionHelper.createDropOut(option.drop_prob));
            }
        }


        return result;
    }

}
