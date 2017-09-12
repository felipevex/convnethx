package convnethx.helper;

import convnethx.layer.model.LayerOption;
import convnethx.type.LayerType;

class LayerOptionHelper {

    public inline static function createInput(width:Int, height:Int, depth:Int):LayerOption {
        return {
            layer_type : LayerType.INPUT,
            out_sx : width,
            out_sy : height,
            out_depth : depth
        }
    }

    public inline static function createFC(numNeurons:Int, activation:LayerType = null, bias_pref:Null<Float> = null, l1_decay_mul:Null<Float> = null, l2_decay_mul:Null<Float> = null):LayerOption {
        return {
            layer_type : LayerType.FC,
            activation : activation,
            num_neurons : numNeurons,
            l1_decay_mul : l1_decay_mul,
            l2_decay_mul : l2_decay_mul,
            bias_pref : bias_pref
        }
    }

    public inline static function createRelu():LayerOption {
        return {
            layer_type : LayerType.RELU
        }
    }

    public inline static function createSigmoid():LayerOption {
        return {
            layer_type : LayerType.SIGMOID
        }
    }

    public inline static function createSoftMax(numClasses:Int):LayerOption {
        return {
            layer_type : LayerType.SOFTMAX,
            num_classes : numClasses
        }
    }

    public inline static function createTANH():LayerOption {
        return {
            layer_type : LayerType.TANH
        }
    }

    public inline static function createMaxOut(groupSize:Int):LayerOption {
        return {
            layer_type : LayerType.MAXOUT
        }
    }

    public inline static function createDropOut(dropProb:Float):LayerOption {
        return {
            layer_type : LayerType.DROPOUT,
            drop_prob : dropProb
        }
    }
}
