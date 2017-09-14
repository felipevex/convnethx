package convnethx.helper;

import convnethx.model.LayerOptionValue;
import convnethx.type.LayerType;

class LayerOptionHelper {

    public inline static function createInput(width:Int, height:Int, depth:Int):LayerOptionValue {
        return {
            layer_type : LayerType.INPUT,
            out_sx : width,
            out_sy : height,
            out_depth : depth
        }
    }

    public inline static function createFC(numNeurons:Int, activation:LayerType = null, bias_pref:Null<Float> = null, l1_decay_mul:Null<Float> = null, l2_decay_mul:Null<Float> = null):LayerOptionValue {
        return {
            layer_type : LayerType.FC,
            activation : activation,
            num_neurons : numNeurons,
            l1_decay_mul : l1_decay_mul,
            l2_decay_mul : l2_decay_mul,
            bias_pref : bias_pref
        }
    }

    public inline static function createRelu():LayerOptionValue {
        return {
            layer_type : LayerType.RELU
        }
    }

    public inline static function createSigmoid():LayerOptionValue {
        return {
            layer_type : LayerType.SIGMOID
        }
    }

    public inline static function createSoftmax(numClasses:Int, ?in_sx:Null<Int> = null, ?in_sy:Null<Int> = null, ?in_depth:Null<Int> = null):LayerOptionValue {
        return {
            layer_type : LayerType.SOFTMAX,
            num_classes : numClasses,

            in_depth : in_depth,
            in_sx : in_sx,
            in_sy : in_sy
        }
    }

    public inline static function createSVM(numClasses:Int):LayerOptionValue {
        return {
            layer_type : LayerType.SVM,
            num_classes : numClasses
        }
    }

    public inline static function createTANH(?in_sx:Null<Int> = null, ?in_sy:Null<Int> = null, ?in_depth:Null<Int> = null):LayerOptionValue {
        return {
            layer_type : LayerType.TANH,
            in_depth : in_depth,
            in_sx : in_sx,
            in_sy : in_sy
        }
    }

    public inline static function createMaxOut(groupSize:Int):LayerOptionValue {
        return {
            layer_type : LayerType.MAXOUT,
            group_size : groupSize
        }
    }

    public inline static function createDropOut(dropProb:Float):LayerOptionValue {
        return {
            layer_type : LayerType.DROPOUT,
            drop_prob : dropProb
        }
    }
}
