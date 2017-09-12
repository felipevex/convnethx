package convnethx.type;

@:enum
abstract LayerType(String) from String to String {
    var INPUT = "input";
    var CONV = "conv";
    var FC = "fc";
    var DROPOUT = "dropout";
    var SOFTMAX = "softmax";
    var REGRESSION = "regression";
    var SVM = "svm";
    var RELU = "relu";
    var SIGMOID = "sigmoid";
    var MAXOUT = "maxout";
    var TANH = "tanh";
    var LRN = "lrn";
    var POOL = "pool";
}