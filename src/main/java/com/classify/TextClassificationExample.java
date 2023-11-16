package com.classify;

import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.guava.io.Files;

import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @author 林湖
 * @version 1.0
 */
public class TextClassificationExample {

    public static final String WORD_VECTORS_PATH = "data/word2vec.bin";
    public static final String TRAIN_FILE_PATH = "data/train.txt";
    public int labelSize = 0;

    public static void main(String[] args) throws IOException {
        TextClassificationExample ccn = new TextClassificationExample();

        int batchSize = 64;     //批次大小
        int vectorSize = 200;   //词向量的维度
        int numEpochs = 100;    //训练轮次
        int truncateReviewsToLength = 50;   //截断文本长度
        int cnnLayerFeatureMaps = 100;      //CNN层的特征图数目
        PoolingType globalPoolingType = PoolingType.MAX;
        Random rng = new Random(12345);
        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        System.out.println("加载词向量并准备遍历训练数据集");
        WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(new File(WORD_VECTORS_PATH));

        //DataSetIterator evalIter = getDataSetIterator(ccn, wordVectors, batchSize, truncateReviewsToLength, rng, "G:/eval.txt");
        DataSetIterator trainIter = getDataSetIterator(ccn, wordVectors, batchSize, truncateReviewsToLength, rng, TRAIN_FILE_PATH);

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.SINGLE)
                .inferenceWorkspaceMode(WorkspaceMode.SINGLE)
                .weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU)
                .updater(Updater.ADAM)
                .convolutionMode(ConvolutionMode.Same)
                .l2(0.0001)
                .graphBuilder()
                .addInputs("input")
                .addLayer("cnn3", new ConvolutionLayer.Builder()
                        .kernelSize(3,vectorSize)
                        .stride(1,vectorSize)
                        .nIn(1)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addLayer("cnn4", new ConvolutionLayer.Builder()
                        .kernelSize(4,vectorSize)
                        .stride(1,vectorSize)
                        .nIn(1)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addLayer("cnn5", new ConvolutionLayer.Builder()
                        .kernelSize(5,vectorSize)
                        .stride(1,vectorSize)
                        .nIn(1)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")
                .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                        .poolingType(globalPoolingType)
                        .dropOut(0.5)
                        .build(), "merge")
                .addLayer("out", new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3*cnnLayerFeatureMaps)
                        .nOut(ccn.labelSize)
                        .build(), "globalPool")
                .setOutputs("out")
                .build();

        ComputationGraph net = new ComputationGraph(config);
        net.init();

        System.out.println("神经网络中每一层的名称与参数数量:");
        for(Layer l : net.getLayers()) {
            System.out.println("\t" + l.conf().getLayer().getLayerName() + "\t" + l.numParams());
        }

        System.out.println("开始训练");
        net.setListeners(new ScoreIterationListener(100));
        for (int i = 0; i < numEpochs; i++) {
            net.fit(trainIter);
//            System.out.println("Epoch " + i + " complete. Starting evaluation:");
//            Evaluation evaluation = net.evaluate(evalIter);
//            System.out.println(evaluation.stats());
        }

        ModelSerializer.writeModel(net, "data\\cnn.model", true);
        List<String> labels = trainIter.getLabels();

        List<String> tests = Files.readLines(new File("data\\test.txt"), StandardCharsets.UTF_8);
        StringBuilder results = new StringBuilder();

        // 输出测试结果
        for (String str : tests) {
            String productName = str.substring(str.indexOf(" ") + 1);
            String type = str.substring(0, str.indexOf(" "));

            INDArray featuresFirstNegative = ((CnnSentenceDataSetIterator) trainIter).loadSingleSentence(productName);
            INDArray predictionsFirstNegative = net.outputSingle(featuresFirstNegative);
            Map<String, Double> values = new HashMap<>();
            for (int i = 0; i < labels.size(); i++) {
                values.put(labels.get(i), predictionsFirstNegative.getDouble(i));
            }
            Map<String, Double> sortedMap = values.entrySet().stream()
                    .sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))  // 根据值降序排序
                    .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (oldValue, newValue) -> oldValue, LinkedHashMap::new));

            results.append("\nType = ").append(type).append(", ProductName = ").append(productName).append("\nCNN 的分析结果 : [");
            for (String key : sortedMap.keySet()) {
                try {
                    BigDecimal b = BigDecimal.valueOf(sortedMap.get(key));
                    double f1 = b.setScale(6, BigDecimal.ROUND_HALF_UP).doubleValue();
                    results.append(key).append("(").append(f1).append("), ");
                } catch (Exception w) {
                }
            }
            results.append("] \n");
        }
        System.out.println(results);
        System.out.println("程序结束.........");
    }

    public static DataSetIterator getDataSetIterator(TextClassificationExample ccn, WordVectors wordVectors, int minibatchSize,
                                                     int maxSentenceLength, Random rng, String file) throws IOException {
        List<String> examples = Files.readLines(new File(file), Charset.forName("utf-8"));
        MySentenceProvider sp = new MySentenceProvider(examples);
        ccn.labelSize = sp.allLabels().size();
        return new CnnSentenceDataSetIterator.Builder()
                .sentenceProvider(sp)
                .wordVectors(wordVectors)
                .minibatchSize(minibatchSize)
                .maxSentenceLength(maxSentenceLength)
                .useNormalizedWordVectors(false)
                .build();
    }

}
