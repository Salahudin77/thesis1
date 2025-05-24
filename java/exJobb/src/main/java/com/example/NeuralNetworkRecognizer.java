package com.example;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryUsage;

public class NeuralNetworkRecognizer {

    public static void main(String[] args) throws Exception {
        File imageFile = new File("testSample/img_2.jpg"); // ← your 28x28 grayscale image

        testNetwork("Easy", easyNetwork(), imageFile);
        testNetwork("Medium", mediumNetwork(), imageFile);
        testNetwork("Hard", hardNetwork(), imageFile);
    }

    static void testNetwork(String name, MultiLayerNetwork model, File imageFile) throws Exception {
        System.out.println("\n===== Running " + name + " Model =====");

        long startTime = System.nanoTime();
        long startMem = getUsedMemory();

        // Load and preprocess image
        NativeImageLoader loader = new NativeImageLoader(28, 28, 1);
        INDArray image = loader.asMatrix(imageFile);
        new ImagePreProcessingScaler(0, 1).transform(image); // normalize 0–1

        // Reshape the image appropriately for the model
        if (name.equals("Hard")) {
            // For CNN: [batch, channels, height, width]
            image = image.reshape(1, 1, 28, 28);
        } else {
            // For dense networks: [batch, features]
            image = image.reshape(1, 28 * 28);
        }

        // Predict
        INDArray output = model.output(image);
        int prediction = Nd4j.argMax(output, 1).getInt(0);

        long endTime = System.nanoTime();
        long endMem = getUsedMemory();

        System.out.println("Prediction: " + prediction);
        System.out.printf("Execution time: %.2f ms%n", (endTime - startTime) / 1e6);
        System.out.printf("Memory used: %.2f MB%n", (endMem - startMem) / 1024.0 / 1024.0);
    }

    static long getUsedMemory() {
        MemoryUsage heapMemoryUsage = ManagementFactory.getMemoryMXBean().getHeapMemoryUsage();
        return heapMemoryUsage.getUsed();
    }

    static MultiLayerNetwork easyNetwork() {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.001))
                .list()
                .layer(new DenseLayer.Builder().nIn(28 * 28).nOut(64)
                        .activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder()
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .setInputType(InputType.feedForward(28 * 28))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        return model;
    }

    static MultiLayerNetwork mediumNetwork() {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.001))
                .list()
                .layer(new DenseLayer.Builder().nIn(28 * 28).nOut(128)
                        .activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nOut(64)
                        .activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder()
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .setInputType(InputType.feedForward(28 * 28))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        return model;
    }

    static MultiLayerNetwork hardNetwork() {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.001))
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(1).nOut(20)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nOut(50)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder().nOut(256).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nOut(10)
                        .build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        return model;
    }
}
