package com.example;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.FileWriter;
import java.lang.management.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

import org.json.JSONArray;
import org.json.JSONObject;
import java.io.File;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryUsage;
import java.text.SimpleDateFormat;
import java.time.Duration;
import java.time.Instant;

public class HardNetworkSolution {

    public static void main(String[] args) throws Exception {
        // Create the network
        MultiLayerNetwork model = createHardNetwork();

        // Measure training performance
        System.out.println("Starting training performance measurement...");
        TrainingResult trainingResult = measureTrainingPerformance(model);

        File imageFile = new File("träningsbilder/testSample/img_1.jpg");

        // Add explicit warm-up phase
        System.out.println("Performing warm-up iterations...");
        for (int i = 0; i < 10; i++) {
            warmupIteration(model, imageFile);
        }
        System.out.println("Warm-up complete, starting benchmark...");

        // Force garbage collection and wait for it to finish
        System.gc();
        Thread.sleep(500);

        int runs = 5;
        List<InferenceResult> inferenceResults = new ArrayList<>();

        for (int i = 0; i < runs; i++) {
            System.out.printf("\n--- Test Run %d ---\n", i + 1);
            InferenceResult result = testNetwork(model, imageFile);
            inferenceResults.add(result);
        }

        // Aggregate metrics for inference
        double avgTime = average(inferenceResults.stream().mapToDouble(r -> r.timeMs).toArray());
        double stdDevTime = stdDev(inferenceResults.stream().mapToDouble(r -> r.timeMs).toArray());

        double avgMemory = average(inferenceResults.stream().mapToDouble(r -> r.memoryUsedMb).toArray());
        double stdDevMemory = stdDev(inferenceResults.stream().mapToDouble(r -> r.memoryUsedMb).toArray());

        double avgPeakMemory = average(inferenceResults.stream().mapToDouble(r -> r.peakMemoryMb).toArray());
        double stdDevPeakMemory = stdDev(inferenceResults.stream().mapToDouble(r -> r.peakMemoryMb).toArray());

        int commonPrediction = mode(inferenceResults.stream().mapToInt(r -> r.prediction).toArray());

        System.out.println("\n===== Average Inference Results After 5 Runs =====");
        System.out.println("Most common prediction: " + commonPrediction);
        System.out.printf("Average execution time: %.2f ms (±%.2f)\n", avgTime, stdDevTime);
        System.out.printf("Average memory used: %.2f MB (±%.2f)\n", avgMemory, stdDevMemory);
        System.out.printf("Average peak memory: %.2f MB (±%.2f)\n", avgPeakMemory, stdDevPeakMemory);

        // Print training performance results
        System.out.println("\n===== Training Performance Results =====");
        System.out.printf("Training time: %.2f ms (%.2f seconds)\n", trainingResult.timeMs, trainingResult.timeMs / 1000);
        System.out.printf("Training memory used: %.2f MB\n", trainingResult.memoryUsedMb);
        System.out.printf("Training peak memory: %.2f MB\n", trainingResult.peakMemoryMb);
        System.out.printf("Training CPU time: %.2f ms\n", trainingResult.cpuTimeMs);
        System.out.printf("Final model accuracy: %.2f%%\n", trainingResult.finalAccuracy * 100);

        // Save combined results to JSON
        saveToJson(inferenceResults, trainingResult, avgTime, stdDevTime, avgMemory,
                stdDevMemory, avgPeakMemory, stdDevPeakMemory, commonPrediction);

        ModelSerializer.writeModel(model, "trained-hard-model.zip", true);
    }

    static class MemoryStats {
        long timestamp;
        long heapUsed;
        long heapCommitted;
        long nonHeapUsed;
        long nonHeapCommitted;
        long[] gcCounts;
        long[] gcTimes;

        MemoryStats() {
            MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
            MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
            MemoryUsage nonHeapUsage = memoryBean.getNonHeapMemoryUsage();

            this.timestamp = System.currentTimeMillis();
            this.heapUsed = heapUsage.getUsed();
            this.heapCommitted = heapUsage.getCommitted();
            this.nonHeapUsed = nonHeapUsage.getUsed();
            this.nonHeapCommitted = nonHeapUsage.getCommitted();

            List<GarbageCollectorMXBean> gcBeans = ManagementFactory.getGarbageCollectorMXBeans();
            this.gcCounts = new long[gcBeans.size()];
            this.gcTimes = new long[gcBeans.size()];

            for (int i = 0; i < gcBeans.size(); i++) {
                GarbageCollectorMXBean gcBean = gcBeans.get(i);
                this.gcCounts[i] = gcBean.getCollectionCount();
                this.gcTimes[i] = gcBean.getCollectionTime();
            }
        }

        JSONObject toJson() {
            JSONObject json = new JSONObject();
            json.put("timestamp", this.timestamp);
            json.put("heap_used_bytes", this.heapUsed);
            json.put("heap_committed_bytes", this.heapCommitted);
            json.put("non_heap_used_bytes", this.nonHeapUsed);
            json.put("non_heap_committed_bytes", this.nonHeapCommitted);
            json.put("heap_used_mb", this.heapUsed / 1024.0 / 1024.0);
            json.put("non_heap_used_mb", this.nonHeapUsed / 1024.0 / 1024.0);

            JSONArray gcCountsArray = new JSONArray();
            JSONArray gcTimesArray = new JSONArray();
            for (int i = 0; i < gcCounts.length; i++) {
                gcCountsArray.put(gcCounts[i]);
                gcTimesArray.put(gcTimes[i]);
            }

            json.put("gc_counts", gcCountsArray);
            json.put("gc_times_ms", gcTimesArray);
            return json;
        }
    }

    static class MemoryTracker implements AutoCloseable {
        private final List<MemoryStats> memoryStatsList = Collections.synchronizedList(new ArrayList<>());
        private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);
        private ScheduledFuture<?> task;
        private AtomicBoolean isRunning = new AtomicBoolean(false);
        private long intervalMs;
        private Instant startTime;
        private Instant endTime;
        private long peakHeapUsage = 0;
        private long peakNonHeapUsage = 0;

        public MemoryTracker(long intervalMs) {
            this.intervalMs = intervalMs;
        }

        public void start() {
            if (isRunning.compareAndSet(false, true)) {
                startTime = Instant.now();
                task = scheduler.scheduleAtFixedRate(() -> {
                    MemoryStats stats = new MemoryStats();
                    memoryStatsList.add(stats);

                    // Track peak memory usage
                    peakHeapUsage = Math.max(peakHeapUsage, stats.heapUsed);
                    peakNonHeapUsage = Math.max(peakNonHeapUsage, stats.nonHeapUsed);
                }, 0, intervalMs, TimeUnit.MILLISECONDS);
            }
        }

        public void stop() {
            if (isRunning.compareAndSet(true, false)) {
                endTime = Instant.now();
                if (task != null) {
                    task.cancel(false);
                }
            }
        }

        public long getPeakHeapUsageBytes() {
            return peakHeapUsage;
        }

        public double getPeakHeapUsageMb() {
            return peakHeapUsage / 1024.0 / 1024.0;
        }

        public long getPeakNonHeapUsageBytes() {
            return peakNonHeapUsage;
        }

        public double getPeakNonHeapUsageMb() {
            return peakNonHeapUsage / 1024.0 / 1024.0;
        }

        public double getTotalPeakMemoryMb() {
            return (peakHeapUsage + peakNonHeapUsage) / 1024.0 / 1024.0;
        }

        public List<MemoryStats> getMemoryStats() {
            return new ArrayList<>(memoryStatsList);
        }

        public JSONObject getSummary() {
            JSONObject summary = new JSONObject();

            if (startTime != null && endTime != null) {
                long durationMs = Duration.between(startTime, endTime).toMillis();
                summary.put("duration_ms", durationMs);
            }

            summary.put("sample_count", memoryStatsList.size());
            summary.put("sample_interval_ms", intervalMs);
            summary.put("peak_heap_mb", getPeakHeapUsageMb());
            summary.put("peak_non_heap_mb", getPeakNonHeapUsageMb());
            summary.put("peak_total_mb", getTotalPeakMemoryMb());

            // Calculate GC activity
            if (!memoryStatsList.isEmpty()) {
                MemoryStats first = memoryStatsList.get(0);
                MemoryStats last = memoryStatsList.get(memoryStatsList.size() - 1);

                if (first.gcCounts.length == last.gcCounts.length) {
                    long totalGcCount = 0;
                    long totalGcTimeMs = 0;

                    for (int i = 0; i < first.gcCounts.length; i++) {
                        totalGcCount += (last.gcCounts[i] - first.gcCounts[i]);
                        totalGcTimeMs += (last.gcTimes[i] - first.gcTimes[i]);
                    }

                    summary.put("gc_count", totalGcCount);
                    summary.put("gc_time_ms", totalGcTimeMs);
                }
            }

            return summary;
        }

        public JSONArray getTimeSeriesData() {
            JSONArray timeSeriesArray = new JSONArray();

            if (!memoryStatsList.isEmpty()) {
                long baseTime = memoryStatsList.get(0).timestamp;

                for (MemoryStats stats : memoryStatsList) {
                    JSONObject dataPoint = new JSONObject();
                    dataPoint.put("time_ms", stats.timestamp - baseTime);
                    dataPoint.put("heap_mb", stats.heapUsed / 1024.0 / 1024.0);
                    dataPoint.put("non_heap_mb", stats.nonHeapUsed / 1024.0 / 1024.0);
                    timeSeriesArray.put(dataPoint);
                }
            }

            return timeSeriesArray;
        }

        @Override
        public void close() {
            stop();
            scheduler.shutdown();
            try {
                if (!scheduler.awaitTermination(500, TimeUnit.MILLISECONDS)) {
                    scheduler.shutdownNow();
                }
            } catch (InterruptedException e) {
                scheduler.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    // Separate classes for training and inference results
    static class InferenceResult {
        int prediction;
        double timeMs;
        double memoryUsedMb;
        double cpuTimeMs;
        double peakMemoryMb;
        JSONObject memoryProfile;

        InferenceResult(int p, double t, double m, double c, double pm, JSONObject mp) {
            prediction = p;
            timeMs = t;
            memoryUsedMb = m;
            cpuTimeMs = c;
            peakMemoryMb = pm;
            memoryProfile = mp;
        }
    }

    static class TrainingResult {
        double timeMs;
        double memoryUsedMb;
        double cpuTimeMs;
        double peakMemoryMb;
        double finalAccuracy;
        JSONObject memoryProfile;

        TrainingResult(double t, double m, double c, double pm, double acc, JSONObject mp) {
            timeMs = t;
            memoryUsedMb = m;
            cpuTimeMs = c;
            peakMemoryMb = pm;
            finalAccuracy = acc;
            memoryProfile = mp;
        }
    }

    static InferenceResult testNetwork(MultiLayerNetwork model, File imageFile) throws Exception {
        System.gc();
        Thread.sleep(100);

        long startMem = getUsedMemory();
        ThreadMXBean threadBean = ManagementFactory.getThreadMXBean();
        boolean cpuTimeEnabled = threadBean.isCurrentThreadCpuTimeSupported();
        long startCpuTime = cpuTimeEnabled ? threadBean.getCurrentThreadCpuTime() : 0;

        // Start memory tracking with 10ms intervals
        try (MemoryTracker memoryTracker = new MemoryTracker(10)) {
            memoryTracker.start();
            long startTime = System.nanoTime();

            // Perform the actual inference
            NativeImageLoader loader = new NativeImageLoader(28, 28, 1);
            INDArray image = loader.asMatrix(imageFile);
            new ImagePreProcessingScaler(0, 1).transform(image);
            image = image.reshape(1, 1, 28, 28);  // Changed for CNN input format
            INDArray output = model.output(image);

            int prediction = Nd4j.argMax(output, 1).getInt(0);

            long endTime = System.nanoTime();
            memoryTracker.stop();

            long endCpuTime = cpuTimeEnabled ? threadBean.getCurrentThreadCpuTime() : 0;
            long endMem = getUsedMemory();

            double timeMs = (endTime - startTime) / 1e6;
            double memoryUsedMb = (endMem - startMem) / 1024.0 / 1024.0;
            double cpuTimeMs = cpuTimeEnabled ? (endCpuTime - startCpuTime) / 1e6 : -1;
            double peakMemoryMb = memoryTracker.getTotalPeakMemoryMb();

            JSONObject memoryProfile = new JSONObject();
            memoryProfile.put("summary", memoryTracker.getSummary());

            System.out.println("Prediction: " + prediction);
            System.out.printf("Execution time (wall): %.2f ms\n", timeMs);
            System.out.printf("CPU time: %.2f ms\n", cpuTimeMs);
            System.out.printf("Memory used (end-start): %.2f MB\n", memoryUsedMb);
            System.out.printf("Peak memory used: %.2f MB\n", peakMemoryMb);
            System.out.printf("Peak heap memory: %.2f MB\n", memoryTracker.getPeakHeapUsageMb());
            System.out.printf("Peak non-heap memory: %.2f MB\n", memoryTracker.getPeakNonHeapUsageMb());

            return new InferenceResult(prediction, timeMs, memoryUsedMb, cpuTimeMs, peakMemoryMb, memoryProfile);
        }
    }

    // New method to measure training performance
    static TrainingResult measureTrainingPerformance(MultiLayerNetwork model) throws Exception {
        System.gc();
        Thread.sleep(100);

        long startMem = getUsedMemory();
        ThreadMXBean threadBean = ManagementFactory.getThreadMXBean();
        boolean cpuTimeEnabled = threadBean.isCurrentThreadCpuTimeSupported();
        long startCpuTime = cpuTimeEnabled ? threadBean.getCurrentThreadCpuTime() : 0;

        double finalAccuracy = 0.0;

        // Start memory tracking with 500ms intervals (longer interval for training)
        try (MemoryTracker memoryTracker = new MemoryTracker(500)) {
            memoryTracker.start();
            long startTime = System.nanoTime();

            // Training configuration
            int batchSize = 256;
            int numEpochs = 10;

            // Load training data
            DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);

            // Set up a listener to track progress
            model.setListeners(new ScoreIterationListener(100));

            System.out.println("Starting training...");
            for (int i = 0; i < numEpochs; i++) {
                System.out.println("Epoch " + (i+1) + "/" + numEpochs);
                model.fit(mnistTrain);

                // Reset the iterator for the next epoch
                mnistTrain.reset();
            }
            System.out.println("Training complete!");

            // Evaluate the model on test data
            DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, 12345);
            Evaluation eval = model.evaluate(mnistTest);
            finalAccuracy = eval.accuracy();
            System.out.println(eval.stats());

            long endTime = System.nanoTime();
            memoryTracker.stop();

            long endCpuTime = cpuTimeEnabled ? threadBean.getCurrentThreadCpuTime() : 0;
            long endMem = getUsedMemory();

            double timeMs = (endTime - startTime) / 1e6;
            double memoryUsedMb = (endMem - startMem) / 1024.0 / 1024.0;
            double cpuTimeMs = cpuTimeEnabled ? (endCpuTime - startCpuTime) / 1e6 : -1;
            double peakMemoryMb = memoryTracker.getTotalPeakMemoryMb();


            JSONObject memoryProfile = new JSONObject();
            memoryProfile.put("summary", memoryTracker.getSummary());


            System.out.println("\n===== Training Performance =====");
            System.out.printf("Training time: %.2f ms (%.2f seconds)\n", timeMs, timeMs / 1000);
            System.out.printf("CPU time: %.2f ms\n", cpuTimeMs);
            System.out.printf("Memory used (end-start): %.2f MB\n", memoryUsedMb);
            System.out.printf("Peak memory used: %.2f MB\n", peakMemoryMb);
            System.out.printf("Peak heap memory: %.2f MB\n", memoryTracker.getPeakHeapUsageMb());
            System.out.printf("Peak non-heap memory: %.2f MB\n", memoryTracker.getPeakNonHeapUsageMb());
            System.out.printf("Final model accuracy: %.2f%%\n", finalAccuracy * 100);

            return new TrainingResult(timeMs, memoryUsedMb, cpuTimeMs, peakMemoryMb, finalAccuracy, memoryProfile);
        }
    }

    static void saveToJson(List<InferenceResult> inferenceResults, TrainingResult trainingResult,
                           double avgTime, double stdDevTime, double avgMemory, double stdDevMemory,
                           double avgPeakMemory, double stdDevPeakMemory, int commonPrediction) throws Exception {

        JSONObject root = new JSONObject();
        JSONArray inferenceRunsArray = new JSONArray();

        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ");
        root.put("timestamp", dateFormat.format(new Date()));
        root.put("system_info", getSystemInfo());

        // Training results - exclude timeseries
        JSONObject trainingObject = new JSONObject();
        trainingObject.put("execution_time_ms", trainingResult.timeMs);
        trainingObject.put("execution_time_sec", trainingResult.timeMs / 1000);
        trainingObject.put("cpu_time_ms", trainingResult.cpuTimeMs);
        trainingObject.put("memory_used_mb", trainingResult.memoryUsedMb);
        trainingObject.put("peak_memory_mb", trainingResult.peakMemoryMb);
        trainingObject.put("final_accuracy", trainingResult.finalAccuracy);
        trainingObject.put("memory_profile", trainingResult.memoryProfile.getJSONObject("summary")); // Only summary

        root.put("training", trainingObject);

        // Inference results - exclude timeseries
        for (int i = 0; i < inferenceResults.size(); i++) {
            InferenceResult r = inferenceResults.get(i);
            JSONObject runObj = new JSONObject();
            runObj.put("run", i + 1);
            runObj.put("prediction", r.prediction);
            runObj.put("execution_time_ms", r.timeMs);
            runObj.put("cpu_time_ms", r.cpuTimeMs);
            runObj.put("memory_used_mb", r.memoryUsedMb);
            runObj.put("peak_memory_mb", r.peakMemoryMb);
            runObj.put("memory_profile", r.memoryProfile.getJSONObject("summary")); // Only summary
            inferenceRunsArray.put(runObj);
        }

        root.put("inference_runs", inferenceRunsArray);
        root.put("inference_summary", new JSONObject()
                .put("average_execution_time_ms", avgTime)
                .put("std_dev_execution_time_ms", stdDevTime)
                .put("average_memory_used_mb", avgMemory)
                .put("std_dev_memory_used_mb", stdDevMemory)
                .put("average_peak_memory_mb", avgPeakMemory)
                .put("std_dev_peak_memory_mb", stdDevPeakMemory)
                .put("most_common_prediction", commonPrediction));

        try (FileWriter file = new FileWriter("hard_memory_results256.json")) {
            file.write(root.toString(4)); // pretty print
            System.out.println("Results exported to hard_memory_results.json");
        }
    }

    static JSONObject getSystemInfo() {
        JSONObject sysInfo = new JSONObject();

        Runtime runtime = Runtime.getRuntime();
        OperatingSystemMXBean osBean = ManagementFactory.getOperatingSystemMXBean();

        sysInfo.put("available_processors", runtime.availableProcessors());
        sysInfo.put("max_memory_mb", runtime.maxMemory() / (1024.0 * 1024.0));
        sysInfo.put("total_memory_mb", runtime.totalMemory() / (1024.0 * 1024.0));
        sysInfo.put("os_name", System.getProperty("os.name"));
        sysInfo.put("os_version", System.getProperty("os.version"));
        sysInfo.put("os_arch", System.getProperty("os.arch"));
        sysInfo.put("java_version", System.getProperty("java.version"));

        return sysInfo;
    }

    static MultiLayerNetwork createHardNetwork() {
        // Complex CNN with multiple convolutional layers
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
                .layer(new DenseLayer.Builder().nOut(256)
                        .activation(Activation.RELU).build())
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

    static long getUsedMemory() {
        return ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed();
    }

    static double average(double[] values) {
        return Arrays.stream(values).average().orElse(0);
    }

    static double stdDev(double[] values) {
        double mean = average(values);
        return Math.sqrt(Arrays.stream(values).map(v -> Math.pow(v - mean, 2)).average().orElse(0));
    }

    static int mode(int[] values) {
        Map<Integer, Integer> freq = new HashMap<>();
        for (int v : values) freq.put(v, freq.getOrDefault(v, 0) + 1);
        return Collections.max(freq.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    static void warmupIteration(MultiLayerNetwork model, File imageFile) throws Exception {
        NativeImageLoader loader = new NativeImageLoader(28, 28, 1);
        INDArray image = loader.asMatrix(imageFile);
        new ImagePreProcessingScaler(0, 1).transform(image);
        image = image.reshape(1, 1, 28, 28);  // Changed for CNN input format
        model.output(image);
    }
}