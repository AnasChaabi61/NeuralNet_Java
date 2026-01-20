package org.anas;


import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;

import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

public class App {
    public static void main(String[] args) throws IOException {
        UIServer uiServer = UIServer.getInstance();

        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        DataSet dataset=Utils.getDataSet();
        dataset.shuffle(123);
        SplitTestAndTrain split = dataset.splitTestAndTrain(0.8);
        DataSet train = split.getTrain();
        DataSet test  = split.getTest();

        DenseLayer layer0= new DenseLayer.Builder()
                .nIn(dataset.getFeatures().columns())
                .nOut(8)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .build();
        OutputLayer layer1= new OutputLayer.Builder()
                .nIn(8)
                .nOut(dataset.getLabels().columns())
                .activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.MCXENT)
                .build();
        MultiLayerConfiguration conf=new NeuralNetConfiguration.Builder()

                .updater(new Adam(0.001))
                .list()
                .layer(0,layer0)
                .layer(1,layer1)
                .build();
        MultiLayerNetwork model= new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(
                new StatsListener(statsStorage, 1),
                new ScoreIterationListener(1)
        );
        for(int epoch=0;epoch<1000;epoch++){
            model.fit(train);

            if (epoch % 25 == 0) {
                INDArray testPred = model.output(test.getFeatures());
                Evaluation eval = new Evaluation(3);
                eval.eval(test.getLabels(),testPred);
                System.out.println(String.format("======epooooch======:%d",epoch));
                System.out.println(eval.stats());
                System.out.println(eval.getConfusionMatrix());
            }

        }

    }
}