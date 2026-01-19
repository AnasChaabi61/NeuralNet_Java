package org.anas;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class Utils {
    public static DataSet getDataSet() throws IOException {

        List<double[]> xList = new ArrayList<>();
        List<int[]> yList = new ArrayList<>();

        Path filePath=Path.of("/home/anas/Documents/Projects/NeuralNet/src/main/resources/Iris.csv");
        try (BufferedReader br = Files.newBufferedReader(filePath)) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.contains("Id")) continue;

                String[] parts = line.split("\\s*,\\s*");
                int target;
                if(parts[5].equals("Iris-setosa")){
                    target=0;
                } else if (parts[5].equals("Iris-versicolor")) {
                    target=1;
                }else {
                    target=2;
                }
                yList.add(new int[]{target});


                xList.add(new double[]{Double.parseDouble(parts[1]), Double.parseDouble(parts[2]), Double.parseDouble(parts[3]),Double.parseDouble(parts[4])});



            }
        }
        INDArray inputs=Nd4j.create(xList.size(),xList.get(0).length);
        INDArray outputs=Nd4j.create(yList.size(),3);
        for (int i=0;i<xList.size();i++){
            inputs.putRow(i,Nd4j.create(xList.get(i)));
            int label=yList.get(i)[0];
            INDArray lables=Nd4j.zeros(1,3);
            lables.putScalar(0,label,1);
            outputs.putRow(i,lables);
        }


        return new DataSet(inputs,outputs);
    }

}
