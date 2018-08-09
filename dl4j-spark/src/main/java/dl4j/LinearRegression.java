package dl4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class LinearRegression {
    private double learningRate = 0.1;
    private double k;
    private double b;

    public LinearRegression(double learningRate, double k, double b) {
        this.learningRate = learningRate;
        this.k = k;
        this.b = b;
    }

    public LinearRegression(double k, double b) {
        this.k = k;
        this.b = b;
    }

    public double fitBGD(INDArray trainData, INDArray labelData) {
        INDArray diff = labelData.sub(trainData.mul(k).add(b));
        k = diff.dup().muli(trainData).sumNumber().doubleValue() / trainData.length() * 2.0 * learningRate + k;
        b = diff.sumNumber().doubleValue() / trainData.length() * 2.0 * learningRate + b;
        return Nd4j.sum(diff.muli(diff).div(trainData.length())).getDouble(0);
    }

    public double fitSGD(INDArray trainData, INDArray labelData) {
        double diff = 0.0;
        for (int i = 0; i < trainData.length(); ++i) {
            double label = labelData.getDouble(i);
            double data = trainData.getDouble(i);
            diff = label - (k * data + b);
            k = k + 2 * diff * data * learningRate;
            b = b + 2 * diff * learningRate;
        }
        return diff * diff;
    }

    public double getK() {
        return k;
    }

    public double getB() {
        return b;
    }

    public static void main(String[] args) {
        LinearRegression model = new LinearRegression(0.1, 0.1);
        double k_label = 125.6;
        double b_label = 10.3;
        INDArray data = Nd4j.getRandom().nextDouble(new int[]{1, 1000});
        INDArray label = data.mul(k_label).add(b_label);
        final int iterations = 1000;
        for (int iter = 0; iter < iterations; ++iter) {
            double loss = model.fitBGD(data, label);
            System.out.println(loss);
        }
        System.out.println("k: " + model.getK());
        System.out.println("b: " + model.getB());
    }
}
