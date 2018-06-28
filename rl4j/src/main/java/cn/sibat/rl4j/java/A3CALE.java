package cn.sibat.rl4j.java;

import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscrete;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscreteConv;
import org.deeplearning4j.rl4j.mdp.ale.ALEMDP;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraphStdConv;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.learning.config.Adam;

import java.io.IOException;

public class A3CALE {
    public static HistoryProcessor.Configuration ALE_HP = new HistoryProcessor.Configuration(
            4,
            84,
            110,
            84,
            84,
            0,
            0,
            4);

    public static A3CDiscrete.A3CConfiguration ALE_A3C =
            new A3CDiscrete.A3CConfiguration(
                    123,
                    10000,
                    8000000,
                    8,
                    32,
                    500,
                    0.1,
                    0.99,
                    10.0
            );

    public static final ActorCriticFactoryCompGraphStdConv.Configuration ALE_NET_A3C =
            new ActorCriticFactoryCompGraphStdConv.Configuration(
                    0.000,
                    new Adam(0.00025),
                    null,
                    false
            );

    public static void main(String[] args) throws IOException {
        DataManager manager = new DataManager(true);
        ALEMDP mdp = null;

        try {
            mdp = new ALEMDP("pong.bin");
        } catch (Exception e) {
            System.out.println("To run this by ale-platform.");
        }

        A3CDiscreteConv<ALEMDP.GameScreen> a3c = new A3CDiscreteConv<ALEMDP.GameScreen>(mdp, ALE_NET_A3C, ALE_HP, ALE_A3C, manager);

        a3c.train();

        a3c.getPolicy().save("ale-a3c.model");

        mdp.close();
    }
}