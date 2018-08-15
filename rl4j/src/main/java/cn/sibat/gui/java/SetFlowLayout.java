package cn.sibat.gui.java;

import javax.swing.*;
import java.awt.*;

public class SetFlowLayout {
    JFrame frame;
    JButton[] button;

    private SetFlowLayout() {
        frame = new JFrame();
        frame.setLayout(new FlowLayout());
        button = new JButton[3];
        for (int i = 0; i < 3; i++) {
            button[i] = new JButton("" + i);
            frame.add(button[i]);
        }
        frame.setBounds(350, 100, 500, 500);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }

    public static void main(String[] args) {
        new SetFlowLayout();
    }
}
