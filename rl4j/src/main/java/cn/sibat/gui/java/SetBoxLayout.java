package cn.sibat.gui.java;

import javax.swing.*;
import java.awt.*;

public class SetBoxLayout extends JFrame {
    public SetBoxLayout() {
        setLayout(new BoxLayout(getContentPane(), BoxLayout.LINE_AXIS));
        getContentPane().setBackground(Color.green);
        add(new Button("123"));
        add(new Button("123"));
        add(new Button("123"));
        setVisible(true);
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        setBounds(100, 50, 700, 400);
    }

    public static void main(String[] args) {
        new SetBoxLayout();
    }
}
