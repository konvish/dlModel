package cn.sibat.gui.java;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

public class KeyBoardListen extends JFrame {
    private String s = null;
    private JLabel label;

    private KeyBoardListen() {
        setLayout(new FlowLayout());
        label = new JLabel();
        addKeyListener(new KeyListener() {
            @Override
            public void keyTyped(KeyEvent e) {
                s = "按下的键是Type：" + e.getKeyChar();
                label.setText(s);
                System.out.println(s);
            }

            @Override
            public void keyReleased(KeyEvent e) {
                s = "释放的键是：" + e.getKeyChar();
                label.setText(s);
                System.out.println(s);
            }

            @Override
            public void keyPressed(KeyEvent e) {
                s = "按下的键是Press：" + e.getKeyChar();
                label.setText(s);
                System.out.println(s);
            }
        });
        add(label);
        setBounds(350, 100, 600, 500);
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        setVisible(true);
    }

    public static void main(String[] args) {
        new KeyBoardListen();
    }
}
