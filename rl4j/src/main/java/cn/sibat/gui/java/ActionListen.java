package cn.sibat.gui.java;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class ActionListen extends JFrame {
    JButton buttonOne = new JButton("按钮");
    JButton buttonTwo = new JButton("melody");
    JButton buttonThree = new JButton("soft");

    private ActionListen() {
        setLayout(new FlowLayout());
        add(buttonOne);
        add(buttonTwo);
        add(buttonThree);
        buttonOne.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent arg0) {
                Container container = getContentPane();
                if (arg0.getSource() == buttonOne) { // 获取当前组件的变量名
                    container.setBackground(Color.blue);
                }
            }
        });
        buttonTwo.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent arg0) {
                String str = arg0.getActionCommand(); // 获取监听事件源的名称字符串
                if (str.equals("melody")) {
                    System.exit(0);
                }
            }
        });
        buttonThree.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent arg0) {
                if (arg0.getSource() == buttonThree) {
                    JOptionPane.showMessageDialog(null,
                            "你点击的按钮是" + arg0.getActionCommand());
                    // 面板信息框
                }
            }
        });
        setBounds(350, 100, 600, 500);
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        setVisible(true);
    }

    public static void main(String[] args) {
        new ActionListen();
    }
}
