package cn.sibat.gui.java;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;

public class Mouse extends JFrame {
    private JLabel label;

    public Mouse() {
        super("mouse");
        Container container = getContentPane();
        container.setLayout(new BorderLayout());
        label = new JLabel();
        container.add(label, BorderLayout.SOUTH);
        setSize(300, 300);
        setVisible(true);
        Handler1 handler1 = new Handler1();
        Handler2 handler2 = new Handler2();
        container.addMouseListener(handler1);
        container.addMouseMotionListener(handler2);
    }

    public class Handler1 implements MouseListener {
        public void mouseClicked(MouseEvent e) {
            label.setText("鼠标点击的坐标[" + e.getX() + "," + e.getY() + "]");
        }

        public void mousePressed(MouseEvent e) {
            label.setText("鼠标经过的坐标[" + e.getX() + "," + e.getY() + "]");
        }

        public void mouseReleased(MouseEvent e) {
            label.setText("鼠标释放的坐标[" + e.getX() + "," + e.getY() + "]");
        }

        public void mouseEntered(MouseEvent e) {
            // JOptionPane.showMessageDialog(null, "鼠标进入窗口");
        }

        public void mouseExited(MouseEvent e) {
            label.setText("鼠标在窗口之外");
        }
    }

    public class Handler2 implements MouseMotionListener {
        public void mouseDragged(MouseEvent e) {
            label.setText("鼠标拖拽的坐标[" + e.getX() + "," + e.getY() + "]");
        }

        public void mouseMoved(MouseEvent e) {
            label.setText("鼠标移动的坐标[" + e.getX() + "," + e.getY() + "]");
        }
    }

    public static void main(String[] args) {
        Mouse g = new Mouse();
        g.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    }
}
