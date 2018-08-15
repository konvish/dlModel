package cn.sibat.gui.java;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

public class Maze extends JFrame {
    private static final int sx = 50; //小方格起始横坐标
    private static final int sy = 50; //小方格起始纵坐标
    private static final int maze_h = 4; //竖直方格数
    private static final int maze_w = 4; //水平方格数
    private static final int w = 40;
    private String[] action_space = {"u", "d", "l", "r"};

    private Graphics jg;
    private int[] rect = {70, 70};
    private boolean isEnd = false;

    public Maze() {
        setTitle("Maze");
        Container p = getContentPane();
        setBounds(100, 100, 250, 250);
        setVisible(true);
        p.setBackground(Color.WHITE);
        setLayout(null);
        setResizable(false);
        this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        try {
            Thread.sleep(500);
        } catch (Exception e) {
            e.printStackTrace();
        }

        //获取专门用于在窗口界面上绘图的对象
        jg = this.getGraphics();

        //绘制游戏区域
        paintComponents(jg);
    }

    public void step(int action) {
        int[] base_action = {0, 0};
        switch (action) {
            case 0:
                if (rect[1] > w)
                    base_action[1] -= w;
                break;
            case 1:
                if (rect[1] < (maze_h - 1) * w)
                    base_action[1] += w;
                break;
            case 2:
                if (rect[0] < (maze_w - 1) * w)
                    base_action[0] += w;
                break;
            case 3:
                if (rect[0] > w)
                    base_action[0] -= w;
                break;
        }

        update(base_action);

        int[] s_ = rect;
        int[] origin = {20, 20};
        int[] hell1_center = {origin[0] + w * 2, origin[1] + w};
        int[] hell2_center = {origin[0] + w, origin[1] + w * 2};
        int[] oval_center = {origin[0] + w * 2, origin[1] + w * 2};
        int reward = 0;
        boolean done = false;

        if (s_[0] == oval_center[0] && s_[1] == oval_center[1]) {
            reward = 1;
            done = true;
            s_[0] = -1;
        } else if ((s_[0] == hell1_center[0] && s_[1] == hell1_center[1]) || (s_[0] == hell2_center[0] && s_[1] == hell2_center[1])) {
            reward = -1;
            done = true;
            s_[0] = -1;
        }

    }

    public void paintComponents(Graphics g) {
        try {
            //设置线条颜色
            g.setColor(Color.BLACK);
            //绘制外层矩形框
            g.drawRect(sx, sy, maze_h * w, maze_w * w);

            for (int i = 0; i < maze_w; i++) {
                //绘制第i条竖直线
                g.drawLine(sx + (i * w), sy, sx + (i * w), sy + maze_w * w);
                //绘制第i条水平线
                g.drawLine(sx, sy + (i * w), sx + maze_h * w, sy + (i * w));
            }

            int[] origin = {70, 70};

            //hell1
            int[] hell1_center = {origin[0] + w * 2, origin[1] + w};
            g.drawRect(hell1_center[0] - 15, hell1_center[1] - 15, 30, 30);
            g.fillRect(hell1_center[0] - 15, hell1_center[1] - 15, 30, 30);

            //he112
            int[] hell2_center = {origin[0] + w, origin[1] + w * 2};
            g.drawRect(hell2_center[0] - 15, hell2_center[1] - 15, 30, 30);
            g.fillRect(hell2_center[0] - 15, hell2_center[1] - 15, 30, 30);

            g.setColor(Color.YELLOW);
            //oval
            int[] oval_center = {origin[0] + w * 2, origin[1] + w * 2};
            g.drawOval(oval_center[0] - 15, oval_center[1] - 15, 30, 30);
            g.fillOval(oval_center[0] - 15, oval_center[1] - 15, 30, 30);

            g.setColor(Color.RED);
            //agent
            g.drawRect(rect[0] - 15, rect[1] - 15, 30, 30);
            g.fillRect(rect[0] - 15, rect[1] - 15, 30, 30);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void reset() {
        jg.clearRect(rect[0] - 15, rect[1] - 15, 31, 31);
        rect[0] = 70;
        rect[1] = 70;
        paintComponents(jg);
    }

    private void update(int[] move) {
        jg.clearRect(rect[0] - 15, rect[1] - 15, 31, 31);
        rect[0] += move[0];
        rect[1] += move[1];
        paintComponents(jg);
    }

    public static void main(String[] args) {
        Maze maze = new Maze();
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
