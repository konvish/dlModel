package cn.sibat.gui.java;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

public class DrawSee extends JFrame {

    private static final int sx = 50; //小方格起始横坐标
    private static final int sy = 50; //小方格起始纵坐标
    private static final int w = 40;
    private static final int rw = 400;
    private int px = 0, py = 0;

    private Graphics jg;
    private int cc = 0; // 被选中的方格个数
    private int[][] map; //存放游戏数据的二维数组
    private boolean isEnd = false;
    private Color rectColor = new Color(0xf5f5f5);

    public DrawSee() {
        Container p = getContentPane();
        setBounds(100, 100, 500, 500);
        setVisible(true);
        p.setBackground(rectColor);
        setLayout(null);
        setResizable(false);
        this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        try {
            //创建游戏数据地图
            map = new int[10][10];
            Thread.sleep(500);
        } catch (Exception e) {
            e.printStackTrace();
        }

        //获取专门用于在窗口界面上绘图的对象
        jg = this.getGraphics();

        //绘制游戏区域
        paintComponents(jg);

        //添加游戏鼠标监听事件，当鼠标点击时触发
        this.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (isEnd) {
                    return;
                }

                //获取鼠标点击的那一点x，y坐标
                int x = e.getX(), y = e.getY();

                //计算当前点击的方格是第几个
                int cx = (x - sx) / w, cy = (y - sy) / w;

                //点击的方格处于游戏区域之外，直接返回
                if (cx < 1 || cy < 1 || cx > 8 || cy > 8) {
                    return;
                }

                //选中的方格个数加一
                cc++;

                compare(cx, cy);
            }
        });
    }

    /**
     * 判断二维数组map中的所有元素是否均为0，
     * 只要有一个不为0，返回false，表示游戏还没有结束；否则返回true表示游戏结束
     *
     * @param map 二维数组
     * @return boolean
     */
    public boolean isEnd(int[][] map) {
        for (int[] ms : map) {
            for (int m : ms) {
                if (m != 0)
                    return false;
            }
        }
        return true;
    }

    private void compare(int cx, int cy) {
        /**
         * 如果cc是1，表示当前一共选中了一个方格，用px，py来记住这个方格的位置
         * 否则，表示现在选中的这个方格要与之前选中的方格比较，决定时候要删除
         */
        if (cc == 1) {
            px = cx;
            py = cy;
            setGrid(cx, cy, Color.LIGHT_GRAY);
        } else {
            //cc大于1，比较两个方格的值是否相同
            removed(map, py, px, cy, cx);
            setGrid(cx, cy, rectColor);
            setGrid(px, py, rectColor);
            cc = 0;
        }

        isEnd = isEnd(map);
        if (isEnd) {
            jg.setColor(Color.RED);
            jg.setFont(new Font("Arial", 0, 62));
            jg.drawString("Game Over!", 100, 220);
        }
    }

    private void removed(int[][] map, int py, int px, int cy, int cx) {
        System.out.println("compare");
    }

    public void paintComponents(Graphics g) {
        try {
            //设置线条颜色
            g.setColor(Color.RED);
            //绘制外层矩形框
            g.drawRect(sx, sy, rw, rw);

            for (int i = 0; i < 10; i++) {
                //绘制第i条竖直线
                g.drawLine(sx + (i * w), sy, sx + (i * w), sy + rw);
                //绘制第i条水平线
                g.drawLine(sx, sy + (i * w), sx + rw, sy + (i * w));

                for (int j = 0; j < 10; j++) {
                    drawString(g, j, i);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void drawString(Graphics g, int x, int y) {
        if (map[x][y] != 0) {
            g.setColor(Color.RED);
            g.setFont(new Font("Arial", 0, 40));
            g.drawString(map[x][y] + "", sx + (y * w) + 5, sy + ((x + 1) * w) - 5);
        }
    }

    /**
     * 制定小方格设置为指定背景颜色
     *
     * @param cx    方格的水平坐标
     * @param cy    方格的垂直坐标
     * @param color 颜色
     */
    private void setGrid(int cx, int cy, Color color) {
        //默认灰色，点击设置为此颜色
        jg.setColor(color);
        // 为该方格填充颜色
        jg.fillRect(sx + (cx * w) + 1, sy + (cy * w) + 1, w - 2, w - 2);
        //重新写数据
        drawString(jg, cy, cx);
    }

    public static void main(String[] args) {
        new DrawSee();
    }
}
