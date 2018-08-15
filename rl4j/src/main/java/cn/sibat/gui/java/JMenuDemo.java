package cn.sibat.gui.java;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class JMenuDemo extends JFrame {

    private JMenuDemo() throws HeadlessException {
        setTitle("记事本");
        setPosition();
        setBounds(300, 200, 500, 300);
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        setVisible(true);
    }

    private void setPosition() {
        JMenuBar bar = new JMenuBar();
        JMenu file = new JMenu("文件");
        JMenu edit = new JMenu("编辑");
        JMenu help = new JMenu("帮助");
        JMenu form = new JMenu("格式");

        //文件下的操作
        JMenuItem newCreate = new JMenuItem("新建");
        JMenuItem open = new JMenuItem("打开");
        JMenuItem save = new JMenuItem("保存");
        JMenuItem saveAs = new JMenuItem("另存为");
        JMenuItem exit = new JMenuItem("退出");
        file.add(newCreate);
        file.add(open);
        file.add(save);
        file.add(saveAs);
        file.addSeparator();
        file.add(exit);

        //格式下的操作
        JCheckBoxMenuItem binary = new JCheckBoxMenuItem("二进制");
        JMenuItem font = new JMenuItem("字体");
        form.add(binary);
        form.add(font);

        //help的操作
        JMenuItem aboutNotepad = new JMenuItem("about notepad");
        help.add(aboutNotepad);

        //创建点击动作
        newCreate.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                new JMenuDemo();
                JOptionPane.showMessageDialog(null, "新建一个窗口");
            }
        });

        //关闭动作
        exit.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                System.exit(0);
            }
        });

        bar.add(file);
        bar.add(edit);
        bar.add(form);
        bar.add(help);
        add(bar, "North");
        JTextArea area = new JTextArea("", 20, 20);
        add(area, "Center");
    }

    public static void main(String[] args) {
        new JMenuDemo();
    }
}
